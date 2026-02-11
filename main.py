from __future__ import print_function

import os
import time
import pickle
import random
import copy
import argparse
import shutil
import numpy as np
import scipy.linalg


import torch.utils.data
from torch import optim
from torchvision.utils import save_image
import torchvision.transforms as transforms

from vae import *
from tqdm import tqdm
from PIL import Image
from dataloader import get_dataloader
from ignite.metrics import FID, InceptionScore
from utils import interpolate, loss_function, plot_loss_curves


im_size = 128

def train_vae(config):
    batch_size = config['batch_size']
    z_size = config['z_size']
    lr = config['lr']
    train_epoch = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    data_dir = config['data_dir']
    layer_count = config['layer_count']
    kl_weight = config['kl_weight']

    vae = VAE(zsize=z_size, layer_count=layer_count)
    vae.cuda()
    vae.train()
    vae.weight_init(mean=0.0, std=0.02)

    optimizer = optim.Adam(vae.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=1e-5)
    
    checkpoint_dir = ckpt_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    min_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    
    epoch_rec_losses = []
    epoch_kl_losses = []
    epoch_total_losses = []

    print("Loading data...")
    train_loader = get_dataloader(
        data_dir, 
        batch_size=batch_size, 
        im_size=im_size, 
        shuffle=True, 
        num_workers=1,
        pin_memory=True
    )
    print("Train set size:", len(train_loader.dataset))
    num_batches_per_epoch = len(train_loader)
    
    for epoch in range(train_epoch):
        print(f"\nEpoch {epoch+1}/{train_epoch}")
        vae.train()

        # if (epoch + 1) % 8 == 0:
        #     optimizer.param_groups[0]['lr'] /= 4
        #     print(f"Epoch {epoch+1}: Learning rate changed to {optimizer.param_groups[0]['lr']:.6f}")

        rec_loss = 0
        kl_loss = 0
        total_loss = 0

        epoch_start_time = time.time()

        batch_pbar = tqdm(
            train_loader,
            total=num_batches_per_epoch,
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}",
        )

        batch_count = 0
        for x in batch_pbar:
            batch_count += 1
            x = x.cuda()  # Move batch to GPU
            optimizer.zero_grad(set_to_none=True)
            rec, mu, logvar = vae(x)

            loss_rec, loss_kl = loss_function(rec, x, mu, logvar, kl_weight)
            loss = loss_rec + loss_kl
            loss.backward()
            optimizer.step()
            
            rec_loss += loss_rec.item()
            kl_loss += loss_kl.item()
            total_loss += loss.item()

            del rec, mu, logvar, loss_rec, loss_kl, loss
        
        # Calculate epoch statistics
        avg_rec_loss = rec_loss / batch_count if batch_count > 0 else 0
        avg_kl_loss = kl_loss / batch_count if batch_count > 0 else 0
        avg_total_loss = total_loss / batch_count if batch_count > 0 else 0
        
        print(f"Train loss: {avg_total_loss:.5f}")
        print(f"rec_loss: {avg_rec_loss:.3f} kl_loss: {avg_kl_loss:.3f} loss: {avg_total_loss:.3f}")

        if avg_total_loss < min_loss:
            min_loss = avg_total_loss
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(vae.state_dict())
        
        epoch_rec_losses.append(avg_rec_loss)
        epoch_kl_losses.append(avg_kl_loss)
        epoch_total_losses.append(avg_total_loss)
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 100 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'avg_rec_loss': avg_rec_loss,
                'avg_kl_loss': avg_kl_loss,
                'avg_total_loss': avg_total_loss,
                'z_size': z_size,
                'layer_count': layer_count,
            }
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save(checkpoint, checkpoint_path)
    
    plot_loss_curves(
        epoch_rec_losses,
        epoch_kl_losses,
        epoch_total_losses,
        train_epoch,
        ckpt_dir,
    )
    
    # save the best model
    if best_model_state is not None:
        best_checkpoint = {
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': optimizer.state_dict(),
            'min_loss': min_loss,
            'z_size': z_size,
            'layer_count': layer_count,
        }
        # Save best epoch checkpoint
        best_checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{best_epoch}.pth')
        torch.save(best_checkpoint, best_checkpoint_path)
        best_copy_path = os.path.join(checkpoint_dir, 'best.pth')
        shutil.copy2(best_checkpoint_path, best_copy_path)
        print(f"Best checkpoint saved: {best_copy_path} (Epoch {best_epoch}, Loss: {min_loss:.6f})")
    
    return best_epoch, min_loss, best_model_state

def eval_vae(config):
    ckpt_dir = config['ckpt_dir']
    batch_size = config['batch_size']
    data_dir = config['data_dir']
    
    # Load checkpoint - support both best.pth and epoch checkpoints
    ckpt_name = config.get('ckpt_name', 'best.pth')
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}.")
    
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    
    # Handle different checkpoint formats
    # best.pth has 'min_loss', epoch checkpoints have 'avg_total_loss'
    loss_value = checkpoint.get('min_loss') or checkpoint.get('avg_total_loss')
    if loss_value is not None:
        print(f"  Loss: {loss_value:.4f}")
    
    # Create model with same configuration
    z_size = checkpoint.get('z_size', 512)
    layer_count = checkpoint.get('layer_count', 5)
    vae = VAE(zsize=z_size, layer_count=layer_count)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.cuda()
    vae.eval()
    print(f"Model loaded: z_size={z_size}, layer_count={layer_count}")
    
    # Initialize Ignite metrics for FID and IS
    print("Initializing FID and Inception Score metrics...")
    fid_metric = FID(device='cuda')
    is_metric = InceptionScore(device='cuda', output_transform=lambda x: x[0])
    
    # Load data
    print("Loading data...")
    eval_loader = get_dataloader(
        data_dir,
        batch_size=batch_size,
        im_size=im_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True
    )
    print("Eval set size:", len(eval_loader.dataset))
    
    # Create output directories
    os.makedirs('results_rec', exist_ok=True)
    os.makedirs('results_gen', exist_ok=True)
    
    # Limit evaluation to first 500 samples (or fewer if dataset is smaller)
    num_eval_batches = min(500 // batch_size, len(eval_loader))
    
    print(f"\nEvaluating with {num_eval_batches * batch_size} samples (max)...")
    
    # Collect samples for visualization
    sample_real_list = []
    sample_recon_list = []
    sample_gen_list = []
    
    with torch.no_grad():
        eval_pbar = tqdm(eval_loader, desc="Evaluating", unit="batch", total=num_eval_batches)
        for batch_idx, eval_x in enumerate(eval_pbar):
            if batch_idx >= num_eval_batches:
                break
            eval_x = eval_x.cuda()  # Move batch to GPU
            # Reconstruction: encode and decode
            x_rec, _, _ = vae(eval_x)
            if batch_idx == 0:  # Save first batch for visualization
                sample_real_list.append(eval_x[:8].cpu())
                sample_recon_list.append(x_rec[:8].cpu())
            
            # Generate fake images from random latent vectors
            z_fake = torch.randn(eval_x.size(0), z_size, device='cuda').view(-1, z_size, 1, 1)
            fake_imgs = vae.decode(z_fake)
            
            if batch_idx == 0:  # Save first batch for visualization
                sample_gen_list.append(fake_imgs[:8].cpu())
            
            # Interpolate to 299x299 for Inception (using PIL as recommended)
            real_imgs_299 = interpolate(eval_x)
            fake_imgs_299 = interpolate(fake_imgs)
            
            # Update FID metric (expects (fake, real) tuple)
            fid_metric.update((fake_imgs_299.cuda(), real_imgs_299.cuda()))
            
            # Update IS metric (expects only fake images)
            is_metric.update(fake_imgs_299.cuda())
    
    print(f"\n{'='*50}")
    # Compute FID and IS
    print("\nCalculating FID...")
    fid_score = fid_metric.compute()
    print(f"FID Score: {fid_score:.4f} (Lower is Better)")
    
    print("Calculating IS...")
    is_score = is_metric.compute()
    print(f"IS Score: {is_score:.4f} (Higher is Better)")
    print(f"{'='*50}\n")
    
    # Save visualization images
    if sample_real_list:
        # Reconstruction samples
        real_samples = torch.cat(sample_real_list, dim=0)
        recon_samples = torch.cat(sample_recon_list, dim=0)
        resultsample = torch.cat([real_samples, recon_samples]) * 0.5 + 0.5
        save_image(resultsample.view(-1, 3, im_size, im_size),
                'results_rec/eval_sample.png', nrow=8)
        print("Reconstruction samples saved: results_rec/eval_sample.png")
        
        # Generation samples
        gen_samples = torch.cat(sample_gen_list, dim=0)
        resultsample = (gen_samples * 0.5 + 0.5)
        save_image(resultsample.view(-1, 3, im_size, im_size),
                'results_gen/eval_sample.png', nrow=8)
        print("Generation samples saved: results_gen/eval_sample.png")
    
    # Save evaluation results
    result_file = os.path.join(ckpt_dir, 'eval_results.txt')
    with open(result_file, 'w') as f:
        f.write(f"Evaluation Results for {ckpt_name}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Epoch: {checkpoint.get('epoch', 'unknown')}\n")
        # Handle different checkpoint formats
        loss_value = checkpoint.get('min_loss') or checkpoint.get('avg_total_loss')
        if loss_value is not None:
            f.write(f"Loss: {loss_value:.6f}\n")
        else:
            f.write(f"Loss: unknown\n")
        f.write(f"FID Score: {fid_score:.4f} (Lower is Better)\n")
        f.write(f"IS Score: {is_score:.4f} (Higher is Better)\n")
        f.write(f"Number of batches evaluated: {num_eval_batches}\n")
    print(f"Evaluation results saved: {result_file}")
    
    return fid_score, is_score

def main(args):
    # Parse arguments
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    batch_size = args['batch_size']
    z_size = args['z_size']
    lr = args['lr']
    num_epochs = args['num_epochs']
    data_dir = args['data_dir']
    
    config = {
        'ckpt_dir': ckpt_dir,
        'batch_size': batch_size,
        'z_size': z_size,
        'lr': lr,
        'num_epochs': num_epochs,
        'data_dir': data_dir,
        'layer_count': args.get('layer_count', 5),
        'kl_weight': args.get('kl_weight', 0.1),
        'ckpt_name': args.get('ckpt_name', 'best.pth'),
    }
    
    if is_eval:
        # Evaluation mode - always use best.pth
        fid_score, is_score = eval_vae(config)
        print(f"Evaluation completed: FID={fid_score:.4f}, IS={is_score:.4f}")
        return
    
    # Training mode
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    best_epoch, min_loss, best_state_dict = train_vae(config)
    
    # Save best checkpoint (already saved in train_vae, but ensure it's there)
    if best_state_dict is not None:
        print(f'Training finished: Best ckpt with loss {min_loss:.6f} @ epoch {best_epoch}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true', help='Evaluation mode')
    parser.add_argument('--ckpt_dir', action='store', type=str, default='./checkpoints', help='eval')
    parser.add_argument('--ckpt_name', action='store', type=str, default='best.pth', help='ckpt_dir')
    parser.add_argument('--batch_size', action='store', type=int, default=8, help='batch_size')
    parser.add_argument('--z_size', action='store', type=int, default=512, help='z_size')
    parser.add_argument('--lr', action='store', type=float, default=0.0001, help='lr')
    parser.add_argument('--num_epochs', action='store', type=int, default=50, help='num_epochs')
    parser.add_argument('--layer_count', action='store', type=int, default=5, help='layer_count')
    parser.add_argument('--kl_weight', action='store', type=float, default=0.1, help='kl_weight')
    parser.add_argument('--data_dir', action='store', type=str, default='/home/ycb410/ycb_ws/vae/datasets/chest_xray/train/', help='data_dir')
    
    main(vars(parser.parse_args()))
