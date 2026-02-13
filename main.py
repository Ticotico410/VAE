from __future__ import print_function

import os
import copy
import math
import shutil
import argparse

import torch
from torch import optim
import torch.nn.functional as F
from torchvision.utils import save_image

from vae import *
from tqdm import tqdm
from ignite.metrics import FID, InceptionScore

from dataLoader import get_dataloader
from utils import loss_function, viz_loss


im_size = 64

def train_vae(config, train_loader):
    lr = config['lr']
    z_size = config['z_size']
    kl_weight = config['kl_weight']
    rec_weight = config['rec_weight']
    train_epoch = config['num_epochs']
    save_per_epoch = config['save_per_epoch']

    ckpt_dir = config['ckpt_dir']
    os.makedirs(ckpt_dir, exist_ok=True)

    # VAE model setup
    model = VAE(z_dim=z_size).cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0)

    min_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    best_optimizer_state = None

    epoch_rec_losses = []
    epoch_kl_losses = []
    epoch_total_losses = []

    # Load data
    print("Loading training dataset...")
    print("Train set size:", len(train_loader.dataset))

    for epoch in range(train_epoch):
        print(f"\nEpoch {epoch+1}/{train_epoch}")
        model.train()

        rec_loss = 0.0
        kl_loss = 0.0
        total_loss = 0.0
        loss_rec_weighted_sum = 0.0
        loss_kl_weighted_sum = 0.0

        z_mean_sum = 0.0
        z_std_sum = 0.0
        mu_std_sum = 0.0
        sigma_mean_sum = 0.0
        active_units_sum = 0.0

        batch_pbar = tqdm(
            train_loader,
            unit="batch",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {rate_fmt}",
        )

        for x in batch_pbar:
            x = x.cuda(non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            mu, logvar = model.encoder(x)
            z = model.reparameterize(mu, logvar)
            rec = model.decoder(z)

            loss_rec, loss_kl = loss_function(rec, x, mu, logvar, use_free_bits=True, free_bits=0.01)
            loss_rec_weighted = loss_rec * rec_weight
            loss_kl_weighted = loss_kl * kl_weight
            loss = loss_rec_weighted + loss_kl_weighted
            loss.backward()
            optimizer.step()

            rec_loss += loss_rec.item()
            kl_loss += loss_kl.item()
            total_loss += loss.item()
            loss_rec_weighted_sum += loss_rec_weighted.item()
            loss_kl_weighted_sum += loss_kl_weighted.item()

            # Debugging
            z_mean_sum += z.mean().item()
            z_std_sum += z.std().item()

            mu_std_sum += mu.std().item()
            sigma = torch.exp(0.5 * logvar)
            sigma_mean_sum += sigma.mean().item()
            mu_var_dim = mu.var(dim=0, unbiased=False)
            active_units = (mu_var_dim > 1e-3).float().sum().item()
            active_units_sum += active_units
        
        # metrics
        batch_count = len(train_loader)
        avg_rec_loss = rec_loss / batch_count
        avg_kl_loss = kl_loss / batch_count
        avg_total_loss = total_loss / batch_count
        avg_rec_weighted_loss = loss_rec_weighted_sum / batch_count
        avg_kl_weighted_loss = loss_kl_weighted_sum / batch_count

        avg_z_mean = z_mean_sum / batch_count
        avg_z_std = z_std_sum / batch_count
        avg_mu_std = mu_std_sum / batch_count
        avg_sigma_mean = sigma_mean_sum / batch_count
        avg_active_units = active_units_sum / batch_count

        print(f"Train loss: {avg_total_loss:.5f}")
        print(f"[Origin] rec_loss: {avg_rec_loss:.5f} kl_loss: {avg_kl_loss:.5f}")
        print(f"[Weight] rec_loss: {avg_rec_weighted_loss:.5f} kl_loss: {avg_kl_weighted_loss:.5f}")
        print(f"[Latent] z_mean={avg_z_mean:.4f}, z_std={avg_z_std:.4f}")
        print(f"[Latent] mu_std={avg_mu_std:.4f}, sigma_mean={avg_sigma_mean:.4f}, active_units={avg_active_units:.1f}")

        if avg_total_loss < min_loss:
            min_loss = avg_total_loss
            best_epoch = epoch + 1
            best_model_state = copy.deepcopy(model.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())

        epoch_rec_losses.append(avg_rec_loss)
        epoch_kl_losses.append(avg_kl_loss)
        epoch_total_losses.append(avg_total_loss)

        # Save checkpoints in a specified interval
        # if (epoch + 1) % save_per_epoch == 0:
        #     checkpoint = {
        #         'epoch': epoch + 1,
        #         'model_state_dict': model.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'avg_rec_loss': avg_rec_loss,
        #         'avg_kl_loss': avg_kl_loss,
        #         'avg_total_loss': avg_total_loss,
        #         'z_size': z_size,
        #     }
        #     checkpoint_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{epoch+1}.pth')
        #     torch.save(checkpoint, checkpoint_path)

    viz_loss(epoch_rec_losses, epoch_kl_losses, train_epoch, ckpt_dir)

    # save the best model
    if best_model_state is not None:
        if best_optimizer_state is None:
            best_optimizer_state = optimizer.state_dict()

        best_checkpoint = {
            'epoch': best_epoch,
            'model_state_dict': best_model_state,
            'optimizer_state_dict': best_optimizer_state,
            'min_loss': min_loss,
            'z_size': z_size,
        }
        best_checkpoint_path = os.path.join(ckpt_dir, f'checkpoint_epoch_{best_epoch}.pth')
        torch.save(best_checkpoint, best_checkpoint_path)
        best_copy_path = os.path.join(ckpt_dir, 'best.pth')
        shutil.copy2(best_checkpoint_path, best_copy_path)
        print(f"Best checkpoint saved: {best_copy_path} (Epoch {best_epoch}, Loss: {min_loss:.6f})")

    return best_epoch, min_loss, best_model_state


def eval_vae(config, test_loader):
    batch_size = config['batch_size']
    ckpt_dir = config['ckpt_dir']

    # Load checkpoint
    ckpt_name = config.get('ckpt_name', 'best.pth')
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    checkpoint = torch.load(ckpt_path, map_location='cuda')
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")

    loss_value = checkpoint.get('min_loss') or checkpoint.get('avg_total_loss')
    if loss_value is not None:
        print(f"  Loss: {loss_value:.4f}")

    # Create model with same configuration
    z_size = checkpoint.get('z_size', 512)
    model = VAE(z_dim=z_size).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded: z_size={z_size}")

    # Initialize Ignite metrics for FID and IS
    print("Initializing FID and Inception Score metrics...")
    fid_metric = FID(device='cuda')
    is_metric = InceptionScore(device='cuda')

    # Load data
    print("Loading testing dataset...")
    print("Test set size:", len(test_loader.dataset))

    os.makedirs('results_rec', exist_ok=True)
    os.makedirs('results_gen', exist_ok=True)

    # Evaluate all samples
    max_eval_samples = len(test_loader.dataset)
    num_eval_batches = math.ceil(max_eval_samples / batch_size)
    print(f"\nEvaluating with {max_eval_samples} samples ...")

    # Preprocess: GPU batch resize to 299 + ImageNet normalize
    mean = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1, 3, 1, 1)

    def to_inception_input(x01: torch.Tensor) -> torch.Tensor:  
        x = F.interpolate(x01, size=(299, 299), mode='bilinear', align_corners=False)
        x = x.clamp(0.0, 1.0)
        x = (x - mean) / std
        return x

    sample_real_list, sample_recon_list, sample_gen_list = [], [], []

    with torch.no_grad():
        eval_pbar = tqdm(test_loader, desc="Evaluating", unit="batch")
        seen = 0

        for batch_idx, eval_x in enumerate(eval_pbar):
            eval_x = eval_x.cuda(non_blocking=True)  # [0,1]

            # Reconstruction
            mu, logvar = model.encoder(eval_x)
            # z = model.reparameterize(mu, logvar)
            z = mu
            x_rec = model.decoder(z)  # [0,1]

            # Generation
            z_fake = torch.randn(eval_x.size(0), z_size, device='cuda')
            fake_imgs = model.decoder(z_fake)  # [0,1]

            # Save first batch for visualization
            if batch_idx == 0:
                sample_real_list.append(eval_x[:8].cpu())
                sample_recon_list.append(x_rec[:8].cpu())
                sample_gen_list.append(fake_imgs[:8].cpu())

            # Prepare inputs for metrics
            real_in = to_inception_input(eval_x)
            fake_in = to_inception_input(fake_imgs)

            fid_metric.update((fake_in, real_in))
            is_metric.update(fake_in)

            seen += eval_x.size(0)
            if seen >= max_eval_samples:
                break

    print(f"\n{'='*50}")
    print("\nCalculating FID...")
    fid_score = fid_metric.compute()
    print(f"FID Score: {float(fid_score):.4f} (Lower is Better)")

    print("Calculating IS...")
    is_out = is_metric.compute()
    if isinstance(is_out, (tuple, list)) and len(is_out) == 2:
        is_score_mean, is_score_std = is_out
        print(f"IS Score: {float(is_score_mean):.4f} ± {float(is_score_std):.4f} (Higher is Better)")
        is_score = is_score_mean
    else:
        is_score = is_out
        print(f"IS Score: {float(is_score):.4f} (Higher is Better)")
    print(f"{'='*50}\n")

    # Save visualization images
    if sample_real_list:
        real_samples = torch.cat(sample_real_list, dim=0)
        recon_samples = torch.cat(sample_recon_list, dim=0)
        save_image(torch.cat([real_samples, recon_samples], dim=0), 'results_rec/eval_sample.png', nrow=8)
        print("Reconstruction samples saved: results_rec/eval_sample.png")

        gen_samples = torch.cat(sample_gen_list, dim=0)
        save_image(gen_samples, 'results_gen/eval_sample.png', nrow=8)
        print("Generation samples saved: results_gen/eval_sample.png")

    return fid_score, is_score


def main(args):
    lr = args['lr']
    is_eval = args['eval']
    z_size = args['z_size']
    batch_size = args['batch_size']
    num_epochs = args['num_epochs']
    ckpt_dir = args['ckpt_dir']
    data_dir = args['data_dir']

    print("Loading data...")
    train_loader, test_loader = get_dataloader(data_dir, batch_size=batch_size, im_size=im_size)
    print(f"Train set size: {len(train_loader.dataset)}, Test set size: {len(test_loader.dataset)}\n")
    
    config = {
        'ckpt_dir': ckpt_dir,
        'batch_size': batch_size,
        'z_size': z_size,
        'lr': lr,
        'num_epochs': num_epochs,
        'data_dir': data_dir,
        'kl_weight': args.get('kl_weight', 0.1),
        'rec_weight': args.get('rec_weight', 10),
        'ckpt_name': args.get('ckpt_name', 'best.pth'),
        'save_per_epoch': args.get('save_per_epoch', 10),
    }
    
    if is_eval:
        fid_score, is_score = eval_vae(config, test_loader)
        print(f"Evaluation completed: FID={fid_score:.4f}, IS={is_score:.4f}")
        return
    
    best_epoch, min_loss, best_state_dict = train_vae(config, train_loader)
    
    # Print best checkpoint
    if best_state_dict is not None:
        print(f'Training finished: Best ckpt with loss {min_loss:.6f} @ epoch {best_epoch}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', action='store', type=float, default=0.00005, help='lr')
    parser.add_argument('--z_size', action='store', type=int, default=64, help='z_size')
    parser.add_argument('--batch_size', action='store', type=int, default=16, help='batch_size')

    parser.add_argument('--num_epochs', action='store', type=int, default=50, help='num_epochs')
    parser.add_argument('--kl_weight', action='store', type=float, default=0.5, help='kl_weight')
    parser.add_argument('--rec_weight', action='store', type=float, default=10, help='rec_weight')
    parser.add_argument('--save_per_epoch', action='store', type=int, default=10, help='save_per_epoch')

    parser.add_argument('--eval', action='store_true', help='eval')
    parser.add_argument('--ckpt_name', action='store', type=str, default='best.pth', help='ckpt_name')
    parser.add_argument('--ckpt_dir', action='store', type=str, default='./checkpoints', help='ckpt_dir')
    parser.add_argument('--data_dir', action='store', type=str, default='/home/ycb410/ycb_ws/vae/datasets/', help='data_dir')
    
    main(vars(parser.parse_args()))
