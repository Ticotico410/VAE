import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from PIL import Image


def interpolate(batch):
    """
    Resize images to 299x299 for Inception network
    Args:
        batch: tensor of shape [B, C, H, W] with values in [-1, 1]
    Returns:
        tensor of shape [B, C, 299, 299] with values in [0, 1]
    """
    arr = []
    for img in batch:
        # Convert from [-1, 1] to [0, 1]
        img_01 = (img + 1.0) / 2.0
        img_01 = torch.clamp(img_01, 0.0, 1.0)
        
        # Convert to PIL Image
        pil_img = transforms.ToPILImage()(img_01.cpu())
        # Resize to 299x299 using BILINEAR interpolation
        resized_img = pil_img.resize((299, 299), Image.BILINEAR)
        # Convert back to tensor
        arr.append(transforms.ToTensor()(resized_img))

    return torch.stack(arr)


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)

    return total_kld


def loss_function(recon_x, x, mu, logvar, kl_weight=0.1):
    batch_size = mu.size(0)
    assert batch_size != 0
    
    # L1 loss
    all_l1 = F.l1_loss(recon_x, x, reduction='none')
    # recon_loss = (all_l1).mean()
    
    # MSE loss
    all_mse = F.mse_loss(recon_x, x, reduction='none')
    recon_loss = (all_mse).mean()

    # KL divergence
    total_kld = kl_divergence(mu, logvar)   # shape [1]
    kl_loss = total_kld[0] * kl_weight
    
    return recon_loss, kl_loss


def plot_loss_curves(
    epoch_rec_losses,
    epoch_kl_losses,
    epoch_total_losses,
    train_epoch,
    ckpt_dir,
    filename="loss_traj.png",
):
    plt.figure(figsize=(12, 5))

    epochs = range(1, train_epoch + 1)

    ax1 = plt.subplot(1, 2, 1)
    plt.plot(epochs, epoch_rec_losses, "b-", label="L1 Loss", linewidth=2)
    plt.plot(epochs, epoch_kl_losses, "r-", label="KL Loss", linewidth=2)
    # plt.plot(epochs, epoch_total_losses, "g-", label="Total Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    ax2 = plt.subplot(1, 2, 2)
    plt.plot(epochs, epoch_rec_losses, "b-", label="L1 Loss", linewidth=2)
    plt.plot(epochs, epoch_kl_losses, "r-", label="KL Loss", linewidth=2)
    # plt.plot(epochs, epoch_total_losses, "g-", label="Total Loss", linewidth=2)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss (Log)", fontsize=12)
    plt.title("Training Loss (Log)", fontsize=14, fontweight="bold")
    plt.yscale("log")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    loss_traj_path = os.path.join(ckpt_dir, filename)
    plt.savefig(loss_traj_path, dpi=300, bbox_inches="tight")
    print(f"Loss trajectory saved: {loss_traj_path}")
    plt.close()
    return loss_traj_path
