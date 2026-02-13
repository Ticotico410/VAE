import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


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


def kl_free_bits(mu, logvar, free_bits=0.0):
    if free_bits is None:
        free_bits = 0.0
    if mu.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    kl_per_dim = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar)
    kl_dim = kl_per_dim.mean(dim=0)
    if free_bits > 0.0:
        kl_dim = torch.clamp(kl_dim, min=free_bits)

    return kl_dim.sum()


def loss_function(recon_x, x, mu, logvar, use_free_bits=False, free_bits=0.0):
    batch_size = mu.size(0)
    assert batch_size != 0
    
    # L1 loss
    all_l1 = F.l1_loss(recon_x, x, reduction='none')
    recon_loss = (all_l1).mean()
    
    # MSE loss
    all_mse = F.mse_loss(recon_x, x, reduction='none')
    # recon_loss = (all_mse).mean()

    # KL divergence
    if use_free_bits:
        kl_loss = kl_free_bits(mu, logvar, free_bits=free_bits)
    else:
        total_kld = kl_divergence(mu, logvar)   # shape [1]
        kl_loss = total_kld[0]
    
    return recon_loss, kl_loss


def viz_loss(
    epoch_rec_losses,
    epoch_kl_losses,
    train_epoch,
    ckpt_dir,
):
    epochs = range(1, train_epoch + 1)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Reconstruction Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, epoch_rec_losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("rec loss")
    plt.title("Reconstruction Loss", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    rec_path = os.path.join(ckpt_dir, "rec_loss.png")
    plt.savefig(rec_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Rec loss saved: {rec_path}")

    # KL Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, epoch_kl_losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel("kl loss")
    plt.title("KL Loss", fontweight="bold")
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    kl_path = os.path.join(ckpt_dir, "kl_loss.png")
    plt.savefig(kl_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"KL loss saved: {kl_path}")

    return rec_path, kl_path
