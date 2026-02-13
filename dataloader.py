import os
import glob
from PIL import Image

import torch
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split


class VAEDataset(Dataset):
    def __init__(self, data_dir, im_size=64, transform=None):
        self.data_dir = data_dir
        self.im_size = im_size
        self.image_paths = []

        if os.path.isdir(data_dir):
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            for ext in extensions:
                self.image_paths.extend(glob.glob(os.path.join(data_dir, ext)))
                self.image_paths.extend(glob.glob(os.path.join(data_dir, ext.upper())))

        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in directory {data_dir}")

        print(f"Found {len(self.image_paths)} images")

        if transform is None:
            self.transform = self._get_default_transform(im_size)
        else:
            self.transform = transform

    def _get_default_transform(self, im_size):
        return transforms.Compose([
            transforms.Resize((im_size, im_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # [0,1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            return self.transform(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return torch.zeros(3, self.im_size, self.im_size)


def get_dataloader(
    data_dir,
    batch_size=16,
    im_size=64,
    split_ratio=0.8,
    seed=42,
    num_workers=1,
    pin_memory=True
):
    full_dataset = VAEDataset(data_dir, im_size=im_size)

    train_size = int(len(full_dataset) * split_ratio)
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False
    )

    return train_loader, test_loader


def main():
    data_dir = "/home/ycb410/ycb_ws/vae/datasets/"
    batch_size = 16
    im_size = 64
    save_dir = "results_dataloader"

    os.makedirs(save_dir, exist_ok=True)

    loader = get_dataloader(
        data_dir=data_dir,
        batch_size=batch_size,
        im_size=im_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    x = next(iter(loader))  # [B,3,H,W]

    save_image(x, os.path.join(save_dir, "batch_grid.png"), nrow=4)

    img0 = x[0]  # [3,H,W]
    save_image(img0[0:1], os.path.join(save_dir, "img0_ch0.png"))
    save_image(img0[1:2], os.path.join(save_dir, "img0_ch1.png"))
    save_image(img0[2:3], os.path.join(save_dir, "img0_ch2.png"))

    # Print information
    print("Saved to:", save_dir)
    print("Batch shape:", x.shape)
    print("Value range:", x.min().item(), x.max().item())
    print("Channel diff (img0):",
        (img0[0] - img0[1]).abs().mean().item(),
        (img0[0] - img0[2]).abs().mean().item())


if __name__ == "__main__":
    main()