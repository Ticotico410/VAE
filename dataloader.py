import os
import glob
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class VAEDataset(Dataset):
    """
    Standard PyTorch Dataset for VAE training
    Uses torchvision transforms for image preprocessing
    """
    def __init__(self, data_dir, im_size=128, transform=None):
        self.data_dir = data_dir
        self.im_size = im_size
        self.image_paths = []
        
        # Find all image files
        if os.path.isdir(data_dir):
            extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
            for ext in extensions:
                self.image_paths.extend(glob.glob(os.path.join(data_dir, ext)))
                self.image_paths.extend(glob.glob(os.path.join(data_dir, ext.upper())))
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in directory {data_dir}")
        
        print(f"Found {len(self.image_paths)} images")
        
        # Use provided transform or create default transform
        if transform is None:
            self.transform = self._get_default_transform(im_size)
        else:
            self.transform = transform
    
    def _get_default_transform(self, im_size):
        """
        Create default transform pipeline:
        1. Resize to target size
        2. Convert to tensor (also converts to [0, 1])
        3. Normalize to [-1, 1]
        """
        return transforms.Compose([
            transforms.Resize((im_size, im_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # Converts PIL Image to tensor and scales to [0, 1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # Load image and ensure it's RGB (handles grayscale conversion)
            img = Image.open(img_path).convert('RGB')
            
            # Apply transforms
            img_tensor = self.transform(img)
            
            return img_tensor
        
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, self.im_size, self.im_size)


def get_dataloader(data_dir, batch_size=16, im_size=128, shuffle=True, num_workers=1, pin_memory=True):
    """
    Create DataLoader with standard PyTorch pipeline
    
    Args:
        data_dir: image directory path
        batch_size: batch size
        im_size: image size (will be resized to im_size x im_size)
        shuffle: whether to shuffle data
        num_workers: number of worker threads for data loading
        pin_memory: whether to use pinned memory for faster GPU transfer
    
    Returns:
        DataLoader object
    """
    dataset = VAEDataset(data_dir, im_size=im_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # Drop incomplete batches for consistent batch size
    )
    return dataloader

