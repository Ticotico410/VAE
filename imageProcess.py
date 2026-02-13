import os
import glob
import random
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image

# Parameters
data_dir = "/home/ycb410/ycb_ws/vae/datasets/chest_xray/train/"
batch_size = 16
im_size = 64
save_dir = "results_resize"

os.makedirs(save_dir, exist_ok=True)
image_paths = glob.glob(os.path.join(data_dir, "**/*.jpeg"), recursive=True)

selected_paths = random.sample(image_paths, batch_size)

transform = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor()
])

images = []
for path in selected_paths:
    img = Image.open(path).convert("RGB")
    images.append(transform(img))

batch = torch.stack(images)

save_image(batch, os.path.join(save_dir, f"resize_{im_size}.png"), nrow=4)

print("Saved to:", os.path.join(save_dir, f"resize_{im_size}.png"))
print("Shape:", batch.shape)
