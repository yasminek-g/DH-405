import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torch.nn.functional as F

# Directory with images
image_folder = 'path/to/your/image_folder'

# Function to pad the image and split it into patches without resizing
def image_to_patches(img_tensor, patch_size):
    # Pad the image to ensure it can be evenly divided into patches
    _, h, w = img_tensor.size()
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    img_tensor = F.pad(img_tensor, (0, pad_w, 0, pad_h), "constant", 0)

    # Calculate patches
    patches = img_tensor.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(3, -1, patch_size, patch_size)
    return patches.permute(1, 0, 2, 3)  # Shape (num_patches, channels, patch_size, patch_size)

# Process all images in the folder
patches_dataset = []
for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        # Load image and convert to tensor
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path).convert("RGB")
        img_tensor = transforms.ToTensor()(img)
        
        # Convert image to patches
        patches = image_to_patches(img_tensor, PATCH_SIZE)
        patches_dataset.append(patches)

# Example output
for idx, patches in enumerate(patches_dataset):
    print(f"Image {idx + 1}: Patches shape {patches.shape}")

# If you need to batch the patches, pad the smaller images to a consistent number of patches
# or handle varying shapes as needed for your MAE model.