import os
import argparse
import math
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
from PIL import Image
from tqdm import tqdm
from model import MAE_ViT  # Only use the MAE_ViT model without the classifier
from utils import setup_seed
import matplotlib.pyplot as plt

# Custom dataset to load images without labels
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir) if fname.endswith('.png')]
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")  # Ensure RGB format
        if self.transform:
            image = self.transform(image)
        return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=20)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--output_model_path', type=str, default='mae-vit-pretrained.pt')

    args = parser.parse_args()
    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)
    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    # Path to your dataset
    data_dir = '/Users/oscargoudet/Desktop/FDH/project/sample_facades_2024_10_08/complete_facades/images'
    
    # Define transformations
    transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize images to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load dataset
    dataset = ImageDataset(data_dir, transform=transform)

    # Split dataset into train and validation sets (80-20 split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Data loaders
    train_dataloader = DataLoader(train_dataset, batch_size=load_batch_size, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(val_dataset, batch_size=load_batch_size, shuffle=False, num_workers=0)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Initialize MAE model
    model = MAE_ViT(image_size=32, patch_size=2, emb_dim=192, encoder_layer=12, encoder_head=3, decoder_layer=4, decoder_head=3, mask_ratio=0.75).to(device)

    # Reconstruction loss (Mean Squared Error)
    reconstruction_loss_fn = torch.nn.MSELoss()

    # Optimizer and learning rate scheduler
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    # Training loop
    for e in range(args.total_epoch):
        model.train()
        train_losses = []
        for img in tqdm(train_dataloader):
            img = img.to(device)
            # Forward pass through encoder and decoder
            predicted_img, mask = model(img)
            
            # Compute reconstruction loss (MSE scaled by mask to emphasize masked regions)
            loss = reconstruction_loss_fn(predicted_img, img) * mask / model.encoder.shuffle.ratio
            loss = torch.mean(loss)
            loss.backward()

            # Optimizer step
            optim.step()
            optim.zero_grad()
            train_losses.append(loss.item())

        lr_scheduler.step()
        avg_train_loss = sum(train_losses) / len(train_losses)
        print(f'Epoch {e}: Avg training loss: {avg_train_loss}')

        # Validation loop
        model.eval()
        val_losses = []
        with torch.no_grad():
            for img in tqdm(val_dataloader):
                img = img.to(device)
                predicted_img, mask = model(img)
                loss = reconstruction_loss_fn(predicted_img, img) * mask / model.encoder.shuffle.ratio
                val_losses.append(loss.mean().item())

            avg_val_loss = sum(val_losses) / len(val_losses)
            print(f'Epoch {e}: Avg validation loss: {avg_val_loss}')

        # Save the model checkpoint
        torch.save(model.state_dict(), args.output_model_path)
    


def visualize_reconstruction(model, dataloader, device, num_images=5):
    model.eval()
    images_shown = 0
    with torch.no_grad():
        for img in dataloader:
            img = img.to(device)
            predicted_img, _ = model(img)

            # Move images to CPU for visualization
            img = img.cpu()
            predicted_img = predicted_img.cpu()

            for i in range(min(num_images, img.size(0))):
                original_image = img[i].permute(1, 2, 0)  # CHW to HWC for plotting
                reconstructed_image = predicted_img[i].permute(1, 2, 0)

                # Unnormalize the images to bring them back to the [0, 1] range
                original_image = torch.clamp(original_image * 0.5 + 0.5, 0, 1)
                reconstructed_image = torch.clamp(reconstructed_image * 0.5 + 0.5, 0, 1)

                # Plot the original and reconstructed images side by side
                fig, axs = plt.subplots(1, 2, figsize=(10, 5))
                axs[0].imshow(original_image)
                axs[0].set_title("Original Image")
                axs[0].axis("off")

                axs[1].imshow(reconstructed_image)
                axs[1].set_title("Reconstructed Image")
                axs[1].axis("off")

                plt.show()

                images_shown += 1
                if images_shown >= num_images:
                    return  # Stop once we've shown the specified number of images

# Example usage after training:
# Load the test data
test_dataloader = DataLoader(val_dataset, batch_size=load_batch_size, shuffle=True, num_workers=0)

# Call the visualization function on the test dataset
visualize_reconstruction(model, test_dataloader, device)


