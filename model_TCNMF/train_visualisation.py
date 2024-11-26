import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from skimage.io import imread_collection
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.util import random_noise
from skimage.restoration import inpaint
from skimage.morphology import square
from skimage.filters import median
import glob
import os
from skimage.color import rgba2rgb

# ---------------------------
# 1. Data Preparation
# ---------------------------

def load_and_preprocess_images(image_paths, image_size=(1024, 1024)):
    images = []
    for img_path in image_paths:
        try:
            # Read image
            img = plt.imread(img_path)
            print(f"Loading image: {img_path}, original shape: {img.shape}")

            # Handle RGBA images
            if img.ndim == 3 and img.shape[2] == 4:
                print(f"Image has 4 channels (RGBA). Converting to RGB.")
                img = rgba2rgb(img)

            # Convert to grayscale if desired; otherwise, keep as RGB
            img = rgb2gray(img)  # Comment out if you want to keep color information

            # Resize image
            img_resized = resize(img, image_size, anti_aliasing=True)
            print(f"Resized image to {image_size}, new shape: {img_resized.shape}")

            # Normalize pixel values to [0, 1]
            img_normalized = img_resized / np.max(img_resized)

            images.append(img_normalized)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            continue  # Skip this image
    # Convert list to NumPy array
    images_array = np.array(images)
    print(f"Final images array shape: {images_array.shape}")
    return images_array

# Define the paths to your image directories
train_images_dir = 'data/complete_facades/images'
test_images_dir = 'data/incomplete_facades/images'  

train_image_paths = glob.glob(os.path.join(train_images_dir, '*.png'))
print(f"Number of training images found: {len(train_image_paths)}")

# Load training images (complete facades)
train_images = load_and_preprocess_images(train_image_paths[:200])

# Flatten images and create data matrix V
# n_samples, img_height, img_width, n_channels = train_images.shape
n_samples, img_height, img_width = train_images.shape

V = train_images.reshape(n_samples, -1)  # Shape: (num_pixels, num_samples)
print(f">>>>>>Training data shape (V): {V.shape}")  # Should be (n_samples, n_features)
# ---------------------------
# 2. Training the NMF Model
# ---------------------------

# Choose the number of components (features)
n_components = 100  # Adjust based on desired detail

# Initialize NMF model
# Initialize NMF model with verbose output
	# •	verbose=0 (default): No output is displayed during training.
	# •	verbose=1: Displays basic information about the convergence at each iteration.
	# •	verbose>1: Provides more detailed output, which can be useful for debugging or in-depth analysis.
nmf_model = NMF(
    n_components=n_components,
    init='nndsvda',
    solver='mu',
    # solver='cd',
    tol=1e-4,
    max_iter=300,
    random_state=42,
    verbose = 1 # Set verbosity level to 1
)

# Fit the model to the training data
W = nmf_model.fit_transform(V)  # W: (num_pixels, n_components)
H = nmf_model.components_       # H: (n_components, num_samples)
# After fitting the model
print(f"Reconstruction error: {nmf_model.reconstruction_err_}")
# ---------------------------
# 3. Reconstructing Incomplete Facades
# ---------------------------

# Load test images (incomplete facades)

test_image_paths = glob.glob(os.path.join(test_images_dir, '*.png'))


test_images = load_and_preprocess_images(test_image_paths[:20])
print(f"Number of test images found: {len(test_image_paths)}")
# print(test_images)
# Introduce missing data (if test images are not already incomplete)
def introduce_missing_data(images, missing_rate=0.2):
    incomplete_images = []
    masks = []
    for img in images:
        mask = np.random.choice([1, 0], size=img.shape, p=[1-missing_rate, missing_rate])
        incomplete_img = img * mask
        incomplete_images.append(incomplete_img)
        masks.append(mask)
    return np.array(incomplete_images), np.array(masks)

# Assuming test images are already incomplete; otherwise, uncomment below
# test_images, test_masks = introduce_missing_data(test_images)

# Flatten test images
test_images_flat = test_images.reshape(test_images.shape[0], -1)  # Shape: (num_samples, num_pixels)
print(f"Test data shape (test_images_flat): {test_images_flat.shape}") 

# Handle missing data
# Create masks where 1 indicates observed data and 0 indicates missing data
test_masks = np.where(test_images_flat > 0, 1, 0)  # Shape: (num_pixels, num_samples)

# Impute missing values (e.g., with zeros)
V_test_imputed = np.nan_to_num(test_images_flat)

# Reconstruct the images
H_test = nmf_model.transform(V_test_imputed)  # Shape: (num_samples, n_components)
V_test_reconstructed = np.dot(H_test, nmf_model.components_)  # Shape: (num_pixels, num_samples)

# Apply the mask to keep original observed pixels
V_test_final = np.where(test_masks == 1, test_images_flat, V_test_reconstructed)

# Reshape reconstructed images
reconstructed_images = V_test_final.reshape(-1, img_height, img_width)

# ---------------------------
# 4. Visualization
# ---------------------------

def display_and_save_images_by_index(incomplete, reconstructed, num_images=20, save_dir="saved_images"):
    os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    for i in range(min(num_images, len(incomplete))):
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        
        # Incomplete image
        axes[0].imshow(incomplete[i], cmap='gray')
        axes[0].set_title('Incomplete Image')
        axes[0].axis('off')
        
        # Reconstructed image
        axes[1].imshow(reconstructed[i], cmap='gray')
        axes[1].set_title('Reconstructed Image')
        axes[1].axis('off')
        
        # Display the images
        plt.show()

        # Save the figure using index as filename
        filename = f'image_{i}.png'
        save_path = os.path.join(save_dir, filename)
        fig.savefig(save_path)
        plt.close(fig)  # Close the figure to avoid memory issues

    print(f"Images saved to {save_dir}")

# Assuming we have the original complete test images; if not, use test_images instead
original_test_images = test_images  # Replace with original images if available
incomplete_test_images = test_images  # Since test_images are incomplete
display_and_save_images_by_index(original_test_images, incomplete_test_images, reconstructed_images)

# ---------------------------
# 5. Evaluation (Optional)
# ---------------------------

from sklearn.metrics import mean_squared_error

def evaluate_reconstruction(original, reconstructed):
    mse = []
    for orig, recon in zip(original, reconstructed):
        mse.append(mean_squared_error(orig.flatten(), recon.flatten()))
    avg_mse = np.mean(mse)
    print(f'Average MSE over test images: {avg_mse}')

# Evaluate reconstruction
evaluate_reconstruction(original_test_images, reconstructed_images)