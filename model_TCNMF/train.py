"""
Reimplementation of the Truncated Cauchy Non-Negative Matrix Factorization (CauchyNMF) algorithm in Python.
Code Structure
	•	Core NMF Functions: CauchyNMF.m, CauchyNNLS.m, CauchyNLS.m, and CauchyOGM.m are essential to the Cauchy NMF implementation.
	•	Helper Functions: CauchyLpzConstt.m, CauchyMedFilter.m, CauchyOutlIndex.m, and CauchySCL.m
	•	Stopping Criterion: GetStopCriterion.m implements stopping rules for convergence.
	•	Outlier Detection: OutlierIX.m handle outlier detection in data.
"""

import numpy as np
from numpy.linalg import norm
from PIL import Image
import glob
import matplotlib.pyplot as plt

# Define the CauchyNMFWithOGM class
class CauchyNMFWithOGM:
    def __init__(self, V, r, max_iter=100, tol=1e-4, gamma=0.1, weighting='plain', lpz_type='plain'):
        """
        Initialize parameters for Truncated Cauchy Non-Negative Matrix Factorization with OGM
        """
        self.V = V
        self.r = r
        self.max_iter = max_iter
        self.tol = tol
        self.gamma = gamma
        self.weighting = weighting.lower()
        self.lpz_type = lpz_type.lower()
        self.m, self.n = V.shape
        self.W = np.random.rand(self.m, self.r)
        self.H = np.random.rand(self.r, self.n)
        
    def compute_lipschitz_constant(self, W, Q):
        """
        Approximate Lipschitz constant based on LPZ_TYPE (here simplified)
        """
        # Use broadcasting to align Q with W dimensions
        Q_expanded = Q.mean(axis=1)[:, np.newaxis] * np.ones((1, W.shape[1]))
        return np.linalg.norm(W.T @ (Q_expanded * W), ord=2)
    
    def get_stop_criterion(self, Y, Grad):
        """
        Compute the stopping criterion based on gradient norm.
        """
        return norm(Y - np.maximum(0, Y - Grad), ord='fro')
    
    def fit(self):
        """
        Fit the CauchyNMF model to factorize V ~ W * H using gradient-based OGM updates for H.
        """
        for iter in range(self.max_iter):
            print("Iteration: ", iter)

            # Compute residual and scale gamma if necessary
            R = np.abs(self.V - np.dot(self.W, self.H))
            if self.gamma < 0:
                self.gamma = np.sqrt(np.sum(R ** 2) / (2 * self.m * self.n))
            Z = (R / self.gamma) ** 2 + 1
            
            # Apply weighting
            Q = 1 / Z if self.weighting == 'plain' else np.ones_like(Z)
            
            # Optimal Gradient Method (OGM) update for H
            L = self.compute_lipschitz_constant(self.W, Q)
            Y = self.H.copy()
            Grad = self.W.T @ (Q * (np.dot(self.W, Y) - self.V))
            alpha0 = 1

            for k in range(10):  # Set a max sub-iteration count
                print("Sub-Iteration: ", k)
                H1 = np.maximum(0, Y - Grad / L)
                alpha1 = (1 + np.sqrt(4 * alpha0 ** 2 + 1)) / 2
                Y = H1 + (alpha0 - 1) * (H1 - self.H) / alpha1
                self.H = H1
                alpha0 = alpha1
                Grad = self.W.T @ (Q * (np.dot(self.W, Y) - self.V))
                print(self.get_stop_criterion(Y, Grad))
                if self.get_stop_criterion(Y, Grad) < self.tol:
                    break
            
            # Update W using a similar non-negative projection
            self.W *= np.dot(Q * self.V, self.H.T) / (np.dot(Q * np.dot(self.W, self.H), self.H.T) + 1e-10)
            
    def transform(self, data):
        """
        Transform the input data using learned W.
        """
        return np.dot(self.W.T, data)
    
    def fit_transform(self):
        """
        Fit the model and return the transformed data.
        """
        self.fit()
        return self.transform(self.V)

# Load and preprocess images without resizing
def load_and_preprocess_images(image_folder, subset_size=100):
    # Collect image paths and limit to subset_size
    image_paths = glob.glob(f"{image_folder}/*.png")[:subset_size]
    print("image process start")
    # image_paths = glob.glob(f"{image_folder}/*.png")

    images = []

    # Find the maximum dimensions across all images to set a consistent size
    max_height, max_width = 0, 0
    for path in image_paths:
        img = Image.open(path)
        max_height = max(max_height, img.height)
        max_width = max(max_width, img.width)

    # Process each image: convert to grayscale, pad to max dimensions, and flatten
    for path in image_paths:
        print(f"Processing image: {path}")
        img = Image.open(path).convert('L')  # Convert to grayscale
        padded_img = Image.new('L', (max_width, max_height))  # Create blank canvas
        padded_img.paste(img, (0, 0))  # Paste original image in the top-left corner
        img_array = np.array(padded_img).flatten()  # Flatten to a vector
        images.append(img_array)

    # Stack all image vectors to create the data matrix V
    V = np.stack(images, axis=1) / 255.0  # Normalize pixel values to [0, 1]
    return V, (max_height, max_width)


# Step 2: Train Cauchy NMF model
image_folder = 'data/training_aug'  # Replace with your actual path
V, target_size = load_and_preprocess_images(image_folder)
print("image processing done")

num_components = 10  # Number of components for NMF
max_iter=10
tol=1e-3
gamma=0.1
weighting='plain'
lpz_type='plain'

print("start training")
# Initialize and train the Cauchy NMF model on the subset data
cauchy_nmf_model = CauchyNMFWithOGM(V=V, r=num_components, max_iter=max_iter, tol=tol, gamma=gamma, weighting=weighting, lpz_type=weighting)
W_H_result = cauchy_nmf_model.fit_transform()

# Step 3: Reconstruct and visualize images
def visualize_reconstruction(V, W, H, target_size, num_images=5):
    # Reshape the flattened images to the original padded size
    reconstructed_images = np.dot(W, H)
    fig, axes = plt.subplots(2, num_images, figsize=(15, 5))
    
    for i in range(num_images):
        # Original image
        original_image = V[:, i].reshape(target_size)
        axes[0, i].imshow(original_image, cmap='gray')
        axes[0, i].axis('off')
        axes[0, i].set_title("Original")

        # Reconstructed image
        reconstructed_image = reconstructed_images[:, i].reshape(target_size)
        axes[1, i].imshow(reconstructed_image, cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_title("Reconstructed")

    plt.show()

# Visualize a few original and reconstructed images
# Provide the maximum image size (height, width) as target_size
visualize_reconstruction(V, cauchy_nmf_model.W, cauchy_nmf_model.H, target_size)