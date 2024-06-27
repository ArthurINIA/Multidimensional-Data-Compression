import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from skimage.io import imread
import os

# Set the output directory
output_dir = os.path.dirname(os.path.abspath(__file__))

# Load the saved image
reconstructed_image_path = os.path.join(output_dir, "reconstructed_image.png")
reconstructed_image_loaded = imread(reconstructed_image_path) / 255.0  # Normalize to [0, 1]
original_image = imread("/Users/arthurchan/Desktop/Codes/ASTRI/data_compression/test.jpeg") / 255.0

# Check if the image was loaded successfully
if reconstructed_image_loaded is None:
    raise FileNotFoundError(f"The reconstructed image could not be loaded. Check the file path and integrity: {reconstructed_image_path}")

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(original_image, reconstructed_image_loaded)
print(f"Mean Squared Error (MSE): {mse}")

# Calculate the Peak Signal-to-Noise Ratio (PSNR)
psnr = peak_signal_noise_ratio(original_image, reconstructed_image_loaded, data_range=1.0)
print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr} dB")

# Display original and reconstructed images side by side for visual comparison
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(original_image)
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(reconstructed_image_loaded)
axes[1].set_title("Reconstructed Image")
axes[1].axis("off")

plt.show()

# Calculate compression ratio
original_size = original_image.nbytes
compressed_size = os.path.getsize(os.path.join(output_dir, "transformed_patches.npz")) + os.path.getsize(os.path.join(output_dir, "dictionary.npy"))
compression_ratio = original_size / compressed_size
print(f"Original size: {original_size} bytes")
print(f"Compressed size: {compressed_size} bytes")
print(f"Compression Ratio: {compression_ratio}")