import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import DictionaryLearning
from skimage import data, img_as_float, img_as_ubyte
from skimage.util import view_as_blocks
from skimage.io import imsave, imread
from scipy.sparse import csr_matrix, save_npz, load_npz
import os

# Function to save CSR matrix to a binary file
def save_csr_matrix(filename, matrix):
    save_npz(filename, matrix)

# Function to load CSR matrix from a binary file
def load_csr_matrix(filename):
    return load_npz(filename)

# Remove existing reconstructed image if it exists
if os.path.exists("reconstructed_image.png"):
    os.remove("reconstructed_image.png")

# Load example image
image = imread("/Users/arthurchan/Desktop/Codes/ASTRI/data_compression/test.jpeg") / 255.0  # Colored image
original_shape = image.shape

plt.imshow(image)
plt.title("Original Image")
plt.show()
print("Loaded and displayed original image")

# Define patch size
print(image.shape)
height, width = image.shape[:2]
patch_height = 4
patch_width = 4
patch_size = (patch_height, patch_width, 3)
n_ter = 40
transform_iter = 40
alpha = 0.1
algorithm = "omp"
n_components = 300  # Number of dictionary components
print("patch_size: ", patch_size, "n_ter: ", n_ter, "transform_iter: ", transform_iter, "alpha: ", alpha)
print("algorithm: ", algorithm, "n_components: ", n_components)

# Ensure the image dimensions are divisible by the patch size
height, width = image.shape[:2]
assert height % patch_height == 0 and width % patch_width == 0, "Image dimensions must be divisible by patch size."

# Extract non-overlapping patches from the image
patches = view_as_blocks(image, block_shape=patch_size)
num_patches_y, num_patches_x = patches.shape[:2]

# Reshape patches for dictionary learning
print("Patches size before reshape: ", patches.shape)
patches = patches.reshape(-1, patch_height * patch_width * patch_size[2])
print("Patches size after reshape: ", patches.shape)  # Should be (4096, 192)

# Perform dictionary learning
print("Performing dictionary learning...")
dl = DictionaryLearning(n_components=n_components, transform_algorithm=algorithm, transform_alpha=alpha, max_iter=n_ter, transform_max_iter=transform_iter)
dictionary = dl.fit(patches).components_
transformed_patches = dl.transform(patches)
print("Dictionary learning completed")

# Convert the sparse matrix to CSR format
transformed_patches_csr = csr_matrix(transformed_patches)

# Save the CSR matrix to a binary file
save_csr_matrix("transformed_patches.npz", transformed_patches_csr)
print("Sparse matrix saved in CSR format")

# Save the dictionary to a binary file
np.save("dictionary.npy", dictionary)

# Load the CSR matrix from the binary file before reconstruction
transformed_patches_csr_loaded = load_csr_matrix("transformed_patches.npz")
transformed_patches_loaded = transformed_patches_csr_loaded.toarray()

# Load the dictionary from the binary file
dictionary_loaded = np.load("dictionary.npy")

# Reconstruct patches using the loaded dictionary
reconstructed_patches = np.dot(transformed_patches_loaded, dictionary_loaded)

# Reshape reconstructed patches back to the original patch shape
reconstructed_patches = reconstructed_patches.reshape(-1, patch_height, patch_width, 3)

# Initialize an empty array for the reconstructed image
reconstructed_image = np.zeros((num_patches_y * patch_height, num_patches_x * patch_width, 3))

# Place reconstructed patches back into the image
patch_index = 0
for i in range(num_patches_y):
    for j in range(num_patches_x):
        reconstructed_image[i * patch_height:(i + 1) * patch_height, j * patch_width:(j + 1) * patch_width, :] = reconstructed_patches[patch_index]
        patch_index += 1

# Crop the reconstructed image to the original dimensions
reconstructed_image = reconstructed_image[:height, :width, :]

# Display the reconstructed image
plt.imshow(reconstructed_image)
plt.title("Reconstructed Image")
plt.show()

# Ensure values are in the range [0, 1]
reconstructed_image = np.clip(reconstructed_image, 0, 1)

# Convert the reconstructed image back to uint8 for saving
reconstructed_image_uint8 = img_as_ubyte(reconstructed_image)

# Save the reconstructed image
imsave("reconstructed_image.png", reconstructed_image_uint8)
print("Reconstructed image saved as 'reconstructed_image.png'")