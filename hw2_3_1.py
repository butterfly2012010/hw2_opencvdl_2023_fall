import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image
image_path = 'E:/YN/opencvdl/Hw2/Dataset_OpenCvDl_Hw2/Q3/closing.png'
image = cv2.imread(image_path)

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Binarize the image
_, binarized_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# Padding the image with zeros
kernel_size = 3
padded_image = np.pad(binarized_image, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='constant', constant_values=0)

# Function to perform dilation
def dilate(image, kernel_size):
    structure_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_height, image_width = image.shape
    dilated_image = np.zeros_like(image)
    
    for i in range(kernel_size//2, image_height - kernel_size//2):
        for j in range(kernel_size//2, image_width - kernel_size//2):
            region = image[i - kernel_size//2:i + kernel_size//2 + 1, j - kernel_size//2:j + kernel_size//2 + 1]
            dilated_image[i, j] = np.max(region * structure_element)
    return dilated_image

# Function to perform erosion
def erode(image, kernel_size):
    structure_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    image_height, image_width = image.shape
    eroded_image = np.zeros_like(image)
    
    for i in range(kernel_size//2, image_height - kernel_size//2):
        for j in range(kernel_size//2, image_width - kernel_size//2):
            region = image[i - kernel_size//2:i + kernel_size//2 + 1, j - kernel_size//2:j + kernel_size//2 + 1]
            eroded_image[i, j] = np.min(region * structure_element)
    return eroded_image

# Perform dilation
dilated_image = dilate(padded_image, kernel_size)

# Perform erosion
closed_image = erode(dilated_image, kernel_size)

# Remove padding from the closed image
final_image = closed_image[kernel_size//2:-kernel_size//2, kernel_size//2:-kernel_size//2]

# Display the image
plt.imshow(final_image, cmap='gray')
plt.title('Closed Image')
plt.axis('off')
plt.show()
