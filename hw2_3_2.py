import cv2
import numpy as np
from matplotlib import pyplot as plt

# Load the image for the opening operation
image_path_opening = 'E:/YN/opencvdl/Hw2/Dataset_OpenCvDl_Hw2/Q3/opening.png'
image_opening = cv2.imread(image_path_opening)

# Convert to grayscale
gray_image_opening = cv2.cvtColor(image_opening, cv2.COLOR_BGR2GRAY)

# Binarize the image
_, binarized_image_opening = cv2.threshold(gray_image_opening, 127, 255, cv2.THRESH_BINARY)

# Padding the image with zeros
kernel_size = 3
padded_image_opening = np.pad(binarized_image_opening, ((kernel_size//2, kernel_size//2), (kernel_size//2, kernel_size//2)), mode='constant', constant_values=0)

# Perform erosion followed by dilation to perform opening operation
# We can use the previously defined erode and dilate functions
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

# Perform erosion
eroded_image_opening = erode(padded_image_opening, kernel_size)

# Perform dilation
opened_image = dilate(eroded_image_opening, kernel_size)

# Remove padding from the opened image
final_image_opening = opened_image[kernel_size//2:-kernel_size//2, kernel_size//2:-kernel_size//2]

# Display the image
plt.imshow(final_image_opening, cmap='gray')
plt.title('Opened Image')
plt.axis('off')
plt.show()
