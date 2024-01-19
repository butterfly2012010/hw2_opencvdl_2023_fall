import numpy as np
import cv2
from scipy.ndimage import binary_erosion, binary_dilation

def load_and_threshold_image(image_path, threshold=127):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Binarize the grayscale image
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    return binary_img

def pad_image(img, kernel_size):
    # Pad the image with zeros based on the kernel size
    pad_size = kernel_size // 2
    return np.pad(img, pad_size, mode='constant', constant_values=0)

def custom_dilation(img, kernel_size):
    # Create a structuring element
    structuring_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    # Perform dilation
    dilated_img = binary_dilation(img, structure=structuring_element).astype(np.uint8) * 255
    return dilated_img

def custom_erosion(img, kernel_size):
    # Create a structuring element
    structuring_element = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    
    # Perform erosion
    eroded_img = binary_erosion(img, structure=structuring_element).astype(np.uint8) * 255
    return eroded_img

def opening_operation(image_path):
    # Load and threshold the image
    binary_img = load_and_threshold_image(image_path)
    
    # Pad the image
    padded_img = pad_image(binary_img, 3)
    
    # Perform erosion then dilation
    eroded_img = custom_erosion(padded_img, 3)
    opened_img = custom_dilation(eroded_img, 3)
    
    # Removing padding
    opened_img = opened_img[1:-1, 1:-1]  # Adjust based on padding size
    return opened_img

# Perform opening operation on the provided image "closing.png"
closing_img_path = '/mnt/data/closing.png'
opened_img = opening_operation(closing_img_path)

# Save and display the result
opened_img_path = '/mnt/data/opened_image.png'
cv2.imwrite(opened_img_path, opened_img)

opened_img_path
