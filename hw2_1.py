import cv2
import numpy as np
from matplotlib import pyplot as plt

# 1-1
# Load the image
image = cv2.imread('/content/coins.jpg')
original_image = image.copy()
processed_image = image.copy()
circle_center_image = image.copy()

# Convert to grayscale
gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

# Remove noise
gray_blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Circle detection
circles = cv2.HoughCircles(gray_blurred,
              cv2.HOUGH_GRADIENT, 1, 20,
              param1=50, param2=30, minRadius=10, maxRadius=30)

# Ensure at least some circles were found
if circles is not None:
    # Convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    # Loop over the (x, y) coordinates and radius of the circles
    for (x, y, r) in circles:
        # Draw the circle in the output image
        cv2.circle(processed_image, (x, y), r, (0, 255, 0), 2)
        # Draw the center of the circle
        cv2.circle(circle_center_image, (x, y), 2, (0, 0, 255), 2)

# Display the images
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
ax[0].set_title('Original Image')
ax[0].axis('off')

ax[1].imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
ax[1].set_title('Processed Image')
ax[1].axis('off')

ax[2].imshow(cv2.cvtColor(circle_center_image, cv2.COLOR_BGR2RGB))
ax[2].set_title('Circles Center Detected')
ax[2].axis('off')

plt.show()



# 1-2
# Count how many coins in the image
circles.shape[0]