import cv2
import matplotlib.pyplot as plt

# Load an image using OpenCV
image_path = "C:/Users/Akhil/Desktop/AIML/CHILLI/dhamini/1 (1).jpg"  # Replace with the path to your image
image = cv2.imread("C:/Users/Akhil/Desktop/AIML/CHILLI/dhamini/1 (1).jpg")

# Convert the image from BGR to RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Plot the original image
plt.figure(figsize=(8, 8))
plt.imshow(image)
plt.title('Original Image')
plt.axis('off')
plt.show()

# Convert the image to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

# Plot the grayscale image
plt.figure(figsize=(8, 8))
plt.imshow(gray_image, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')
plt.show()

# Extract the red channel from the image
red_channel = image[:, :, 0]

# Plot the red channel image
plt.figure(figsize=(8, 8))
plt.imshow(red_channel, cmap='Reds')
plt.title('Red Channel')
plt.axis('off')
plt.show()

# Extract the green channel from the image
green_channel = image[:, :, 1]

# Plot the green channel image
plt.figure(figsize=(8, 8))
plt.imshow(green_channel, cmap='Greens')
plt.title('Green Channel')
plt.axis('off')
plt.show()

# Extract the blue channel from the image
blue_channel = image[:, :, 2]

# Plot the blue channel image
plt.figure(figsize=(8, 8))
plt.imshow(blue_channel, cmap='Blues')
plt.title('Blue Channel')
plt.axis('off')
plt.show()
