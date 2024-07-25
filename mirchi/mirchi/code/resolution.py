import cv2

# Load an image using OpenCV
image_path = "image.jpg"  # Replace with the path to your image
image = cv2.imread("C:/Users/Akhil/Desktop/AIML/CHILLI/jyothika/3 (1).jpg")

# Extract image resolution (width and height)
image_resolution = image.shape[:2]  # Tuple of (height, width)

# Calculate average pixel value
pixel_values = cv2.mean(image)

print("Image Resolution: ", image_resolution)
print("Pixel Values (BGR): ", pixel_values[:3])
print("Pixel Values (Grayscale): ", pixel_values[0])
