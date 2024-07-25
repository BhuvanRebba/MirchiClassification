import matplotlib.pyplot as plt
import cv2
import os

# Set the path to your image dataset directory
dataset_dir = "C:/Users/Akhil/Desktop/AIML/CHILLI/dhamini"

# Get a list of all image files in the dataset directory
image_files = [os.path.join(dataset_dir, f) for f in os.listdir(dataset_dir) if f.endswith(".jpg") or f.endswith(".png")]

# Loop through the image files and plot histograms of pixel intensities using matplotlib
for image_file in image_files:
    # Load the image using OpenCV
    image = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)  # Load the image as grayscale
    
    # Flatten the image to a 1D array for histogram calculation
    pixel_values = image.flatten()
    
    # Plot a histogram of pixel intensities
    plt.hist(pixel_values, bins=256, range=(0, 255))
    plt.title("Pixel Intensity Histogram")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.show()
