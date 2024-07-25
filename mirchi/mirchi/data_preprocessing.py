import cv2
import os
import numpy as np

# Define the path to your dataset
dataset_path = "C:/Users/Akhil/Desktop/CHILLI"

# Define the desired image size
image_size = (256, 256)

# Create an empty list to store preprocessed images and labels
preprocessed_images = []
labels = []

# Iterate through the images in the dataset directory
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        # Load the image using OpenCV
        image_path = os.path.join(root, file)
        image = cv2.imread(image_path)
 
        # Resize the image to the desired size
        image = cv2.resize(image, image_size)

        # Perform any additional preprocessing steps, such as normalization or data augmentation

        # Append the preprocessed image to the list
        preprocessed_images.append(image)

        # Extract the label from the file name or directory path, depending on your dataset structure
        label = os.path.basename(root) # Assumes that the directory name is the label
        # Alternatively, you can extract the label from the file name using string processing techniques
        # label = file.split("_")[0] # Assumes that the label is the first part of the file name before an underscore
        labels.append(label)

# Convert the lists to NumPy arrays for further processing
preprocessed_images = np.array(preprocessed_images)
labels = np.array(labels)

# Perform any additional processing or splitting of the dataset, such as train-test split or data shuffling, as needed

# Now you can use the preprocessed_images and labels for training your machine learning model
import numpy as np
from sklearn.model_selection import train_test_split

# Assume you have already preprocessed your images and labels
preprocessed_images = ... # shape: (num_samples, image_height, image_width, num_channels)
labels = ... # shape: (num_samples,)

# Perform train-test split with a 80-20 ratio (80% for training, 20% for testing)
train_images, test_images, train_labels, test_labels = train_test_split(preprocessed_images, labels, test_size=0.2, random_state=42)

# Optionally, you can also perform validation set split using train_test_split function
# Perform train-validation-test split with 70-15-15 ratio (70% for training, 15% for validation, 15% for testing)
train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.15, random_state=42)

# Now you have train_images, train_labels for training, val_images, val_labels for validation, and test_images, test_labels for testing
# You can use these arrays to feed into your machine learning model for training, validation, and testing respectively

