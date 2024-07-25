import matplotlib.pyplot as plt
import os

# Set the path to your image dataset directory
dataset_dir = "C:/Users/Akhil/Desktop/AIML/CHILLI"

# Define the categories or classes in your dataset
categories = ["dhamini", "teja", "jyothika"]

# Count the number of images in each category
image_counts = []
for category in categories:
    category_dir = os.path.join(dataset_dir, category)
    image_files = [f for f in os.listdir(category_dir) if f.endswith(".jpg") or f.endswith(".png")]
    image_counts.append(len(image_files))

# Create a bar graph
plt.bar(categories, image_counts)
plt.title("Image Dataset Categories")
plt.xlabel("Category")
plt.ylabel("Image Count")
plt.show()