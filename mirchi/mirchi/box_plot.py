import cv2
import matplotlib.pyplot as plt

# Example image dataset
dataset = ["C:/Users/Akhil/Desktop/AIML/CHILLI/jyothika/3 (1).jpg", "C:/Users/Akhil/Desktop/AIML/CHILLI/jyothika/3 (2).jpg", "C:/Users/Akhil/Desktop/AIML/CHILLI/jyothika/3 (3).jpg", "C:/Users/Akhil/Desktop/AIML/CHILLI/jyothika/3 (4).jpg"]

# Extract hue, saturation, and intensity values from image dataset
hue_values = []
saturation_values = []
intensity_values = []

for image_path in dataset:
    # Load image in HSV color space
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract hue, saturation, and value channels
    hue = hsv[:, :, 0].flatten()
    saturation = hsv[:, :, 1].flatten()
    intensity = hsv[:, :, 2].flatten()
    
    # Append values to respective lists
    hue_values.extend(hue)
    saturation_values.extend(saturation)
    intensity_values.extend(intensity)

# Create a box plot for hue, saturation, and intensity values
fig, ax = plt.subplots()
ax.boxplot([hue_values, saturation_values, intensity_values])
ax.set_xticklabels(['Hue', 'Saturation', 'Intensity'])
ax.set_title('Image Dataset Box Plot (HSI)')
ax.set_ylabel('Values')

# Show the plot
plt.show()
