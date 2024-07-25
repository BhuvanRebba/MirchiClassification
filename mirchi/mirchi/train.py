import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Example image dataset
dataset = ["C:/Users/Akhil/Desktop/AIML/CHILLI/dhamini/1 (1).jpg", "C:/Users/Akhil/Desktop/AIML/CHILLI/dhamini/1 (2).jpg", "C:/Users/Akhil/Desktop/AIML/CHILLI/dhamini/1 (3).jpg", "C:/Users/Akhil/Desktop/AIML/CHILLI/dhamini/1 (4).jpg"]
labels = [0, 1, 0, 1]  # Example labels for binary classification

# Extract features (hue, saturation, intensity, and RGB) from image dataset
features = []
for image_path in dataset:
    # Load image in HSV color space
    image = cv2.imread(image_path)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract hue, saturation, and value channels
    hue = hsv[:, :, 0].flatten()
    saturation = hsv[:, :, 1].flatten()
    intensity = hsv[:, :, 2].flatten()
    
    # Extract RGB values
    r = image[:, :, 0].flatten()
    g = image[:, :, 1].flatten()
    b = image[:, :, 2].flatten()
    
    # Combine features into a single feature vector and append to features list
    feature_vector = np.concatenate((hue, saturation, intensity, r, g, b), axis=0)
    features.append(feature_vector)

# Convert features list to NumPy array
X = np.array(features)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict labels for test set
y_pred = model.predict(X_test)

# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}".format(accuracy))
