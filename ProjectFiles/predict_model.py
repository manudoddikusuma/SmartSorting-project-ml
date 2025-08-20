import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

# --- Configuration ---
model_path = 'healthy_vs_rotten.h5' # Path to your trained model
IMG_SIZE = (224, 224) # Image size used during training
# Important: Ensure these class names match the order in train_generator.class_indices
# You can check this by running train_model.py and looking at the "Train classes:" output.
# It's usually {'fresh': 0, 'rotten': 1} or {'rotten': 0, 'fresh': 1}
# Adjust this list if 'rotten' comes before 'fresh' based on your output.
class_names = ['fresh', 'rotten'] # Example, adjust if needed

# --- 1. Load the Trained Model ---
print("--- Loading the trained model ---")
try:
    model = load_model(model_path)
    print(f"Model '{model_path}' loaded successfully.")
    model.summary()
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure 'healthy_vs_rotten.h5' is in the same directory as this script.")
    exit() # Exit if model cannot be loaded

# --- 2. Evaluate Model on Test Data (Optional, but Recommended) ---
# If your train_model.py is still open, you can get test_generator from there.
# If not, you'll need to re-initialize it here.
# For simplicity, we'll re-initialize test_generator for evaluation purposes.

from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_dir = 'dataset_split/test'

# Only rescaling for test images
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=32, # Same batch size as training
    class_mode='binary', # Must match how it was trained
    shuffle=False # Important for consistent evaluation
)

print("\n--- Evaluating Model on Test Data ---")
if test_generator.samples > 0:
    loss, accuracy = model.evaluate(test_generator)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
else:
    print("No test images found. Skipping evaluation.")


# --- 3. Function to Predict a Single Image ---
def predict_single_image(image_path):
    print(f"\n--- Predicting for image: {image_path} ---")
    try:
        img = load_img(image_path, target_size=IMG_SIZE)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = img_array / 255.0 # Rescale pixels to [0,1]

        predictions = model.predict(img_array)
        # For binary classification with softmax on 2 units, predictions will be like [[prob_class0, prob_class1]]
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        confidence = predictions[0][predicted_class_index]

        predicted_class_name = class_names[predicted_class_index]

        print(f"Raw predictions: {predictions}")
        print(f"Predicted Class: {predicted_class_name}")
        print(f"Confidence: {confidence:.2f}")

        return predicted_class_name, confidence

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return None, None

# --- Example Usage for Single Image Prediction ---
print("\n--- Running Example Single Image Predictions ---")

# Replace these paths with actual image paths from your 'dataset_split/test' folder
# or any other image you want to test.
# Make sure the paths are correct relative to where you run the script.

# Example 1: Fresh image (replace with an actual path from your test/fresh folder)
fresh_image_path = os.path.join('dataset_split', 'test', 'fresh','186.jpg') # Adjust if your test/fresh folder has direct images
# The original split_data.py created subfolders like apple_fresh inside fresh, so you might need to adjust paths
# Example: If your fresh images are directly in 'dataset_split/test/fresh', use:
# fresh_image_path = os.path.join('dataset_split', 'test', 'fresh', 'some_fresh_image.jpg')
# You need to manually check the content of 'dataset_split/test/fresh' and 'dataset_split/test/rotten'
# to get valid image paths.

# Let's try to find an actual image from dataset_split/test/fresh
test_fresh_dir = os.path.join('dataset_split', 'test', 'fresh')
sample_fresh_image = None
for root, _, files in os.walk(test_fresh_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            sample_fresh_image = os.path.join(root, f)
            break
    if sample_fresh_image:
        break

if sample_fresh_image:
    predict_single_image(sample_fresh_image)
else:
    print(f"Could not find a sample fresh image in {test_fresh_dir}")

print("-" * 30)

# Example 2: Rotten image (replace with an actual path from your test/rotten folder)
rotten_image_path = os.path.join('dataset_split', 'test', 'rotten', 'apple_rotten', '0_100.jpg') # Adjust similarly
test_rotten_dir = os.path.join('dataset_split', 'test', 'rotten')
sample_rotten_image = None
for root, _, files in os.walk(test_rotten_dir):
    for f in files:
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            sample_rotten_image = os.path.join(root, f)
            break
    if sample_rotten_image:
        break

if sample_rotten_image:
    predict_single_image(sample_rotten_image)
else:
    print(f"Could not find a sample rotten image in {test_rotten_dir}")