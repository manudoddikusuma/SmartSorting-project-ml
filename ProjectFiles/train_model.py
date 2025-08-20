import os
import shutil
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

# --- 1. Data Splitting Logic ---
# This will take images from static/assets/dataset and split them into dataset_split/train/fresh, dataset_split/train/rotten etc.
source_dir = 'static/assets/dataset'
target_dir = 'dataset_split'
classes_binary = ['fresh', 'rotten'] # Confirmed binary classification categories
split_ratio = {'train': 0.7, 'val': 0.15, 'test': 0.15}

print("--- Starting Data Splitting ---")
# Clean previous split if it exists
if os.path.exists(target_dir):
    print(f"Removing existing {target_dir} directory...")
    shutil.rmtree(target_dir)

# Create base folders for splits (e.g., dataset_split/train/fresh)
for split in split_ratio:
    for cls in classes_binary:
        os.makedirs(os.path.join(target_dir, split, cls), exist_ok=True)

# Go through each main class (fresh/rotten) to collect and split images
for cls in classes_binary:
    class_root_dir = os.path.join(source_dir, cls)
    image_paths_in_class = []

    # Walk through subfolders (like 'apple', 'banana' etc.) to find all images for this 'fresh' or 'rotten' class
    for root, _, files in os.walk(class_root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                image_paths_in_class.append(os.path.join(root, file))

    if not image_paths_in_class:
        print(f"⚠️ Warning: No images found in {class_root_dir}")
        continue

    random.shuffle(image_paths_in_class)
    total_images = len(image_paths_in_class)
    train_end = int(total_images * split_ratio['train'])
    val_end = train_end + int(total_images * split_ratio['val'])

    train_images = image_paths_in_class[:train_end]
    val_images = image_paths_in_class[train_end:val_end]
    test_images = image_paths_in_class[val_end:]

    print(f"Splitting {cls} images: Train={len(train_images)}, Val={len(val_images)}, Test={len(test_images)}")

    # Copy images to their respective split folders
    for img_list, split_name in [(train_images, 'train'), (val_images, 'val'), (test_images, 'test')]:
        for src_path in img_list:
            # The destination path will be like dataset_split/train/fresh/image.jpg
            filename = os.path.basename(src_path)
            dst_path = os.path.join(target_dir, split_name, cls, filename)
            try:
                shutil.copy2(src_path, dst_path)
            except Exception as e:
                print(f"❌ Failed to copy {filename} to {dst_path}: {e}")

    print("✅ Data splitting and copying complete!")


# --- 2. Data Generators ---
train_dir = os.path.join(target_dir, 'train')
val_dir = os.path.join(target_dir, 'val')
test_dir = os.path.join(target_dir, 'test') # Note: Test generator will be used for final evaluation later

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Data augmentation for training images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Only rescaling for validation and test images
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Load images using flow_from_directory for binary classification
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary' # Set to 'binary' for 2 classes (fresh/rotten)
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary' # Set to 'binary' for 2 classes
)

test_generator = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary', # Set to 'binary' for 2 classes
    shuffle=False # Keep order for later prediction/evaluation
)

print("\n--- Data Generators Initialized ---")
print("Train classes:", train_generator.class_indices) # Should show {'fresh': 0, 'rotten': 1} or vice-versa
print("Validation classes:", val_generator.class_indices)
print("Test classes:", test_generator.class_indices)
print(f"Found {train_generator.samples} training images.")
print(f"Found {val_generator.samples} validation images.")
print(f"Found {test_generator.samples} test images.")


# --- 3. Model Building, Compilation, and Training ---

# Initialize VGG16 base model (pre-trained on ImageNet)
vgg16 = VGG16(include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Freeze VGG16 layers so their weights are not updated during training
for layer in vgg16.layers:
    layer.trainable = False

# Build the custom classification head on top of VGG16's output
x = Flatten()(vgg16.output)
# Output layer for binary classification (2 classes: fresh, rotten)
# Using softmax and 2 units with sparse_categorical_crossentropy is consistent for binary with integer labels
output = Dense(2, activation='softmax')(x)

# Create the final model
model = Model(inputs=vgg16.input, outputs=output)

# Print model summary to see the architecture and number of parameters
print("\n--- Model Summary ---")
model.summary()

# Define Early Stopping callback to prevent overfitting
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Define Adam optimizer with a learning rate
opt = Adam(learning_rate=0.0001)

# Compile the model
# Using sparse_categorical_crossentropy because class_mode='binary' provides integer labels (0 or 1)
model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
print("\n--- Starting Model Training ---")
EPOCHS = 15 # Number of training epochs
# Calculate steps per epoch based on the number of samples and batch size
STEPS_PER_EPOCH = train_generator.samples // train_generator.batch_size
VALIDATION_STEPS = val_generator.samples // val_generator.batch_size

# Fit the model to the training data
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=val_generator,
    validation_steps=VALIDATION_STEPS,
    callbacks=[early_stopping] # Apply early stopping
)

print("\n--- Model Training Completed ---")

# Save the trained model for future use (e.g., in a Flask application)
model_save_path = 'healthy_vs_rotten.h5'
model.save(model_save_path)
print(f"Model saved to {model_save_path}")