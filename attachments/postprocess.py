import os
import shutil
import random
from path import directory_output, output_folder

# Create output directory if it doesn't exist
os.makedirs(directory_output, exist_ok=True)

# Percentage of images to use for validation
validation_split = 0.2

# Loop through each class folder
for class_folder in os.listdir(output_folder):
    class_path = os.path.join(output_folder, class_folder)

    # Create class subdirectories in the output directory for training and validation
    train_class_dir = os.path.join(directory_output, 'train', class_folder)
    val_class_dir = os.path.join(directory_output, 'validation', class_folder)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(val_class_dir, exist_ok=True)

    # Get list of image files in the class folder
    image_files = os.listdir(class_path)

    # Calculate number of images for validation
    num_validation_images = int(len(image_files) * validation_split)

    # Randomly select validation images
    validation_images = random.sample(image_files, num_validation_images)

    # Move validation images to validation directory
    for img in validation_images:
        src = os.path.join(class_path, img)
        dst = os.path.join(val_class_dir, img)
        shutil.copy(src, dst)

    # Move remaining images to training directory
    for img in image_files:
        src = os.path.join(class_path, img)
        dst = os.path.join(train_class_dir, img)
        shutil.copy(src, dst)
