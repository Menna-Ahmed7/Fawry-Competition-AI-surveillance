import shutil
import random
import os

def split_dataset(image_dir, label_dir, dataset_dir, train_ratio=0.8):
    """
    Splits the dataset into train and validation sets.
    - image_dir: Directory containing the input images.
    - label_dir: Directory containing the corresponding label files.
    - dataset_dir: Directory where the split dataset will be stored.
    - train_ratio: Proportion of images to use for training (default 0.8).
    """
    # Create train/val directories for images and labels
    for split in ["train", "val"]:
        os.makedirs(os.path.join(dataset_dir, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "labels", split), exist_ok=True)
    
    # List images and shuffle them
    images = sorted([f for f in os.listdir(image_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    random.shuffle(images)
    
    # Calculate the split sizes
    train_size = int(train_ratio * len(images))
    train_images = images[:train_size]
    val_images = images[train_size:]
    
    # Copy training files
    for img in train_images:
        # Copy image
        shutil.copy(os.path.join(image_dir, img), os.path.join(dataset_dir, "images", "train", img))
        # Determine corresponding label file name (adjust extension if needed)
        label = os.path.splitext(img)[0] + ".txt"
        if os.path.exists(os.path.join(label_dir, label)):
            shutil.copy(os.path.join(label_dir, label), os.path.join(dataset_dir, "labels", "train", label))
        else:
            print(f"Warning: Label file for {img} not found in {label_dir}.")
    
    # Copy validation files
    for img in val_images:
        shutil.copy(os.path.join(image_dir, img), os.path.join(dataset_dir, "images", "val", img))
        label = os.path.splitext(img)[0] + ".txt"
        if os.path.exists(os.path.join(label_dir, label)):
            shutil.copy(os.path.join(label_dir, label), os.path.join(dataset_dir, "labels", "val", label))
        else:
            print(f"Warning: Label file for {img} not found in {label_dir}.")
    
    print("Dataset split complete.")

if __name__ == '__main__':
    # Define the paths (adjust these paths to your environment)
    image_dir = r"D:\GItHub Reops\yolov11\02_selected_img"
    label_dir = r"D:\GItHub Reops\yolov11\02_selected_labels"
    dataset_dir = r"img_02/dataset"
    
    split_dataset(image_dir, label_dir, dataset_dir, train_ratio=0.8)
