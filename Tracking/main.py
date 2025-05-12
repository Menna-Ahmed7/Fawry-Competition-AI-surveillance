import os
import shutil

def main():
    # Define input and output directories (assumed to be in the same folder as this script)
    input_images_dir = r"c:\Users\11abd\Downloads\surveillance-for-retail-stores\tracking\train\05\img1"
    input_labels_dir = r"c:\Users\11abd\Downloads\labels\labels_05\labels"
    output_images_dir = "05_selected_img"
    output_labels_dir = "05_selected_labels"

    # Create output directories if they don't exist
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    # List and sort image files (assumes file names determine the order correctly)
    image_files = sorted([
        f for f in os.listdir(input_images_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
    ])

    # Define the frame interval: keep the first image and every 50th image thereafter.
    frame_interval = 15
    selected_images = []

    # Loop over images using their index (index 0 is frame 1)
    for idx, image in enumerate(image_files):
        # Keep first image or every image where (index + 1) is a multiple of frame_interval.
        if idx == 0 or (idx + 1) % frame_interval == 0:
            selected_images.append(image)

    print(f"Selected {len(selected_images)} images out of {len(image_files)} total images.")

    # Copy selected images and corresponding label files
    for image in selected_images:
        # Copy the image file
        src_image_path = os.path.join(input_images_dir, image)
        dst_image_path = os.path.join(output_images_dir, image)
        shutil.copy2(src_image_path, dst_image_path)

        # Construct the corresponding label file name
        base_name, _ = os.path.splitext(image)
        label_file = base_name + ".txt"
        src_label_path = os.path.join(input_labels_dir, label_file)
        dst_label_path = os.path.join(output_labels_dir, label_file)

        # Copy the label file if it exists; otherwise, log a warning.
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)
        else:
            print(f"Warning: Label file '{src_label_path}' not found.")

    print("Done copying selected images and label files.")

if __name__ == "__main__":
    main()
