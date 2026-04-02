import cv2
import os
# PATHS
DATASET_DIR = "data/raw/images"
OUTPUT_DIR = "outputs/augmentation/flips"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Automatically detect defect folders (exclude Good)
defect_folders = [
    f for f in os.listdir(DATASET_DIR)
    if f != "Good" and os.path.isdir(os.path.join(DATASET_DIR, f))]
print("Defect folders found for augmentation:", defect_folders)
# FLIP AUGMENTATION
for defect in defect_folders:
    input_dir = os.path.join(DATASET_DIR, defect)
    output_dir = os.path.join(OUTPUT_DIR, defect)
    os.makedirs(output_dir, exist_ok=True)
    for img_name in os.listdir(input_dir):
        img_path = os.path.join(input_dir, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue
        # Resize for consistency
        img = cv2.resize(img, (512, 512))
        name, ext = os.path.splitext(img_name)
        # Original
        cv2.imwrite(os.path.join(output_dir, f"{name}_orig{ext}"), img)
        # Horizontal flip
        h_flip = cv2.flip(img, 1)
        cv2.imwrite(os.path.join(output_dir, f"{name}_hflip{ext}"), h_flip)
        # Vertical flip
        v_flip = cv2.flip(img, 0)
        cv2.imwrite(os.path.join(output_dir, f"{name}_vflip{ext}"), v_flip)
        # Horizontal + Vertical flip
        hv_flip = cv2.flip(img, -1)
        cv2.imwrite(os.path.join(output_dir, f"{name}_hvflip{ext}"), hv_flip)
print("Flip augmentation completed successfully.")