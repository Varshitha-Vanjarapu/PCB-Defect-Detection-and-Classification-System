import cv2
import os
import numpy as np
DATASET_DIR = "data/raw/images"
GOOD_DIR = os.path.join(DATASET_DIR, "Good")
MASK_DIR = "outputs/masks"
os.makedirs(MASK_DIR, exist_ok=True)
# Load ONE good reference image (demo-safe)
good_name = os.listdir(GOOD_DIR)[0]
good_img = cv2.imread(os.path.join(GOOD_DIR, good_name))
good_img = cv2.resize(good_img, (512, 512))
good_gray = cv2.cvtColor(good_img, cv2.COLOR_BGR2GRAY)
good_gray = cv2.GaussianBlur(good_gray, (5, 5), 0)
# Automatically detect defect folders
defect_folders = [
    f for f in os.listdir(DATASET_DIR)
    if f != "Good" and os.path.isdir(os.path.join(DATASET_DIR, f))]

print("Defect folders found:", defect_folders)

for defect in defect_folders:
    defect_path = os.path.join(DATASET_DIR, defect)
    out_path = os.path.join(MASK_DIR, defect)
    os.makedirs(out_path, exist_ok=True)

    for img_name in os.listdir(defect_path):
        img = cv2.imread(os.path.join(defect_path, img_name))
        if img is None:
            continue

        img = cv2.resize(img, (512, 512))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(good_gray, gray)
        _, mask = cv2.threshold(diff, 12, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 2), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        cv2.imwrite(os.path.join(out_path, img_name), mask)

print("Image subtraction completed.")
