import os
import shutil
import random

IMAGE_DIR = "data/raw/images"
OUTPUT_BASE = "prepared_dataset"

TRAIN_DIR = os.path.join(OUTPUT_BASE, "train")
TEST_DIR = os.path.join(OUTPUT_BASE, "test")

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

print("Preparing 80-20 split from folder structure...")

for defect in os.listdir(IMAGE_DIR):

    defect_path = os.path.join(IMAGE_DIR, defect)

    if not os.path.isdir(defect_path):
        continue

    images = os.listdir(defect_path)

    if len(images) == 0:
        continue

    random.shuffle(images)

    split_index = int(0.8 * len(images))

    train_images = images[:split_index]
    test_images = images[split_index:]

    for img in train_images:
        src = os.path.join(defect_path, img)
        dst_folder = os.path.join(TRAIN_DIR, defect)
        os.makedirs(dst_folder, exist_ok=True)
        shutil.copy(src, dst_folder)

    for img in test_images:
        src = os.path.join(defect_path, img)
        dst_folder = os.path.join(TEST_DIR, defect)
        os.makedirs(dst_folder, exist_ok=True)
        shutil.copy(src, dst_folder)

print("80-20 Train/Test split completed successfully.")