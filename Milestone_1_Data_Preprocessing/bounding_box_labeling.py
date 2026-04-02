import cv2
import os

MASK_DIR = "outputs/masks"
BOX_DIR = "outputs/bounding_boxes"
IMAGE_DIR = "data/raw/images"
os.makedirs(BOX_DIR, exist_ok=True)
# Detect defect folders dynamically
defect_folders = [
    f for f in os.listdir(MASK_DIR)
    if os.path.isdir(os.path.join(MASK_DIR, f))]

print("Defect folders for labeling:", defect_folders)

for defect in defect_folders:
    mask_folder = os.path.join(MASK_DIR, defect)
    image_folder = os.path.join(IMAGE_DIR, defect)
    box_folder = os.path.join(BOX_DIR, defect)
    os.makedirs(box_folder, exist_ok=True)
    for mask_name in os.listdir(mask_folder):
        mask_path = os.path.join(mask_folder, mask_name)
        image_path = os.path.join(image_folder, mask_name)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(image_path)
        # SAFETY CHECK (MOST IMPORTANT)
        if mask is None or img is None:
            continue

        img = cv2.resize(img, (512, 512))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 4 or h < 4:
                continue

            cv2.rectangle(img,(x, y),(x + w, y + h),(0, 0, 255),1)
            font=cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, defect, (x, y - 5), font, 0.4, (0, 0, 255), 1, cv2.LINE_AA)
        # ALWAYS save labeled image
        cv2.imwrite(os.path.join(box_folder, mask_name), img)

print("Bounding box labeling completed successfully.")
