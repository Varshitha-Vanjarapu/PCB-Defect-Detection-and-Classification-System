import cv2
import os

MASK_DIR = "outputs/masks"
ROI_DIR = "outputs/rois"
IMAGE_DIR = "data/raw/images"

os.makedirs(ROI_DIR, exist_ok=True)

defect_folders = [
    f for f in os.listdir(MASK_DIR)
    if os.path.isdir(os.path.join(MASK_DIR, f))
]

print("Defect folders for ROI extraction:", defect_folders)

PADDING = 40   # important for context

for defect in defect_folders:

    mask_folder = os.path.join(MASK_DIR, defect)
    image_folder = os.path.join(IMAGE_DIR, defect)
    roi_folder = os.path.join(ROI_DIR, defect)

    os.makedirs(roi_folder, exist_ok=True)

    for mask_name in os.listdir(mask_folder):

        mask_path = os.path.join(mask_folder, mask_name)
        image_path = os.path.join(image_folder, mask_name)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(image_path)

        if mask is None or img is None:
            continue

        # resize BOTH image and mask
        img = cv2.resize(img, (512, 512))
        mask = cv2.resize(mask, (512, 512))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        roi_count = 0

        for cnt in contours:

            x, y, w, h = cv2.boundingRect(cnt)

            if w < 5 or h < 5:
                continue

            # add padding
            x1 = max(0, x - PADDING)
            y1 = max(0, y - PADDING)
            x2 = min(512, x + w + PADDING)
            y2 = min(512, y + h + PADDING)

            roi = img[y1:y2, x1:x2]

            roi_count += 1

            roi_name = f"{os.path.splitext(mask_name)[0]}_roi_{roi_count}.png"

            cv2.imwrite(os.path.join(roi_folder, roi_name), roi)

        if roi_count == 0:
            roi_none_name = f"{os.path.splitext(mask_name)[0]}_roi_none.png"
            cv2.imwrite(os.path.join(roi_folder, roi_none_name), img)

print("ROI extraction completed successfully.")