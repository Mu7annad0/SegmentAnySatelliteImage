import cv2
import numpy as np
import os
import glob
from tqdm import tqdm
import shutil

folder = '/Users/mu7annad.0gmail.com/Documents/gradDataset/loveDa'
dest_folder = '/Users/mu7annad.0gmail.com/Documents/gradDataset/loveDa_yolo'

# Define directory paths for images and masks
for split in ['Train', 'Valid']:
    SRC_DIR = f'{folder}/{split}'
    DEST_DIR = f'{dest_folder}/{split}'

    # Create directories for images and labels
    os.makedirs(os.path.join(DEST_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(DEST_DIR, 'labels'), exist_ok=True)

    # Get mask file paths
    mask_paths = glob.glob(os.path.join(SRC_DIR, "*/masks_png/*.png"))
    mask_paths.sort()

    # Iterate through mask files
    for mask_path in tqdm(mask_paths):
        img = cv2.imread(mask_path)
        h, w = img.shape[0:2]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = np.zeros((8, img.shape[0], img.shape[1]), np.uint8)

        polygons = [[], [], [], [], [], [], [], []]

        mask_classes = [
            ("no-data",     0),
            ("background",  1),
            ("building",    2),
            ("road",        3),
            ("water",       4),
            ("barren",      5),
            ("forest",      6),
            ("agriculture", 7)]

        # Detect contours for each mask class
        for mask_class in mask_classes:
            color_range_lower = np.array((mask_class[1]), np.uint8)
            color_range_upper = np.array((mask_class[1]), np.uint8)

            color_range_seg = cv2.inRange(img, color_range_lower, color_range_upper)

            (contours, _) = cv2.findContours(color_range_seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                polygons[mask_class[1]].append(contours)

        # Convert polygons to YOLO labels
        mask_name = os.path.split(mask_path)[-1].split('.')[0]

        labels = []

        for id, classes_datas in enumerate(polygons):
            if classes_datas:
                for polygon in classes_datas[0]:
                    label = f'{(id - 2)} '

                    for xy in polygon:
                        label += f'{round(xy[0][0]/h, 5)} {round(xy[0][1]/w, 5)} '

                    label += '\n'
                    labels.append(label)

        # Write YOLO labels to text files
        label_dest_path = os.path.join(DEST_DIR, 'labels', f'{mask_name}.txt')
        with open(label_dest_path, 'w') as text:
            text.writelines(labels)

        # Copy images to destination folder
        img_src_path = mask_path.replace("/masks_png/", "/images_png/")
        img_dest_path = os.path.join(DEST_DIR, 'images', f'{mask_name}.png')
        shutil.copyfile(img_src_path, img_dest_path)
