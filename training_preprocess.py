import cv2
import numpy as np
import os
import glob

def resize_with_aspect_ratio_and_padding(img, target_size=(28, 28)):
    # Calculate the ratio of the target dimensions
    h, w = img.shape[:2]
    scale = min(target_size[1] / w, target_size[0] / h)

    # New dimensions
    new_w, new_h = int(scale * w), int(scale * h)
    resized_img = cv2.resize(img, (new_w, new_h))

    # Prepare padding
    top = (target_size[0] - new_h) // 2
    bottom = target_size[0] - new_h - top
    left = (target_size[1] - new_w) // 2
    right = target_size[1] - new_w - left

    # Add padding
    padded_img = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_img

dataset_base_path = 'data/testing_data'  # Base path of your dataset
target_size = (28, 14)

for label_folder in os.listdir(dataset_base_path):
    folder_path = os.path.join(dataset_base_path, label_folder)
    if not os.path.isdir(folder_path):
        continue

    for image_path in glob.glob(os.path.join(folder_path, '*.png')):
        img = cv2.imread(image_path)
        resized_padded_img = resize_with_aspect_ratio_and_padding(img, target_size=target_size)

        # Save the processed image
        cv2.imwrite(image_path, resized_padded_img)  # This will overwrite the original image
