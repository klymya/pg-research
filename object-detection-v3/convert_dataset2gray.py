import os
import shutil
import argparse

import cv2
from tqdm import tqdm


def get_image_paths(dataset_path):
    image_paths = []
    image_paths = os.listdir(dataset_path)
    image_paths = [os.path.join(dataset_path, image_path) for image_path in image_paths]
    return image_paths


parser = argparse.ArgumentParser(description='Convert dataset to gray')
parser.add_argument('--input', type=str, required=True, help='Input dataset path')
args = parser.parse_args()

gray_dataset_path = args.input + "_gray"

# Copy dataset
shutil.copytree(args.input, gray_dataset_path)

# Remove cache files
os.remove(os.path.join(gray_dataset_path, "labels", "train.cache"))
os.remove(os.path.join(gray_dataset_path, "labels", "val.cache"))

# Convert images to gray
img_paths = get_image_paths(os.path.join(gray_dataset_path, "images", "train")) + \
    get_image_paths(os.path.join(gray_dataset_path, "images", "val"))

for img_path in tqdm(img_paths):
    img = cv2.imread(img_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_path, img_gray)

# Update project file
project_file_path = os.path.join(gray_dataset_path, "project.yaml")
with open(project_file_path, "r") as f:
    project_file = f.read()

project_file = project_file.replace(args.input, gray_dataset_path)

with open(project_file_path, "w") as f:
    f.write(project_file)
