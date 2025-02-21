import os
import glob
import cv2
import numpy as np
import shutil
from tqdm.contrib.concurrent import process_map


BASE_DATASET_PTH = "/home/klymy/data/people_detection/filtered_v3_margin_15_median_blur"
BASE_PTH = "/home/klymy/data/people_detection/filtered_v3_margin_15_median_blur/images"

shutil.copytree(
    "/home/klymy/data/people_detection/filtered_v3_margin_15",
    BASE_DATASET_PTH
)

for i in ["project.yaml", "project_algotest.yaml", "project_holdout.yaml"]:
    with open(os.path.join(BASE_DATASET_PTH, i), "r") as f:
        lines = f.readlines()
    lines[0] = f"path: {BASE_DATASET_PTH}\n"

    with open(os.path.join(BASE_DATASET_PTH, i), "w") as f:
        f.writelines(lines)


img_paths = []
for i in ["train", "val", "tst", "holdout", "algotest"]:
    pth = os.path.join(BASE_PTH, i)
    img_paths += [os.path.join(pth, j) for j in glob.glob(os.path.join(pth, "*.jpg"))]

print(len(img_paths))


def blur(img, kernel_size=(15, 15)):
    # return cv2.GaussianBlur(img, kernel_size, 0)
    return cv2.medianBlur(img, kernel_size[0])

def adjast_contrast_brightness(img, alpha=1, beta=0):
    return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

def adjast_gamma(img, gamma=1):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(img, table)


def processor(img_pth):
   img = cv2.imread(img_pth)
   h, _, _ = img.shape
   kernel_size = np.random.randint(13, 19, 1)[0]
   if h < 500:
       kernel_size /= 2
       kernel_size = int(kernel_size)
   if kernel_size % 2 == 0:
      kernel_size += 1

   alpha = 1.1 + np.random.randint(-2, 2, 1)[0] / 10
   beta = np.random.randint(5, 15, 1)[0]
   gamma = 1.5 + np.random.randint(-5, 5, 1)[0] / 10

   processed_img = adjast_gamma(
       adjast_contrast_brightness(blur(img, (kernel_size, kernel_size)), alpha=alpha, beta=beta), gamma=gamma)

   cv2.imwrite(img_pth, processed_img)


np.random.seed(42)
r = process_map(processor, img_paths, max_workers=8)
