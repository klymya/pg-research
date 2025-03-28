import os
import glob
from functools import partial
import cv2
import numpy as np
import shutil
from tqdm.contrib.concurrent import process_map


BASE_DATASET_PTHS = [
    "/home/klymy/data/people_detection/filtered_v3_margin_15_pixilized",
    "/home/klymy/data/people_detection/filtered_v3_margin_15_pixilized_sharped",
]


def pixelize(img: np.ndarray, scale_factor: float = 2) -> np.ndarray:
    """
    Pixelize the image by scaling it down and up.
    """
    # Downscale the image
    img = cv2.resize(img, None, fx=1/scale_factor, fy=1/scale_factor, interpolation=cv2.INTER_LINEAR)
    # Upscale the image
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    return img


def sharpen(img: np.ndarray) -> np.ndarray:
    # Create the sharpening kernel 
    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    # Sharpen the image
    sharpened_image = cv2.filter2D(img, -1, kernel)
    return sharpened_image


def processor(img_pth, do_sharpening: bool = False):
    img = cv2.imread(img_pth)

    w = img.shape[1]
    if w != 640:
        img = cv2.resize(img, (640, 480), cv2.INTER_AREA)

    if do_sharpening:
        img = sharpen(img)

    processed_img = pixelize(img, scale_factor=1.5)

    cv2.imwrite(img_pth, processed_img)


processor_with_sharpening = partial(processor, do_sharpening=True)


for dataset_pth, img_processor, desc in zip(
    BASE_DATASET_PTHS,
    (processor, processor_with_sharpening),
    ("Pixelizing images", "Pixelizing and sharpening images"),
):

    base_pth = os.path.join(dataset_pth, "images")

    shutil.rmtree(dataset_pth, ignore_errors=True)
    shutil.copytree(
        "/home/klymy/data/people_detection/filtered_v3_margin_15",
        dataset_pth
    )

    # Remove cache files
    cache_pths = glob.glob(os.path.join(dataset_pth, "labels", "*.cache"))
    print(cache_pths)
    for cache_pth in cache_pths:
        os.remove(cache_pth)

    for i in ["project.yaml", "project_algotest.yaml", "project_holdout.yaml"]:
        with open(os.path.join(dataset_pth, i), "r") as f:
            lines = f.readlines()
        lines[0] = f"path: {dataset_pth}\n"

        with open(os.path.join(dataset_pth, i), "w") as f:
            f.writelines(lines)

    img_paths = []
    for i in ["train", "val", "tst", "holdout", "algotest"]:
        pth = os.path.join(base_pth, i)
        img_paths += [os.path.join(pth, j) for j in glob.glob(os.path.join(pth, "*.jpg"))]
    print(len(img_paths))

    r = process_map(img_processor, img_paths, max_workers=8, desc=desc)
