import glob
import os
import shutil

import yaml
import cv2


def create_project_file(file_pth: str, dataset_pth: str, label_mapping: dict, val_folder: str = "images/val") -> None:
    data = {
        "path": dataset_pth,
        "train": "images/train",
        "val": val_folder,
        "kpt_shape": [1, 2],
    }

    with open(file_pth, "wt") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=None)

    with open(file_pth, "at") as f:
        yaml.dump({"names": label_mapping}, f, sort_keys=False)


def get_files_glob(directory, ext="*.txt"):
    pattern = os.path.join(directory, '**', ext)
    txt_files = glob.glob(pattern, recursive=True)
    return txt_files


def clahe(img):
    # Improve contrast using histogram equalization (adaptive for better results)
    lab = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe_img = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe_img.apply(l)
    enhanced_image = cv2.merge((l, a, b))
    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2RGB)
    return enhanced_image


def cp_dataset(base_pth: str, src: str, dst: str):
    """
    Copy dataset from src to dst.
    """
    src_pth = os.path.join(base_pth, src)
    dst_pth = os.path.join(base_pth, dst)
    if os.path.exists(dst_pth):
        shutil.rmtree(dst_pth)
    shutil.copytree(src_pth, dst_pth)

    cache_files = get_files_glob(dst_pth, "*.cache")
    for i in cache_files:
        os.remove(i)

    # create project file
    create_project_file(os.path.join(dst_pth, "project.yaml"), dst_pth, {0: "person"})
    create_project_file(os.path.join(dst_pth, "project_holdout.yaml"), dst_pth, {0: "person"}, "images/holdout")
