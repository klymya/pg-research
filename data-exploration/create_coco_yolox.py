
import json
from concurrent.futures import ProcessPoolExecutor
import os
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

N = 10
M = 10000

max_workers = 4

path2data = "../../../db/obj-det/color_data/tst.csv"
path2imgs = "tmp_dataset/frames"
path2annotations = "tmp_dataset/labels-tst.json"
path2errors = "errors-tst.txt"

df = pd.read_csv(path2data)
df.head()
avi_paths = df.avi_path.unique()


def get_frames(avi_path, frame_nums):
    frames = {}
    count = 0
    cap = cv2.VideoCapture(avi_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count in frame_nums:
            frames[count] = frame
        count += 1

    cap.release()
    return frames


def process_avi_file(in_data):
    df = pd.read_csv(path2data)

    avi_path, avi_idx = in_data
    out = ""
    images = []
    annotations = []
    try:
        avi_name = avi_path.split("/")[-1].replace(".avi", "")

        avi_df = df[df["avi_path"] == avi_path]
        frame_numbers = avi_df["frame_n"].unique()[::N]
        camera_type = avi_df["camera_type"].values[0]
        w = avi_df["frame_w"].values[0]
        h = avi_df["frame_h"].values[0]

        frames = get_frames(avi_path, frame_numbers)
        for frame_n in frame_numbers:
            img_name = f"{avi_name}__{frame_n}.jpg"
            img_id = int(avi_idx * M + frame_n)

            images.append({
                "id": img_id,
                "license": None,
                "coco_url": None,
                "flickr_url": None,
                "width": int(w),
                "height": int(h),
                "file_name": img_name,
                "date_captured": str(datetime.now())
            })

            frame_df = avi_df[avi_df["frame_n"] == frame_n]

            img = frames[frame_n]
            axis = (1)
            if camera_type in ["SPCA2281B-L0", "SPCA2281B-L1"]:
                axis = (0, 1)

            img = np.flip(img, axis=axis).astype(np.uint8)

            for idx, row in frame_df.iterrows():
                annotations.append({
                    "id": img_id * M + idx,
                    "image_id": img_id,
                    "category_id": 0,
                    "segmentation": None,
                    "area": None,
                    "bbox": [int(i) for i in [row["x_left"], row["y_top"], row["width"], row["height"]]],
                    "iscrowd": None,
                })

            cv2.imwrite(os.path.join(path2imgs, img_name), img)
    except Exception as e:
        out = f"{avi_path} - {e}\n"
    return images, annotations, out


def run(f, my_iter):
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(f, my_iter), total=len(my_iter)))
    return results


avi_data = [(i, idx) for idx, i in enumerate(avi_paths)]

results = run(process_avi_file, avi_data)


imgs = []
annot = []
errors = []
for i in results:
    imgs += i[0]
    annot += i[1]
    errors.append(i[2])


with open(path2errors, "wt") as f:
    f.writelines(errors)


with open(path2annotations, "wt") as f:
    json.dump({
        "info": {"description": ""},
        "licenses": [],
        "images": imgs,
        "annotations": annot,
        "categories": [
            {"supercategory": None, "id": 0, "name": "person"},
        ]
    }, f)
