# %%
import os
import sys
import json
import shutil

import pandas as pd
import sqlite3
import cv2
import numpy as np
import yaml
from tqdm.contrib.concurrent import thread_map

sys.path.append("/home/klymy/projects/PG_PyTraining")
from common.pg.cam_params import get_camparams

# %%
DB_PTH = "/home/klymy/projects/researches/data-exploration/people_detection.db"
DATABASE_BASE_PTH = "/home/klymy/data/people_detection"
IMG_SUB_PTH = "images"
LABELS_SUB_PTH = "labels"

# %%
class SQLLiteWrapper:
    def __init__(self, pth: str) -> None:
        self.conn = sqlite3.connect(pth)

    def run_query(self, query: str) -> pd.DataFrame:
        return pd.read_sql_query(query, self.conn)

    def get_dets(self, seq_pth: str) -> pd.DataFrame:
        return self.run_query(
            f"SELECT frame_n, frame_h, frame_w, x_left , y_top, width, height, pos_x, pos_y "\
            f"FROM boxes WHERE seq_path == \"{seq_pth}\";"
        )

# %%
def read_avi(avi_pth: str, fps: float = None) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(avi_pth)
    try:
        campparams = get_camparams(avi_pth)
    except AssertionError:
        with open("no_cam_params.txt", "at") as f:
            f.write(f"{avi_pth}\n")
        return {}

    anchor = 1
    if fps is not None:
        original_fps = int(cap.get(cv2.CAP_PROP_FPS))

        anchor = int(original_fps / fps)

    frames = {}
    count = -1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        count += 1
        if count % anchor:
            continue
        frame = cv2.flip(frame, campparams.FLIP_CODE)
        frames[count] = frame

    cap.release()

    return frames


# %%
def create_label_txt(pth: str, detections: pd.DataFrame) -> None:
    with open(pth, "wt") as f:
        for _, det in detections.iterrows():
            #frame_n, frame_h, frame_w, x_left , y_top, width, height, pos_x, pos_y
            w = det["width"] / det["frame_w"]
            h = det["height"] / det["frame_h"]
            x_c = det["x_left"] / det["frame_w"] + w / 2
            y_c = det["y_top"] / det["frame_h"] + h / 2
            x = det["pos_x"] / det["frame_w"]
            y = det["pos_y"] / det["frame_h"]
            f.write(f"0 {x_c} {y_c} {w} {h} {x} {y}\n")


def process_video(
    avi_pth: str, seq_pth: str, imgs_folder: str, labels_folder: str, fps: float = 0.2
) -> None:
    # read video
    frames = read_avi(avi_pth, fps)

    # read dets
    db_conn = SQLLiteWrapper(DB_PTH)
    df_boxes = db_conn.get_dets(seq_pth)

    # filter frames and dets
    intersected_frames = set(frames.keys()).intersection(df_boxes["frame_n"].values)
    frames = {k: v for k, v in frames.items() if k in intersected_frames}
    df_boxes = df_boxes[df_boxes["frame_n"].isin(intersected_frames)]

    # save in yolo format
    base_name = os.path.splitext(os.path.basename(avi_pth))[0]
    for i in frames:
        tmp_name = f"{base_name}_{i:05}"
        cv2.imwrite(os.path.join(imgs_folder, f"{tmp_name}.jpg"), frames[i])
        create_label_txt(os.path.join(labels_folder, f"{tmp_name}.txt"), df_boxes[df_boxes["frame_n"] == i])
        
        # frame = frames[i].copy()
        # boxes = df_boxes[df_boxes["frame_n"] == i][["x_left", "y_top", "width", "height", "pos_x", "pos_y"]].values
        # for box in boxes:
        #     cv2.rectangle(frame, box[:2], box[:2] + box[2:4], (0, 255, 0), 2)
        #     cv2.circle(frame, box[-2:], 3, (0, 255, 0), 6)
        
        # plt.figure(figsize=(8, 8))
        # plt.imshow(frame)

        # break

# %%
def create_project_file(
    file_pth: str, dataset_pth: str, label_mapping: dict, kpt_shape: list = [1, 2], val_folder: str = "images/val"
) -> None:
    data = {
        "path": dataset_pth,
        "train": "images/train",
        "val": val_folder,
        "kpt_shape": kpt_shape,
    }

    with open(file_pth, "wt") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=None)

    with open(file_pth, "at") as f:
        yaml.dump({"names": label_mapping}, f, sort_keys=False)


def thread_wrapper(indata):
    avi_pth, seq_pth, imgs_folder, labels_folder, fps = indata
    process_video(avi_pth, seq_pth, imgs_folder, labels_folder, fps)

def create_subset(
    name: str, base_img_pth: str, base_label_pth: str, seq_names: list[str], df_video: pd.DataFrame, fps: float = 0.2,
    n_workers: int = 4,
) -> None:
    img_pth = os.path.join(base_img_pth, name)
    label_pth = os.path.join(base_label_pth, name)

    os.makedirs(img_pth, exist_ok=True)
    os.makedirs(label_pth, exist_ok=True)

    # in parallel process each seq file
    subset_df = df_video[df_video["seq_name"].isin(seq_names)]
    subset_df["img_pth"] = img_pth
    subset_df["label_pth"] = label_pth
    subset_df["fps"] = fps
    print(subset_df.shape, len(seq_names))

    data = subset_df[["avi_path", "seq_path", "img_pth", "label_pth", "fps"]].values
    thread_map(
        thread_wrapper,
        data,
        desc=f"process {name}", max_workers=n_workers
    )


def create_dataset(
    dataset_pth: str,
    data_splitting: dict[str, list[str]],
    df_video: pd.DataFrame,
    n_workers: int = 4,
    rm_exist: bool = False,
    subset: list[str] | None = None,
    fps: float = 0.2,
) -> None:
    # create folder struct
    imgs_pth = os.path.join(dataset_pth, IMG_SUB_PTH)
    labels_pth = os.path.join(dataset_pth, LABELS_SUB_PTH)

    if rm_exist:
        shutil.rmtree(dataset_pth, ignore_errors=True)
        os.makedirs(dataset_pth, exist_ok=True)
        os.makedirs(imgs_pth, exist_ok=True)
        os.makedirs(labels_pth, exist_ok=True)

    # process train, val, holdout
    if subset is None:
        subset = list(data_splitting.keys())
    for k in subset:
        create_subset(k, imgs_pth, labels_pth, data_splitting[k], df_video, fps=fps, n_workers=n_workers)

    # create project file
    create_project_file(os.path.join(dataset_pth, "project.yaml"), dataset_pth, {0: "person"}, [1, 2])
    create_project_file(
        os.path.join(dataset_pth, "project_holdout.yaml"), dataset_pth, {0: "person"}, [1, 2], "images/holdout")
    create_project_file(
        os.path.join(dataset_pth, "project_algotest.yaml"), dataset_pth, {0: "person"}, [1, 2], "images/algotest")

# %%

if __name__ == "__main__":
    # %%
    with open("/home/klymy/projects/researches/data-exploration/video_similarity/splitting.json", "rt") as f:
        data_splitting = json.load(f)

    df_video = pd.read_csv("/home/klymy/projects/researches/data-exploration/seq_files.csv")
    df_video.dropna(inplace=True)
    print(df_video.shape)
    df_video.head()
    create_dataset(
        os.path.join(DATABASE_BASE_PTH, "baseline_10sec"),
        data_splitting,
        df_video,
        n_workers=8,
        rm_exist=True,
        fps=0.1
    )

# %%
