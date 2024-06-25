import os
import shutil
import json

import pandas as pd
import yaml
from tqdm.contrib.concurrent import thread_map

"""
Example of project yaml:
```yaml
# Ultralytics YOLO 🚀, AGPL-3.0 license
# COCO128 dataset https://www.kaggle.com/ultralytics/coco128 (first 128 images from COCO train2017) by Ultralytics
# Example usage: yolo train data=coco128.yaml
# parent
# ├── ultralytics
# └── datasets
#     └── coco128  ← downloads here (7 MB)


# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: ../datasets/position_db_pose  # dataset root dir
train: images/train  # train images (relative to 'path') 128 images
val: images/val  # val images (relative to 'path') 128 images
test: images/test  # test images (optional)

kpt_shape: [1, 2]

# Classes
names:
  0: person
```

Example of annotaion txt file
```txt
0 0.726953125 0.3953125 0.27265625 0.20104166666666667 0.63359375 0.453125
0 0.288671875 0.35885416666666664 0.16484375 0.17395833333333333 0.36015625 0.41458333333333336
0 0.507421875 0.1359375 0.10703125 0.271875 0.49296875 0.25104166666666666
0 0.59765625 0.07864583333333333 0.090625 0.15729166666666666 0.58203125 0.10833333333333334
```

Algo:
1. Read the annotaion and images;
2. Filter: remove: images without annotations, with errors, duplicates;
3. Select specific classes, merge if needed;
4. Map labels to class ids;
5. Convert to list of dict:
```python
{
    "id",
    "path",
    "objects": {
        "id",
        "cls",
        "bbox",
        "position"
    }
}
```
6. Split on train and val
7. Create YOLO dataset:
    1. label txt
    2. copy images
    3. create project.yaml file
"""


def dataframes2objects(df_annotations: pd.DataFrame, df_images: pd.DataFrame, labels: list | None = None) -> tuple:
    if labels is None:
        label_counts = df_annotations["label"].value_counts().reset_index().to_dict(orient="records")
        labels = [i["label"] for i in label_counts]
    label_mapping = {idx: i for idx, i in enumerate(labels)}
    reverse_label_mapping = {i: idx for idx, i in enumerate(labels)}

    objects = []
    for _, row in df_images.iterrows():
        pth = os.path.join(row["batch"], row["folder"], row["name"])
        dets = []
        for _, row_det in df_annotations[df_annotations["img_id"] == row["id"]].iterrows():
            dets.append({
                "id": row_det["id"],
                "cls": reverse_label_mapping[row_det["label"]],
                "xtl": row_det["xtl"] / row["width"],
                "ytl": row_det["ytl"] / row["height"],
                "xbr": row_det["xbr"] / row["width"],
                "ybr": row_det["ybr"] / row["height"],
                "x": row_det["x"] / row["width"],
                "y": row_det["y"] / row["height"],
            })
        objects.append({
            "id": row["id"],
            "group_id": row["group_id"],
            "path": pth,
            "objects": dets,
        })

    return objects, label_mapping


def create_project_file(file_pth: str, dataset_pth: str, label_mapping: dict) -> None:
    data = {
        "path": dataset_pth,
        "train": "images/train",
        "val": "images/val",
        "kpt_shape": [1, 2],
    }

    with open(file_pth, "wt") as f:
        yaml.dump(data, f, sort_keys=False, default_flow_style=None)

    with open(file_pth, "at") as f:
        yaml.dump({"names": label_mapping}, f, sort_keys=False)


def create_label_txt(args) -> None:
    pth, obj = args
    with open(os.path.join(pth, f"{obj['id']}.txt"), "wt") as f:
        for det in obj["objects"]:
            w = det["xbr"] - det["xtl"]
            h = det["ybr"] - det["ytl"]
            x_c = det["xtl"] + w / 2
            y_c = det["ytl"] + h / 2
            f.write(f"{det['cls']} {x_c} {y_c} {w} {h} {det['x']} {det['y']}\n")


def cp_img(args) -> None:
    base_pth, pth, obj = args
    shutil.copy(
        os.path.join(base_pth, obj["path"]),
        os.path.join(pth, f"{obj['id']}{os.path.splitext(obj['path'])[-1]}")
    )


def create_yolo_dataset(
    objects: list, label_mapping: dict, base_imgs_pth: str, dataset_pth: str, split_pth: str
) -> None:
    """_summary_

    Args:
        objects (list): list of objects. Object is an image with annotation:
        {
            "id": str,
            "group_id": str,
            "path": str,
            "objects": List[dict]: [
                {
                    "id": str,
                    "cls": int,
                    "xtl": float,
                    "ytl": float,
                    "xbr": float,
                    "ybr": float,
                    "x": float,
                    "y": float
                }
            ]
            
        }.
        label_mapping (dict): labels mapping, e.g. {0: desk, 1: chair}.
        base_imgs_pth (str): _description_
        dataset_pth (str): _description_
        val_size (float, optional): _description_. Defaults to 0.1.
        random_state (int, optional): _description_. Defaults to 42.
    """
    imgs_pth = os.path.join(dataset_pth, "images")
    labels_pth = os.path.join(dataset_pth, "labels")

    shutil.rmtree(dataset_pth, ignore_errors=True)
    os.makedirs(dataset_pth)
    os.makedirs(imgs_pth)
    os.makedirs(os.path.join(imgs_pth, "train"))
    os.makedirs(os.path.join(imgs_pth, "val"))
    os.makedirs(labels_pth)
    os.makedirs(os.path.join(labels_pth, "train"))
    os.makedirs(os.path.join(labels_pth, "val"))

    with open(split_pth, "r") as f:
        split = json.load(f)
    train_ids = split["train"]
    val_ids = split["test"]
    print(len(train_ids), len(val_ids))

    train_objects = list(filter(lambda x: x["id"] in train_ids, objects))
    val_objects = list(filter(lambda x: x["id"] in val_ids, objects))

    thread_map(create_label_txt,
               zip([os.path.join(labels_pth, "train")] * len(train_ids), train_objects), desc="train labels")
    thread_map(create_label_txt,
               zip([os.path.join(labels_pth, "val")] * len(val_ids), val_objects), desc="val labels")

    thread_map(
        cp_img,
        zip([base_imgs_pth] * len(train_ids), [os.path.join(imgs_pth, "train")] * len(train_ids), train_objects),
        desc="train images", max_workers=4
    )
    thread_map(
        cp_img,
        zip([base_imgs_pth] * len(val_ids), [os.path.join(imgs_pth, "val")] * len(val_ids), val_objects),
        desc="val images", max_workers=4
    )

    create_project_file(os.path.join(dataset_pth, "project.yaml"), dataset_pth, label_mapping)
