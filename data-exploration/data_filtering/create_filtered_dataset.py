# %%
# Import necessary libraries
import os
import json
import sys
import shutil

from tqdm import tqdm

sys.path.append("/home/klymy/projects/researches/data-exploration/video_splitting")
from splitter import create_project_file


# %%
BASE_PTH = "/home/klymy/data/people_detection"
BASE_DATASET = "filtered_v1_10_sec"
FILTERED_DATASET = "filtered_v2_10_sec"

SUBSETS = ["train", "val", "holdout", "algotest"]

# %%
def copy_file(src):
    dst = i.replace(BASE_DATASET, FILTERED_DATASET)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst)

for subset in SUBSETS:
    with open(f"filtered_frames_{BASE_DATASET}_{subset}.json", "rt") as f:
        data = json.load(f)

    assert data["correct_count"] == len(data["imgs"])
    assert data["correct_count"] == len(data["labels"])

    for i in tqdm(data["imgs"], desc=f"{subset} imgs"):
       copy_file(i)

    for i in tqdm(data["labels"], desc=f"{subset} imgs"):
       copy_file(i)

dataset_pth = os.path.join(BASE_PTH, FILTERED_DATASET)
# create project file
create_project_file(os.path.join(dataset_pth, "project.yaml"), dataset_pth, {0: "person"}, [1, 2])
create_project_file(
    os.path.join(dataset_pth, "project_holdout.yaml"), dataset_pth, {0: "person"}, [1, 2], "images/holdout")
create_project_file(
    os.path.join(dataset_pth, "project_algotest.yaml"), dataset_pth, {0: "person"}, [1, 2], "images/algotest")

# %%
