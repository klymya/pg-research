import os

import pandas as pd
import numpy as np
from tqdm import tqdm


"""
How to join:
- by id
- by IoU*: the area of desk should be at least N% of the area of desk_FULL, N >= 75%
"""


GROUPS = {
    "chair": ["office_chair", "dinning_chair", "lounge_chair", "bar_stool"],
    "table": ["desk", "coffee_table", "meeting_room_table", "desk_FULL", "coffee_table_FULL", "meeting_room_table_FULL"],
    "cabinet": ["cabinet", "cabinet_FULL"],
    # "laptop": ["laptop_closed", "laptop_opened"],
}

LABEL_MAPING = {k: i for i, j in GROUPS.items() for k in j}

JOIN_OBJECTS = [
    ("desk", "desk_FULL"),
    ("meeting_room_table", "meeting_room_table_FULL"),
    ("coffee_table", "coffee_table_FULL"),
    ("cabinet", "cabinet_FULL")
]


def calculate_iou_and_covered_area(box1: list, box2: list) -> float:
    """
    Calculate the fraction of the area of box1 covered by box2.

    Parameters:
    box1 (list or array): Bounding box in the format [xtl, ytl, xbr, ybr]
    box2 (list or array): Bounding box in the format [xtl, ytl, xbr, ybr]

    Returns:
    float: covered_fraction (float): Fraction of box1's area covered by box2
    """
    
    # Extract coordinates for clarity
    xtl1, ytl1, xbr1, ybr1 = box1
    xtl2, ytl2, xbr2, ybr2 = box2

    # Determine the coordinates of the intersection rectangle
    xi1 = max(xtl1, xtl2)
    yi1 = max(ytl1, ytl2)
    xi2 = min(xbr1, xbr2)
    yi2 = min(ybr1, ybr2)

    # Calculate the area of the intersection rectangle
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Calculate the area of bounding box
    box1_area = (xbr1 - xtl1) * (ybr1 - ytl1)

    # Calculate the fraction of box1's area covered by box2
    covered_fraction = inter_area / box1_area if box1_area != 0 else 0

    return covered_fraction


def process_objects_merge(df, name, take_full_version = True, coverage = 0.9):
    df1 = df[df["label"] == name]
    df2 = df[df["label"] == f"{name}_FULL"]

    merge_ids = pd.merge(df1, df2, on="ID", how="inner")["ID"].unique()

    remaining_df1 = df1[~df1["ID"].isin(merge_ids)]
    remaining_df2 = df2[~df2["ID"].isin(merge_ids)]

    # Create empty DataFrame to store the additional merged rows
    additional_merged_rows = []

    # Iterate over remaining rows and check covered_fraction
    for idx1, row1 in remaining_df1.iterrows():
        for idx2, row2 in remaining_df2.iterrows():
            box1 = [row1.xtl, row1.ytl, row1.xbr, row1.ybr]
            box2 = [row2.xtl, row2.ytl, row2.xbr, row2.ybr]
            covered_fraction = calculate_iou_and_covered_area(box1, box2)
            if covered_fraction >= coverage:
                if take_full_version:
                    row = row2
                else:
                    row = row1
                additional_merged_rows.append(row)
                remaining_df1 = remaining_df1.drop(idx1)
                remaining_df2 = remaining_df2.drop(idx2)
                break  # Only merge each row once
            elif  0.4 < covered_fraction < 0.5:
                print(covered_fraction)

    # Convert to DataFrame
    additional_merged_df = pd.DataFrame(additional_merged_rows)
    if take_full_version:
        merged_df = df2[df2["ID"].isin(merge_ids)]
    else:
        merged_df = df1[df1["ID"].isin(merge_ids)]

    # Combine all DataFrames
    res = pd.concat([merged_df, additional_merged_df, remaining_df1, remaining_df2], ignore_index=True)
    if res.shape[0] > 1:
        pass
    return res


def merge_objects(df, take_full_version, join_objects, th=0.7):

    processed_dfs = []
    for i in tqdm(df["img_id"].unique()):
        tmp_df = df[df["img_id"] == i]
        non_mergable_df = tmp_df[~tmp_df["label"].isin(np.concatenate(join_objects))]
        # break

        mergable_dfs = []
        for names in join_objects:
            df_to_process = tmp_df[tmp_df["label"].isin(names)]
            if df_to_process.shape[0] == 0:
                continue
            mergable_dfs.append(process_objects_merge(df_to_process, names[0], take_full_version, th))
        
        res_df = pd.concat([non_mergable_df] + mergable_dfs, ignore_index=True)
        processed_dfs.append(res_df)

    final_df = pd.concat(processed_dfs, ignore_index=True)
    return final_df
