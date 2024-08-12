from concurrent.futures import ThreadPoolExecutor
import os

import pandas as pd
from tqdm import tqdm

os.makedirs("tmp/results", exist_ok=True)

df = pd.read_csv("tmp/data.csv")

file_paths = df["seq_path"].values


def parse_seq(seq_path):
    with open(seq_path, "r") as f:
        file_path = f.readline().strip()

    fixed_file_path = file_path.replace("\\", "/").replace("/pg-fs01/db", "/mnt/pg-fs01/")
    nearest_file_path = seq_path.replace(".seq", "")

    if os.path.exists(file_path):
        return file_path
    elif os.path.exists(nearest_file_path):
        return nearest_file_path
    elif os.path.exists(fixed_file_path):
        return fixed_file_path
    else:
        return ""


def run(f, my_iter):
    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(executor.map(f, my_iter), total=len(my_iter)))
    return results


res = run(parse_seq, file_paths)
df["video_path"] = res

df.to_csv("tmp/data_video_paths_upd_nearest_first.csv", index=False)
