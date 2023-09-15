from concurrent.futures import ThreadPoolExecutor
import threading
import os

import pandas as pd
from tqdm import tqdm

os.makedirs("results", exist_ok=True)

df = pd.read_csv("data_deduplicated.csv")

file_paths = df["seq_path"].values
file_names = df["seq_name"].values
video_paths = df["video_path"].values

seq_data = list(zip(file_paths, video_paths, file_names))


def parse_seq(in_data):
    seq_path, avi_path, seq_name = in_data
    try:
        thread_id = threading.get_ident()

        with open(seq_path, "r") as f:
            lines = f.readlines()

        if not len(lines):
            raise Exception("Empty seq file")

        lines = [line.strip() for line in lines]

        # avi_path = lines[0]
        altitude = ""
        camera_type = ""
        frame_h = ""
        frame_w = ""

        for line in lines:
            if "CEILING_HEIGHT" in line:
                altitude = line.split()[-1]
            if "Camera Type" in line:
                camera_type = line.split("|")[-1]
            if "Frame Width" in line:
                frame_w = line.split("|")[-1]
            if "Frame Height" in line:
                frame_h = line.split("|")[-1]
            if line.startswith("frame") and "POSITION_" in line:
                vals = line.split()
                frame_n = vals[1]
                user_id = vals[3].split("_")[-1]
                # bounding box: left, top, width, height
                left = vals[4]
                top = vals[5]
                width = vals[6]
                height = vals[7]
                pos = vals[-1].split("_")[-1]
                # splitter = "*"
                if "*" in pos:
                    splitter = "*"
                elif "-" in pos:
                    splitter = "-"
                else:
                    splitter = "_"
                pos_x, pos_y = pos.split(splitter)
                with open(f"results/{thread_id}.txt", "a") as outfile:
                    outfile.write(",".join([
                        seq_name, seq_path, avi_path, frame_h, frame_w, camera_type, altitude, frame_n, user_id,
                        left, top, width, height, pos_x, pos_y
                    ]) + "\n")
        return f"{seq_path} - OK\n"
    except Exception as e:
        out = f"{seq_path} - {e}\n"
        return out


def run(f, my_iter):
    with ThreadPoolExecutor(max_workers=32) as executor:
        results = list(tqdm(executor.map(f, my_iter), total=len(my_iter)))
    return results


errors = run(parse_seq, seq_data)
with open("statuses.txt", "w") as f:
    f.writelines(errors)
