import os
import sys
from multiprocessing import Pool

import pandas as pd
import sqlite3
import cv2
import numpy as np
import tqdm
import time

import group_images



DB_PATH = "../people_detection.db"
SEQ_CSV_PATH = "../seq_files.csv"
RES_FOLDER_PATH = "embeddings"




def process_avi(indata):
    try:
        avi_pth, device_idx = indata
        N = 50
        cap = cv2.VideoCapture(avi_pth)

        frames = []
        count = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if not count % N:
                frames.append(frame)

        cap.release()

        embedder = group_images.Embedder(f"cuda:{device_idx}")
        embeddings = [embedder.extract_embedding_from_img(frame).detach().numpy() for frame in frames]

        file_path = os.path.join(RES_FOLDER_PATH, os.path.splitext(os.path.basename(avi_pth))[0])
        with open(file_path, "wb") as f:
            np.save(f, embeddings)

        return avi_pth, "OK"
    except Exception as e:
        return avi_pth, repr(e)


if os.path.exists(SEQ_CSV_PATH):
    df = pd.read_csv(SEQ_CSV_PATH)
else:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """SELECT DISTINCT seq_name, seq_path, avi_path, frame_h, frame_w, camera_type, altitude
        FROM boxes;
        """, conn)
    df.to_csv("../seq_files.csv", index=False)

print(df.shape)


t0 = time.time()
avi_paths = df["avi_path"].values
device_idxs = [i % 3 for i in range(len(avi_paths))]

with Pool(4) as p:
    r = list(tqdm.tqdm(p.imap(process_avi, zip(avi_paths, device_idxs)), total=len(avi_paths)))
t1 = time.time()
print((t1 - t0) // 60, (t1 - t0) % 60)

with open("logs.txt", "wt") as f:
    for i in r:
        f.write(f"{i[1]} {i[0]}\n")
