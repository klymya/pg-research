import os
import glob

import pandas as pd
import tqdm


IN_FOLDER = "tmp"
BENCKMARK_PATHS = [
    "/mnt/pg-fs01/Current/Benchmarks/PG1_AlgoDev_DLtest02_RV9",
    "/mnt/pg-fs01/Current/Benchmarks/PG1_AlgoDev_DLtest_RV59",
    "/mnt/pg-fs01/Current/Benchmarks/PG1_AlgoDev_VLC_RV5_V2",
    "/mnt/pg-fs01/Current/DB/CV_PeopleTracking/Benchmarks/Mixed/VLC_MultiplePeopleCrossing/10fps",
]

seq_paths = []
for data_pth in glob.glob(os.path.join(IN_FOLDER, "*.txt")):
    with open(data_pth, "rt") as f:
        seq_paths += f.readlines()

seq_paths = set(seq_paths)
seq_paths = [i[:-1] for i in seq_paths]

data = []
for pth in tqdm.tqdm(seq_paths, desc="seq paths processing"):
    data.append({
        "seq_path": pth,
        "is_algo_test": any([i in pth for i in BENCKMARK_PATHS]),
        "vlc": "vlc" in pth.lower()
    })

pd.DataFrame(data).to_csv(os.path.join(IN_FOLDER, "data.csv"), index=False)
