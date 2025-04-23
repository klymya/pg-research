import argparse
import glob
import os

import pandas as pd


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        help='Path to resuls root folder',
                        required=True)

    args = vars(parser.parse_args())

    file_paths = glob.glob(os.path.join(args['i'], "*/*/*.csv"))

    dfs = []
    for file_pth in file_paths:
        if os.path.exists(file_pth):
            dfs.append(pd.read_csv(file_pth))
        else:
            print(f"File not found: {file_pth}")

    df = pd.concat(dfs, ignore_index=True)

    pd.DataFrame({
        "Type": ["avg", "median"],
        "Precision": [df["Precision"].mean().round(2), df["Precision"].median().round(2)],
        "Recall": [df["Recall"].mean().round(2), df["Recall"].median().round(2)],
        "F1 score": [df["F1 score"].mean().round(2), df["F1 score"].median().round(2)],
    }).to_csv(
        os.path.join(args['i'], "avg_score.csv"),
        index=False
    )
