import argparse
import glob
import os
import json
import pickle

import tqdm
import matplotlib.pyplot as plt


def main(ipath, opath):
    with open(os.path.join(ipath, "area_params.json"), 'rt') as f:
        params = json.load(f)

    with open(glob.glob(os.path.join(ipath, "*.pickle"))[0], 'rb') as f:
        vayyar_data = pickle.load(f)

    if not os.path.isdir(opath):
        os.mkdir(opath)

    for idx, frame in tqdm.tqdm(enumerate(vayyar_data), total=len(vayyar_data)):
        points = frame['data']['Payload']['points_Locs']
        abs_vals = frame['data']['Payload']['points_absValues']
        loc = frame['data']['Payload']['targets_Locs']
        count = frame['data']['Payload']['targets_targetsCount']

        plt.figure(figsize=(10, 10))
        plt.xlim(params['area_dims'][:2])
        plt.ylim(0, params['area_dims'][3])
        plt.xlabel('X')
        plt.ylabel('Y')

        plt.scatter(0, 0, marker='s', s=400, c='black', label="sensor")
        try:
            plt.scatter(points[:, 0], points[:, 1], c=abs_vals, label="point cloud")
        except:
            print(idx)

        if count:
            if count > 1:
                for i in range(loc.shape[0]):
                    plt.scatter(*loc[i, :2].T, marker='^', s=200, c='red', label='target')
            else:
                plt.scatter(*loc[:2].T, marker='^', s=200, c='red', label='target')

        plt.legend()

        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')

        plt.savefig(os.path.join(opath, f"{idx:03d}.png"), facecolor='white', edgecolor='none')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Renders vayyar data on video from PG sensor.")
    parser.add_argument("--input", help="Path to folder with input data.")
    parser.add_argument("--output",  help="Name of output AVI video file.")
    args = parser.parse_args()

    main(args.input, args.output)
