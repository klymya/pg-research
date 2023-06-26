import argparse
import glob
import os
import json
import pickle

import numpy as np
import cv2
import tqdm

from point_mapper import get_nearest_idxs, PointsMapper


def main(ipath, video_name, opath):
    with open(os.path.join(ipath, "params.json"), 'rt') as f:
        params = json.load(f)

    with open(glob.glob(os.path.join(ipath, "*.pickle"))[0], 'rb') as f:
        vayyar_data = pickle.load(f)

    video_timestamps = []
    with open(glob.glob(os.path.join(ipath, "*.txt"))[0], 'rt') as f:
        video_timestamps = np.array([float(i) for i in f.readlines()])

    cap = cv2.VideoCapture(os.path.join(ipath, video_name))
    imgs = []
    while True:
        ret, img = cap.read()
        if not ret:
            break
        imgs.append(img)
    cap.release()

    vayyar_timestamps = np.array([i['ts'] for i in vayyar_data])
    ts_mapping_idxs = get_nearest_idxs(vayyar_timestamps, video_timestamps)

    mapper = PointsMapper(
        params["cam_type"],
        params["cam_height_cm"],
        params["radar_position_m"],
        params["radar_direction_deg"]
    )

    writer = cv2.VideoWriter(
        opath,
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        10,
        (imgs[0].shape[1], imgs[0].shape[0])
    )

    for idx in tqdm.tqdm(range(len(vayyar_data))):
        video_idx = ts_mapping_idxs[idx]
        if video_idx == -1:
            tmp = np.zeros_like(imgs[0], dtype=np.uint8) + 255
        else:
            tmp = imgs[video_idx].copy()

        point_cloud = vayyar_data[idx]['data']['Payload']['points_Locs']
        if point_cloud.ndim == 1:
            point_cloud = point_cloud[None]

        targets = vayyar_data[idx]['data']['Payload']['targets_Locs']
        if targets.ndim == 1:
            targets = targets[None]

        for p in mapper.meters2pixels(point_cloud[:, :2]):
            cv2.circle(tmp, p, 2, (255, 0, 0), 2)

        for p in mapper.meters2pixels(targets[:, :2]):
            cv2.rectangle(tmp, p - 5, p + 5, (0, 255, 0), -1)

        cv2.putText(
            tmp, f"Frame # {idx}", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2
        )

        cv2.putText(
            tmp, "Point Cloud 2D projection on the floor", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )

        cv2.putText(
            tmp, "target 2D projection on the floor", (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
        )

        writer.write(tmp)

    writer.release()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Renders vayyar data on video from PG sensor.")
    parser.add_argument("--input", help="Path to folder with input data.")
    parser.add_argument("--video-name", help="Name of the input video file.")
    parser.add_argument("--output",  help="Name of output AVI video file.")
    args = parser.parse_args()

    main(args.input, args.video_name, args.output)
