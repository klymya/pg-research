import argparse
import glob
import os
import json
import pickle

import numpy as np
import cv2
import tqdm

from sort import sort
from point_mapper import PointsMapper, get_nearest_idxs
from benchmark_admin.bmark_files_factory import make_seq


def vayyar_to_ts_dict(data):
    traj_points = []
    for frame in tqdm.tqdm(data):
        points = frame['data']['Payload']['targets_Locs']
        if points.ndim == 1:
            points = points[None]
        traj_points.append(points[:, :2])

    # track association
    sort.KalmanPointTracker.count = 0
    sort_tracker = sort.Sort(l2_threshold=0.5)

    for idx, i in enumerate(traj_points):
        sort_tracker.update(i)

    # track to ts dict
    data_points = {}
    for track in sort_tracker.trackers:
        for point, t in zip(track.raw, track.ts):
            if t not in data_points:
                data_points[t - 1] = []
            data_points[t - 1].append((track.id, point))

    for track in sort_tracker.dead_trackers:
        for point, t in zip(track.raw, track.ts):
            if t not in data_points:
                data_points[t - 1] = []
            data_points[t - 1].append((track.id, point))

    return data_points


def main(ipath, video_name):
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
    ts_mapping_idxs = get_nearest_idxs(vayyar_timestamps, video_timestamps, dist_th=0.1)

    ts_dict = vayyar_to_ts_dict(vayyar_data)

    mapper = PointsMapper(
        params["cam_type"],
        params["cam_height_cm"],
        params["radar_position_m"],
        params["radar_direction_deg"]
    )

    writer = cv2.VideoWriter(
        os.path.join(ipath, "PG1_aligned_video.avi"),
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
        10,
        (imgs[0].shape[1], imgs[0].shape[0])
    )

    n_frames = len(list(filter(lambda x: x > -1, ts_mapping_idxs)))

    seq_lines = []
    skip_count = 0
    for idx in tqdm.tqdm(range(len(vayyar_data))):
        video_idx = ts_mapping_idxs[idx]
        if video_idx == -1:
            skip_count += 1
            continue

        tmp = imgs[video_idx].copy()

        if idx not in ts_dict:
            continue

        for tpoint in ts_dict[idx]:
            x_m, y_m = np.round(mapper.transform_point(tpoint[1][None])[0], 1)
            x_px, y_px = mapper.meters2pixels(tpoint[1][None])[0]
            x_px = imgs[0].shape[1] - x_px
            seq_lines.append(
                f"\nframe {idx - skip_count} "
                f"ProcThread::Run TRACK_USER_ID_{tpoint[0]} {x_px} {y_px} {50} {50} "
                f"POSITION_{x_m}*{y_m} STATUS_UNKNOWN HEIGHT_0 ORIGINALPOSITION_{x_px}*{y_px}"
            )

        writer.write(tmp)

    writer.release()

    make_seq(os.path.join(ipath, "PG1_aligned_video.avi"), params["cam_height_cm"], n_frames, 10)
    with open(os.path.join(ipath, "PG1_aligned_video.avi.seq"), 'at') as f:
        f.writelines(seq_lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Renders vayyar data on video from PG sensor.")
    parser.add_argument("--input", help="Path to folder with input data.")
    parser.add_argument("--video-name", help="Name of the input video file.")
    args = parser.parse_args()

    main(args.input, args.video_name)
