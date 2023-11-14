import argparse
import glob
import os
import json
import pickle

import numpy as np
import cv2
import tqdm

from common.pg.pg_geometry import PgCamera


def get_nearest_idxs(base_ts: np.ndarray, aligned_ts: np.ndarray, dist_th: float = 0.1) -> np.ndarray:
    """Gets indexes of nearest timestamps to base_ts from aligned_ts.
    In casen if min distante is higer than `dist_th` returns -1.

    Args:
        base_ts (np.ndarray): timestamps to which need to find nearest.
        aligned_ts (np.ndarray): timestamps from which need to find nearest
        dist_th (float, optional): distance threshold in sec. Defaults to 0.1.

    Returns:
        np.ndarray: indexes of nearest items from aligned_ts
    """
    dist_mtx = np.abs(base_ts.reshape(-1, 1) - aligned_ts.reshape(1, -1))

    min_idxs = np.argmin(dist_mtx, axis=1)
    dists = []
    for i, j in zip(min_idxs, dist_mtx):
        dists.append(j[i])
    dists = np.array(dists)

    min_idxs[dists > dist_th] = -1

    return min_idxs


class PointsMapper:
    def __init__(self, cam_type: str, cam_altitude_cm: int, radar_pos: list, radar_dir_deg: float) -> None:
        self._cam_type = cam_type
        self._cam_altitude_cm = cam_altitude_cm
        self._radar_pos = radar_pos
        self._radar_dir_deg = radar_dir_deg

    def meters2pixels(self, points: np.ndarray) -> np.ndarray:
        transformed_points = self._transform_point(self._radar_pos, self._radar_dir_deg, points[:, :2])
        if points.shape[1] == 3:
            transformed_points = np.hstack([transformed_points, points[:, 2:]])

        return self._meters2pixels(
            self._cam_altitude_cm,
            self._cam_type,
            transformed_points
        )

    @staticmethod
    def _transform_point(t: np.ndarray, a: float, points: np.ndarray) -> np.ndarray:
        """Translate and rotate point.

        Args:
            t (np.ndarray): transition [x, y]
            a (float): rotation angle in deg
            points (np.ndarray):  list of points [x, y] to transform

        Returns:
            np.ndarray: transformed points
        """
        a_rad = np.deg2rad(a)
        return (np.array([
            [np.cos(a_rad), -np.sin(a_rad), t[0]],
            [np.sin(a_rad), np.cos(a_rad), t[1]],
            [0, 0, 1],
        ]) @ np.hstack([points, np.ones((points.shape[0], 1))]).T)[:-1].T

    @staticmethod
    def _meters2pixels(cam_h_cm: int, cam_type: str, points: np.ndarray) -> np.ndarray:
        """Convert real world coordinates in meter to camera coordinates in pixels.

        Args:
            cam_h_cm (int): camera altitude in cm
            cam_type (str): camera type: Cam1, Cam2 or Cam3
            points (np.ndarray): points [[x, y]] to transform

        Returns:
            np.ndarray: transformed points
        """
        camera = PgCamera(cam_h_cm * 10, cam_type)
        return camera.cm2pixel(points * 100).astype(int)


def main(ipath, opath):
    with open(os.path.join(ipath, "params.json"), 'rt') as f:
        params = json.load(f)

    with open(glob.glob(os.path.join(ipath, "*.pickle"))[0], 'rb') as f:
        vayyar_data = pickle.load(f)

    video_timestamps = []
    with open(glob.glob(os.path.join(ipath, "*.txt"))[0], 'rt') as f:
        video_timestamps = np.array([float(i) for i in f.readlines()])

    cap = cv2.VideoCapture(glob.glob(os.path.join(ipath, "*.avi"))[0])
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
    parser.add_argument("--output",  help="Name of output AVI video file.")
    args = parser.parse_args()

    main(args.input, args.output)
