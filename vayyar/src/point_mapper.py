import numpy as np

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
        transformed_points = self.transform_point(points[:, :2])
        if points.shape[1] == 3:
            transformed_points = np.hstack([transformed_points, points[:, 2:]])

        return self._meters2pixels(
            self._cam_altitude_cm,
            self._cam_type,
            transformed_points
        )

    def transform_point(self, points: np.ndarray) -> np.ndarray:
        return self._transform_point(self._radar_pos, self._radar_dir_deg, points)

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