import numpy as np

from common.pg.pg_geometry import PgCamera


class PointsMapper:
    def __init__(self, cam_type: str, cam_altitude_cm: int, radar_pos: list, radar_dir_deg: float) -> None:
        self._cam_type = cam_type
        self._cam_altitude_cm = cam_altitude_cm
        self._radar_pos = radar_pos
        self._radar_dir_deg = radar_dir_deg

    def meters2pixels(self, points: np.ndarray) -> np.ndarray:
        return self._meters2pixels(
            self._cam_altitude_cm,
            self._cam_type,
            self._transform_point(self._radar_pos, self._radar_dir_deg, points[:, :2])
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
