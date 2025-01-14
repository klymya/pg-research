import json
import time

from preprocessing_utils import Detection, preprocessing
from detection_area import DetectionArea, CameraConfiguration
from fisheye_camera import CameraCalibrationParameters
from xy_storage import XYStorage


class Preprocessor:
    def __init__(self, data_pth: str, db_name: str | None = None) -> None:
        self._xy_storage = XYStorage(db_name)

        self._camera_configs: dict[CameraConfiguration] = {}
        self._detection_areas: dict[DetectionArea] = {}
        self._sensor_serial_numbers: list[str] = []
        self._load_sensors_params(data_pth)

    def _load_sensors_params(self, data_pth: str) -> None:
        with open(data_pth, "r") as f:
            data = json.load(f)

        for serial_num, params in data.items():
            calib_params = CameraCalibrationParameters(
                params["LENS_PARAMS"]["CX"],
                params["LENS_PARAMS"]["CY"],
                params["LENS_PARAMS"]["FX"],
                params["LENS_PARAMS"]["FY"],
                params["LENS_PARAMS"]["K0"],
                params["LENS_PARAMS"]["K1"],
                params["LENS_PARAMS"]["K2"],
                params["LENS_PARAMS"]["K3"],
            )
            cam_config = CameraConfiguration(
                calib_params,
                params["LENS_PARAMS"]["FOV_X"],
                params["LENS_PARAMS"]["FOV_Y"],
                params["HEIGHT"],
                params["IMG_W"],
                params["IMG_H"],
            )
            da = DetectionArea(cam_config)

            self._sensor_serial_numbers.append(serial_num)
            self._camera_configs[serial_num] = cam_config
            self._detection_areas[serial_num] = da

    @staticmethod
    def get_timestamp_sec() -> int:
        return int(time.time())

    def preprocess_detection(
        self, detections: list[Detection], sensor_serial_number: str, ts: int | None = None
    ) -> None:
        if sensor_serial_number not in self._sensor_serial_numbers:
            raise ValueError(f"Unknown sensor serial number: {sensor_serial_number}")

        if ts is None:
            ts = self.get_timestamp_sec()
        xy_points = preprocessing(
            detections,
            self._detection_areas[sensor_serial_number],
            self._camera_configs[sensor_serial_number]
        )

        for point in xy_points:
            self._xy_storage.append_data_point(sensor_serial_number, point.x, point.y, ts)
