import json
import os
import pytest
from unittest.mock import MagicMock
from preprocessor import Preprocessor
from common import DEFAULT_CAMERA_CONFIG, DEFAULT_CALIBRATION_PARAMS
from preprocessing_utils import Detection
from geometry import Rectangle, Point


@pytest.fixture
def tmp_json_file(tmp_path):
    data = {
        "sensor_1": {
            "LENS_PARAMS": {
                "CX": DEFAULT_CALIBRATION_PARAMS.CX,
                "CY": DEFAULT_CALIBRATION_PARAMS.CY,
                "FX": DEFAULT_CALIBRATION_PARAMS.FX,
                "FY": DEFAULT_CALIBRATION_PARAMS.FY,
                "K0": DEFAULT_CALIBRATION_PARAMS.K0,
                "K1": DEFAULT_CALIBRATION_PARAMS.K1,
                "K2": DEFAULT_CALIBRATION_PARAMS.K2,
                "K3": DEFAULT_CALIBRATION_PARAMS.K3,
                "FOV_X": DEFAULT_CAMERA_CONFIG.fov_angle_x,
                "FOV_Y": DEFAULT_CAMERA_CONFIG.fov_angle_x
            },
            "HEIGHT": DEFAULT_CAMERA_CONFIG.height_cm,
            "IMG_W": DEFAULT_CAMERA_CONFIG.image_w,
            "IMG_H": DEFAULT_CAMERA_CONFIG.image_h,
        }
    }
    file_path = tmp_path / "sensors.json"
    with open(file_path, "w") as f:
        json.dump(data, f)
    yield file_path
    os.remove(file_path)

def test_load_sensors_params(tmp_json_file):
    preprocessor = Preprocessor(tmp_json_file)
    
    # Check that only one sensor is added
    assert len(preprocessor._sensor_serial_numbers) == 1
    assert "sensor_1" in preprocessor._sensor_serial_numbers

    # Check each camera config parameter
    cam_config = preprocessor._camera_configs["sensor_1"]
    assert cam_config.calibration_params.CX == DEFAULT_CALIBRATION_PARAMS.CX
    assert cam_config.calibration_params.CY == DEFAULT_CALIBRATION_PARAMS.CY
    assert cam_config.calibration_params.FX == DEFAULT_CALIBRATION_PARAMS.FX
    assert cam_config.calibration_params.FY == DEFAULT_CALIBRATION_PARAMS.FY
    assert cam_config.calibration_params.K0 == DEFAULT_CALIBRATION_PARAMS.K0
    assert cam_config.calibration_params.K1 == DEFAULT_CALIBRATION_PARAMS.K1
    assert cam_config.calibration_params.K2 == DEFAULT_CALIBRATION_PARAMS.K2
    assert cam_config.calibration_params.K3 == DEFAULT_CALIBRATION_PARAMS.K3
    assert cam_config.fov_angle_x == DEFAULT_CAMERA_CONFIG.fov_angle_x
    assert cam_config.fov_angle_x == DEFAULT_CAMERA_CONFIG.fov_angle_x
    assert cam_config.height_cm == DEFAULT_CAMERA_CONFIG.height_cm
    assert cam_config.image_w == DEFAULT_CAMERA_CONFIG.image_w
    assert cam_config.image_h == DEFAULT_CAMERA_CONFIG.image_h

    # Check that DetectionArea has the same config
    da = preprocessor._detection_areas["sensor_1"]
    assert da.m_cam == cam_config

def test_preprocess_detection(mocker, tmp_json_file):
    preprocessor = Preprocessor(tmp_json_file)
    mocker.patch.object(preprocessor._xy_storage, 'append_data_point', MagicMock())

    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9),
        Detection(bbox=Rectangle.from_xywh(10, 0, 50, 50), position=Point(10, 0), confidence=0.8)
    ]

    preprocessor.preprocess_detection(detections, "sensor_1")

    preprocessor._xy_storage.append_data_point.assert_called_once_with("sensor_1", -17, 10, mocker.ANY)
