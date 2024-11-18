import pytest
from fisheye_camera import (
    CameraCalibrationParameters,
    FisheyeInternal,
    distort,
    undistort,
    project,
    pixel2cm,
    cm2pixel,
    camera_calibration_parameters_to_string
)
from geometry import Point
from common import DEFAULT_CALIBRATION_PARAMS, W, H, CAMERA_H, TABLE_H

def test_camera_calibration_parameters_to_string_default():
    params = CameraCalibrationParameters(0, 1, 2, 3, 4, 5, 6, 7)
    expected = """
    CX: 0
    CY: 1
    FX: 2
    FY: 3
    K0: 4
    K1: 5
    K2: 6
    K3: 7
    """
    result = camera_calibration_parameters_to_string(params)
    assert result.strip() == expected.strip()

def test_normalize_center():
    p_in = Point(float(W) / 2, float(H) / 2)
    p_out = FisheyeInternal.normalize(p_in, DEFAULT_CALIBRATION_PARAMS)
    assert p_out.x == pytest.approx(-0.063, abs=1e-3)
    assert p_out.y == pytest.approx(-0.037, abs=1e-3)

def test_denormalize_center():
    p_in = Point(-0.063, -0.037)
    p_out = FisheyeInternal.denormalize(p_in, DEFAULT_CALIBRATION_PARAMS)
    assert p_out.x == pytest.approx(float(W) / 2, abs=1)
    assert p_out.y == pytest.approx(float(H) / 2, abs=1)

def test_distort_normalized_center():
    p_in = Point(0.0, 0.0)
    p_out = FisheyeInternal.distort_normalized(p_in, DEFAULT_CALIBRATION_PARAMS)
    assert p_out.x == pytest.approx(p_in.x, abs=1e-3)
    assert p_out.y == pytest.approx(p_in.y, abs=1e-3)

def test_distort_normalized_bottom_right():
    p_in = Point(1, 1)
    p_out = FisheyeInternal.distort_normalized(p_in, DEFAULT_CALIBRATION_PARAMS)
    assert p_out.x == pytest.approx(0.844, abs=1e-3)
    assert p_out.y == pytest.approx(0.844, abs=1e-3)

def test_distort_center():
    p_in = Point(float(W) / 2, float(H) / 2)
    p_out = distort(p_in, DEFAULT_CALIBRATION_PARAMS)
    assert p_out.x == pytest.approx(float(W) / 2, abs=1)
    assert p_out.y == pytest.approx(float(H) / 2, abs=1)

def test_distort_top_left():
    p_in = Point(0.0, 0.0)
    p_out = distort(p_in, DEFAULT_CALIBRATION_PARAMS)
    assert p_out.x == pytest.approx(96, abs=1)
    assert p_out.y == pytest.approx(71, abs=1)

def test_undistort_center():
    p_in = Point(float(W) / 2, float(H) / 2)
    p_out = undistort(p_in, DEFAULT_CALIBRATION_PARAMS)
    assert p_out.x == pytest.approx(float(W) / 2, abs=1)
    assert p_out.y == pytest.approx(float(H) / 2, abs=1)

def test_undistort_top_left():
    p_in = Point(0, 0)
    p_out = undistort(p_in, DEFAULT_CALIBRATION_PARAMS)
    assert p_out.x == pytest.approx(-461, abs=1)
    assert p_out.y == pytest.approx(-343, abs=1)

def test_project_zero_height():
    p_in = Point(float(W) / 4, float(H) / 4)
    p_out = project(p_in, CAMERA_H, 0, DEFAULT_CALIBRATION_PARAMS)
    assert p_out.x == pytest.approx(float(W) / 4, abs=1)
    assert p_out.y == pytest.approx(float(H) / 4, abs=1)

def test_project_positive_height():
    p_in = Point(float(W) / 4, float(H) / 4)
    p_out = project(p_in, CAMERA_H, TABLE_H, DEFAULT_CALIBRATION_PARAMS)
    assert p_out.x == pytest.approx(124, abs=1)
    assert p_out.y == pytest.approx(93, abs=1)

def test_project_negative_height():
    p_in = Point(float(W) / 4, float(H) / 4)
    p_out = project(p_in, CAMERA_H, -TABLE_H, DEFAULT_CALIBRATION_PARAMS)
    assert p_out.x == pytest.approx(196, abs=1)
    assert p_out.y == pytest.approx(146, abs=1)

def test_pixel2cm_center():
    p_in = Point(float(W) / 2, float(H) / 2)
    p_out = pixel2cm(p_in, CAMERA_H, DEFAULT_CALIBRATION_PARAMS)
    assert p_out.x == pytest.approx(-17, abs=1)
    assert p_out.y == pytest.approx(10, abs=1)

def test_cm2pixel_ground_level():
    p_in = Point(200.0, 200.0)
    p_out = cm2pixel(p_in, CAMERA_H, 0, DEFAULT_CALIBRATION_PARAMS)
    assert p_out.x == pytest.approx(468, abs=1)
    assert p_out.y == pytest.approx(112, abs=1)

def test_cm2pixel_table_level():
    p_in = Point(200.0, 200.0)
    p_out = cm2pixel(p_in, CAMERA_H, TABLE_H, DEFAULT_CALIBRATION_PARAMS)
    assert p_out.x == pytest.approx(500, abs=1)
    assert p_out.y == pytest.approx(80, abs=1)
