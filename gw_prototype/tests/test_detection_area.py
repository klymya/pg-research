import pytest
from detection_area import DetectionArea, DetectionAreaInternal
from geometry import Rectangle, Point
from common import DEFAULT_CAMERA_CONFIG

def test_deg2rad_default():
    assert DetectionAreaInternal.deg2rad(1) == pytest.approx(0.018, abs=1e-3)
    assert DetectionAreaInternal.deg2rad(180) == pytest.approx(3.142, abs=1e-3)

def test_detection_area_constructor():
    da = DetectionArea(DEFAULT_CAMERA_CONFIG)
    rect_cm = da.get_da_rectangle_cm()
    assert rect_cm.w == 845
    assert rect_cm.h == 491

def test_is_point_inside_px_true():
    da = DetectionArea(DEFAULT_CAMERA_CONFIG)
    p = Point(320, 240)
    assert da.is_point_inside_px(p)

def test_is_point_inside_px_false():
    da = DetectionArea(DEFAULT_CAMERA_CONFIG)
    p = Point(90, 100)
    assert not da.is_point_inside_px(p)

def test_is_rect_intersect_px_true():
    da = DetectionArea(DEFAULT_CAMERA_CONFIG)
    r = Rectangle.from_xywh(320, 240, 50, 50)
    assert da.is_rect_intersect_px(r)

def test_is_rect_intersect_px_false():
    da = DetectionArea(DEFAULT_CAMERA_CONFIG)
    r = Rectangle.from_xywh(10, 0, 50, 50)
    assert not da.is_rect_intersect_px(r)
