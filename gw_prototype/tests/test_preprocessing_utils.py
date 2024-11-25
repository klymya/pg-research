import pytest

from preprocessing_utils import filter_detections_by_da, Detection, detection2xy, preprocessing
from detection_area import DetectionArea
from geometry import Rectangle, Point
from common import DEFAULT_CAMERA_CONFIG, DEFAULT_CALIBRATION_PARAMS, CAMERA_H


def test_filter_detections_by_da_all_inside():
    da = DetectionArea(DEFAULT_CAMERA_CONFIG)
    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9),
        Detection(bbox=Rectangle.from_xywh(330, 250, 50, 50), position=Point(330, 250), confidence=0.8)
    ]
    filtered = filter_detections_by_da(detections, da)
    assert len(filtered) == 2


def test_filter_detections_by_da_all_outside():
    da = DetectionArea(DEFAULT_CAMERA_CONFIG)
    detections = [
        Detection(bbox=Rectangle.from_xywh(10, 0, 50, 50), position=Point(10, 0), confidence=0.9),
        Detection(bbox=Rectangle.from_xywh(20, 10, 50, 50), position=Point(20, 10), confidence=0.8)
    ]
    filtered = filter_detections_by_da(detections, da)
    assert len(filtered) == 0


def test_filter_detections_by_da_some_inside():
    da = DetectionArea(DEFAULT_CAMERA_CONFIG)
    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9),
        Detection(bbox=Rectangle.from_xywh(10, 0, 50, 50), position=Point(10, 0), confidence=0.8)
    ]
    filtered = filter_detections_by_da(detections, da)
    assert len(filtered) == 1
    assert filtered[0].position.x == 320
    assert filtered[0].position.y == 240


def test_detection2xy():
    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9),
    ]
    point = detection2xy(detections, DEFAULT_CALIBRATION_PARAMS, CAMERA_H)[0]
    assert point.x ==  -17
    assert point.y ==  10

def test_preprocessing():
    da = DetectionArea(DEFAULT_CAMERA_CONFIG)
    detections = [
        Detection(bbox=Rectangle.from_xywh(320, 240, 50, 50), position=Point(320, 240), confidence=0.9),
        Detection(bbox=Rectangle.from_xywh(10, 0, 50, 50), position=Point(10, 0), confidence=0.8)
    ]
    points = preprocessing(detections, da, DEFAULT_CAMERA_CONFIG)
    assert len(points) == 1
    assert points[0].x == -17
    assert points[0].y == 10
