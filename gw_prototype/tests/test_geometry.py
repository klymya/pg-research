import pytest
from geometry import Rectangle, Point, is_point_inside_rectangle, calculate_iou

def test_is_point_inside_rectangle_true():
    r = Rectangle.from_xywh(0.0, 0.0, 10.0, 10.0)
    p = Point(1, 1)
    assert is_point_inside_rectangle(p, r)

def test_is_point_inside_rectangle_false():
    r = Rectangle.from_xywh(0.0, 0.0, 10.0, 10.0)
    p = Point(11, 11)
    assert not is_point_inside_rectangle(p, r)

def test_calculate_iou_default():
    r1 = Rectangle.from_xywh(0.0, 0.0, 10.0, 10.0)
    r2 = Rectangle.from_xywh(5.0, 0.0, 5.0, 10.0)
    assert calculate_iou(r1, r2) == 0.5
