import pytest
from geometry import Point
from aoi import AreaOfInterest

def test_initialization():
    polygon = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]
    aoi = AreaOfInterest("123", "Test Area", polygon, 60)
    assert aoi._id == "123"
    assert aoi._name == "Test Area"
    assert aoi._polygon == polygon
    assert aoi._reporting_period == 60

def test_reporting_period_setter():
    polygon = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]
    aoi = AreaOfInterest("123", "Test Area", polygon, 60)
    aoi.reporting_period = 120
    assert aoi.reporting_period == 120

    with pytest.raises(ValueError):
        aoi.reporting_period = 0

def test_name_setter():
    polygon = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]
    aoi = AreaOfInterest("123", "Test Area", polygon, 60)
    aoi.name = "New Name"
    assert aoi.name == "New Name"

    with pytest.raises(ValueError):
        aoi.name = ""

def test_polygon_setter():
    polygon1 = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]
    polygon2 = [Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2)]
    aoi = AreaOfInterest("123", "Test Area", polygon1, 60)
    aoi.polygon = polygon2
    assert aoi.polygon == polygon2

def test_is_point_inside():
    polygon = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)]
    aoi = AreaOfInterest("123", "Test Area", polygon, 60)

    inside_point = Point(0.5, 0.5)
    outside_point = Point(1.5, 1.5)
    edge_point = Point(1, 0)

    assert aoi.is_point_inside(inside_point) == True
    assert aoi.is_point_inside(outside_point) == False
    assert aoi.is_point_inside(edge_point) == True
