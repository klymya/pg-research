import pytest
from aggregator import Aggregator
from geometry import Point
from aoi import AreaOfInterest

@pytest.fixture
def mock_storage(mocker):
    return mocker.patch('aggregator.XYStorage')

@pytest.fixture
def aggregator(mock_storage):
    return Aggregator()

def test_set_and_get_aoi(aggregator):
    aoi1 = AreaOfInterest(area_id="1", name="tst", polygon=[], reporting_period_sec=10)
    aoi2 = AreaOfInterest(area_id="2", name="tst", polygon=[], reporting_period_sec=20)
    aggregator.set_aoi('sensor1', [aoi1, aoi2])

    aois = aggregator.get_aoi('sensor1')
    assert aois == [aoi1, aoi2]

def test_get_aoi_by_report_period(aggregator):
    aoi1 = AreaOfInterest(area_id="1", name="tst", polygon=[], reporting_period_sec=60)
    aoi2 = AreaOfInterest(area_id="2", name="tst", polygon=[], reporting_period_sec=20)
    aggregator.set_aoi('sensor1', [aoi1])
    aggregator.set_aoi('sensor2', [aoi2])

    ts = 100
    aois = aggregator._get_aoi_by_report_period(ts)
    assert aois == [(aoi2, 'sensor2')]

    ts = 120
    aois = aggregator._get_aoi_by_report_period(ts)
    assert aois == [(aoi1, 'sensor1'), (aoi2, 'sensor2')]

def test_count_numbers_in_range():
    assert Aggregator._count_numbers_in_range(10, 5, 0, 20) == 4
    assert Aggregator._count_numbers_in_range(10, 5, 5, 15) == 2
    assert Aggregator._count_numbers_in_range(10, 5, 0, 10) == 0
    assert Aggregator._count_numbers_in_range(10, 5, 0, 5) == 0

# def test_calculate_count(mocker):
#     aoi = mocker.Mock(spec=AreaOfInterest)
#     aoi.reporting_period = 10
#     aoi.is_point_inside.side_effect = lambda point: point.x == 1 and point.y == 1

#     points = [(10, Point(1, 1)), (20, Point(1, 1)), (30, Point(2, 2))]
#     start_ts = 0
#     end_ts = 40

#     count = Aggregator._calculate_count(aoi, points, start_ts, end_ts)
#     assert count == 1

def test_calculate_count_no_points(mocker):
    aoi = mocker.Mock(spec=AreaOfInterest)
    aoi.reporting_period = 30
    aoi.is_point_inside.side_effect = lambda point: point.x == 1 and point.y == 1

    start_ts = 0
    end_ts = 30

    points = []

    count = Aggregator._calculate_count(aoi, points, start_ts, end_ts)
    assert count == 0

def test_calculate_count_one_point(mocker):
    aoi = mocker.Mock(spec=AreaOfInterest)
    aoi.reporting_period = 30
    aoi.is_point_inside.side_effect = lambda point: point.x == 1 and point.y == 1

    start_ts = 0
    end_ts = 30

    points = [
        (4, Point(1, 1)),
    ]

    count = Aggregator._calculate_count(aoi, points, start_ts, end_ts)
    assert count == 1

def test_calculate_count_points_half_time(mocker):
    aoi = mocker.Mock(spec=AreaOfInterest)
    aoi.reporting_period = 30
    aoi.is_point_inside.side_effect = lambda point: point.x == 1 and point.y == 1

    start_ts = 0
    end_ts = 30

    points = [
        (4, Point(1, 1)),
        (4, Point(1, 1)),
        (9, Point(1, 1)),
        (9, Point(1, 1)),
        (14, Point(1, 1)),
        (14, Point(1, 1)),
    ]

    count = Aggregator._calculate_count(aoi, points, start_ts, end_ts)
    assert count == 1

def test_calculate_count_points_all_time(mocker):
    aoi = mocker.Mock(spec=AreaOfInterest)
    aoi.reporting_period = 30
    aoi.is_point_inside.side_effect = lambda point: point.x == 1 and point.y == 1

    start_ts = 0
    end_ts = 30

    points = [
        (4, Point(1, 1)),
        (4, Point(1, 1)),
        (9, Point(1, 1)),
        (9, Point(1, 1)),
        (14, Point(1, 1)),
        (14, Point(1, 1)),
        (19, Point(1, 1)),
        (19, Point(1, 1)),
        (24, Point(1, 1)),
        (24, Point(1, 1)),
        (29, Point(1, 1)),
        (29, Point(1, 1)),
    ]

    count = Aggregator._calculate_count(aoi, points, start_ts, end_ts)
    assert count == 2

def test_check_aois(mocker, mock_storage):
    mock_storage_instance = mock_storage.return_value
    mock_storage_instance.query_data_points.return_value = [
        (10, 'sensor1', 1, 1),
        (15, 'sensor1', 1, 1),
        (20, 'sensor1', 2, 2)
    ]

    aoi = mocker.Mock(spec=AreaOfInterest)
    aoi.reporting_period = 30
    aoi.id = 'aoi1'
    aoi.is_point_inside.side_effect = lambda point: point.x == 1 and point.y == 1

    aggregator = Aggregator()
    aggregator.set_aoi('sensor1', [aoi])

    mocker.patch('time.time', return_value=30)
    counts = aggregator.check_aois()

    assert counts == [('sensor1', 'aoi1', 1)]
