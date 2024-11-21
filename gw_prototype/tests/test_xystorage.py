import os

import time
import pytest
from xy_storage import XYStorage


@pytest.fixture
def storage():
    storage = XYStorage(db_name="test_gw.db")
    yield storage
    storage.close()
    os.remove("test_gw.db")


def test_append_data_point(storage):
    # Test with default timestamp
    timestamp = int(time.time()) - 100
    storage.append_data_point(sensor_serial_number=123, x=10, y=20, timestamp=timestamp)
    data_points = storage.query_data_points(sensor_serial_number=123, start_timestamp=0)
    time.sleep(1e-1)
    assert len(data_points) == 1
    assert data_points[0][1:] == (123, 10, 20)

    # Test with provided timestamp
    timestamp = int(time.time()) - 1000
    storage.append_data_point(sensor_serial_number=123, x=30, y=40, timestamp=timestamp)
    time.sleep(1e-1)
    data_points = storage.query_data_points(sensor_serial_number=123, start_timestamp=0)
    assert len(data_points) == 2
    assert data_points[1] == (timestamp, 123, 30, 40)


def test_query_data_points(storage):
    timestamp1 = int(time.time()) - 2000
    timestamp2 = int(time.time()) - 1000
    storage.append_data_point(sensor_serial_number=123, x=10, y=20, timestamp=timestamp1)
    storage.append_data_point(sensor_serial_number=123, x=30, y=40, timestamp=timestamp2)
    time.sleep(1e-1)

    # Query with a range that includes both data points
    data_points = storage.query_data_points(sensor_serial_number=123, start_timestamp=timestamp1 - 1, end_timestamp=timestamp2 + 1)
    assert len(data_points) == 2

    # Query with a range that includes only the second data point
    data_points = storage.query_data_points(sensor_serial_number=123, start_timestamp=timestamp2 - 1, end_timestamp=timestamp2 + 1)
    assert len(data_points) == 1
    assert data_points[0] == (timestamp2, 123, 30, 40)


def test_remove_old_data_points(storage):
    timestamp1 = int(time.time()) - 3600  # 1 hour ago
    timestamp2 = int(time.time()) - 1800  # 30 minutes ago
    storage.append_data_point(sensor_serial_number=123, x=10, y=20, timestamp=timestamp1)
    storage.append_data_point(sensor_serial_number=123, x=30, y=40, timestamp=timestamp2)
    time.sleep(1e-1)

    # Remove data points older than 45 minutes
    storage.remove_old_data_points(threshold_min=45)
    data_points = storage.query_data_points(sensor_serial_number=123, start_timestamp=0)
    assert len(data_points) == 1
    assert data_points[0] == (timestamp2, 123, 30, 40)


def test_multiple_cases(storage):
    timestamp1 = int(time.time()) - 3600  # 1 hour ago
    timestamp2 = int(time.time()) - 1800  # 30 minutes ago
    timestamp3 = int(time.time()) - 900   # 15 minutes ago
    storage.append_data_point(sensor_serial_number=123, x=10, y=20, timestamp=timestamp1)
    storage.append_data_point(sensor_serial_number=123, x=30, y=40, timestamp=timestamp2)
    storage.append_data_point(sensor_serial_number=456, x=50, y=60, timestamp=timestamp3)
    time.sleep(1e-1)

    # Query for sensor 123
    data_points = storage.query_data_points(sensor_serial_number=123, start_timestamp=0)
    assert len(data_points) == 2

    # Query for sensor 456
    data_points = storage.query_data_points(sensor_serial_number=456, start_timestamp=0)
    assert len(data_points) == 1
    assert data_points[0] == (timestamp3, 456, 50, 60)

    # Remove old data points
    storage.remove_old_data_points(threshold_min=45)
    data_points = storage.query_data_points(sensor_serial_number=123, start_timestamp=0)
    assert len(data_points) == 1
    assert data_points[0] == (timestamp2, 123, 30, 40)

    data_points = storage.query_data_points(sensor_serial_number=456, start_timestamp=0)
    assert len(data_points) == 1
    assert data_points[0] == (timestamp3, 456, 50, 60)
