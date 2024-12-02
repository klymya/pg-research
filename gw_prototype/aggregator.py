import time
import math

from xy_storage import XYStorage
from aoi import AreaOfInterest
from geometry import Point


DEFAULT_SENSOR_REPORTING_PERIOD_SEC = 5


class Aggregator:
    def __init__(self, sensors_serials: list[str]) -> None:
        self._storage = XYStorage()
        self._sensor_serials = sensors_serials
        self._aois = {}
        self._sensor_reporting_periods = {}

    def set_aoi(self, sensor_serial_number: str, aois: list[AreaOfInterest]) -> None:
        if sensor_serial_number not in self._sensor_serials:
            raise ValueError(f"Sensor with serial number {sensor_serial_number} does not exist")
        self._aois[sensor_serial_number] = aois

    def get_aoi(self, sensor_serial_number: str) -> list[AreaOfInterest]:
        if sensor_serial_number not in self._sensor_serials:
            raise ValueError(f"Sensor with serial number {sensor_serial_number} does not exist")
        return self._aois.get(sensor_serial_number, [])

    def _get_aoi_by_report_period(self, ts_sec: int) -> list[tuple[AreaOfInterest, str]]:
        aois2check = []
        for serial_num, aois in self._aois.items():
            for aoi in aois:
                if ts_sec % aoi.reporting_period == 0:
                    aois2check.append((aoi, serial_num))
        return aois2check

    @staticmethod
    def _count_numbers_in_range(anchor: int, period: int, start: int, end: int) -> int:
        if not start <= anchor < end:
            return 0

        count = 0
        current = anchor

        # Check numbers in the positive direction
        while current < end:
            count += 1
            current += period

        # Check numbers in the negative direction
        current = anchor - period
        while current >= start:
            count += 1
            current -= period

        return count

    @staticmethod
    def _group_points_by_ts(points: list[tuple[int, Point]]) -> dict[int, list[Point]]:
        points_by_ts = {}
        for ts, point in points:
            if ts not in points_by_ts:
                points_by_ts[ts] = [point]
            else:
                points_by_ts[ts].append(point)
        return points_by_ts

    @staticmethod
    def _calculate_count(aoi: AreaOfInterest, points: list[tuple[int, Point]], start_ts: int, end_ts: int) -> int:
        points_by_ts = {}
        anchor_ts = 0
        for ts, point in points:
            anchor_ts = ts
            if ts not in points_by_ts:
                points_by_ts[ts] = [point]
            else:
                points_by_ts[ts].append(point)
        points_by_ts = Aggregator._group_points_by_ts(points)
        anchor_ts = min(points_by_ts.keys()) if len(points_by_ts) > 0 else 0

        count = 0
        for ts, points in points_by_ts.items():
            for point in points:
                if aoi.is_point_inside(point):
                    count += 1
        if count > 0:
            count = int(math.ceil(count / \
                Aggregator._count_numbers_in_range(anchor_ts, DEFAULT_SENSOR_REPORTING_PERIOD_SEC, start_ts, end_ts)))
        return count

    def _get_points(self, serial_num: str, start_ts: int, end_ts: int) -> list[tuple[int, Point]]:
        data_points = self._storage.query_data_points(serial_num, start_ts, end_ts)
        return [(ts, Point(x, y)) for ts, _, x, y in data_points]

    def check_aois(self) -> list[tuple[str, str, int, int]]:
        ts = int(time.time())

        counts = []
        aois = self._get_aoi_by_report_period(ts)
        for aoi, serial_num in aois:
            data_points = self._get_points(serial_num, ts - aoi.reporting_period, ts)
            count = self._calculate_count(aoi, data_points, ts - aoi.reporting_period, ts)
            counts.append((serial_num, aoi.id, count, ts))

        return counts

    def get_xys(self) -> list[tuple[str, int, int, int]]:
        ts = int(time.time())

        xys = []
        for serial_num in self._sensor_serials:
            data_points = self._get_points(serial_num, ts - DEFAULT_SENSOR_REPORTING_PERIOD_SEC, ts)
            points_by_ts = Aggregator._group_points_by_ts(data_points)
            last_points = points_by_ts.get(max(points_by_ts.keys()), []) if len(points_by_ts) > 0 else []
            for point in last_points:
                xys.append((serial_num, ts, point.x, point.y))
        return xys
