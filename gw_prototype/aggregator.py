import time
import math

from xy_storage import XYStorage
from aoi import AreaOfInterest
from geometry import Point


DEFAULT_SENSOR_REPORTING_PERIOD_SEC = 5


class Aggregator:
    def __init__(self) -> None:
        self._storage = XYStorage()
        self._aois = {}
        self._sensor_reporting_periods = {}

    def set_aoi(self, sensor_serial_number: str, aois: list[AreaOfInterest]) -> None:
        self._aois[sensor_serial_number] = aois

    def get_aoi(self, sensor_serial_number: str) -> list[AreaOfInterest]:
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
    def _calculate_count(aoi: AreaOfInterest, points: list[tuple[int, Point]], start_ts: int, end_ts: int) -> int:
        points_by_ts = {}
        anchor_ts = 0
        for ts, point in points:
            anchor_ts = ts
            if ts not in points_by_ts:
                points_by_ts[ts] = [point]
            else:
                points_by_ts[ts].append(point)

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

    def check_aois(self) -> None:
        ts = int(time.time())

        counts = []
        aois = self._get_aoi_by_report_period(ts)
        for aoi, serial_num in aois:
            data_points = self._get_points(serial_num, ts - aoi.reporting_period, ts)
            count = self._calculate_count(aoi, data_points, ts - aoi.reporting_period, ts)
            counts.append((serial_num, aoi.id, count))

        return counts

    def run(self) -> None:
        while True:
            self.check_aois()
            time.sleep(1)
