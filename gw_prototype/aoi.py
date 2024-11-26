from geometry import Point


class AreaOfInterest:
    def __init__(self, area_id: str, name: str, polygon: list[Point], reporting_period_sec: int) -> None:
        self._id = area_id
        self._name = name
        self._polygon = polygon
        self._reporting_period = reporting_period_sec

    @property
    def id(self) -> str:
        return self._id

    @property
    def reporting_period(self) -> int:
        return self._reporting_period

    @reporting_period.setter
    def reporting_period(self, period_sec: int) -> None:
        if period_sec <= 0:
            raise ValueError("Reporting period must be positive")
        self._reporting_period = period_sec

    @property
    def polygon(self) -> list[Point]:
        return self._polygon

    @polygon.setter
    def polygon(self, polygon: list[Point]) -> None:
        self._polygon = polygon

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        if not name:
            raise ValueError("Name cannot be empty")
        self._name = name

    @staticmethod
    def _cross_product(o, a, b):
        # Calculate the cross product of vector OA and OB
        # A positive cross product indicates a counter-clockwise turn,
        # a negative cross product indicates a clockwise turn,
        # and zero indicates a collinear point.
        return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x)

    def is_point_inside(self, point: Point) -> bool:
        n_pos = 0
        n_neg = 0
        for i in range(len(self._polygon)):
            a = self._polygon[i]
            b = self._polygon[(i + 1) % len(self._polygon)]
            if self._cross_product(a, b, point) < 0:
                n_neg += 1
            else:
                n_pos += 1

        if n_neg > 0 and n_pos > 0:
            return False
        else:
            return True
