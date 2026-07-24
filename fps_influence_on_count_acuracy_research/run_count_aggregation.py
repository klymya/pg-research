"""Run PG2 count aggregation over parsed detection logs by reusing the emulator.

This is a thin adapter around the aggregation pipeline in
``~/projects/algo_system_emulator`` (``src/gw/aggregator.py``), which is itself a
port of the gateway's active aggregation path (``PG2_GW/src/rawDetect.c`` ->
``sendAgreggateCount`` with ``ENABLE_SMOOTHING=1``). No aggregation logic is
reimplemented here: we import :class:`Aggregator`, feed it the detections parsed
out of the gateway log (see ``parse_logs_to_csv.py``), and drive it exactly like
``video_processor.get_counts()``.

Data adapter:
    - Each input row carries already-translated ``cm`` detection coordinates
      (the gateway's ``translateDetection`` output), so we build
      :class:`Point` directly from ``cm`` and skip the emulator's pixel->cm
      preprocessing.
    - A single catch-all AOI covers all space, so the aggregated value is a
      whole-room person count comparable to the ``det.txt`` ground truth.
    - Smoothing is enabled (median window 3 + EMA alpha 0.25 + 20% buffer) to
      match the deployed gateway build.
    - ``--detection-period`` subsamples the input to model a sensor reporting at
      1/N fps (native rate ~10s) -- the independent variable of the study. Two
      methods (``--subsample``): 'row' (default) keeps every k-th frame, which
      preserves the real inter-report timing; 'time' keeps one frame per N-sec
      grid window. The count *output* period (``--period``) stays fixed so
      results share a common grid.

The aggregator reports at multiples of the output reporting period; each report
tick emits one output row (including ticks whose count is 0).

Output CSV columns:
    timestamp   wall-clock time of the report tick (same format as the input)
    count       aggregated person count reported for that tick

Usage:
    python run_count_aggregation.py                        # 10s, row-decimation (default)
    python run_count_aggregation.py --detection-period 20  # -> aggregated_counts_p20_row.csv
    python run_count_aggregation.py --detection-period 20 --subsample time
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_EMULATOR = Path.home() / "projects" / "algo_system_emulator"
LOG_TS_FORMAT = "%b %d %H:%M:%S"  # e.g. "Jul 23 05:19:25" (year not in syslog)

# Catch-all AOI half-extent in cm: far larger than any translated coordinate, so
# every detection falls inside and the count is a whole-room total.
CATCH_ALL_EXTENT_CM = 100_000

# Sensor detection reporting periods to study (sec). Native rate is ~10s (0.1 fps);
# larger values subsample the input to simulate a slower-reporting sensor.
DETECTION_PERIOD_CHOICES = [10, 20, 30, 60]
NATIVE_PERIOD_SEC = 10  # native sensor cadence, used to derive the row-decimation factor


def _load_emulator(emulator_root: Path):
    """Import the emulator's aggregation package and return its symbols."""
    src = emulator_root / "src"
    if not src.is_dir():
        raise SystemExit(f"Emulator source not found: {src}")
    sys.path.insert(0, str(src))
    from gw.aggregator import Aggregator  # noqa: E402
    from gw.aoi import AreaOfInterest  # noqa: E402
    from gw.geometry import Point  # noqa: E402

    return Aggregator, AreaOfInterest, Point


def _make_catch_all_aoi(AreaOfInterest, Point, reporting_period_sec: int):
    """A single AOI whose convex polygon encloses the whole coordinate space."""
    b = CATCH_ALL_EXTENT_CM
    # Same winding as the project's aois.json (top-left, top-right, bottom-*).
    polygon = [Point(-b, b), Point(b, b), Point(b, -b), Point(-b, -b)]
    return AreaOfInterest("room", "room", polygon, reporting_period_sec)


def load_frames(
    detections_csv: Path, Point, year: int
) -> tuple[list[tuple[int, list]], tuple[int, int]]:
    """Adapt parsed-log rows into per-frame ``(epoch_sec, [Point, ...])`` records.

    A frame is one telemetry packet (one row); its points are that packet's
    detections. Returns the frames plus the ``(min, max)`` timestamp bounds of
    *all* frames (including detection-free ones), so aggregation spans the whole
    log and emits ``count=0`` reports for empty-room periods, matching the
    gateway's timer-driven reporting.
    """
    frames: list[tuple[int, list]] = []
    with detections_csv.open(newline="") as fh:
        for row in csv.DictReader(fh):
            dt = datetime.strptime(row["timestamp"], LOG_TS_FORMAT).replace(
                year=year, tzinfo=timezone.utc
            )
            ts = int(dt.timestamp())
            pts = [Point(*det["cm"]) for det in json.loads(row["detections"])]
            frames.append((ts, pts))
    if not frames:
        raise SystemExit("No frames found in input; nothing to aggregate.")
    ts_all = [ts for ts, _ in frames]
    return frames, (min(ts_all), max(ts_all))


def subsample_row(frames: list[tuple[int, list]], period_sec: int) -> list[tuple[int, list]]:
    """Keep every ``k``-th frame, ``k = period_sec / NATIVE_PERIOD_SEC`` (default).

    Row decimation: taking every k-th packet the sensor actually sent preserves
    the real inter-report timing character (jitter compounds naturally) instead
    of snapping reports onto a fixed grid. This is the faithful model of a
    slower real sensor, so it is the default.
    """
    k = max(1, round(period_sec / NATIVE_PERIOD_SEC))
    return frames[::k]


def subsample_time(frames: list[tuple[int, list]], period_sec: int) -> list[tuple[int, list]]:
    """Keep one frame per ``period_sec`` wall-clock window (grid bucketing).

    The first frame entering each new time bucket is kept, whether or not it has
    detections (an empty report is still a report). Guarantees an exact mean
    cadence of ``period_sec`` but introduces grid-quantization jitter a real
    sensor would not have -- provided as an alternative for comparison.
    """
    kept: list[tuple[int, list]] = []
    last_bucket: int | None = None
    for ts, pts in frames:
        bucket = ts // period_sec
        if bucket != last_bucket:
            kept.append((ts, pts))
            last_bucket = bucket
    return kept


SUBSAMPLERS = {"row": subsample_row, "time": subsample_time}


def frames_to_points(frames: list[tuple[int, list]]) -> list[tuple[int, object]]:
    """Flatten per-frame records into the aggregator's ``(epoch_sec, Point)`` list."""
    return [(ts, p) for ts, pts in frames for p in pts]


def aggregate(
    points: list[tuple[int, object]],
    ts_bounds: tuple[int, int],
    Aggregator,
    aoi,
    period_sec: int,
) -> list[tuple[int, int]]:
    """Drive the aggregator across the timeline, collecting per-tick counts.

    Mirrors ``video_processor.get_counts()``: step second-by-second and let the
    aggregator decide when a report is due (``ts % reporting_period == 0``).
    """
    aggregator = Aggregator(period_sec, smoothing_enabled=True)
    aggregator.set_aois([aoi])
    aggregator.set_points(points)

    start_ts, end_ts = ts_bounds
    results: list[tuple[int, int]] = []
    for ts in range(start_ts, end_ts + 1):
        _, counts = aggregator.check_aois(ts)
        if counts:  # non-empty only on report ticks
            _, count = counts[0]  # single catch-all AOI
            results.append((ts, count))
    return results


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--detections", type=Path, default=here / "artifacts" / "detections.csv")
    parser.add_argument("--out", type=Path, default=None,
                        help="output CSV (default: artifacts/aggregated_counts_p<detection-period>.csv)")
    parser.add_argument("--detection-period", type=int, default=10, choices=DETECTION_PERIOD_CHOICES,
                        help="sensor detection reporting period in seconds (input fps: 1 frame per N sec)")
    parser.add_argument("--subsample", choices=list(SUBSAMPLERS), default="row",
                        help="how to reduce fps: 'row' = keep every k-th frame (realistic timing, default); "
                             "'time' = one frame per N-sec grid window")
    parser.add_argument("--period", type=int, default=60, help="count output reporting period in seconds")
    parser.add_argument("--emulator", type=Path, default=DEFAULT_EMULATOR)
    parser.add_argument("--year", type=int, default=2026, help="year for timestamps (syslog omits it)")
    args = parser.parse_args()

    if not args.detections.is_file():
        raise SystemExit(f"Detections CSV not found: {args.detections}")

    out = args.out or here / "artifacts" / f"aggregated_counts_p{args.detection_period}_{args.subsample}.csv"

    Aggregator, AreaOfInterest, Point = _load_emulator(args.emulator)
    aoi = _make_catch_all_aoi(AreaOfInterest, Point, args.period)

    frames, ts_bounds = load_frames(args.detections, Point, args.year)
    sampled = SUBSAMPLERS[args.subsample](frames, args.detection_period)
    points = frames_to_points(sampled)
    results = aggregate(points, ts_bounds, Aggregator, aoi, args.period)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "count"])
        for ts, count in results:
            stamp = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(LOG_TS_FORMAT)
            writer.writerow([stamp, count])

    print(
        f"Detection period {args.detection_period}s [{args.subsample}]: "
        f"kept {len(sampled)}/{len(frames)} frames "
        f"({len(points)} detections) -> {len(results)} count reports "
        f"(output period {args.period}s) -> {out}"
    )


if __name__ == "__main__":
    main()
