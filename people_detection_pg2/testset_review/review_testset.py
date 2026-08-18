"""Summarise PG2 test-case captures.

A test-case folder contains one frame per file, named like::

    4.2_H_cam_249_menivim10_rnd_revc_0065_1784813613071_180.tif
    <prefix..>_H_cam_<height_cm>_<env..>_<frame_idx>_<timestamp_ms>_<duration_ms>.tif

The camera height (cm) is the token after ``H_cam_``. The last three
underscore-separated tokens before the extension are the frame index, the
capture timestamp (ms since epoch) and the frame-transfer duration (ms).

For each test case (one subfolder per case) the script reports:

* per-frame FPS statistics (avg, median, std, z-score > 3 outliers)
* number of frames, first/last frame index, missing-index ranges
* covered wall-clock time in seconds, and the wall-clock time lost to gaps
* camera height in cm

Missing indexes are collapsed into ranges rather than listed one by one.
A run of consecutive misses 9, 10, 11, 12 becomes ``9-12``; a periodic
drop such as 9, 11, 13, 15 becomes ``9-15 every 2nd frame`` (step 2). Each
range carries the wall-clock time it cost: measured from the timestamps
bracketing the hole for consecutive runs, estimated as
``count * nominal frame period`` for periodic ones.

Usage:
    python review_testset.py --input <folder-with-testcases> \
        --output <folder-for-json>
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

FRAME_EXTS = {".tif", ".tiff"}
HEIGHT_RE = re.compile(r"H_cam_(\d+)")
Z_OUTLIER_THRESHOLD = 3.0


@dataclass
class Frame:
    index: int
    timestamp_ms: int
    duration_ms: int
    filename: str


def parse_frame_name(path: Path) -> Frame | None:
    """Extract (index, timestamp_ms, duration_ms) from a frame filename.

    Returns ``None`` if the filename does not match the expected template.
    The last three underscore-separated tokens of the stem are treated as
    ``<frame_idx>_<timestamp_ms>_<duration_ms>``; everything before is env
    metadata and is ignored here (height is parsed separately).
    """
    stem = path.stem
    parts = stem.split("_")
    if len(parts) < 4:
        return None
    try:
        idx = int(parts[-3])
        ts = int(parts[-2])
        dur = int(parts[-1])
    except ValueError:
        return None
    return Frame(index=idx, timestamp_ms=ts, duration_ms=dur, filename=path.name)


def parse_camera_height_cm(name: str) -> int | None:
    m = HEIGHT_RE.search(name)
    return int(m.group(1)) if m else None


def _ordinal(n: int) -> str:
    if 10 <= n % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def group_missing_indexes(missing: list[int]) -> list[tuple[int, int, int, int]]:
    """Collapse sorted missing indexes into ``(start, end, step, count)`` runs.

    A run is a maximal arithmetic progression. Step 1 runs (a contiguous hole)
    are always emitted; a step > 1 run needs at least 3 members before it is
    called a pattern, since any two indexes trivially share a "constant" step
    and would otherwise turn every scattered pair into a fake periodic drop.
    Anything left over is emitted as a single-index run.
    """
    groups: list[tuple[int, int, int, int]] = []
    i, n = 0, len(missing)
    while i < n:
        if i + 1 < n:
            step = missing[i + 1] - missing[i]
            j = i + 1
            while j + 1 < n and missing[j + 1] - missing[j] == step:
                j += 1
            count = j - i + 1
            if step == 1 or count >= 3:
                groups.append((missing[i], missing[j], step, count))
                i = j + 1
                continue
        groups.append((missing[i], missing[i], 1, 1))
        i += 1
    return groups


def format_missing_range(start: int, end: int, step: int, count: int) -> str:
    if count == 1:
        return str(start)
    if step == 1:
        return f"{start}-{end}"
    return f"{start}-{end} every {_ordinal(step)} frame"


def nominal_period_ms(frames: list[Frame]) -> float | None:
    """Median per-index frame period, in ms.

    Dividing each timestamp delta by its index delta makes this robust to the
    very gaps it is used to price: a hole or a periodic drop stretches the raw
    period between the surviving frames, but leaves the per-index period alone.
    """
    per_index = [
        (b.timestamp_ms - a.timestamp_ms) / (b.index - a.index)
        for a, b in zip(frames, frames[1:])
        if b.index > a.index and b.timestamp_ms > a.timestamp_ms
    ]
    return statistics.median(per_index) if per_index else None


def missing_ranges(
    missing: list[int], frames: list[Frame], period_ms: float | None
) -> list[dict]:
    """Describe each missing-index run, with the wall-clock time it cost.

    Consecutive runs are priced from the timestamps of the frames bracketing
    the hole, minus the one frame period that a normal step would have taken.
    Periodic runs cannot be priced that way — frames survive inside the span —
    so they fall back to ``count * period``.
    """
    ts_by_index = {f.index: f.timestamp_ms for f in frames}
    present = sorted(ts_by_index)

    out: list[dict] = []
    for start, end, step, count in group_missing_indexes(missing):
        seconds: float | None = None
        source = "unknown"
        if step == 1:
            before = [i for i in present if i < start]
            after = [i for i in present if i > end]
            if before and after and period_ms is not None:
                hole_ms = ts_by_index[after[0]] - ts_by_index[before[-1]] - period_ms
                seconds = max(0.0, hole_ms) / 1000.0
                source = "measured"
        if seconds is None and period_ms is not None:
            seconds = count * period_ms / 1000.0
            source = "estimated"

        out.append({
            "range": format_missing_range(start, end, step, count),
            "start": start,
            "end": end,
            "step": step,
            "count": count,
            "missing_time_s": seconds,
            "missing_time_source": source,
        })
    return out


def zscore_outliers(values: list[float], threshold: float) -> list[int]:
    """Return indexes into ``values`` whose |z-score| exceeds ``threshold``.

    Requires at least 2 samples and non-zero std; otherwise returns [].
    """
    if len(values) < 2:
        return []
    mean = statistics.fmean(values)
    std = statistics.pstdev(values)
    if std == 0:
        return []
    return [i for i, v in enumerate(values) if abs((v - mean) / std) > threshold]


def summarise_case(case_dir: Path) -> dict:
    files = sorted(p for p in case_dir.iterdir() if p.suffix.lower() in FRAME_EXTS)
    frames: list[Frame] = []
    unparsed: list[str] = []
    for p in files:
        parsed = parse_frame_name(p)
        if parsed is None:
            unparsed.append(p.name)
        else:
            frames.append(parsed)

    if not frames:
        return {
            "name": case_dir.name,
            "num_frames": 0,
            "unparsed_files": unparsed,
            "error": "no parseable frames found",
        }

    frames.sort(key=lambda f: f.index)

    heights = {parse_camera_height_cm(f.filename) for f in frames}
    heights.discard(None)
    camera_height_cm: int | list[int] | None
    if len(heights) == 1:
        camera_height_cm = heights.pop()
    elif not heights:
        camera_height_cm = None
    else:
        camera_height_cm = sorted(heights)  # inconsistent → surface all values

    indexes = [f.index for f in frames]
    expected = set(range(indexes[0], indexes[-1] + 1))
    missing = sorted(expected - set(indexes))
    duplicates = sorted({i for i in indexes if indexes.count(i) > 1})

    timestamps = [f.timestamp_ms for f in frames]
    covered_time_s = (timestamps[-1] - timestamps[0]) / 1000.0

    period_ms = nominal_period_ms(frames)
    gaps = missing_ranges(missing, frames, period_ms)
    missing_time_s = sum(g["missing_time_s"] or 0.0 for g in gaps)

    # Per-frame period → per-frame FPS. Skip zero/negative diffs (would come
    # from duplicate timestamps and produce inf/undefined FPS).
    periods_ms = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
    fps_values = [1000.0 / dt for dt in periods_ms]

    fps_stats: dict = {"num_intervals": len(fps_values)}
    if fps_values:
        fps_stats["avg"] = statistics.fmean(fps_values)
        fps_stats["median"] = statistics.median(fps_values)
        fps_stats["std"] = statistics.pstdev(fps_values) if len(fps_values) > 1 else 0.0
        fps_stats["min"] = min(fps_values)
        fps_stats["max"] = max(fps_values)
        outlier_idx = zscore_outliers(fps_values, Z_OUTLIER_THRESHOLD)
        fps_stats["outliers"] = [
            {
                "interval_index": i,
                "between_frames": [frames[i].index, frames[i + 1].index],
                "fps": fps_values[i],
                "period_ms": periods_ms[i],
            }
            for i in outlier_idx
        ]

    durations = [f.duration_ms for f in frames]
    duration_stats = {
        "avg": statistics.fmean(durations),
        "median": statistics.median(durations),
        "std": statistics.pstdev(durations) if len(durations) > 1 else 0.0,
        "min": min(durations),
        "max": max(durations),
    }

    summary: dict = {
        "name": case_dir.name,
        "camera_height_cm": camera_height_cm,
        "num_frames": len(frames),
        "frame_index_range": [indexes[0], indexes[-1]],
        "expected_num_frames": len(expected),
        "num_missing_frames": len(missing),
        "missing_frame_ranges": gaps,
        "missing_time_s": missing_time_s,
        "nominal_frame_period_ms": period_ms,
        "duplicate_frame_indexes": duplicates,
        "has_no_skips": not missing and not duplicates,
        "covered_time_s": covered_time_s,
        "first_timestamp_ms": timestamps[0],
        "last_timestamp_ms": timestamps[-1],
        "fps": fps_stats,
        "transfer_duration_ms": duration_stats,
    }
    if unparsed:
        summary["unparsed_files"] = unparsed
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Folder containing one subfolder per test case.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Folder in which to write testset_review.json.",
    )
    parser.add_argument(
        "--filename",
        default="testset_review.json",
        help="Output JSON filename (default: testset_review.json).",
    )
    args = parser.parse_args()

    case_dirs = sorted(p for p in args.input.iterdir() if p.is_dir())
    if not case_dirs:
        raise SystemExit(f"No test-case subfolders found in {args.input}")

    cases = [summarise_case(d) for d in case_dirs]

    args.output.mkdir(parents=True, exist_ok=True)
    out_path = args.output / args.filename
    payload = {
        "input_folder": str(args.input.resolve()),
        "num_cases": len(cases),
        "cases": cases,
    }
    out_path.write_text(json.dumps(payload, indent=2))

    for c in cases:
        if c.get("error"):
            print(f"{c['name']}: {c['error']}")
            continue
        fps_avg = c["fps"].get("avg")
        fps_txt = f"{fps_avg:.2f}" if fps_avg is not None else "n/a"
        print(
            f"{c['name']}: {c['num_frames']} frames, "
            f"H={c['camera_height_cm']} cm, "
            f"{c['covered_time_s']:.2f} s, "
            f"fps avg={fps_txt}"
        )
        for g in c["missing_frame_ranges"]:
            lost = (
                f"{g['missing_time_s']:.2f} s {g['missing_time_source']}"
                if g["missing_time_s"] is not None else "time unknown"
            )
            plural = "frame" if g["count"] == 1 else "frames"
            print(f"    missing {g['range']}  ({g['count']} {plural}, {lost})")
        if c["missing_frame_ranges"]:
            print(f"    missing total: {c['num_missing_frames']} frames, "
                  f"{c['missing_time_s']:.2f} s")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
