"""Summarise PG2 test-case captures.

A test-case folder contains one frame per file, named like::

    4.2_H_cam_249_menivim10_rnd_revc_0065_1784813613071_180.tif
    <prefix..>_H_cam_<height_cm>_<env..>_<frame_idx>_<timestamp_ms>_<duration_ms>.tif

The camera height (cm) is the token after ``H_cam_``. The last three
underscore-separated tokens before the extension are the frame index, the
capture timestamp (ms since epoch) and the frame-transfer duration (ms).

For each test case (one subfolder per case) the script reports:

* per-frame FPS statistics (avg, median, std, z-score > 3 outliers)
* number of frames, first/last frame index, list of missing indexes
* covered wall-clock time in seconds
* camera height in cm

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
        "missing_frame_indexes": missing,
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
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
