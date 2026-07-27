"""Per-frame count accuracy vs ground truth at several detection rates (fps).

Reads the parsed detections (``detections.csv``) and, for each simulated sensor
rate, checks how often a frame's raw detection count already equals the
ground-truth room count -- with no aggregation. Lower rates are simulated by
row-decimation: keep every k-th frame, ``k = detection_period / 10`` (native
rate is ~10s / 0.1 fps):

    1/10 fps -> every frame        (k=1)
    1/20 fps -> every 2nd frame    (k=2)
    1/30 fps -> every 3rd frame    (k=3)
    1/60 fps -> every 6th frame    (k=6)

A frame is "correct" when ``num_detections == gt_num_person``. Both columns are
already present in ``detections.csv`` (see ``parse_logs_to_csv.py``).

Output CSV columns:
    detection_period_sec   simulated reporting period (s)
    fps                    "1/N" label
    row_step               k (every k-th row kept)
    frames                 number of kept frames
    correct                frames where count matched GT
    pct_correct            percentage correct

Usage:
    python per_frame_accuracy.py
    python per_frame_accuracy.py --detections artifacts/detections.csv \
        --out artifacts/per_frame_accuracy.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

NATIVE_PERIOD_SEC = 10
DETECTION_PERIODS = [10, 20, 30, 60]


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detections", type=Path, default=here / "artifacts" / "detections.csv")
    parser.add_argument("--out", type=Path, default=here / "artifacts" / "per_frame_accuracy.csv")
    args = parser.parse_args()

    if not args.detections.is_file():
        raise SystemExit(f"Detections CSV not found: {args.detections}")

    with args.detections.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("No frames found in input.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["detection_period_sec", "fps", "row_step", "frames", "correct", "pct_correct"])
        for period in DETECTION_PERIODS:
            k = max(1, round(period / NATIVE_PERIOD_SEC))
            kept = rows[::k]
            correct = sum(1 for r in kept if r["num_detections"] == r["gt_num_person"])
            frames = len(kept)
            pct = 100.0 * correct / frames
            writer.writerow([period, f"1/{period}", k, frames, correct, f"{pct:.2f}"])
            print(f"1/{period:<2} fps (every {k} row): {correct}/{frames} correct = {pct:.2f}%")

    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
