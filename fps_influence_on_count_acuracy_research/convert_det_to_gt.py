"""Convert the ground-truth marks file (``det.txt``) into a GT count file.

``det.txt`` records how many people are in the room from a given wall-clock time
onward (a step function), e.g.::

    05:19:25 0 ppl
    06:08:21 1 ppl
    06:36:14 2 ppl

This emits one row per threshold mark with columns ``timestamp,count``. Each
mark holds until the next one, so a consumer reads the ground truth at any time
``t`` by taking the last row whose timestamp is ``<= t``.

Timestamps are prefixed with the log date (``--date``, default ``Jul 23``) so the
GT file uses the same ``"Jul 23 HH:MM:SS"`` format as ``detections.csv`` and
``aggregated_counts.csv`` and joins against them directly.

Parsing of ``det.txt`` is reused from :mod:`parse_logs_to_csv` (``load_marks``)
rather than duplicated.

Usage:
    python convert_det_to_gt.py
    python convert_det_to_gt.py --marks ControlledENV/det.txt \
        --out ControlledENV/gt_counts.csv --format csv --date "Jul 23"
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from parse_logs_to_csv import load_marks


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--marks", type=Path, default=here / "ControlledENV" / "det.txt")
    parser.add_argument("--out", type=Path, default=here / "ControlledENV" / "gt_counts.csv")
    parser.add_argument("--format", choices=["csv", "json"], default="csv")
    parser.add_argument("--date", default="Jul 23", help="date prefix for timestamps (det.txt omits it)")
    args = parser.parse_args()

    marks = load_marks(args.marks)
    if not marks:
        raise SystemExit(f"No ground-truth marks found in {args.marks}")

    rows = [{"timestamp": f"{args.date} {tod.strftime('%H:%M:%S')}", "count": count} for tod, count in marks]

    out = args.out.with_suffix(".json" if args.format == "json" else ".csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.format == "json":
        out.write_text(json.dumps(rows, indent=2) + "\n")
    else:
        with out.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["timestamp", "count"])
            writer.writeheader()
            writer.writerows(rows)

    print(f"Wrote {len(rows)} ground-truth threshold marks -> {out}")


if __name__ == "__main__":
    main()
