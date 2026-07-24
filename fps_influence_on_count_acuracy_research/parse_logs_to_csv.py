"""Parse a PG2 gateway telemetry log into a per-frame detection CSV.

The gateway (``~/projects/PG2_GW``, ``src/coap.c`` -> ``processDetections``)
logs one line per received telemetry packet followed by zero or more detection
lines for that same packet::

    Telemetry from sensor: <ipv6>
    Received telemetry data, length: <N>, from thing <thing-id>
    Detection translated for sensor <id>: pixel(px,py) -> cm(cx,cy). conf = <c>; box: x = <x>, y = <y>, w = <w>, h = <h>
    Detection filtered out due to low confidence: <c>; box: ...

Each ``Received telemetry data`` line opens a frame; the ``Detection
translated`` lines that follow (until the next telemetry line) are the accepted
person detections for that frame. ``Detection filtered out`` lines are rejected
by the gateway and do not count as detections.

Ground truth is taken from a marks file (``det.txt``) describing how many people
were in the room from a given wall-clock time onward, e.g.::

    05:19:25 0 ppl
    06:08:21 1 ppl
    06:36:14 2 ppl

Each mark holds until the next one (a step function over time of day).

Output CSV columns:
    timestamp              original syslog timestamp of the frame
    num_detections         count of accepted detections in the frame
    detections             JSON list of {cm, conf, box} for each detection
    gt_num_person          ground-truth person count from the marks file

Frames with no detections are kept with ``num_detections = 0`` and an empty
detections list.

Usage:
    python parse_logs_to_csv.py
    python parse_logs_to_csv.py --logs ControlledENV/logs.txt \
        --marks ControlledENV/det.txt --out ControlledENV/detections.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass, field
from datetime import time
from pathlib import Path

# Jul 23 05:19:25 pg2gw pg2gw[749]: <message>
LINE_RE = re.compile(
    r"^(?P<ts>[A-Za-z]{3}\s+\d+\s+(?P<tod>\d{2}:\d{2}:\d{2}))\s+\S+\s+\S+:\s+(?P<msg>.*)$"
)

RECEIVED_RE = re.compile(r"^Received telemetry data, length: (?P<length>\d+), from thing (?P<thing>\S+)")

DETECTION_RE = re.compile(
    r"^Detection translated for sensor (?P<sensor>\S+): "
    r"pixel\((?P<px>-?\d+),(?P<py>-?\d+)\) -> cm\((?P<cx>-?\d+),(?P<cy>-?\d+)\)\. "
    r"conf = (?P<conf>[\d.]+); "
    r"box: x = (?P<x>-?\d+), y = (?P<y>-?\d+), w = (?P<w>-?\d+), h = (?P<h>-?\d+)"
)

MARK_RE = re.compile(r"^(?P<tod>\d{1,2}:\d{2}:\d{2})\s+(?P<count>\d+)\s*ppl")


def parse_tod(text: str) -> time:
    """Parse an ``HH:MM:SS`` time-of-day string."""
    h, m, s = (int(p) for p in text.split(":"))
    return time(h, m, s)


def load_marks(path: Path) -> list[tuple[time, int]]:
    """Load ground-truth marks as a sorted list of ``(time_of_day, count)``."""
    marks: list[tuple[time, int]] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        m = MARK_RE.match(line)
        if not m:
            continue
        marks.append((parse_tod(m["tod"]), int(m["count"])))
    marks.sort(key=lambda kv: kv[0])
    return marks


def gt_for(tod: time, marks: list[tuple[time, int]]) -> int | None:
    """Return the ground-truth count in effect at ``tod`` (step function).

    Returns ``None`` for frames that precede the first mark.
    """
    current: int | None = None
    for mark_tod, count in marks:
        if mark_tod <= tod:
            current = count
        else:
            break
    return current


@dataclass
class Frame:
    timestamp: str
    tod: time
    detections: list[dict] = field(default_factory=list)


def parse_frames(path: Path) -> list[Frame]:
    """Group log lines into per-telemetry-packet frames."""
    frames: list[Frame] = []
    current: Frame | None = None

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        msg = m["msg"]

        recv = RECEIVED_RE.match(msg)
        if recv:
            current = Frame(timestamp=m["ts"], tod=parse_tod(m["tod"]))
            frames.append(current)
            continue

        det = DETECTION_RE.match(msg)
        if det and current is not None:
            current.detections.append(
                {
                    "cm": [int(det["cx"]), int(det["cy"])],
                    "pixel": [int(det["px"]), int(det["py"])],
                    "conf": float(det["conf"]),
                    "box": [int(det["x"]), int(det["y"]), int(det["w"]), int(det["h"])],
                }
            )
        # 'Detection filtered out ...' and other lines are ignored on purpose.

    return frames


def main() -> None:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--logs", type=Path, default=here / "ControlledENV" / "logs.txt")
    parser.add_argument("--marks", type=Path, default=here / "ControlledENV" / "det.txt")
    parser.add_argument("--out", type=Path, default=here / "ControlledENV" / "detections.csv")
    args = parser.parse_args()

    marks = load_marks(args.marks)
    frames = parse_frames(args.logs)

    with args.out.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["timestamp", "num_detections", "detections", "gt_num_person"])
        for frame in frames:
            gt = gt_for(frame.tod, marks)
            writer.writerow(
                [
                    frame.timestamp,
                    len(frame.detections),
                    json.dumps(frame.detections, separators=(",", ":")),
                    "" if gt is None else gt,
                ]
            )

    total_det = sum(len(f.detections) for f in frames)
    print(f"Parsed {len(frames)} frames, {total_det} detections -> {args.out}")


if __name__ == "__main__":
    main()
