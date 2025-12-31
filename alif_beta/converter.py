#!/usr/bin/env python3
"""
Parse pg2gw journal log and export detections into a JSON list of dicts.

UPDATED LOGIC (important):
- Events are grouped by TELEMETRY MESSAGE, not by their own line timestamp.
- Each exported record uses the TELEMETRY MESSAGE timestamp (the timestamp of the
  "Telemetry from sensor" line that starts the message block).
- This matches the plotting logic: one "message timestamp" can contain multiple detections
  (accepted / filtered_confidence / filtered_novelty).

Event types:
- detection            (Detection translated ... pixel(...) -> cm(...). conf=...; box: ...)
- filtered_confidence  (Detection filtered out due to low confidence: ...; box: ...)
- filtered_novelty     (Detection filtered out by GMM: OUTLIER|NOVEL (x=..., y=..., w=..., h=..., conf=...))

Device id logic:
- Prefer "for sensor <id>" on the detection line (or "Detection translated for sensor <id>").
- Else use the most recent "from thing <id>" seen in the log (context).

Timestamps:
- Syslog lines don't include year; default year=2025 (override via --year).
- Output timestamp is ISO-8601 with Z, e.g. "2025-12-19T10:31:25Z"
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

MONTHS = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4,
    "May": 5, "Jun": 6, "Jul": 7, "Aug": 8,
    "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# --- Keywords --------------------------------------------------------------
KW_TELEMETRY = "Telemetry from sensor"  # telemetry message boundary

# --- Regexes ---------------------------------------------------------------

# Example prefix: "Dec 18 14:48:03"
RE_TS_PREFIX = re.compile(
    r"^(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<hms>\d{2}:\d{2}:\d{2})\s+"
)

# Context / device markers
RE_FROM_THING = re.compile(r"\bfrom thing\s+(?P<id>\S+)\b")
RE_FOR_SENSOR = re.compile(r"\bfor sensor\s+(?P<id>\S+)\b")

# detection translated:
RE_DET_TRANSLATED = re.compile(
    r"Detection translated(?: for sensor\s+(?P<sensor>\S+))?:\s*"
    r"pixel\((?P<pix_x>-?\d+),(?P<pix_y>-?\d+)\)\s*->\s*"
    r"cm\((?P<cm_x>-?\d+),(?P<cm_y>-?\d+)\)\.\s*"
    r"conf\s*=\s*(?P<conf>[0-9]*\.?[0-9]+)\s*;\s*"
    r"box:\s*x\s*=\s*(?P<bx>-?\d+),\s*y\s*=\s*(?P<by>-?\d+),\s*w\s*=\s*(?P<bw>-?\d+),\s*h\s*=\s*(?P<bh>-?\d+)"
)

# filtered confidence:
RE_FILT_CONF = re.compile(
    r"Detection filtered out due to low confidence:\s*(?P<conf>[0-9]*\.?[0-9]+)\s*;\s*"
    r"box:\s*x\s*=\s*(?P<bx>-?\d+),\s*y\s*=\s*(?P<by>-?\d+),\s*w\s*=\s*(?P<bw>-?\d+),\s*h\s*=\s*(?P<bh>-?\d+)"
)

# filtered novelty (GMM):
RE_FILT_GMM = re.compile(
    r"Detection filtered out by GMM:\s*(?P<label>OUTLIER|NOVEL)\s*\(\s*"
    r"x\s*=\s*(?P<bx>-?\d+),\s*y\s*=\s*(?P<by>-?\d+),\s*w\s*=\s*(?P<bw>-?\d+),\s*h\s*=\s*(?P<bh>-?\d+),\s*"
    r"conf\s*=\s*(?P<conf>[0-9]*\.?[0-9]+)\s*\)"
)


@dataclass
class ParseContext:
    current_device_id: Optional[str] = None
    current_telemetry_ts: Optional[str] = None  # ISO timestamp of current telemetry message


def parse_timestamp(line: str, year: int) -> Optional[str]:
    """
    Returns ISO-8601 string with Z suffix, or None if the line has no timestamp prefix.
    Assumes log time is UTC/GMT.
    """
    m = RE_TS_PREFIX.match(line)
    if not m:
        return None
    mon = m.group("mon")
    day = int(m.group("day"))
    hms = m.group("hms")
    month = MONTHS.get(mon)
    if not month:
        return None

    dt = datetime.strptime(f"{year:04d}-{month:02d}-{day:02d} {hms}", "%Y-%m-%d %H:%M:%S")
    dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def build_record(
    *,
    timestamp: str,
    device_id: Optional[str],
    rec_type: str,
    pixel_x: Optional[int],
    pixel_y: Optional[int],
    cm_x: Optional[int],
    cm_y: Optional[int],
    confidence: float,
    box: Dict[str, int],
    gmm_label: Optional[str] = None,
) -> Dict[str, Any]:
    rec: Dict[str, Any] = {
        "timestamp": timestamp,  # TELEMETRY MESSAGE timestamp
        "device_id": device_id,
        "type": rec_type,
        "pixel_x": pixel_x,
        "pixel_y": pixel_y,
        "cm_x": cm_x,
        "cm_y": cm_y,
        "confidence": confidence,
        "box": box,
    }
    if gmm_label is not None:
        rec["gmm_label"] = gmm_label  # optional extra info (OUTLIER/NOVEL)
    return rec


def parse_log_file(path: Path, year: int) -> List[Dict[str, Any]]:
    """
    IMPORTANT:
    - Only emit detection records if we are inside a telemetry message block.
    - The timestamp used for emitted records is the telemetry message timestamp.
    """
    ctx = ParseContext()
    out: List[Dict[str, Any]] = []

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n")
            line_ts = parse_timestamp(line, year)
            if line_ts is None:
                continue

            # Update device context if we see "from thing <id>"
            m_thing = RE_FROM_THING.search(line)
            if m_thing:
                ctx.current_device_id = m_thing.group("id")

            # Telemetry boundary: start a new message block
            if KW_TELEMETRY in line:
                ctx.current_telemetry_ts = line_ts
                continue

            # We only count detections that occur inside a telemetry message block
            if ctx.current_telemetry_ts is None:
                continue

            # If the line itself contains "for sensor <id>", prefer that for this record
            m_sensor = RE_FOR_SENSOR.search(line)
            sensor_id = m_sensor.group("id") if m_sensor else None

            # 1) detection translated
            m = RE_DET_TRANSLATED.search(line)
            if m:
                device_id = m.group("sensor") or sensor_id or ctx.current_device_id
                out.append(
                    build_record(
                        timestamp=ctx.current_telemetry_ts,  # <- TELEMETRY timestamp
                        device_id=device_id,
                        rec_type="detection",
                        pixel_x=int(m.group("pix_x")),
                        pixel_y=int(m.group("pix_y")),
                        cm_x=int(m.group("cm_x")),
                        cm_y=int(m.group("cm_y")),
                        confidence=float(m.group("conf")),
                        box={
                            "x": int(m.group("bx")),
                            "y": int(m.group("by")),
                            "w": int(m.group("bw")),
                            "h": int(m.group("bh")),
                        },
                    )
                )
                continue

            # 2) filtered out due to low confidence
            m = RE_FILT_CONF.search(line)
            if m:
                device_id = sensor_id or ctx.current_device_id
                out.append(
                    build_record(
                        timestamp=ctx.current_telemetry_ts,  # <- TELEMETRY timestamp
                        device_id=device_id,
                        rec_type="filtered_confidence",
                        pixel_x=None,
                        pixel_y=None,
                        cm_x=None,
                        cm_y=None,
                        confidence=float(m.group("conf")),
                        box={
                            "x": int(m.group("bx")),
                            "y": int(m.group("by")),
                            "w": int(m.group("bw")),
                            "h": int(m.group("bh")),
                        },
                    )
                )
                continue

            # 3) filtered out by GMM novelty/outlier
            m = RE_FILT_GMM.search(line)
            if m:
                device_id = sensor_id or ctx.current_device_id
                out.append(
                    build_record(
                        timestamp=ctx.current_telemetry_ts,  # <- TELEMETRY timestamp
                        device_id=device_id,
                        rec_type="filtered_novelty",
                        pixel_x=None,
                        pixel_y=None,
                        cm_x=None,
                        cm_y=None,
                        confidence=float(m.group("conf")),
                        box={
                            "x": int(m.group("bx")),
                            "y": int(m.group("by")),
                            "w": int(m.group("bw")),
                            "h": int(m.group("bh")),
                        },
                        gmm_label=m.group("label"),
                    )
                )
                continue

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "log_path",
        nargs="?",
        default="/mnt/data/pg2gw.log",
        help="Path to pg2gw.log (default: /mnt/data/pg2gw.log)",
    )
    ap.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output JSON path (default: <log_path>.parsed.json)",
    )
    ap.add_argument(
        "--year",
        type=int,
        default=2025,
        help="Year to use for timestamps (log lines don't include year). Default: 2025",
    )
    args = ap.parse_args()

    log_path = Path(args.log_path)
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    out_path = Path(args.output) if args.output else log_path.with_suffix(log_path.suffix + ".parsed.json")

    records = parse_log_file(log_path, year=args.year)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(records)} records to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
