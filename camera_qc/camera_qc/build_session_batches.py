"""Build one ref/dut batch per capture session, per chosen reference unit.

Some datasets are laid out as one folder per capture session — a time of day
such as `5.07PM/` — holding one image of every unit, rather than the
`batch_<n>/{ref,dut}/` split that `build_batch_all.py` expects.

Comparing across sessions is what makes those datasets hard to read: the same
module measured in two sessions can differ by more than the tolerance band, so
a cross-session verdict mixes capture conditions with module quality. Giving
each session its own reference removes that: every DUT is compared against a
module photographed minutes earlier, under the same light, in the same rig.

For each session and each `--ref-serial`, this writes

    <out>/<session>_ref<serial>/ref/    the reference unit's capture
    <out>/<session>_ref<serial>/dut/    every other unit in that session
    <out>/<session>_ref<serial>/manifest.csv

plus an `index.csv` at the top listing every batch built.

Usage:
    python -m camera_qc.build_session_batches \
        --src ~/Pictures/RevC_Camera_eval --ref-serial 119 --ref-serial 120 \
        --out ./revc
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from .build_batch_all import IMAGE_EXTS, canonical_name, md5, place


log = logging.getLogger(__name__)


# `6000000-1001-000119-2628.jpg` -> serial 119.
SERIAL_RE = re.compile(r"^\d+-\d+-(\d+)-\d+$")

# `5.07PM` / `11.30AM` -> a sortable minute-of-day. Sessions must order by
# clock time, not by filename: `10.00AM` sorts before `5.07PM` lexically but
# after it in a 12-hour reading, and the reverse for `6.02PM` vs `6.13PM`.
SESSION_RE = re.compile(r"^(\d{1,2})[.:](\d{2})\s*(AM|PM)$", re.I)


@dataclass(frozen=True)
class Capture:
    session: str
    serial: int
    path: Path


def session_minutes(name: str) -> int | None:
    m = SESSION_RE.match(name.strip())
    if not m:
        return None
    hour, minute, meridiem = int(m.group(1)), int(m.group(2)), m.group(3).upper()
    if not (1 <= hour <= 12 and minute < 60):
        return None
    hour = hour % 12 + (12 if meridiem == "PM" else 0)
    return hour * 60 + minute


def serial_of(path: Path) -> int | None:
    m = SERIAL_RE.match(path.stem)
    return int(m.group(1)) if m else None


def discover_sessions(src: Path) -> list[str]:
    """Session folder names, ordered by clock time."""
    found = []
    for child in sorted(src.iterdir()):
        if not child.is_dir():
            continue
        minutes = session_minutes(child.name)
        if minutes is None:
            log.debug("Skipping %s — not a recognisable session name", child.name)
            continue
        found.append((minutes, child.name))
    return [name for _m, name in sorted(found)]


def collect_session(src: Path, session: str) -> dict[int, Capture]:
    """The captures in one session, keyed by serial.

    One image per unit is expected. A repeated serial is reported rather than
    silently resolved — within a single session there is no later capture to
    prefer, so the right fix is in the source folder.
    """
    captures: dict[int, Capture] = {}
    for path in sorted((src / session).iterdir()):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS:
            continue
        serial = serial_of(path)
        if serial is None:
            log.warning("%s/%s — filename does not carry a serial, skipping",
                        session, path.name)
            continue
        if serial in captures:
            raise SystemExit(
                f"{session} holds two captures of unit {serial}: "
                f"{captures[serial].path.name} and {path.name}. Remove one — "
                f"a session is expected to hold one image per unit."
            )
        captures[serial] = Capture(session, serial, path)
    return captures


def suffixed_name(capture: Capture) -> str:
    """`…-000114-2628.jpg` in `5.07PM` -> `…-000114-2628_5.07PM.jpg`.

    Applied to every image so a file lifted out of its folder still says which
    session it came from.
    """
    name = canonical_name(capture.path)
    stem, _, ext = name.rpartition(".")
    return f"{stem}_{capture.session}.{ext}"


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["dut", "unit", "serial", "session", "source_path", "md5"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build-session-batches",
        description=(
            "Build one ref/dut batch per capture session per reference unit, "
            "so every DUT is compared against a reference shot in the same "
            "session."
        ),
    )
    p.add_argument("--src", type=Path, required=True,
                   help="Folder holding the session subfolders (e.g. 5.07PM/).")
    p.add_argument("--out", type=Path, required=True,
                   help="Destination root; one subfolder per batch is created.")
    p.add_argument("--ref-serial", type=int, action="append", required=True,
                   metavar="N",
                   help="Unit serial to use as the reference. Repeat the flag "
                        "for more than one (a batch is built per session per "
                        "serial).")
    p.add_argument("--symlink", action="store_true",
                   help="Symlink to the source images instead of copying.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be built without writing anything.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing batch folders under --out.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Verbose logging (DEBUG level).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-5s  %(message)s",
    )

    src = args.src.expanduser()
    out = args.out.expanduser()
    if not src.is_dir():
        raise SystemExit(f"Source folder not found: {src}")

    sessions = discover_sessions(src)
    if not sessions:
        raise SystemExit(
            f"No session folders under {src}. Expected names like '5.07PM'.")
    log.info("Found %d session(s): %s", len(sessions), ", ".join(sessions))

    by_session = {s: collect_session(src, s) for s in sessions}
    for s in sessions:
        serials = sorted(by_session[s])
        log.info("%-9s %d unit(s): %s", s, len(serials),
                 ", ".join(f"#{n}" for n in serials))

    index_rows = []
    built = skipped = 0
    for session in sessions:
        captures = by_session[session]
        for ref_serial in args.ref_serial:
            batch = f"{session}_ref{ref_serial}"
            if ref_serial not in captures:
                log.warning("%s — no capture of the reference unit #%d, "
                            "skipping this batch", batch, ref_serial)
                skipped += 1
                continue
            duts = [c for serial, c in sorted(captures.items())
                    if serial != ref_serial]
            if not duts:
                log.warning("%s — no DUT units besides the reference, skipping",
                            batch)
                skipped += 1
                continue

            ref_capture = captures[ref_serial]
            rows = [{
                "dut": suffixed_name(c),
                "unit": canonical_name(c.path),
                "serial": c.serial,
                "session": c.session,
                "source_path": str(c.path),
                "md5": "" if args.dry_run else md5(c.path),
            } for c in duts]

            log.info("%-18s ref #%d, %d DUT(s): %s", batch, ref_serial,
                     len(duts), ", ".join(f"#{c.serial}" for c in duts))
            index_rows.append({
                "batch": batch,
                "session": session,
                "ref_serial": ref_serial,
                "ref_image": suffixed_name(ref_capture),
                "n_dut": len(duts),
                "dut_serials": " ".join(str(c.serial) for c in duts),
            })

            if args.dry_run:
                built += 1
                continue

            batch_dir = out / batch
            for sub in ("ref", "dut"):
                d = batch_dir / sub
                if d.is_dir() and any(d.iterdir()):
                    if not args.force:
                        raise SystemExit(
                            f"{d} already exists and is not empty. Re-run with "
                            f"--force to rebuild.")
                    for f in d.iterdir():
                        f.unlink()
                d.mkdir(parents=True, exist_ok=True)

            place(ref_capture.path, batch_dir / "ref" / suffixed_name(ref_capture),
                  args.symlink)
            for c in duts:
                place(c.path, batch_dir / "dut" / suffixed_name(c), args.symlink)
            write_manifest(batch_dir / "manifest.csv", rows)
            built += 1

    if not args.dry_run and index_rows:
        out.mkdir(parents=True, exist_ok=True)
        with (out / "index.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(index_rows[0]))
            w.writeheader()
            w.writerows(index_rows)
        log.info("Index written to %s", out / "index.csv")

    log.info("%s %d batch(es)%s",
             "Would build" if args.dry_run else "Built", built,
             f"; skipped {skipped}" if skipped else "")
    return 0


if __name__ == "__main__":
    sys.exit(main())
