"""Merge per-run capture batches into one global batch for a single QC pass.

Each capture run lives in its own `batch_<n>/` folder holding `ref/` and
`dut/`. A module that failed in `batch_n` gets re-worked and re-shot in a
later batch, so when the same DUT image name appears in several batches the
capture from the **highest-numbered** batch is the one that counts.

This builds `batch_all/` with

    batch_all/ref/   <- the reference image, taken from one chosen batch
    batch_all/dut/   <- one image per DUT, latest batch wins
    batch_all/manifest.csv

which is then evaluated in the usual way:

    python -m camera_qc --ref batch_all/ref --dut batch_all/dut \
        --scene chessboard --out batch_all/report.pdf

Usage:
    python -m camera_qc.build_batch_all [--src ~/Pictures] [--out ./batch_all]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path


log = logging.getLogger(__name__)


IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})

# `batch_7` -> 7. Ordering is numeric, not lexicographic, so batch_10 sorts
# after batch_9 rather than after batch_1.
BATCH_RE = re.compile(r"^batch_(\d+)$")

# Finder/Explorer duplicate suffixes: "foo copy.jpg", "foo_copy.jpg",
# "foo copy 2.jpg", "foo (1).jpg". These are the same capture as "foo.jpg"
# and must collapse onto it rather than count as a separate DUT.
COPY_SUFFIX_RE = re.compile(r"(?:[ _-]copy(?:[ _-]?\d+)?|[ _]\(\d+\))$", re.I)


def unit_filename(serial: str | int) -> str:
    """`163` -> the full DUT filename for that serial."""
    return f"6000001-1001-{int(serial):06d}-2634.jpg"


# Facts about a unit that the optical metrics cannot express. Carried into
# manifest.csv and surfaced in the report, so a unit rejected for a non-optical
# reason is not silently read as an optical pass.
# Units that failed for a reason sharpness cannot see. The optical
# measurement is still taken and reported; the verdict is forced to FAIL so a
# unit rejected on electronics is not counted as accepted just because its
# optics are fine.
FORCED_FAILURES: dict[str, str] = {
    # Empty by design. #135 was replaced for a communication fault, but this
    # report judges optics: it passes that check at 157% of reference and is
    # reported as PASS, with the electronics failure recorded in UNIT_NOTES
    # and tracked wherever acceptance is decided. Add an entry here only when
    # a non-optical rejection should override the optical verdict in *this*
    # report.
}


UNIT_NOTES: dict[str, str] = {
    unit_filename(119): (
        "Camera module replaced once: the first module was shot in batch_1, "
        "batch_10 and batch_11 and never reached the sharpness floor, so it "
        "was swapped; the replacement is the batch_12 capture, which passes. "
        "Caveat on the split: the first module's three captures measure 508, "
        "478 and 514 — a 1.08x spread, well inside the 1.35x capture "
        "repeatability measured on the reference module itself (#106, "
        "re-shot in all 13 batches, 758-1020). They therefore cannot be told "
        "apart from the images, and if the test log records a second "
        "replacement among them this split has to be corrected."
    ),
    unit_filename(135): (
        "Camera module replaced for a communication fault — an electronics "
        "failure, not an optical one. Failed on electronics only; it appears "
        "once here and its optical result stands on its own."
    ),
}


@dataclass(frozen=True)
class Candidate:
    """One capture of one DUT, found in one batch."""
    batch_index: int
    batch_name: str
    path: Path

    @property
    def is_verbatim(self) -> bool:
        """True when the filename is already the canonical name, i.e. it is
        not a `... copy.jpg` duplicate."""
        return self.path.name == canonical_name(self.path)


def canonical_name(path: Path) -> str:
    """The DUT identity: filename with any copy-suffix stripped and the
    extension lower-cased, so `6000001-1001-000177-2634_copy.JPG` and
    `6000001-1001-000177-2634.jpg` are recognised as the same unit."""
    stem = COPY_SUFFIX_RE.sub("", path.stem)
    return f"{stem}{path.suffix.lower()}"


def list_images(folder: Path) -> list[Path]:
    """Image files directly inside `folder`, sorted. Non-images (.DS_Store,
    Thumbs.db, stray PDFs) are ignored."""
    if not folder.is_dir():
        return []
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def discover_batches(src: Path) -> list[tuple[int, Path]]:
    """All `batch_<n>` folders under `src`, ordered oldest to newest."""
    found: list[tuple[int, Path]] = []
    for child in src.iterdir():
        if not child.is_dir():
            continue
        m = BATCH_RE.match(child.name)
        if m:
            found.append((int(m.group(1)), child))
    return sorted(found, key=lambda t: t[0])


def collect_dut(
    batches: list[tuple[int, Path]],
    include_ref: bool = False,
) -> dict[str, list[Candidate]]:
    """Map each DUT's canonical name to every capture of it, ordered oldest
    batch first. The last entry of each list is the winner.

    `include_ref` also folds each batch's `ref/` image into the pool. Those
    are captures of a real module too, so when the global reference is a
    *different* module they are ordinary DUT captures — see `--ref-from`.
    """
    by_dut: dict[str, list[Candidate]] = {}
    for index, batch_dir in batches:
        images = list_images(batch_dir / "dut")
        ref_images = list_images(batch_dir / "ref") if include_ref else []
        if not images and not ref_images:
            log.warning("%s has no images — skipping", batch_dir.name)
            continue
        for path in images + ref_images:
            key = canonical_name(path)
            by_dut.setdefault(key, []).append(
                Candidate(index, batch_dir.name, path)
            )
        extra = f" + {len(ref_images)} ref" if ref_images else ""
        log.info("%-9s %2d DUT image(s)%s", batch_dir.name, len(images), extra)
    return by_dut


def pick_winner(candidates: list[Candidate]) -> Candidate:
    """Latest batch wins. Within one batch prefer the verbatim filename over
    its `... copy.jpg` twin, then the lexicographically first name, so the
    choice never depends on directory iteration order."""
    newest = max(c.batch_index for c in candidates)
    same_batch = [c for c in candidates if c.batch_index == newest]
    return min(same_batch, key=lambda c: (not c.is_verbatim, c.path.name))


def batch_suffixed_name(candidate: Candidate) -> str:
    """`6000001-...-2634.jpg` in batch_7 -> `6000001-...-2634_batch07.jpg`.

    Applied to every image, not only the repeated ones, so a filename in the
    merged folder always says which run it came from and the report table can
    be read without the manifest. Sorting stays grouped by unit, so all the
    captures of one module sit next to each other; the batch number is
    zero-padded so they then sort in batch order rather than batch_10 before
    batch_1.
    """
    name = canonical_name(candidate.path)
    stem, _, ext = name.rpartition(".")
    return f"{stem}_batch{candidate.batch_index:02d}.{ext}"


def drop_intra_batch_clones(candidates: list[Candidate]) -> list[Candidate]:
    """Remove files that are byte-identical to another capture in the *same*
    batch — `foo.jpg` and its `foo copy.jpg` twin are one capture, and a
    batch-number suffix cannot tell them apart.

    Re-shoots of the same unit in *different* batches are always kept; that is
    the whole point of this mode.
    """
    kept: list[Candidate] = []
    seen: set[tuple[int, str]] = set()
    # Verbatim names first, so `foo.jpg` survives and `foo copy.jpg` is the one
    # dropped rather than the other way round.
    for c in sorted(candidates, key=lambda c: (c.batch_index,
                                               not c.is_verbatim,
                                               c.path.name)):
        fingerprint = (c.batch_index, md5(c.path))
        if fingerprint in seen:
            log.debug("Dropping intra-batch clone %s (identical to a file "
                      "already kept from %s)", c.path, c.batch_name)
            continue
        seen.add(fingerprint)
        kept.append(c)
    return kept


Selection = list[tuple[str, str, Candidate]]   # (output filename, unit, capture)


def select_latest_per_unit(by_dut: dict[str, list[Candidate]]) -> Selection:
    """One image per unit, from the highest-numbered batch it appears in.
    The output keeps the plain unit filename."""
    return [(unit, unit, pick_winner(cands))
            for unit, cands in sorted(by_dut.items())]


def select_all_captures(by_dut: dict[str, list[Candidate]]) -> Selection:
    """Every capture of every unit, each named `<unit>_batch<n>.jpg`."""
    selection: Selection = []
    used: dict[str, int] = {}
    for unit, cands in sorted(by_dut.items()):
        for c in drop_intra_batch_clones(cands):
            name = batch_suffixed_name(c)
            # Two *different* images sharing a unit and a batch (e.g. a
            # `copy` that was re-shot rather than duplicated) would collide
            # on the suffixed name; number them rather than overwrite.
            if name in used:
                used[name] += 1
                stem, _, ext = name.rpartition(".")
                name = f"{stem}_{used[name]}.{ext}"
                log.warning("Name collision for unit %s in %s — writing %s",
                            unit, c.batch_name, name)
            else:
                used[name] = 1
            selection.append((name, unit, c))
    return selection


def resolve_ref(src: Path, ref_from: str) -> Path:
    """The single reference image of the chosen batch."""
    ref_dir = src / ref_from / "ref"
    images = list_images(ref_dir)
    if not images:
        raise SystemExit(f"No reference image in {ref_dir}")
    if len(images) > 1:
        raise SystemExit(
            f"Expected exactly one reference image in {ref_dir}, found "
            f"{len(images)}: {[p.name for p in images]}"
        )
    return images[0]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def place(source: Path, dest: Path, symlink: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if symlink:
        dest.symlink_to(source.resolve())
    else:
        shutil.copy2(source, dest)


def prepare_out_dir(out: Path, force: bool) -> None:
    """`batch_all` must be rebuilt from scratch — a stale image left behind
    from a previous run would silently join the evaluation."""
    for sub in ("ref", "dut"):
        existing = out / sub
        if existing.is_dir() and any(existing.iterdir()):
            if not force:
                raise SystemExit(
                    f"{existing} already exists and is not empty. "
                    f"Re-run with --force to rebuild it from scratch."
                )
            shutil.rmtree(existing)
        existing.mkdir(parents=True, exist_ok=True)


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["dut", "unit", "source_batch", "source_path", "captures",
              "seen_in", "md5", "note", "unit_note",
              "forced_verdict", "forced_reason"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build-batch-all",
        description=(
            "Merge batch_<n>/dut folders into one batch_all/dut, keeping the "
            "capture from the highest-numbered batch for any DUT that appears "
            "in more than one batch."
        ),
    )
    p.add_argument("--src", type=Path, default=Path.home() / "Pictures",
                   help="Folder containing the batch_<n> directories "
                        "(default: ~/Pictures).")
    p.add_argument("--out", type=Path, default=Path("./batch_all"),
                   help="Destination batch folder (default: ./batch_all).")
    p.add_argument("--ref-from", default="batch_0", metavar="BATCH",
                   help="Batch whose ref/ image becomes the global reference "
                        "(default: batch_0).")
    p.add_argument("--keep-duplicates", action="store_true",
                   help="Keep every capture instead of one per unit, naming "
                        "each file <unit>_batch<n>.jpg so same-named images "
                        "from different batches coexist. Byte-identical "
                        "clones inside one batch are still dropped.")
    p.add_argument("--include-ref-images", action="store_true",
                   help="Also treat every batch's ref/ image as a DUT "
                        "capture. Use when --ref-from names a different "
                        "module than the other batches used as their "
                        "reference, so that module's captures are not lost.")
    p.add_argument("--symlink", action="store_true",
                   help="Symlink to the source images instead of copying "
                        "them. Saves space; breaks if the sources move.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would be built without writing anything.")
    p.add_argument("--force", action="store_true",
                   help="Wipe an existing out/ref and out/dut before writing.")
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

    batches = discover_batches(src)
    if not batches:
        raise SystemExit(f"No batch_<n> folders under {src}")
    log.info("Found %d batches: %s to %s",
             len(batches), batches[0][1].name, batches[-1][1].name)

    ref_src = resolve_ref(src, args.ref_from)
    by_dut = collect_dut(batches, include_ref=args.include_ref_images)
    if not by_dut:
        raise SystemExit("No DUT images found in any batch.")

    # The reference module is the yardstick, not a unit under test — measuring
    # it against another capture of itself reports capture repeatability, not
    # module quality, and would sit in the table as an ordinary pass/fail row.
    ref_key = canonical_name(ref_src)
    dropped = by_dut.pop(ref_key, None)
    if dropped is not None:
        log.info("Excluded the reference module %s from the DUT set "
                 "(%d capture(s) in %s)", ref_key, len(dropped),
                 ", ".join(sorted({c.batch_name for c in dropped})))

    total_captures = sum(len(c) for c in by_dut.values())
    repeats = {k: v for k, v in by_dut.items() if len(v) > 1}

    # (output filename, unit, chosen capture) — one entry per file to write.
    selection = (
        select_all_captures(by_dut) if args.keep_duplicates
        else select_latest_per_unit(by_dut)
    )

    if args.keep_duplicates:
        log.info("%d captures across all batches -> %d DUT images kept "
                 "(%d unit(s), %d of them shot in more than one batch)",
                 total_captures, len(selection), len(by_dut), len(repeats))
    else:
        log.info("%d captures across all batches -> %d unique DUT images "
                 "(%d appeared in more than one batch)",
                 total_captures, len(selection), len(repeats))
        for key in sorted(repeats):
            chosen = next(c for name, unit, c in selection if unit == key)
            log.debug("%s: %s -> keeping %s", key,
                      ", ".join(c.batch_name for c in by_dut[key]),
                      chosen.batch_name)

    rows = []
    for name, unit, chosen in selection:
        rows.append({
            "dut": name,
            "unit": unit,
            "source_batch": chosen.batch_name,
            "source_path": str(chosen.path),
            "captures": len(by_dut[unit]),
            "seen_in": " ".join(c.batch_name for c in by_dut[unit]),
            "md5": "" if args.dry_run else md5(chosen.path),
            "note": "",
            "unit_note": UNIT_NOTES.get(unit, ""),
            "forced_verdict": "fail" if unit in FORCED_FAILURES else "",
            "forced_reason": FORCED_FAILURES.get(unit, ""),
        })

    if args.dry_run:
        log.info("Dry run — nothing written. Reference would be %s (%s).",
                 ref_src.name, args.ref_from)
        for row in rows:
            print(f"{row['dut']}  <- {row['source_batch']}"
                  f"  ({row['captures']} capture(s): {row['seen_in']})")
        return 0

    prepare_out_dir(out, args.force)
    place(ref_src, out / "ref" / ref_src.name, args.symlink)
    for name, _unit, chosen in selection:
        place(chosen.path, out / "dut" / name, args.symlink)

    manifest = out / "manifest.csv"
    write_manifest(manifest, rows)

    verb = "Linked" if args.symlink else "Copied"
    log.info("%s 1 reference (%s from %s) and %d DUT images into %s",
             verb, ref_src.name, args.ref_from, len(selection), out)
    log.info("Provenance written to %s", manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
