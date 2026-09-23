"""Build the curated acceptance set: the mechanical batch merge, corrected by
the rework history that the filenames cannot carry.

`build_batch_all.py` assumes a repeated serial means the *same* module was
re-shot after rework, so the latest capture wins. That is true for most units,
but not all — sometimes the module itself was swapped, and the two captures
are then two different physical modules that happen to share a serial label.
Only the test log knows which is which, so the exceptions are declared here.

Everything below is asserted, not assumed: if a declared batch holds no
capture of its unit, or a unit contributes a different number of images than
declared, the build fails rather than quietly producing a wrong headcount.

Usage:
    python -m camera_qc.build_acceptance_set [--out ./batch_acceptance]
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

from .evaluate import PACKAGE_CONFIG
from .metrics import load_rgb, sharpness_metrics
from .verdict import load_config, within_tolerance
from .build_batch_all import (
    FORCED_FAILURES,
    UNIT_NOTES,
    Candidate,
    drop_intra_batch_clones,
    batch_suffixed_name,
    canonical_name,
    collect_dut,
    discover_batches,
    md5,
    pick_winner,
    place,
    prepare_out_dir,
    resolve_ref,
    unit_filename as unit,
)


log = logging.getLogger(__name__)


# --- The rework history -----------------------------------------------------

# Batch whose ref/ image is the yardstick. batch_0's reference is unit 105.
REFERENCE_BATCH = "batch_0"

# Units where a repeated serial is a *different physical module*, not another
# capture of the same one: the module was replaced between these runs, so each
# listed batch contributes its own DUT rather than superseding the earlier one.
# Captures in batches not listed here are earlier states of the *first* listed
# module and are superseded as usual.
MODULE_REPLACEMENTS: dict[str, tuple[str, ...]] = {
    # Camera swapped once. The batch_12 module failed and was replaced; both
    # it and its replacement are counted, so the failure is not lost.
    # b10 and b11 are re-shoots of the batch_1 module: 478 / 514 / 508 sit
    # inside the +/-15% capture repeatability measured on the reference
    # module itself, so they cannot be separated into distinct modules from
    # the images. If the test log says otherwise, split them here.
    unit("119"): ("batch_1", "batch_12"),   # see UNIT_NOTES
    unit("163"): ("batch_12", "batch_13"),
    unit("177"): ("batch_12", "batch_13"),
}

# Unit notes are shared with build_batch_all (see UNIT_NOTES there).

# How many images each named unit must contribute. Guards the headline count
# against a re-run over changed source folders.
EXPECTED_CAPTURES: dict[str, int] = {
    unit("119"): 2,
    unit("135"): 1,
    unit("163"): 2,
    unit("177"): 2,
}

# Total DUT images the curated set must contain: 76 units from the mechanical
# merge, plus the second module behind serials 119, 163 and 177.
EXPECTED_TOTAL_DUT = 79


# --- Selection --------------------------------------------------------------


MODULE = "module"                     # the capture representing a module
SUPERSEDED_FAILURE = "superseded-failure"   # an earlier capture that failed
SUPERSEDED_PASS = "superseded-pass"         # an earlier capture that passed
REFERENCE_SELF = "reference-self-check"     # the reference, judged against itself

Selection = list[tuple[str, str, Candidate, str, str]]  # +note


def add_superseded(
    selection: Selection,
    by_dut: dict[str, list[Candidate]],
    ref_image_path: Path,
    scene: str,
    config_path: Path | None,
    mode: str,
) -> Selection:
    """Append every superseded capture that fails the sharpness tolerance.

    The curated set answers "which modules are good". Adding the failures back
    answers "what went wrong on the way there" in the same report, without
    also dragging in superseded captures that merely passed — those are
    redundant re-shoots of a module already counted once.

    The tolerance comes from the same config the report uses, so a capture is
    called failing here exactly when the report would fail it.
    """
    config = load_config(config_path or PACKAGE_CONFIG)
    tol = float(config["scenes"][scene]["tolerances_pct"]["sharpness"])
    direction = config["metric_direction"]["sharpness"]
    ref_sharp = sharpness_metrics(load_rgb(ref_image_path))["sharpness"]
    log.info("Reference sharpness %.1f; superseded captures below %.1f "
             "(%s, tol %.0f%%) are kept as failures",
             ref_sharp, ref_sharp * (1 - tol / 100), scene, tol)

    kept_paths = {c.path for _n, _u, c, _r, _note in selection}
    added: Selection = []
    for unit_name, candidates in sorted(by_dut.items()):
        for c in candidates:
            if c.path in kept_paths:
                continue
            value = sharpness_metrics(load_rgb(c.path))["sharpness"]
            passing = within_tolerance(ref_sharp, value, direction, tol)
            if passing and mode != "all":
                log.debug("%s: %s superseded and passing (%.1f) — not added",
                          unit_name, c.batch_name, value)
                continue
            role = SUPERSEDED_PASS if passing else SUPERSEDED_FAILURE
            added.append((batch_suffixed_name(c), unit_name, c, role, ""))
            log.info("%s: adding superseded %s from %s (%.1f, %.0f%% of ref)",
                     unit_name, "pass" if passing else "FAILURE",
                     c.batch_name, value, value / ref_sharp * 100)
    return selection + added


def add_reference_self_check(
    selection: Selection, ref_src: Path, ref_batch: str
) -> Selection:
    """Evaluate the reference image as a DUT against itself.

    A control row, not a unit under test: it must read exactly 100% of
    reference, so it shows at a glance that the comparison pipeline is wired
    up correctly. Any other value means something is wrong with the tooling,
    not with a module.
    """
    index = int(ref_batch.split("_")[1])
    c = Candidate(index, ref_batch, ref_src)
    note = ("The reference image evaluated against itself — a pipeline "
            "control, not a module under test. It must read exactly 100% of "
            "reference; anything else indicates a tooling fault.")
    log.info("Adding the reference %s as a self-check DUT row", ref_src.name)
    return selection + [
        (batch_suffixed_name(c), canonical_name(ref_src), c, REFERENCE_SELF, note)
    ]


def select_acceptance(
    by_dut: dict[str, list[Candidate]],
    suffix_all: bool = False,
) -> Selection:
    """One entry per module: (output filename, unit, capture, role).

    Latest batch wins per unit — except the units in MODULE_REPLACEMENTS,
    which contribute one image per declared module, named with a `_batch<nn>`
    suffix so the modules are distinguishable. `suffix_all` applies that
    suffix to every image, needed once superseded captures join the set and
    plain names would collide.
    """
    selection: Selection = []
    for name, candidates in sorted(by_dut.items()):
        declared = MODULE_REPLACEMENTS.get(name)
        if declared is None:
            chosen = pick_winner(candidates)
            out = batch_suffixed_name(chosen) if suffix_all else name
            selection.append((out, name, chosen, MODULE, ""))
            continue

        by_batch: dict[str, list[Candidate]] = {}
        for c in candidates:
            by_batch.setdefault(c.batch_name, []).append(c)

        missing = [b for b in declared if b not in by_batch]
        if missing:
            raise SystemExit(
                f"{name} is declared as replaced in {', '.join(declared)}, but "
                f"{', '.join(missing)} holds no capture of it. Present in: "
                f"{', '.join(sorted(by_batch))}. Fix MODULE_REPLACEMENTS or "
                f"the source batches."
            )

        # Each declared batch marks where a NEW module starts. Every other
        # capture is an earlier state of the module whose declared batch is
        # the latest at or before it (a capture predating them all belongs to
        # the first module). Each module is then represented by its own
        # latest capture, exactly as latest-wins works for a single unit.
        def num(batch: str) -> int:
            return int(batch.split("_")[1])

        declared_nums = sorted(num(b) for b in declared)
        groups: dict[int, list[Candidate]] = {n: [] for n in declared_nums}
        for batch, caps in by_batch.items():
            earlier = [d for d in declared_nums if d <= num(batch)]
            groups[earlier[-1] if earlier else declared_nums[0]].extend(caps)

        for module_no, start_batch in enumerate(declared_nums, start=1):
            chosen = pick_winner(groups[start_batch])
            others = sorted({c.batch_name for c in groups[start_batch]}
                            - {chosen.batch_name}, key=num)
            superseded = f" (supersedes {', '.join(others)})" if others else ""
            note = (
                f"Module {module_no} of {len(declared_nums)} behind this "
                f"serial — the camera module was replaced, so these are "
                f"different physical modules sharing one serial label. This "
                f"one is measured from its last capture, in {chosen.batch_name}"
                + (f", superseding {', '.join(others)}." if others else ".")
            )
            selection.append(
                (batch_suffixed_name(chosen), name, chosen, MODULE, note))
            log.info("%s: module %d of %d kept from %s%s -> %s",
                     name, module_no, len(declared_nums), chosen.batch_name,
                     superseded, batch_suffixed_name(chosen))
    return selection


def verify(selection: Selection, expect_total: int) -> None:
    """Fail the build if the curated set does not match what was declared.

    Counts only the module entries: EXPECTED_CAPTURES declares how many
    physical modules a serial stands for, which does not change when the
    superseded failures are added alongside them.
    """
    counts: dict[str, int] = {}
    for _name, u, _c, role, _note in selection:
        if role == MODULE:
            counts[u] = counts.get(u, 0) + 1

    problems: list[str] = []
    for u, expected in sorted(EXPECTED_CAPTURES.items()):
        actual = counts.get(u, 0)
        if actual != expected:
            problems.append(
                f"{u}: expected {expected} capture(s) in the set, found {actual}"
            )
    if len(selection) != expect_total:
        problems.append(
            f"total DUT images: expected {expect_total}, built {len(selection)}"
        )
    if problems:
        raise SystemExit(
            "Curated set does not match the declared rework history:\n  "
            + "\n  ".join(problems)
        )

    for u, expected in sorted(EXPECTED_CAPTURES.items()):
        log.info("verified %s contributes %d capture(s)", u, expected)
    log.info("verified total: %d DUT images", len(selection))


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["dut", "unit", "role", "source_batch", "source_path",
              "captures", "seen_in", "md5", "note", "unit_note",
              "forced_verdict", "forced_reason"]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="build-acceptance-set",
        description=(
            "Build the curated acceptance set: one image per module, with the "
            "declared module replacements counted as separate modules."
        ),
    )
    p.add_argument("--src", type=Path, default=Path.home() / "Pictures",
                   help="Folder containing the batch_<n> directories "
                        "(default: ~/Pictures).")
    p.add_argument("--out", type=Path, default=Path("./batch_acceptance"),
                   help="Destination batch folder (default: "
                        "./batch_acceptance).")
    p.add_argument("--ref-from", default=REFERENCE_BATCH, metavar="BATCH",
                   help=f"Batch whose ref/ image is the reference "
                        f"(default: {REFERENCE_BATCH}).")
    p.add_argument("--superseded", choices=("drop", "failures", "all"),
                   default="drop",
                   help="What to do with captures superseded by a later one. "
                        "'drop' (default): one image per module. 'failures': "
                        "also keep the superseded captures that fail the "
                        "sharpness tolerance, so the rework history appears "
                        "alongside the accepted modules. 'all': keep every "
                        "capture. Anything but 'drop' implies batch-suffixed "
                        "filenames throughout.")
    p.add_argument("--include-failures", action="store_true",
                   help="Alias for --superseded failures.")
    p.add_argument("--include-ref-images", action="store_true",
                   help="Also treat every batch's ref/ image as a DUT "
                        "capture. Needed when --ref-from names a different "
                        "module than the other batches used as their "
                        "reference, so that module's captures are not lost.")
    p.add_argument("--ref-as-dut", action="store_true",
                   help="Also evaluate the reference image as a DUT, against "
                        "itself. A control row that must read 100%% of "
                        "reference; it verifies the comparison pipeline "
                        "rather than a module.")
    p.add_argument("--scene", default="chessboard",
                   choices=("chessboard", "ceiling"),
                   help="Scene preset whose sharpness tolerance decides which "
                        "superseded captures count as failures (default: "
                        "chessboard). Use the same preset you report with.")
    p.add_argument("--config", type=Path, default=None,
                   help="Override the bundled tolerance config.yaml.")
    p.add_argument("--expect-dut", type=int, default=None,
                   metavar="N",
                   help=f"Fail unless the set contains exactly N DUT images "
                        f"(default: {EXPECTED_TOTAL_DUT}, or no total check "
                        f"with --include-failures, whose count follows from "
                        f"the measurements).")
    p.add_argument("--symlink", action="store_true",
                   help="Symlink to the source images instead of copying.")
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

    # `foo.jpg` and its byte-identical `foo copy.jpg` twin are one capture.
    # Left in, they would both claim the same `_batch<nn>` output name once
    # superseded captures join the set, and one would silently overwrite the
    # other.
    by_dut = {u: drop_intra_batch_clones(c) for u, c in by_dut.items()}

    ref_key = canonical_name(ref_src)
    dropped = by_dut.pop(ref_key, None)
    if dropped is not None:
        log.info("Excluded the reference module %s from the DUT set", ref_key)

    superseded = "failures" if args.include_failures else args.superseded
    selection = select_acceptance(by_dut, suffix_all=superseded != "drop")
    n_modules = len(selection)
    verify(selection, args.expect_dut or EXPECTED_TOTAL_DUT)
    if superseded != "drop":
        selection = add_superseded(
            selection, by_dut, ref_src, args.scene, args.config, superseded)
    if args.ref_as_dut:
        selection = add_reference_self_check(selection, ref_src, args.ref_from)

    rows = []
    for name, u, chosen, role, entry_note in selection:
        parts = []
        if role == SUPERSEDED_FAILURE:
            kept = " / ".join(
                c.batch_name for _n, un, c, r, _x in selection
                if un == u and r == MODULE)
            parts.append(
                f"Superseded capture — an earlier state of a module that was "
                f"re-worked or replaced and is accepted from {kept}. Recorded "
                f"so the failure is not lost.")
        if entry_note:
            parts.append(entry_note)
        note = " ".join(parts)
        rows.append({
            "dut": name,
            "unit": u,
            "role": role,
            "source_batch": chosen.batch_name,
            "source_path": str(chosen.path),
            # The reference self-check row's unit is not in by_dut — it was
            # removed as the yardstick — so fall back to the row's own capture.
            "captures": len(by_dut.get(u, [chosen])),
            "seen_in": " ".join(c.batch_name for c in by_dut.get(u, [chosen])),
            "md5": "" if args.dry_run else md5(chosen.path),
            "note": note,
            "unit_note": UNIT_NOTES.get(u, ""),
            # A control row is never a unit, so it is never force-failed.
            "forced_verdict": ("fail" if u in FORCED_FAILURES
                               and role != REFERENCE_SELF else ""),
            "forced_reason": (FORCED_FAILURES.get(u, "")
                              if role != REFERENCE_SELF else ""),
        })

    extra = len(selection) - n_modules
    log.info("%d module(s) under test, compared against %s "
             "— %d module(s) in total including the reference",
             n_modules, ref_src.name, n_modules + 1)
    if extra:
        by_role: dict[str, int] = {}
        for _n, _u, _c, role, _x in selection:
            if role != MODULE:
                by_role[role] = by_role.get(role, 0) + 1
        detail = ", ".join(f"{v} {k}" for k, v in sorted(by_role.items()))
        log.info("plus %d extra image(s) (%s) — %d images in total",
                 extra, detail, len(selection))
    for u, note in sorted(UNIT_NOTES.items()):
        log.info("note — %s: %s", u, note)

    if args.dry_run:
        log.info("Dry run — nothing written.")
        for row in rows:
            print(f"{row['dut']}  <- {row['source_batch']}"
                  f"  ({row['captures']} capture(s): {row['seen_in']})")
        return 0

    prepare_out_dir(out, args.force)
    place(ref_src, out / "ref" / ref_src.name, args.symlink)
    for name, _u, chosen, _role, _note in selection:
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
