"""Run every native background model (fast and slow) and tabulate F1 per case.

Sweeps all pure-NumPy models in :mod:`scr.models` over the ``busStation`` and
``canoe`` sequences at 0.1 fps and QVGA (320x240), saving both the fast and slow
foreground masks and their background-model images, then scores them and writes
one CSV per case.

For each model the dual-rate pipeline is run with that model in both roles; the
fast mask (`--mask fast`) captures the model at the fast learning rate and the
slow mask (`--mask slow`) at the slow rate. Because the fast and slow models are
independent in the pipeline, these per-model fast/slow results cover every
fast x slow combination at the mask level without redundant runs.

Output layout::

    <results-dir>/
      <model>__fast/<case>/{bin%06d.png, background/bg%06d.png}
      <model>__slow/<case>/{...}
      eval_summary.csv        # cases x (model__mask), from summarize_results.py
      busStation.csv          # rows: models, cols: [fast, slow]  (F1)
      canoe.csv

Evaluation reuses ``summarize_results.py`` with its ``--groundtruth-dataset``
option (every model variant is scored against ``data_0_1fps`` groundtruth), and
the resulting table is reshaped into one models x [fast, slow] CSV per case.

Usage:
    python run_model_experiments.py
    python run_model_experiments.py --results-dir results_models --dry-run
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger("run_model_experiments")

MODELS = (
    "running_average",
    "adaptive_median",
    "running_gaussian",
    "mixture_of_gaussians",
    "vibe",
    "frame_difference",
    "static_frame_difference",
    "eigenbackground",
)
"""Native models to sweep (see :func:`scr.backends.available_backends`)."""

CASES = ("busStation", "canoe")
"""Sequences to evaluate."""

MASKS = ("fast", "slow")
"""Dual-rate outputs saved and scored per model."""

DATASET = "data_0_1fps"
"""0.1 fps source dataset (also the fixed groundtruth for scoring)."""

RESIZE = (320, 240)
"""QVGA target resolution (width, height)."""

FAST_ALPHA = 0.1
"""Learning rate of the fast model."""

SLOW_ALPHA = 0.001
"""Learning rate of the slow model."""


def run_one(
    python: str,
    main_script: Path,
    model: str,
    mask: str,
    case: str,
    dataset: Path,
    results_dir: Path,
    dry_run: bool,
) -> bool:
    """Run ``main.py`` for one (model, mask, case) and save masks + backgrounds.

    Args:
        python: Python interpreter to invoke.
        main_script: Path to ``main.py``.
        model: Native backend name used for both the fast and slow model.
        mask: Which dual-rate output to persist (``fast`` or ``slow``).
        case: Sequence name.
        dataset: 0.1 fps dataset root.
        results_dir: Root of the results tree.
        dry_run: When ``True``, log the command without executing it.

    Returns:
        ``True`` on success (or dry run), ``False`` if the run failed.
    """
    output_dir = results_dir / f"{model}__{mask}" / case
    command = [
        python,
        str(main_script),
        "--input", str(dataset / case),
        "--output", str(output_dir),
        "--implementation", "native",
        "--fast-backend", model,
        "--slow-backend", model,
        "--mask", mask,
        "--resize", str(RESIZE[0]), str(RESIZE[1]),
        "--fast-alpha", str(FAST_ALPHA),
        "--slow-alpha", str(SLOW_ALPHA),
        "--dump-background",
        "--progress", "0",
    ]

    if dry_run:
        logger.info("DRY-RUN %s", " ".join(command))
        return True

    logger.info("Running %-24s mask=%-4s case=%s", model, mask, case)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        logger.error("main.py failed for %s/%s/%s", model, mask, case)
        return False
    return True


def summarize(
    python: str,
    summarize_script: Path,
    results_dir: Path,
    dataset: Path,
    dry_run: bool,
) -> Path:
    """Score the whole tree with ``summarize_results.py`` against a fixed GT.

    Args:
        python: Python interpreter to invoke.
        summarize_script: Path to ``summarize_results.py``.
        results_dir: Root of the results tree.
        dataset: Dataset whose groundtruth every model variant is scored against.
        dry_run: When ``True``, log the command without executing it.

    Returns:
        Path of the written ``eval_summary.csv``.
    """
    summary_csv = results_dir / "eval_summary.csv"
    command = [
        python,
        str(summarize_script),
        "--results-dir", str(results_dir),
        "--groundtruth-dataset", str(dataset),
        "--output", str(summary_csv),
    ]
    if dry_run:
        logger.info("DRY-RUN %s", " ".join(command))
        return summary_csv

    logger.info("Summarising -> %s", summary_csv)
    subprocess.run(command, check=True)
    return summary_csv


def write_per_case_tables(summary_csv: Path, results_dir: Path) -> list[Path]:
    """Reshape the summary into one models x [fast, slow] CSV per case.

    Args:
        summary_csv: The ``eval_summary.csv`` (rows: cases, cols: model__mask).
        results_dir: Directory receiving the per-case CSVs.

    Returns:
        The written per-case CSV paths.
    """
    summary = pd.read_csv(summary_csv).set_index("case")
    written: list[Path] = []
    for case in CASES:
        if case not in summary.index:
            logger.warning("Case %s missing from summary", case)
            continue
        row = summary.loc[case]
        table = {model: {mask: float("nan") for mask in MASKS} for model in MODELS}
        for variant, value in row.items():
            model, _, mask = str(variant).rpartition("__")
            if model in table and mask in MASKS:
                table[model][mask] = value
        frame = pd.DataFrame.from_dict(table, orient="index")[list(MASKS)]
        frame = frame.reindex(MODELS)
        frame.index.name = "model"

        destination = results_dir / f"{case}.csv"
        frame.to_csv(destination, float_format="%.4f")
        written.append(destination)
        logger.info("Wrote %s\n%s", destination, frame.to_string(float_format="%.4f"))
    return written


def run_pair(
    python: str,
    main_script: Path,
    fast_model: str,
    slow_model: str,
    case: str,
    dataset: Path,
    results_dir: Path,
    mask: str,
    dry_run: bool,
) -> bool:
    """Run ``main.py`` for one (fast_model, slow_model, case) pair.

    Args:
        python: Python interpreter to invoke.
        main_script: Path to ``main.py``.
        fast_model: Native backend used for the fast model.
        slow_model: Native backend used for the slow model.
        case: Sequence name.
        dataset: 0.1 fps dataset root.
        results_dir: Root of the results tree.
        mask: Which output to persist (``fast`` or ``slow``); also names the
            variant folder so both can be saved for the same pair.
        dry_run: When ``True``, log the command without executing it.

    Returns:
        ``True`` on success (or dry run), ``False`` if the run failed.
    """
    output_dir = results_dir / f"{fast_model}__{slow_model}__{mask}" / case
    command = [
        python,
        str(main_script),
        "--input", str(dataset / case),
        "--output", str(output_dir),
        "--implementation", "native",
        "--fast-backend", fast_model,
        "--slow-backend", slow_model,
        "--mask", mask,
        "--resize", str(RESIZE[0]), str(RESIZE[1]),
        "--fast-alpha", str(FAST_ALPHA),
        "--slow-alpha", str(SLOW_ALPHA),
        "--dump-background",
        "--progress", "0",
    ]

    if dry_run:
        logger.info("DRY-RUN %s", " ".join(command))
        return True

    logger.info("Running fast=%-20s slow=%-20s case=%s", fast_model, slow_model, case)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        logger.error("main.py failed for %s x %s / %s", fast_model, slow_model, case)
        return False
    return True


def write_pair_fast_slow_tables(summary_csv: Path, results_dir: Path) -> list[Path]:
    """Reshape the summary into one pairs x [fast_f1, slow_f1] CSV per case.

    Each row is a (fast_model, slow_model) pair with the F1 of its fast mask and
    of its slow mask.

    Args:
        summary_csv: The ``eval_summary.csv`` (rows: cases, cols: fast__slow__mask).
        results_dir: Directory receiving the per-case CSVs.

    Returns:
        The written per-case CSV paths.
    """
    summary = pd.read_csv(summary_csv).set_index("case")
    written: list[Path] = []
    for case in CASES:
        if case not in summary.index:
            logger.warning("Case %s missing from summary", case)
            continue
        row = summary.loc[case]
        records = [
            {
                "fast_model": fast,
                "slow_model": slow,
                "fast_f1": row.get(f"{fast}__{slow}__fast", float("nan")),
                "slow_f1": row.get(f"{fast}__{slow}__slow", float("nan")),
            }
            for fast in MODELS
            for slow in MODELS
        ]
        frame = pd.DataFrame(records)

        destination = results_dir / f"{case}_pairs.csv"
        frame.to_csv(destination, index=False, float_format="%.4f")
        written.append(destination)
        logger.info(
            "Wrote %s (%d pairs)\n%s",
            destination,
            len(frame),
            frame.head(10).to_string(index=False, float_format="%.4f"),
        )
    return written


def run(args: argparse.Namespace) -> int:
    """Run the sweep (roles or pairs), summarise and write per-case tables.

    Args:
        args: Parsed command line arguments.

    Returns:
        The number of failed model runs.
    """
    python = sys.executable
    here = Path(__file__).resolve().parent
    main_script = here / "main.py"
    summarize_script = here / "summarize_results.py"
    dataset = Path(DATASET)

    failures = 0
    if args.pairs:
        logger.info(
            "Sweeping %d x %d fast/slow pairs x %d cases x 2 masks at 0.1 fps, QVGA %dx%d",
            len(MODELS), len(MODELS), len(CASES), RESIZE[0], RESIZE[1],
        )
        for fast_model in MODELS:
            for slow_model in MODELS:
                for case in CASES:
                    for mask in MASKS:
                        if not run_pair(
                            python, main_script, fast_model, slow_model, case,
                            dataset, args.results_dir, mask, args.dry_run,
                        ):
                            failures += 1
    else:
        logger.info(
            "Sweeping %d models x %d masks x %d cases at 0.1 fps, QVGA %dx%d",
            len(MODELS), len(MASKS), len(CASES), RESIZE[0], RESIZE[1],
        )
        for model in MODELS:
            for mask in MASKS:
                for case in CASES:
                    if not run_one(
                        python, main_script, model, mask, case, dataset,
                        args.results_dir, args.dry_run,
                    ):
                        failures += 1

    summary_csv = summarize(
        python, summarize_script, args.results_dir, dataset, args.dry_run
    )
    if not args.dry_run:
        if args.pairs:
            write_pair_fast_slow_tables(summary_csv, args.results_dir)
        else:
            write_per_case_tables(summary_csv, args.results_dir)

    logger.info("Done: %d failed runs", failures)
    return failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command line interface.

    Args:
        argv: Argument list to parse; ``sys.argv[1:]`` when omitted.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run every native background model (fast and slow) on busStation "
            "and canoe at 0.1 fps / QVGA, save masks and backgrounds, and write "
            "one F1 CSV per case."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results_models"),
        help="root directory for the collected masks, backgrounds and CSVs",
    )
    parser.add_argument(
        "--pairs",
        action="store_true",
        help=(
            "run all fast x slow model pairs (64), saving the fast and slow "
            "masks + backgrounds of each, and write a pairs x [fast_f1, slow_f1] "
            "CSV per case, instead of each model in both roles"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands without executing them",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list to parse; ``sys.argv[1:]`` when omitted.

    Returns:
        A process exit code: ``0`` when every run succeeded, ``1`` otherwise.
    """
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return 1 if run(args) else 0


if __name__ == "__main__":
    sys.exit(main())
