"""Summarise pipeline results into a per-sequence x per-fps F-Measure table.

Consumes the ``results/`` tree produced by ``run_experiments.py`` — one
sub-folder per frame-rate variant, each holding one sub-folder of masks per
sequence:

    results/
      basic/   <sequence>/bin%06d.png     <- scored against data/<sequence>
      1fps/    <sequence>/bin%06d.png      <- scored against data_1fps/<sequence>
      0_5fps/  <sequence>/bin%06d.png      <- scored against data_0_5fps/<sequence>
      0_1fps/  <sequence>/bin%06d.png      <- scored against data_0_1fps/<sequence>

Each ``results/<variant>/<sequence>`` is scored with :func:`evaluate.evaluate`
against the groundtruth of the dataset it came from (``basic`` -> ``data``,
``<suffix>`` -> ``data_<suffix>``), and the F-Measures are written to a CSV
whose **rows are sequences (cases)** and **columns are frame-rate variants**:

    case,basic,1fps,0_5fps,0_1fps
    boats,0.421,0.4103,0.3980,0.3555
    ...
    average,0.51,0.49,0.47,0.42

Cells with no result or no groundtruth are left empty; a trailing ``average``
row reports the mean over the populated cells of each column.

Usage:
    python summarize_results.py                      # results/ -> results/eval_summary.csv
    python summarize_results.py -r results -o out.csv
    python summarize_results.py --metric recall      # tabulate a different metric
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from pathlib import Path

from evaluate import evaluate

logger = logging.getLogger("summarize_results")

METRICS = ("f_measure", "precision", "recall", "iou")
"""Selectable metrics, matching :class:`evaluate.EvaluationResult` fields."""

_LABEL_FPS = re.compile(r"^(\d+)(?:_(\d+))?fps$")
"""Parses a variant label such as ``0_5fps`` into its numeric frame rate."""


def dataset_folder_for(label: str) -> str:
    """Map a results variant label to its source dataset folder name.

    Args:
        label: Variant sub-folder name, e.g. ``basic`` or ``0_5fps``.

    Returns:
        ``"data"`` for ``basic``, otherwise ``data_<label>``.
    """
    return "data" if label == "basic" else f"data_{label}"


def label_fps(label: str) -> float:
    """Return the frame rate a variant label encodes, for column ordering.

    Args:
        label: Variant sub-folder name.

    Returns:
        The rate in fps; ``inf`` for ``basic`` (native rate) so it sorts first.
    """
    if label == "basic":
        return float("inf")
    match = _LABEL_FPS.match(label)
    if not match:
        return 0.0
    whole, fraction = match.groups()
    return float(f"{whole}.{fraction}") if fraction else float(whole)


def discover_variants(results_dir: Path) -> list[str]:
    """List the frame-rate variant folders under ``results_dir``.

    Args:
        results_dir: Root of the results tree.

    Returns:
        The variant labels, ordered by decreasing frame rate.
    """
    labels = [path.name for path in results_dir.iterdir() if path.is_dir()]
    return sorted(labels, key=label_fps, reverse=True)


def discover_cases(results_dir: Path, variants: list[str]) -> list[str]:
    """List every sequence name appearing under any variant.

    Args:
        results_dir: Root of the results tree.
        variants: Variant labels to scan.

    Returns:
        The sequence names, sorted alphabetically.
    """
    cases: set[str] = set()
    for variant in variants:
        variant_dir = results_dir / variant
        cases.update(path.name for path in variant_dir.iterdir() if path.is_dir())
    return sorted(cases)


def score_cell(
    results_dir: Path,
    data_root: Path,
    variant: str,
    case: str,
    metric: str,
    fg_threshold: int,
    groundtruth_dataset: Path | None = None,
) -> float | None:
    """Score a single (case, variant) cell.

    Args:
        results_dir: Root of the results tree.
        data_root: Directory holding the source ``data*`` datasets.
        variant: Variant label (column).
        case: Sequence name (row).
        metric: Attribute of the evaluation result to return.
        fg_threshold: Foreground threshold forwarded to the evaluator.
        groundtruth_dataset: When set, every variant is scored against this
            dataset's groundtruth (``<dataset>/<case>/groundtruth``) instead of
            deriving the dataset from the variant name. Use this when the
            variant dimension is something other than a frame-rate (e.g. a model
            name) but all runs share one groundtruth.

    Returns:
        The metric value, or ``None`` when the masks or groundtruth are missing.
    """
    result_dir = results_dir / variant / case
    if groundtruth_dataset is not None:
        groundtruth_dir = groundtruth_dataset / case / "groundtruth"
    else:
        groundtruth_dir = data_root / dataset_folder_for(variant) / case / "groundtruth"
    if not result_dir.is_dir() or not groundtruth_dir.is_dir():
        return None

    try:
        result = evaluate(groundtruth_dir, result_dir, fg_threshold)
    except (FileNotFoundError, ValueError) as error:
        logger.warning("Skipping %s/%s: %s", variant, case, error)
        return None
    return float(getattr(result, metric))


def build_table(
    results_dir: Path,
    data_root: Path,
    variants: list[str],
    cases: list[str],
    metric: str,
    fg_threshold: int,
    groundtruth_dataset: Path | None = None,
) -> dict[tuple[str, str], float]:
    """Score every populated (case, variant) cell.

    Args:
        results_dir: Root of the results tree.
        data_root: Directory holding the source ``data*`` datasets.
        variants: Variant labels (columns).
        cases: Sequence names (rows).
        metric: Attribute of the evaluation result to tabulate.
        fg_threshold: Foreground threshold forwarded to the evaluator.
        groundtruth_dataset: Optional fixed groundtruth dataset, see
            :func:`score_cell`.

    Returns:
        A mapping of ``(case, variant)`` to metric value for populated cells.
    """
    table: dict[tuple[str, str], float] = {}
    for case in cases:
        for variant in variants:
            value = score_cell(
                results_dir,
                data_root,
                variant,
                case,
                metric,
                fg_threshold,
                groundtruth_dataset,
            )
            if value is not None:
                table[(case, variant)] = value
                logger.info("%-12s %-8s %s=%.4f", case, variant, metric, value)
    return table


def write_csv(
    output_path: Path,
    variants: list[str],
    cases: list[str],
    table: dict[tuple[str, str], float],
) -> None:
    """Write the score table to a CSV with an appended average row.

    Args:
        output_path: Destination CSV path.
        variants: Variant labels, used as column order.
        cases: Sequence names, used as row order.
        table: Populated ``(case, variant)`` scores.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", *variants])

        for case in cases:
            writer.writerow(
                [case, *(_format(table.get((case, variant))) for variant in variants)]
            )

        writer.writerow(
            ["average", *(_format(_column_mean(table, variant, cases)) for variant in variants)]
        )


def _column_mean(
    table: dict[tuple[str, str], float], variant: str, cases: list[str]
) -> float | None:
    """Return the mean of a column's populated cells, or ``None`` if empty."""
    values = [table[(case, variant)] for case in cases if (case, variant) in table]
    return sum(values) / len(values) if values else None


def _format(value: float | None) -> str:
    """Format a metric value for CSV output; empty string for ``None``."""
    return "" if value is None else f"{value:.4f}"


def run(args: argparse.Namespace) -> int:
    """Build the summary table and write it to CSV.

    Args:
        args: Parsed command line arguments.

    Returns:
        The number of populated cells written.

    Raises:
        FileNotFoundError: If the results directory does not exist.
        NotADirectoryError: If the results path is not a directory.
    """
    if not args.results_dir.exists():
        raise FileNotFoundError(f"results directory does not exist: {args.results_dir}")
    if not args.results_dir.is_dir():
        raise NotADirectoryError(f"results path is not a directory: {args.results_dir}")

    variants = discover_variants(args.results_dir)
    cases = discover_cases(args.results_dir, variants)
    logger.info(
        "Summarising %d cases x %d variants (%s)",
        len(cases),
        len(variants),
        ", ".join(variants),
    )

    table = build_table(
        args.results_dir,
        args.data_root,
        variants,
        cases,
        args.metric,
        args.fg_threshold,
        args.groundtruth_dataset,
    )

    output_path = args.output or args.results_dir / "eval_summary.csv"
    write_csv(output_path, variants, cases, table)
    logger.info("Wrote %d cells to %s", len(table), output_path)
    return len(table)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command line interface.

    Args:
        argv: Argument list to parse; ``sys.argv[1:]`` when omitted.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Score every results/<variant>/<sequence> against its groundtruth "
            "and write an F-Measure CSV (rows: sequences, columns: frame rates)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-r",
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="root of the results tree to evaluate",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="destination CSV; defaults to <results-dir>/eval_summary.csv",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="directory holding the source data* datasets (for groundtruth)",
    )
    parser.add_argument(
        "--groundtruth-dataset",
        type=Path,
        default=None,
        help=(
            "score every variant against this dataset's groundtruth "
            "(<dataset>/<case>/groundtruth) instead of deriving it from the "
            "variant name; use when variants are model names, not frame rates"
        ),
    )
    parser.add_argument(
        "--metric",
        choices=METRICS,
        default="f_measure",
        help="metric to tabulate",
    )
    parser.add_argument(
        "--fg-threshold",
        type=int,
        default=127,
        help="prediction values above this are treated as foreground",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list to parse; ``sys.argv[1:]`` when omitted.

    Returns:
        A process exit code: ``0`` on success, ``1`` on a handled error.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        run(args)
    except (FileNotFoundError, NotADirectoryError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
