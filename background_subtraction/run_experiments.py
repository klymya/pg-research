"""Batch-run the pipeline over every ``data*`` dataset via ``main.py``.

Discovers the dataset roots produced by ``downsample_dataset.py`` (the original
``data/`` plus every ``data_<rate>fps/`` variant), runs ``main.py`` on each
sequence with ``subprocess``, and collects the binary masks under a single
``results/`` tree with one subfolder per frame-rate variant:

    results/
      basic/     <- data/           (native ~24 fps)
      1fps/      <- data_1fps/
      0_5fps/    <- data_0_5fps/    (0.5 fps, i.e. "1/5 fps" in shorthand)
      0_1fps/    <- data_0_1fps/    (0.1 fps, i.e. "1/10 fps")
        <sequence>/bin%06d.png

The ``data/`` root maps to ``basic``; every ``data_<suffix>`` root maps to
``<suffix>``, so the result folder names stay filesystem-safe and mirror the
datasets on disk.

Any argument this script does not recognise is forwarded verbatim to
``main.py``, so pipeline parameters can be set for the whole batch:

    # defaults
    python run_experiments.py

    # forward pipeline options to every run
    python run_experiments.py --mask slow --fast-alpha 0.02 --slow-alpha 0.002

    # preview the commands without executing them
    python run_experiments.py --dry-run
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("run_experiments")


def results_label(dataset_name: str) -> str:
    """Map a dataset folder name to its results subfolder name.

    Args:
        dataset_name: Dataset directory name, e.g. ``data`` or ``data_0_5fps``.

    Returns:
        ``"basic"`` for the native ``data`` root, otherwise the suffix after
        ``data_`` (e.g. ``data_0_5fps`` -> ``0_5fps``).
    """
    if dataset_name == "data":
        return "basic"
    return dataset_name.removeprefix("data_")


def discover_datasets(data_root: Path) -> list[Path]:
    """Find the dataset roots to process under ``data_root``.

    Args:
        data_root: Directory containing ``data`` and ``data_<rate>fps`` folders.

    Returns:
        The matching dataset directories, sorted by name.
    """
    return sorted(
        path
        for path in data_root.iterdir()
        if path.is_dir()
        and (path.name == "data" or (path.name.startswith("data_") and path.name.endswith("fps")))
    )


def discover_sequences(dataset_dir: Path) -> list[Path]:
    """Find the sequence directories inside a dataset root.

    A sequence is any sub-directory that holds an ``input/`` folder.

    Args:
        dataset_dir: A dataset root such as ``data_1fps``.

    Returns:
        The sequence directories, sorted by name.
    """
    return sorted(
        path for path in dataset_dir.iterdir() if (path / "input").is_dir()
    )


def run_sequence(
    python: str,
    main_script: Path,
    sequence_dir: Path,
    output_dir: Path,
    main_args: list[str],
    dry_run: bool,
) -> bool:
    """Run ``main.py`` on a single sequence.

    Args:
        python: Python interpreter to invoke.
        main_script: Path to ``main.py``.
        sequence_dir: Input sequence directory.
        output_dir: Directory receiving the sequence's masks.
        main_args: Extra arguments forwarded to ``main.py``.
        dry_run: When ``True``, log the command without executing it.

    Returns:
        ``True`` if the run succeeded (or was skipped for a dry run), else
        ``False``.
    """
    command = [
        python,
        str(main_script),
        "--input",
        str(sequence_dir),
        "--output",
        str(output_dir),
        *main_args,
    ]

    if dry_run:
        logger.info("DRY-RUN %s", " ".join(command))
        return True

    logger.info("Running %s -> %s", sequence_dir, output_dir)
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        logger.error(
            "main.py failed for %s (exit code %d)", sequence_dir, completed.returncode
        )
        return False
    return True


def run(args: argparse.Namespace, main_args: list[str]) -> int:
    """Run the whole batch and return the number of failed sequences.

    Args:
        args: Parsed runner arguments.
        main_args: Extra arguments forwarded to ``main.py``.

    Returns:
        The count of sequences whose ``main.py`` run failed.
    """
    datasets = discover_datasets(args.data_root)
    if not datasets:
        logger.warning("No data* datasets found under %s", args.data_root)
        return 0

    logger.info(
        "Found %d datasets: %s",
        len(datasets),
        ", ".join(dataset.name for dataset in datasets),
    )

    total = 0
    failures = 0
    started = time.perf_counter()
    for dataset in datasets:
        label = results_label(dataset.name)
        sequences = discover_sequences(dataset)
        logger.info("=== %s -> results/%s (%d sequences) ===", dataset.name, label, len(sequences))

        for sequence in sequences:
            output_dir = args.results_dir / label / sequence.name
            total += 1
            if not run_sequence(
                sys.executable, args.main_script, sequence, output_dir, main_args, args.dry_run
            ):
                failures += 1

    elapsed = time.perf_counter() - started
    logger.info(
        "Done: %d sequences, %d failed, %.1fs elapsed", total, failures, elapsed
    )
    return failures


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    """Define and parse the command line interface.

    Unrecognised arguments are returned separately for forwarding to ``main.py``.

    Args:
        argv: Argument list to parse; ``sys.argv[1:]`` when omitted.

    Returns:
        A ``(namespace, forwarded_args)`` tuple.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the background-subtraction pipeline over every data* dataset "
            "and collect the masks under results/<variant>/<sequence>/. "
            "Unknown arguments are forwarded to main.py."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    here = Path(__file__).resolve().parent
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("."),
        help="directory containing the data* dataset folders",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="root directory for the collected masks",
    )
    parser.add_argument(
        "--main-script",
        type=Path,
        default=here / "main.py",
        help="path to the pipeline runner main.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the commands without executing them",
    )
    return parser.parse_known_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list to parse; ``sys.argv[1:]`` when omitted.

    Returns:
        A process exit code: ``0`` when every sequence succeeded, ``1``
        otherwise.
    """
    args, main_args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    failures = run(args, main_args)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
