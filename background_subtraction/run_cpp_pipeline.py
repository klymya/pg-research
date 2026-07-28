"""Run the C++ Sign-of-Life pipeline on the QVGA busStation/canoe cases.

Builds ``sign_of_life_cpp`` if needed, then runs the compiled binary on the QVGA
dataset (``data_0_1fps_qvga``, already 320x240 so it matches the notebook's
``RESIZE = (320, 240)``) and writes, per case, the raw foreground mask, the mask
after morphology, and the ``sign_of_life.txt`` report under
``results_cpp/<case>/``.

Configuration parity with ``pipeline_experiments.ipynb``: the C++ tool's
compiled defaults already equal the notebook's parameters -- native
``running_gaussian`` (alpha 0.1), Gaussian denoise k=5, HSV shadow veto
(0.40-0.95, |dS|<=40, |dH|<=12), morphology open=3/close=5. The two CLI-tunable
values, ``min_area`` and the Sign-of-Life ``area_threshold``, are passed
explicitly here (150 and 500) to mirror the notebook exactly.

Usage:
    python run_cpp_pipeline.py
    python run_cpp_pipeline.py --dataset data_0_1fps_qvga --output results_cpp
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("run_cpp_pipeline")

HERE = Path(__file__).resolve().parent
CPP_PROJECT = HERE / "sign_of_life_cpp"

# Parameters mirroring pipeline_experiments.ipynb.
CASES = ("busStation", "canoe")
MIN_AREA = 150
SOL_AREA = 500


def ensure_binary(project_dir: Path) -> Path:
    """Return the built ``sign_of_life`` executable, building it if missing.

    Args:
        project_dir: The ``sign_of_life_cpp`` project directory.

    Returns:
        Path to the executable.

    Raises:
        FileNotFoundError: If the project directory does not exist.
        RuntimeError: If the CMake configure or build step fails.
    """
    if not project_dir.is_dir():
        raise FileNotFoundError(f"C++ project not found: {project_dir}")

    build_dir = project_dir / "build"
    binary = build_dir / "sign_of_life"
    if binary.is_file():
        logger.info("Using existing binary %s", binary)
        return binary

    logger.info("Building the C++ pipeline (%s)", project_dir)
    steps = [
        ["cmake", "-S", str(project_dir), "-B", str(build_dir),
         "-DCMAKE_BUILD_TYPE=Release"],
        ["cmake", "--build", str(build_dir), "-j"],
    ]
    for command in steps:
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"command failed: {' '.join(command)}\n{result.stderr}"
            )
    if not binary.is_file():
        raise RuntimeError(f"build finished but binary is missing: {binary}")
    return binary


def run_case(binary: Path, input_dir: Path, output_dir: Path) -> None:
    """Run the C++ pipeline over one case's input folder.

    Args:
        binary: The ``sign_of_life`` executable.
        input_dir: Folder of QVGA input frames for the case.
        output_dir: Destination folder for the masks and report.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist.
        RuntimeError: If the pipeline exits with a non-zero status.
    """
    if not input_dir.is_dir():
        raise FileNotFoundError(f"case input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(binary),
        str(input_dir),
        str(output_dir),
        "--min-area", str(MIN_AREA),
        "--sol-area", str(SOL_AREA),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"pipeline failed for {input_dir}\n{result.stderr}")
    logger.info("%s", result.stdout.strip())


def run(args: argparse.Namespace) -> int:
    """Build the binary and run every requested case."""
    binary = ensure_binary(CPP_PROJECT)
    for case in args.cases:
        input_dir = args.dataset / case / "input"
        output_dir = args.output / case
        logger.info("Case %s: %s -> %s", case, input_dir, output_dir)
        run_case(binary, input_dir, output_dir)
    logger.info("Done. Results in %s", args.output)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command line interface."""
    parser = argparse.ArgumentParser(
        description="Run the C++ Sign-of-Life pipeline on the QVGA cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=HERE / "data_0_1fps_qvga",
        help="QVGA dataset root (each case has an 'input' folder)",
    )
    parser.add_argument(
        "--output", type=Path, default=HERE / "results_cpp",
        help="destination root for per-case results",
    )
    parser.add_argument(
        "--cases", nargs="+", default=list(CASES), help="cases to process"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        return run(args)
    except (FileNotFoundError, RuntimeError) as error:
        logger.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
