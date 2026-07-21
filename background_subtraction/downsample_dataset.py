"""Downsample the CDNET-14 sequences under ``data/`` to a low frame rate.

The "Sign of Life" prototype targets low-framerate deployment, so this utility
rebuilds every sequence at a reduced rate (``--target-fps``, default 0.1 fps —
one frame every ten seconds) and writes the result to a parallel tree whose
name encodes the rate (``data_0_1fps/`` by default, ``data_1fps/`` for
``--target-fps 1``). Pass ``--output`` to override the folder name.

What it does, per sequence:

* **Source frame rate** is taken from the folder name when it encodes one
  (CDNET's lowFramerate convention, where the underscore is a decimal point:
  ``tramCrossroad_1fps`` -> 1 fps, ``turnpike_0_5fps`` -> 0.5 fps). Sequences
  whose name carries no rate are assumed to run at ``--default-fps`` (24 fps).
* **Subsampling.** Keeping every ``round(source_fps / target_fps)``-th frame
  yields the requested output rate. Both ``input/`` and ``groundtruth/`` are
  subsampled at the **same** frame numbers so each mask still matches its frame:

  | source fps | step to 0.1 fps | step to 1 fps |
  | ---------- | --------------- | ------------- |
  | 24 (default) | 240 (frames 1, 241, ...) | 24 (frames 1, 25, ...) |
  | 1  (tramCrossroad) | 10 | 1 (every frame) |
  | 0.5 (turnpike) | 5 | 1 (every frame, clamped) |

  A sequence already at or below the target rate cannot be upsampled; the step
  clamps to 1 and every frame is kept.
* **Original frame numbers are preserved** — a kept frame keeps its filename
  (``in000025.jpg`` stays ``in000025.jpg``, not renumbered to ``in000002``).
  This is deliberate: it keeps the copied ``temporalROI.txt``, whose range is
  expressed in original frame numbers, valid without recomputation.
* **ROI and temporal-ROI files are copied verbatim** (``ROI.bmp``, ``ROI.jpg``,
  ``temporalROI.txt``, and any other top-level file of the sequence).

The CDNET directory layout is assumed:

    data/<sequence>/input/in%06d.jpg
    data/<sequence>/groundtruth/gt%06d.png
    data/<sequence>/{ROI.bmp, ROI.jpg, temporalROI.txt}

Usage:
    python downsample_dataset.py                       # 0.1 fps -> data_0_1fps/
    python downsample_dataset.py --target-fps 1        # 1 fps   -> data_1fps/
    python downsample_dataset.py -i data -o data_custom --target-fps 0.5
    python downsample_dataset.py --default-fps 30 -v
"""

from __future__ import annotations

import argparse
import logging
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("downsample_dataset")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
"""Image extensions recognised inside ``input/`` directories."""

FPS_IN_NAME = re.compile(r"_(\d+)(?:_(\d+))?fps$")
"""Matches a trailing ``_<int>fps`` or ``_<int>_<int>fps`` rate in a name."""

FRAME_NUMBER = re.compile(r"(\d+)")
"""Extracts the frame number from a stem such as ``in000240``."""


@dataclass
class SequenceStats:
    """Outcome of downsampling a single sequence.

    Attributes:
        name: Sequence directory name.
        source_fps: Frame rate inferred for the source sequence.
        step: Frame stride applied to reach the target rate.
        total_frames: Number of input frames before subsampling.
        kept_frames: Number of input frames written.
        kept_masks: Number of groundtruth masks written.
    """

    name: str
    source_fps: float
    step: int
    total_frames: int
    kept_frames: int
    kept_masks: int


def parse_source_fps(name: str, default_fps: float) -> float:
    """Infer the source frame rate from a sequence name.

    CDNET's lowFramerate sequences encode the rate as a suffix in which the
    underscore stands in for a decimal point, e.g. ``turnpike_0_5fps`` is
    0.5 fps and ``tramCrossroad_1fps`` is 1 fps.

    Args:
        name: Sequence directory name.
        default_fps: Rate assumed when the name encodes none.

    Returns:
        The inferred frames-per-second value.
    """
    match = FPS_IN_NAME.search(name)
    if not match:
        return default_fps
    whole, fraction = match.groups()
    return float(f"{whole}.{fraction}") if fraction else float(whole)


def default_output_name(target_fps: float) -> str:
    """Build the CDNET-style output folder name for a target rate.

    The rate is rendered with the underscore standing in for the decimal point,
    matching CDNET's lowFramerate naming (0.1 -> ``data_0_1fps``, 1.0 ->
    ``data_1fps``, 0.5 -> ``data_0_5fps``).

    Args:
        target_fps: Desired output frame rate.

    Returns:
        The folder name, e.g. ``data_0_1fps``.
    """
    return f"data_{f'{target_fps:g}'.replace('.', '_')}fps"


def compute_step(source_fps: float, target_fps: float) -> int:
    """Compute the frame stride mapping ``source_fps`` onto ``target_fps``.

    Args:
        source_fps: Frame rate of the source sequence.
        target_fps: Desired output frame rate.

    Returns:
        The stride, i.e. keep every N-th frame; never below 1.

    Raises:
        ValueError: If either rate is not positive.
    """
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError("frame rates must be positive")
    return max(1, round(source_fps / target_fps))


def list_images(directory: Path) -> list[Path]:
    """Return the image files of a directory sorted by name.

    Args:
        directory: Directory to scan.

    Returns:
        The image paths in name order, which matches numeric order for the
        zero-padded CDNET naming scheme.
    """
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def frame_number(path: Path) -> int | None:
    """Extract the frame number embedded in a filename.

    Args:
        path: Image path such as ``in000240.jpg``.

    Returns:
        The parsed frame number, or ``None`` when the name carries no digits.
    """
    match = FRAME_NUMBER.search(path.stem)
    return int(match.group(1)) if match else None


def downsample_sequence(
    sequence_dir: Path,
    output_dir: Path,
    target_fps: float,
    default_fps: float,
) -> SequenceStats | None:
    """Downsample one sequence into ``output_dir``.

    Args:
        sequence_dir: Source sequence directory (holding ``input/``).
        output_dir: Destination sequence directory, created if missing.
        target_fps: Desired output frame rate.
        default_fps: Rate assumed when the name encodes none.

    Returns:
        The per-sequence statistics, or ``None`` when the sequence has no
        ``input/`` directory and is skipped.
    """
    input_dir = sequence_dir / "input"
    if not input_dir.is_dir():
        logger.warning("Skipping %s: no input/ directory", sequence_dir.name)
        return None

    source_fps = parse_source_fps(sequence_dir.name, default_fps)
    step = compute_step(source_fps, target_fps)
    if source_fps < target_fps:
        logger.warning(
            "%s is %.2f fps, below the %.2f fps target; cannot upsample, "
            "keeping every frame",
            sequence_dir.name,
            source_fps,
            target_fps,
        )
    frames = list_images(input_dir)
    selected = frames[::step]

    groundtruth_dir = sequence_dir / "groundtruth"
    out_input = output_dir / "input"
    out_groundtruth = output_dir / "groundtruth"
    out_input.mkdir(parents=True, exist_ok=True)
    if groundtruth_dir.is_dir():
        out_groundtruth.mkdir(parents=True, exist_ok=True)

    kept_masks = 0
    for frame_path in selected:
        shutil.copy2(frame_path, out_input / frame_path.name)

        number = frame_number(frame_path)
        if groundtruth_dir.is_dir() and number is not None:
            mask_path = groundtruth_dir / f"gt{number:06d}.png"
            if mask_path.exists():
                shutil.copy2(mask_path, out_groundtruth / mask_path.name)
                kept_masks += 1
            else:
                logger.debug("No groundtruth for frame %d in %s", number, sequence_dir.name)

    copy_auxiliary_files(sequence_dir, output_dir)

    stats = SequenceStats(
        name=sequence_dir.name,
        source_fps=source_fps,
        step=step,
        total_frames=len(frames),
        kept_frames=len(selected),
        kept_masks=kept_masks,
    )
    logger.info(
        "%-22s %6.2f fps  step %-4d  %5d -> %-4d frames  (%d masks)",
        stats.name,
        stats.source_fps,
        stats.step,
        stats.total_frames,
        stats.kept_frames,
        stats.kept_masks,
    )
    return stats


def copy_auxiliary_files(sequence_dir: Path, output_dir: Path) -> None:
    """Copy the sequence's top-level files (ROI images, temporal ROI) verbatim.

    Args:
        sequence_dir: Source sequence directory.
        output_dir: Destination sequence directory.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sequence_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, output_dir / path.name)


def convert_dataset(
    input_root: Path,
    output_root: Path,
    target_fps: float,
    default_fps: float,
) -> list[SequenceStats]:
    """Downsample every sequence found under ``input_root``.

    Args:
        input_root: Directory holding the source sequence sub-directories.
        output_root: Directory receiving the downsampled tree.
        target_fps: Desired output frame rate.
        default_fps: Rate assumed for sequences whose name encodes none.

    Returns:
        The per-sequence statistics for every sequence that was converted.

    Raises:
        FileNotFoundError: If ``input_root`` does not exist.
        NotADirectoryError: If ``input_root`` is not a directory.
    """
    if not input_root.exists():
        raise FileNotFoundError(f"input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"input root is not a directory: {input_root}")

    sequences = sorted(path for path in input_root.iterdir() if path.is_dir())
    logger.info(
        "Downsampling %d sequences from %s to %.3f fps -> %s",
        len(sequences),
        input_root,
        target_fps,
        output_root,
    )

    results: list[SequenceStats] = []
    for sequence_dir in sequences:
        stats = downsample_sequence(
            sequence_dir,
            output_root / sequence_dir.name,
            target_fps,
            default_fps,
        )
        if stats is not None:
            results.append(stats)

    kept = sum(stats.kept_frames for stats in results)
    logger.info("Done: %d sequences, %d frames written to %s", len(results), kept, output_root)
    return results


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command line interface.

    Args:
        argv: Argument list to parse; ``sys.argv[1:]`` when omitted.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Downsample the CDNET-14 sequences to a low frame rate, "
            "subsampling input and groundtruth while copying ROI and "
            "temporal-ROI files verbatim."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", type=Path, default=Path("data"), help="source dataset root"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="destination root; defaults to data_<rate>fps derived from --target-fps",
    )
    parser.add_argument(
        "--target-fps", type=float, default=0.1, help="output frame rate"
    )
    parser.add_argument(
        "--default-fps",
        type=float,
        default=24.0,
        help="assumed source rate when the sequence name encodes none",
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

    output = args.output or Path(default_output_name(args.target_fps))
    try:
        convert_dataset(args.input, output, args.target_fps, args.default_fps)
    except (FileNotFoundError, NotADirectoryError, ValueError, OSError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
