"""Run the 4-layer "Sign of Life" pipeline over a frame sequence.

Reads an ordered image sequence from disk, feeds it through
:class:`scr.pipeline.SignOfLifePipeline` and writes one binary PNG mask per
frame, following the CDNET-14 result convention (``bin%06d.png``).

Example:
    python main.py --input data/canoe --output results/canoe
    python main.py -i data/traffic/input -o results/traffic \\
        --fast-backend running_average --slow-backend subsense --feature lbp
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

import cv2

from scr import PipelineConfig, SignOfLifePipeline
from scr.backends import available_backends
from scr.config import (
    BGSConfig,
    BlobConfig,
    DenoiseMethod,
    DualRateConfig,
    DwellConfig,
    FeatureConfig,
    FeatureKind,
    PreprocessConfig,
    RegistrationMethod,
    ShadowConfig,
)
from scr.structures import PipelineResult

logger = logging.getLogger("background_subtraction")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
"""Image extensions recognised when scanning an input directory."""

MASK_SOURCES = ("filtered", "static", "fast", "slow")
"""Selectable pipeline outputs, see :func:`select_mask`."""

FRAME_NUMBER_PATTERN = re.compile(r"(\d+)")
"""Extracts the frame number from a file stem such as ``in000123``."""

PYBGS_BACKENDS = frozenset({"vibe", "subsense", "lobster", "pawcs"})
"""Backends that ignore the learning rate and cannot be paired with themselves."""


def discover_frames(input_path: Path) -> list[Path]:
    """Collect the frame files of a sequence in display order.

    A CDNET sequence directory holding an ``input/`` subdirectory is accepted
    directly, as is the ``input/`` directory itself.

    Args:
        input_path: Sequence directory or directory of images.

    Returns:
        The image paths sorted by name, which matches numeric order for the
        zero-padded CDNET naming scheme.

    Raises:
        FileNotFoundError: If the directory does not exist or holds no images.
        NotADirectoryError: If ``input_path`` is not a directory.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"input path does not exist: {input_path}")
    if not input_path.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {input_path}")

    nested = input_path / "input"
    if nested.is_dir():
        logger.info("Using nested sequence directory %s", nested)
        input_path = nested

    frames = sorted(
        path
        for path in input_path.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not frames:
        raise FileNotFoundError(f"no images found in {input_path}")
    return frames


def output_name(frame_path: Path, index: int) -> str:
    """Derive the CDNET result filename for a frame.

    Args:
        frame_path: Path of the source frame, e.g. ``in000123.jpg``.
        index: Zero-based position in the sequence, used when the source name
            carries no digits.

    Returns:
        The result filename, e.g. ``bin000123.png``.
    """
    match = FRAME_NUMBER_PATTERN.search(frame_path.stem)
    number = int(match.group(1)) if match else index + 1
    return f"bin{number:06d}.png"


def select_mask(result: PipelineResult, source: str):
    """Pick the mask to persist from a per-frame result.

    Args:
        result: Output of a single pipeline step.
        source: One of :data:`MASK_SOURCES`.

    Returns:
        The selected ``uint8`` mask with values ``0``/``255``.

    Raises:
        ValueError: If ``source`` is not a known mask name.
    """
    if source == "filtered":
        return result.filtered_mask
    if source == "static":
        return result.masks.static
    if source == "fast":
        return result.masks.fast
    if source == "slow":
        return result.masks.slow
    raise ValueError(f"unknown mask source {source!r}; expected one of {MASK_SOURCES}")


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Assemble a :class:`PipelineConfig` from parsed arguments.

    Args:
        args: Parsed command line arguments.

    Returns:
        The pipeline configuration.
    """
    return PipelineConfig(
        preprocess=PreprocessConfig(
            denoise=DenoiseMethod(args.denoise),
            kernel_size=args.denoise_kernel,
            registration=RegistrationMethod(args.registration),
        ),
        feature=FeatureConfig(kind=FeatureKind(args.feature)),
        shadow=ShadowConfig(enabled=not args.no_shadow_veto),
        dual_rate=DualRateConfig(
            fast=BGSConfig(
                backend=args.fast_backend,
                learning_rate=args.fast_alpha,
                threshold=args.threshold,
            ),
            slow=BGSConfig(
                backend=args.slow_backend,
                learning_rate=args.slow_alpha,
                threshold=args.threshold,
            ),
            warmup_frames=args.warmup_frames,
        ),
        blob=BlobConfig(min_area=args.min_area, max_area=args.max_area),
        dwell=DwellConfig(
            dwell_frames=args.dwell_frames,
            max_missing_frames=args.max_missing_frames,
        ),
    )


def warn_on_degenerate_pairing(fast_backend: str, slow_backend: str) -> None:
    """Warn when the dual-rate pair cannot produce a static cue.

    ``pybgs`` algorithms expose no learning rate and no constructor arguments,
    so two instances of the same one are identical and their masks cancel out
    in ``slow AND NOT fast``, leaving every output frame empty.

    Args:
        fast_backend: Name of the fast backend.
        slow_backend: Name of the slow backend.
    """
    if fast_backend.lower() != slow_backend.lower():
        return
    if fast_backend.lower() not in PYBGS_BACKENDS:
        return
    logger.warning(
        "Both models use the pybgs backend %r, which ignores the learning "
        "rate; the fast and slow masks will be identical and every output "
        "mask will be empty. Pair it with a rate-controlled fast backend "
        "such as running_average or mog2 instead.",
        fast_backend,
    )


def run(args: argparse.Namespace) -> int:
    """Process a sequence and write one binary mask per frame.

    Args:
        args: Parsed command line arguments.

    Returns:
        The number of masks written.
    """
    frames = discover_frames(args.input)
    if args.limit is not None:
        frames = frames[: args.limit]

    args.output.mkdir(parents=True, exist_ok=True)
    warn_on_degenerate_pairing(args.fast_backend, args.slow_backend)

    pipeline = SignOfLifePipeline.from_config(build_config(args))
    logger.info(
        "Processing %d frames from %s -> %s (mask=%s)",
        len(frames),
        args.input,
        args.output,
        args.mask,
    )

    written = 0
    events = 0
    for index, frame_path in enumerate(frames):
        image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Skipping unreadable frame %s", frame_path)
            continue

        result = pipeline.process(image)
        mask = select_mask(result, args.mask)

        destination = args.output / output_name(frame_path, index)
        if not cv2.imwrite(str(destination), mask):
            raise OSError(f"failed to write {destination}")
        written += 1

        for event in result.events:
            events += 1
            logger.info(
                "Sign of life: frame %d (%s), track %d, bbox %s",
                index,
                frame_path.name,
                event.track_id,
                event.bbox,
            )

        if args.progress and index and index % args.progress == 0:
            logger.info("  %d/%d frames", index, len(frames))

    logger.info("Wrote %d masks, %d sign-of-life events", written, events)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command line interface.

    Args:
        argv: Argument list to parse; ``sys.argv[1:]`` when omitted.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the 4-layer non-NN background subtraction pipeline over a "
            "frame sequence and write binary PNG masks."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    io_group = parser.add_argument_group("input/output")
    io_group.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="sequence directory; an 'input' subdirectory is used when present",
    )
    io_group.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="directory receiving the binary bin%%06d.png masks",
    )
    io_group.add_argument(
        "--mask",
        choices=MASK_SOURCES,
        default="filtered",
        help="which pipeline output to persist",
    )
    io_group.add_argument(
        "--limit", type=int, default=None, help="process at most this many frames"
    )
    io_group.add_argument(
        "--progress",
        type=int,
        default=100,
        help="log progress every N frames; 0 disables",
    )

    layer0 = parser.add_argument_group("layer 0 - preprocessing")
    layer0.add_argument(
        "--denoise",
        choices=[method.value for method in DenoiseMethod],
        default=DenoiseMethod.GAUSSIAN.value,
    )
    layer0.add_argument(
        "--denoise-kernel", type=int, default=5, help="odd kernel size >= 3"
    )
    layer0.add_argument(
        "--registration",
        choices=[method.value for method in RegistrationMethod],
        default=RegistrationMethod.NONE.value,
    )

    layer12 = parser.add_argument_group("layers 1/2 - model and invariance")
    layer12.add_argument(
        "--feature",
        choices=[kind.value for kind in FeatureKind],
        default=FeatureKind.GRAY.value,
        help="differencing metric",
    )
    layer12.add_argument(
        "--no-shadow-veto",
        action="store_true",
        help="disable the HSV brightness-ratio shadow filter",
    )
    layer12.add_argument(
        "--fast-backend",
        choices=available_backends(),
        default="running_average",
        help="backend of the fast model",
    )
    layer12.add_argument(
        "--slow-backend",
        choices=available_backends(),
        default="running_average",
        help="backend of the slow model",
    )
    layer12.add_argument(
        "--fast-alpha", type=float, default=0.1, help="fast learning rate"
    )
    layer12.add_argument(
        "--slow-alpha", type=float, default=0.001, help="slow learning rate"
    )
    layer12.add_argument(
        "--threshold", type=float, default=25.0, help="native backend difference threshold"
    )

    layer3 = parser.add_argument_group("layer 3 - dual rate and dwell time")
    layer3.add_argument("--warmup-frames", type=int, default=30)
    layer3.add_argument("--min-area", type=int, default=150)
    layer3.add_argument("--max-area", type=int, default=None)
    layer3.add_argument("--dwell-frames", type=int, default=60)
    layer3.add_argument("--max-missing-frames", type=int, default=15)

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
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
