"""Resize every image in a folder to 320x240 with OpenCV.

Reads all images from an input folder, resizes each to ``320 x 240`` (QVGA,
width x height) using ``cv2.resize`` with area interpolation (the project's
convention for downscaling), and writes them to an output folder under the same
file names.

Usage:
    python resize_images.py --input path/to/src --output path/to/dst
    python resize_images.py -i src -o dst --size 640 480
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import cv2

logger = logging.getLogger("resize_images")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
"""Image extensions recognised when scanning the input folder."""


def resize_folder(
    input_dir: Path, output_dir: Path, size: tuple[int, int]
) -> int:
    """Resize every image in ``input_dir`` and write it to ``output_dir``.

    Args:
        input_dir: Folder to read images from.
        output_dir: Folder to write the resized images to; created if missing.
        size: Target ``(width, height)`` in pixels.

    Returns:
        The number of images written.

    Raises:
        FileNotFoundError: If ``input_dir`` does not exist or holds no images.
        NotADirectoryError: If ``input_dir`` is not a directory.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"input folder does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"input path is not a directory: {input_dir}")

    images = sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"no images found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            logger.warning("Skipping unreadable image %s", path.name)
            continue

        resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        destination = output_dir / path.name
        if not cv2.imwrite(str(destination), resized):
            raise OSError(f"failed to write {destination}")
        written += 1

    logger.info("Resized %d/%d images to %dx%d -> %s", written, len(images),
                size[0], size[1], output_dir)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command line interface."""
    parser = argparse.ArgumentParser(
        description="Resize all images in a folder to 320x240 with OpenCV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i", "--input", type=Path, required=True, help="input folder of images"
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True, help="output folder for resized images"
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("WIDTH", "HEIGHT"),
        default=(320, 240),
        help="target width and height",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        resize_folder(args.input, args.output, (args.size[0], args.size[1]))
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
