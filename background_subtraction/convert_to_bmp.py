"""Mirror a dataset tree and re-encode every image as BMP.

Walks ``--input`` recursively, writes each image found (``.jpg``, ``.jpeg``,
``.png``, ``.bmp``) to the parallel path under ``--output`` with a ``.bmp``
extension, and copies every non-image file verbatim so auxiliary files such as
``temporalROI.txt`` stay in place.

The CDNET directory layout is assumed but not required:

    data_0_1fps_qvga/<sequence>/input/in%06d.jpg   -> ...input/in%06d.bmp
    data_0_1fps_qvga/<sequence>/groundtruth/gt%06d.png -> ...groundtruth/gt%06d.bmp
    data_0_1fps_qvga/<sequence>/{ROI.bmp, ROI.jpg, temporalROI.txt}

Note: ``ROI.jpg`` is re-encoded as ``ROI.bmp`` and overwrites the sibling
``ROI.bmp`` written earlier in the same run; both encode the same mask, so the
result is idempotent regardless of iteration order.

Usage:
    python convert_to_bmp.py                              # data_0_1fps_qvga -> data_0_1fps_qvga_bmp
    python convert_to_bmp.py -i src_tree -o dst_tree
    python convert_to_bmp.py -i src -o dst -v
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import cv2

logger = logging.getLogger("convert_to_bmp")

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")
"""Image extensions recognised when scanning the input tree."""


def convert_tree(input_root: Path, output_root: Path) -> tuple[int, int]:
    """Mirror ``input_root`` under ``output_root``, converting images to BMP.

    Args:
        input_root: Directory tree to read.
        output_root: Directory tree to write; created if missing.

    Returns:
        A ``(converted, copied)`` tuple: the number of images re-encoded and
        the number of non-image files copied verbatim.

    Raises:
        FileNotFoundError: If ``input_root`` does not exist.
        NotADirectoryError: If ``input_root`` is not a directory.
        OSError: If a BMP write fails.
    """
    if not input_root.exists():
        raise FileNotFoundError(f"input root does not exist: {input_root}")
    if not input_root.is_dir():
        raise NotADirectoryError(f"input root is not a directory: {input_root}")

    converted = 0
    copied = 0
    for path in sorted(input_root.rglob("*")):
        if not path.is_file():
            continue
        destination = output_root / path.relative_to(input_root)
        destination.parent.mkdir(parents=True, exist_ok=True)

        if path.suffix.lower() in IMAGE_SUFFIXES:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                logger.warning("Skipping unreadable image %s", path)
                continue
            bmp_destination = destination.with_suffix(".bmp")
            if not cv2.imwrite(str(bmp_destination), image):
                raise OSError(f"failed to write {bmp_destination}")
            converted += 1
        else:
            shutil.copy2(path, destination)
            copied += 1

    logger.info(
        "Converted %d images and copied %d files from %s -> %s",
        converted, copied, input_root, output_root,
    )
    return converted, copied


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command line interface.

    Args:
        argv: Argument list to parse; ``sys.argv[1:]`` when omitted.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Mirror a dataset tree, re-encoding every image as BMP and "
            "copying non-image files verbatim."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("data_0_1fps_qvga"),
        help="source dataset root",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data_0_1fps_qvga_bmp"),
        help="destination dataset root",
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
        convert_tree(args.input, args.output)
    except (FileNotFoundError, NotADirectoryError, OSError) as error:
        logger.error("%s", error)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
