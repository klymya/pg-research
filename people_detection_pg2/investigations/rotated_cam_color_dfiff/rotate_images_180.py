"""Rotate every image in a folder 180 degrees, overwriting the originals."""

import argparse
from pathlib import Path

from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def rotate_folder(folder: Path) -> None:
    images = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    if not images:
        print(f"No images found in {folder}")
        return

    for path in images:
        with Image.open(path) as img:
            # Apply EXIF orientation first so the saved pixels match what the viewer shows,
            # then rotate 180. Save back in the original format to preserve the extension.
            oriented = ImageOps.exif_transpose(img)
            rotated = oriented.rotate(180)
            rotated.save(path, format=img.format)
        print(f"rotated {path.name}")

    print(f"Done. {len(images)} image(s) rotated in {folder}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "folder",
        nargs="?",
        default=str(Path(__file__).parent / "images" / "new"),
        help="Folder of images to rotate in place (default: ./images/new).",
    )
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    rotate_folder(folder)


if __name__ == "__main__":
    main()
