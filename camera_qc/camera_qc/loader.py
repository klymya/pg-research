"""Load the single reference image and the N DUT images."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from .metrics import load_rgb


log = logging.getLogger(__name__)


IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


@dataclass
class LoadedInputs:
    ref_path: Path
    ref_image: np.ndarray
    dut_paths: list[Path]
    dut_images: list[np.ndarray]


def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def load_inputs(ref_dir: Path | str, dut_dir: Path | str) -> LoadedInputs:
    """Read the ref/ and dut/ folders and return loaded RGB arrays.

    - ref/ must contain exactly one image.
    - dut/ must contain at least one image.
    - All DUT images must match the reference shape.
    """
    ref_dir = Path(ref_dir)
    dut_dir = Path(dut_dir)

    ref_files = _list_images(ref_dir)
    if len(ref_files) == 0:
        raise ValueError(
            f"No reference image found in {ref_dir}. "
            f"Place exactly one .jpg/.png in this folder."
        )
    if len(ref_files) > 1:
        raise ValueError(
            f"Expected exactly one reference image in {ref_dir}, found "
            f"{len(ref_files)}: {[p.name for p in ref_files]}"
        )

    dut_files = _list_images(dut_dir)
    if not dut_files:
        raise ValueError(
            f"No DUT images found in {dut_dir}. "
            f"Add one or more .jpg/.png captures."
        )

    ref_path = ref_files[0]
    log.info("Loading reference: %s", ref_path)
    ref_image = load_rgb(ref_path)
    log.info("Reference shape: %s; found %d DUT image(s)",
             ref_image.shape, len(dut_files))

    dut_images: list[np.ndarray] = []
    for p in tqdm(dut_files, desc="Loading DUT images", unit="img"):
        img = load_rgb(p)
        if img.shape != ref_image.shape:
            raise ValueError(
                f"DUT image shape mismatch: {p.name} is {img.shape}, "
                f"reference {ref_path.name} is {ref_image.shape}. "
                f"Re-capture so all images have the same resolution."
            )
        dut_images.append(img)

    return LoadedInputs(
        ref_path=ref_path,
        ref_image=ref_image,
        dut_paths=dut_files,
        dut_images=dut_images,
    )
