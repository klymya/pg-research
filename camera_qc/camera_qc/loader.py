"""Load the single reference image and the N DUT images."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from .metrics import load_rgb
from .preprocess import MODE as PREPROCESS_MODE, preprocess_rgb


log = logging.getLogger(__name__)


IMAGE_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})


@dataclass
class LoadedInputs:
    ref_path: Path
    ref_image: np.ndarray
    dut_paths: list[Path]
    dut_images: list[np.ndarray]
    preprocess_ref: bool = False
    preprocess_dut: bool = False

    @property
    def preprocess_asymmetric(self) -> bool:
        """True when only one side went through the capture pipeline — the
        metric deltas then mix processing effects with sensor differences.
        """
        return self.preprocess_ref != self.preprocess_dut


def _list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _dump_preprocessed(image_rgb: np.ndarray, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / name
    if not cv2.imwrite(str(out_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
        raise ValueError(f"Could not write preprocessed image: {out_path}")
    log.debug("Wrote preprocessed image %s", out_path)


def load_inputs(
    ref_dir: Path | str,
    dut_dir: Path | str,
    preprocess_ref: bool = False,
    preprocess_dut: bool = False,
    save_preprocessed: Path | str | None = None,
) -> LoadedInputs:
    """Read the ref/ and dut/ folders and return loaded RGB arrays.

    - ref/ must contain exactly one image.
    - dut/ must contain at least one image.
    - All DUT images must match the reference shape.

    `preprocess_ref` / `preprocess_dut` apply the ALIF white balance -> CLAHE
    capture pipeline (see `preprocess.py`) in memory, after loading; the source
    folders are never modified. `save_preprocessed` optionally dumps the
    processed images under `<dir>/ref/` and `<dir>/dut/` for inspection.
    """
    ref_dir = Path(ref_dir)
    dut_dir = Path(dut_dir)
    dump_dir = Path(save_preprocessed) if save_preprocessed is not None else None

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
    if preprocess_ref:
        log.info("Preprocessing reference (%s)", PREPROCESS_MODE)
        ref_image = preprocess_rgb(ref_image)
        if dump_dir is not None:
            _dump_preprocessed(ref_image, dump_dir / "ref", ref_path.name)
    log.info("Reference shape: %s; found %d DUT image(s)",
             ref_image.shape, len(dut_files))

    dut_images: list[np.ndarray] = []
    desc = "Loading DUT images" + (" (preprocessing)" if preprocess_dut else "")
    for p in tqdm(dut_files, desc=desc, unit="img"):
        img = load_rgb(p)
        if img.shape != ref_image.shape:
            raise ValueError(
                f"DUT image shape mismatch: {p.name} is {img.shape}, "
                f"reference {ref_path.name} is {ref_image.shape}. "
                f"Re-capture so all images have the same resolution."
            )
        if preprocess_dut:
            img = preprocess_rgb(img)
            if dump_dir is not None:
                _dump_preprocessed(img, dump_dir / "dut", p.name)
        dut_images.append(img)

    return LoadedInputs(
        ref_path=ref_path,
        ref_image=ref_image,
        dut_paths=dut_files,
        dut_images=dut_images,
        preprocess_ref=preprocess_ref,
        preprocess_dut=preprocess_dut,
    )
