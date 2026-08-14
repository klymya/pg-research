"""ALIF white balance + CLAHE, the same pipeline used to build the training
datasets.

Ported verbatim from
`people_detection_pg2/cam4_revB/alif_awb.py` (`white_balance_vectorized`) and
`people_detection_pg2/cam4_revB/preprocess_dataset.py` (`apply_clahe`), so that
camera_qc stays self-contained and usable outside this repo. Keep the two
copies in sync: an image preprocessed here must be bit-identical to one
preprocessed by `preprocess_dataset.py --mode wb_alif_clahe` (that script also
offers a sharpness augmentation stage, which is deliberately *not* copied — it
is a random training-time augmentation, not part of the capture pipeline).

All functions take and return RGB uint8 arrays, matching `metrics.load_rgb`.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np


log = logging.getLogger(__name__)

MODE = "wb_alif_clahe"  # the single pipeline this module implements

# === Shared Constants ===
CORRECTION_MATRIX = np.array([
    [1.86,  0.00, -0.05],  # R
    [-0.16, 0.95, -0.53],  # G
    [-0.53, 0.07,  2.27]   # B
], dtype=np.float32)

CLAHE_CLIP_LIMIT = 3.0
CLAHE_TILE_GRID_SIZE = (8, 8)


# === Utilities ===

def srgb_oetf(lum: np.ndarray) -> np.ndarray:
    """sRGB Opto-Electronic Transfer Function (OETF)"""
    out = np.empty_like(lum)
    mask = lum <= 0.0031308
    out[mask] = 12.92 * lum[mask]
    out[~mask] = 1.055 * np.power(lum[~mask], 1 / 2.4) - 0.055
    return out


def prepare_gamma_lut() -> np.ndarray:
    """Create a 256-entry gamma LUT using sRGB OETF"""
    lut = np.empty(256, dtype=np.uint8)
    for i in range(256):
        lum = i / 255.0
        gamma = srgb_oetf(np.array([lum], dtype=np.float32))[0]
        lut[i] = np.clip(np.round(gamma * 255), 0, 255).astype(np.uint8)
    return lut


# === Vectorized White Balance ===

def white_balance_vectorized(image: np.ndarray) -> np.ndarray:
    """
    Vectorized white balance method.
    This method combines color correction and gray-world white balancing.
    """
    assert image.dtype == np.uint8 and image.ndim == 3 and image.shape[2] == 3

    img_float = image.astype(np.float32)
    corrected = np.tensordot(img_float, CORRECTION_MATRIX.T, axes=([2], [1]))
    corrected = np.clip(corrected, 0, 255)

    lut = prepare_gamma_lut()
    corrected_gamma = lut[corrected.astype(np.uint8)]

    mean_rgb = corrected_gamma.mean(axis=(0, 1), dtype=np.float32)
    mean_gray = mean_rgb.mean()
    scale = mean_gray / mean_rgb
    balanced = corrected_gamma.astype(np.float32) * scale

    balanced = np.clip(balanced, 0, 255).astype(np.uint8)
    return balanced


# === CLAHE ===

def apply_clahe(image_rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(image_rgb.copy(), cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_GRID_SIZE
    )
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


# === Public entry point ===

def preprocess_rgb(image_rgb: np.ndarray) -> np.ndarray:
    """Apply the full capture pipeline: ALIF white balance -> CLAHE."""
    if image_rgb.dtype != np.uint8 or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(
            f"Expected an RGB uint8 image, got dtype={image_rgb.dtype}, "
            f"shape={image_rgb.shape}"
        )
    return apply_clahe(white_balance_vectorized(image_rgb))
