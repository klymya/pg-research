"""Shared utilities for the sensor color/quality comparison notebooks.

Used by:
  - sensor_color_comparison.ipynb         (raw old-vs-new comparison)
  - sensor_color_comparison_postccm.ipynb (old vs CCM-corrected new)

The per-image `*_stats` functions all take an RGB uint8 image and return a
small dict, so they plug straight into `side_by_side` for the per-section
paired tables. `headline_metrics` is the canonical metric pack used by the
paired delta summary — extending it there propagates to both notebooks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Mapping

import cv2
import numpy as np
import pandas as pd


# RGB channel index + matplotlib color, used for histogram plots.
CHANNELS: tuple[tuple[str, int, str], ...] = (
    ("R", 0, "tab:red"),
    ("G", 1, "tab:green"),
    ("B", 2, "tab:blue"),
)


# ----- I/O -----------------------------------------------------------------

def load_rgb(path: Path | str) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def load_scenes(
    folder: Path, scenes: Iterable[int], ext: str = "jpg"
) -> list[np.ndarray]:
    return [load_rgb(Path(folder) / f"{i}.{ext}") for i in scenes]


# ----- Tabulation ----------------------------------------------------------

def side_by_side(
    stat_fn: Callable[[np.ndarray], Mapping[str, float]],
    images: Mapping[str, list[np.ndarray]],
    sensors: tuple[str, ...] = ("old", "new"),
) -> pd.DataFrame:
    """One row per scene; columns are a MultiIndex (metric, sensor) with the
    sensors in the order given so old and new sit next to each other.
    Number of scenes is inferred from `images[sensors[0]]`.
    """
    sensors = tuple(sensors)
    n_scenes = len(images[sensors[0]])
    rows = [
        {"scene": i, "sensor": s, **stat_fn(images[s][i])}
        for i in range(n_scenes) for s in sensors
    ]
    df = pd.DataFrame(rows)
    metrics = [c for c in df.columns if c not in ("scene", "sensor")]
    wide = df.pivot(index="scene", columns="sensor", values=metrics)
    return wide.reindex(columns=pd.MultiIndex.from_product([metrics, list(sensors)]))


# ----- Per-pixel basics ----------------------------------------------------

def luminance(img: np.ndarray) -> np.ndarray:
    """BT.601 luminance from an RGB uint8 image."""
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


# ----- Per-image statistic blocks -----------------------------------------

def exposure_stats(img: np.ndarray) -> dict:
    y = luminance(img)
    return {
        "mean_Y": float(y.mean()),
        "shadow_clip_%": float((y == 0).mean() * 100),
        "highlight_clip_%": float((y == 255).mean() * 100),
        "p1": float(np.percentile(y, 1)),
        "p99": float(np.percentile(y, 99)),
    }


def contrast_stats(img: np.ndarray) -> dict:
    y = luminance(img)
    return {
        "std_Y": float(y.std()),
        "rms_contrast": float(y.std() / (y.mean() + 1e-6)),
    }


def color_balance_stats(img: np.ndarray) -> dict:
    r = float(img[..., 0].mean())
    g = float(img[..., 1].mean())
    b = float(img[..., 2].mean())
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(float)
    gray = (r + g + b) / 3
    return {
        "mean_R": r, "mean_G": g, "mean_B": b,
        "R/G": r / (g + 1e-6),
        "B/G": b / (g + 1e-6),
        "gray_world_dev": float(np.linalg.norm([r - gray, g - gray, b - gray])),
        "Lab_a*": float(lab[..., 1].mean() - 128),
        "Lab_b*": float(lab[..., 2].mean() - 128),
    }


def hsv_sat_stats(img: np.ndarray) -> dict:
    s = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)[..., 1]
    return {"mean_S": float(s.mean()), "median_S": float(np.median(s))}


def quality_stats(img: np.ndarray) -> dict:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise = float((gray.astype(float) - blurred.astype(float)).std())
    return {"sharpness_lap_var": sharp, "noise_hf_std": noise}


def fft_sharpness(img: np.ndarray, radius_frac: float = 1 / 8) -> float:
    """Mean magnitude of the FFT spectrum outside a low-frequency central
    disk. The disk radius is `radius_frac * min(H, W)`; everything inside
    (DC + low frequencies) is masked out so the result is dominated by
    fine-texture / edge energy and is largely invariant to global
    illumination changes.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(spectrum)
    h, w = gray.shape
    cy, cx = h / 2, w / 2
    yy, xx = np.ogrid[:h, :w]
    radius = radius_frac * min(h, w)
    high_freq_mask = (yy - cy) ** 2 + (xx - cx) ** 2 > radius ** 2
    return float(mag[high_freq_mask].mean())


def fft_sharpness_stats(img: np.ndarray) -> dict:
    return {"sharpness_fft_mean": fft_sharpness(img)}


def headline_metrics(img: np.ndarray) -> dict:
    """Canonical metric pack for the paired delta summary. Adding a key here
    propagates the new column into both notebooks' delta tables.
    """
    y = luminance(img)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(float)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    return {
        "L": float(y.mean()),
        "a*": float(lab[..., 1].mean() - 128),
        "b*": float(lab[..., 2].mean() - 128),
        "R": float(img[..., 0].mean()),
        "G": float(img[..., 1].mean()),
        "B": float(img[..., 2].mean()),
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "noise": float((gray.astype(float) - blurred.astype(float)).std()),
        "sharpness_fft": fft_sharpness(img),
    }


# ----- CCM (used by the post-CCM notebook) --------------------------------

# Param vector layout matches color_correction.c macro order:
#   index: 0     1     2     3     4     5     6     7     8
#   macro: C_RR  C_RG  C_RB  C_GR  C_GG  C_GB  C_BR  C_BG  C_BB
CCM_MACRO_NAMES: tuple[str, ...] = (
    "C_RR", "C_RG", "C_RB",
    "C_GR", "C_GG", "C_GB",
    "C_BR", "C_BG", "C_BB",
)


def build_ccm(macros: Mapping[str, float]) -> np.ndarray:
    """Pack the 9 #define values into the 3x3 matrix M such that
    out_pixel = M @ in_pixel. Mirrors the column-major layout of
    color_correction.c: row 0 (R_out) uses [C_RR, C_GR, C_BR].
    """
    m = macros
    return np.array([
        [m["C_RR"], m["C_GR"], m["C_BR"]],
        [m["C_RG"], m["C_GG"], m["C_BG"]],
        [m["C_RB"], m["C_GB"], m["C_BB"]],
    ], dtype=np.float64)


def apply_ccm(img_u8: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Apply CCM in normalized float space, clip to [0, 1], return uint8 RGB.
    Mirrors firmware behavior: 0..255 in, clipped 0..255 out.
    """
    x = img_u8.astype(np.float64) / 255.0
    y = x @ M.T
    np.clip(y, 0.0, 1.0, out=y)
    return (y * 255.0).round().astype(np.uint8)
