"""Image metrics, ported verbatim from
people_detection_pg2/investigations/rotated_cam_color_dfiff/sensor_metrics.py.

Only the headline pack is wired into the QC pipeline; the per-image stat
blocks (exposure/contrast/color_balance/etc.) are kept available for future
report sections without re-deriving them.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_rgb(path: Path | str) -> np.ndarray:
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def luminance(img: np.ndarray) -> np.ndarray:
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def exposure_stats(img: np.ndarray) -> dict:
    y = luminance(img)
    return {
        "mean_Y": float(y.mean()),
        "shadow_clip_%": float((y == 0).mean() * 100),
        "highlight_clip_%": float((y == 255).mean() * 100),
        "p1": float(np.percentile(y, 1)),
        "p99": float(np.percentile(y, 99)),
    }


def fft_sharpness(img: np.ndarray, radius_frac: float = 1 / 8) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float64)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(spectrum)
    h, w = gray.shape
    cy, cx = h / 2, w / 2
    yy, xx = np.ogrid[:h, :w]
    radius = radius_frac * min(h, w)
    high_freq_mask = (yy - cy) ** 2 + (xx - cx) ** 2 > radius ** 2
    return float(mag[high_freq_mask].mean())


def headline_metrics(img: np.ndarray) -> dict:
    """Canonical metric pack used by the QC verdict. Order matches the report
    table.
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


HEADLINE_KEYS: tuple[str, ...] = (
    "L", "a*", "b*", "R", "G", "B", "sharpness", "noise", "sharpness_fft",
)
