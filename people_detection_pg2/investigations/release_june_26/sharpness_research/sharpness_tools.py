"""Sharpness / softness measurement and calibration for the release_june_26
camera-comparison images.

Metrics are aligned with `camera_qc.metrics.headline_metrics` so numbers from
this module are directly comparable with the QC verdict pipeline.

Operators
---------
- `soften(img, strength, sigma=1.0)` — Gaussian blend at fixed sigma;
  strength in [0, 1] (0 = no change, 1 = full blur).
- `sharpen(img, amount, sigma=1.0)` — unsharp mask;
  amount >= 0 (0 = no change, higher = sharper).

Both operators use a fixed Gaussian kernel `sigma` and vary a scalar
strength/amount, so bisection is well-behaved and monotone across the
full working range.

Calibration
-----------
`find_parameter_for_delta` bisects the operator parameter to land on a
target sharpness change (percent of source sharpness). Sharpness is
monotone in both `sigma` (soften) and `amount` (sharpen) over their
normal working ranges, so bisection is stable.

`equalize_sharpness` picks the right direction (soften or sharpen) and
returns the image whose sharpness best matches the reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal

import cv2
import numpy as np


# ----------------------------------------------------------------------------
# I/O helpers (thin, so users of this module don't need camera_qc)
# ----------------------------------------------------------------------------

def load_rgb(path: str | Path) -> np.ndarray:
    """Read an image and return it as HxWx3 uint8 RGB."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _as_gray_u8(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.uint8, copy=False)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


# ----------------------------------------------------------------------------
# Sharpness / softness metrics
# ----------------------------------------------------------------------------

def laplacian_variance(img: np.ndarray) -> float:
    """Variance of the Laplacian on grayscale.

    Matches `camera_qc.metrics.headline_metrics()["sharpness"]`.
    Higher = sharper. Robust, cheap, sensitive to defocus and noise.
    """
    gray = _as_gray_u8(img)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def fft_high_freq_energy(img: np.ndarray, radius_frac: float = 1 / 8) -> float:
    """Mean magnitude outside a central low-freq disk in the FFT.

    Matches `camera_qc.metrics.headline_metrics()["sharpness_fft"]`.
    Higher = sharper. Less sensitive to isolated noise than Laplacian var.
    """
    gray = _as_gray_u8(img).astype(np.float64)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    mag = np.abs(spectrum)
    h, w = gray.shape
    cy, cx = h / 2, w / 2
    yy, xx = np.ogrid[:h, :w]
    radius = radius_frac * min(h, w)
    high_freq_mask = (yy - cy) ** 2 + (xx - cx) ** 2 > radius ** 2
    return float(mag[high_freq_mask].mean())


def tenengrad(img: np.ndarray, ksize: int = 3) -> float:
    """Mean squared Sobel gradient magnitude — classic edge-based sharpness.

    Higher = sharper. Provided as a cross-check for `laplacian_variance`.
    """
    gray = _as_gray_u8(img)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize)
    return float((gx * gx + gy * gy).mean())


def noise_std(img: np.ndarray, ksize: int = 5) -> float:
    """High-frequency residual std after a small Gaussian.

    Matches `camera_qc.metrics.headline_metrics()["noise"]`. Lower = cleaner.
    Included here because sharpening always couples sharpness with noise —
    experiments should report both together.
    """
    gray = _as_gray_u8(img).astype(float)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0).astype(float)
    return float((gray - blurred).std())


def softness_index(img: np.ndarray) -> float:
    """Reciprocal of laplacian_variance, scaled for readability.

    softness_index = 1000 / laplacian_variance. Purely a convenience view;
    use `laplacian_variance` for anything quantitative.
    """
    lv = laplacian_variance(img)
    return float(1000.0 / lv) if lv > 0 else float("inf")


METRIC_FUNCS: dict[str, Callable[[np.ndarray], float]] = {
    "laplacian": laplacian_variance,
    "fft": fft_high_freq_energy,
    "tenengrad": tenengrad,
}


def sharpness_report(img: np.ndarray) -> dict:
    """One-stop report for a single image."""
    return {
        "laplacian": laplacian_variance(img),
        "fft": fft_high_freq_energy(img),
        "tenengrad": tenengrad(img),
        "noise": noise_std(img),
        "softness_index": softness_index(img),
    }


# ----------------------------------------------------------------------------
# Operators
# ----------------------------------------------------------------------------

def soften(img: np.ndarray, strength: float, sigma: float = 1.0) -> np.ndarray:
    """Blend the image with its Gaussian blur.

    out = (1 - strength) * img + strength * blur(img, sigma)

    `strength` in [0, 1]: 0 returns the input, 1 returns the fully blurred
    image. Fixed `sigma` keeps the transition smooth (varying sigma with
    ksize=0 snaps to discrete kernel sizes and makes bisection unstable).
    """
    if strength <= 0:
        return img.copy()
    strength = float(min(strength, 1.0))
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    out = (1.0 - strength) * img.astype(np.float32) + strength * blurred.astype(np.float32)
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def sharpen(img: np.ndarray, amount: float, sigma: float = 1.0) -> np.ndarray:
    """Unsharp mask: out = img + amount * (img - blur(img, sigma)).

    amount == 0 returns the input. amount ~0.3 = subtle, ~1.0 = obvious,
    >2.0 tends to over-shoot edges and amplify noise.
    """
    if amount == 0:
        return img.copy()
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=float(sigma), sigmaY=float(sigma))
    high = img.astype(np.float32) - blurred.astype(np.float32)
    out = img.astype(np.float32) + float(amount) * high
    # np.round (banker's rounding) rather than astype-truncation: truncation
    # biases small negative perturbations to -1 and drops small positive ones
    # to 0, so tiny amounts still shift ~40% of pixels by 1 gray-level.
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


# ----------------------------------------------------------------------------
# Calibration: pick the parameter that lands on a target sharpness delta
# ----------------------------------------------------------------------------

Direction = Literal["sharpen", "soften"]


@dataclass
class CalibrationResult:
    direction: Direction
    param_name: str          # "strength" (soften) or "amount" (sharpen)
    param_value: float
    baseline_sharpness: float
    achieved_sharpness: float
    achieved_delta_pct: float  # (achieved - baseline) / baseline * 100
    target_delta_pct: float
    metric: str
    kernel_sigma: float      # Gaussian sigma used by the operator


def _measure(img: np.ndarray, metric: str) -> float:
    return METRIC_FUNCS[metric](img)


def find_parameter_for_delta(
    img: np.ndarray,
    target_delta_pct: float,
    metric: str = "laplacian",
    kernel_sigma: float = 1.0,
    tol_pct: float = 0.1,
    coarse_grid: int = 65,
    fine_grid: int = 65,
    strength_bracket: tuple[float, float] = (0.0, 0.6),
    amount_bracket: tuple[float, float] = (0.0, 0.6),
) -> CalibrationResult:
    """Pick the operator parameter that lands closest to the target delta.

    Two-pass grid search:
      1. Coarse pass over the full parameter bracket.
      2. Fine pass in a narrow window around the coarse best point.

    Grid search is used instead of bisection because the sharpness response
    is not strictly monotone on 8-bit inputs: `np.rint` + `clip` inject small
    discontinuities that trap a plain bisection near jumps. A dense grid is
    O(n) instead of O(log n) but n = 129 total evals × ~5 ms is still under
    a second per calibration on a 240x320 image.

    `target_delta_pct` is signed (negative -> soften, positive -> sharpen).
    `tol_pct` is only used to early-exit when the coarse pass already lands
    within tolerance.
    """
    baseline = _measure(img, metric)
    if baseline <= 0:
        raise ValueError(f"Baseline {metric} sharpness is not positive ({baseline}).")

    if abs(target_delta_pct) < tol_pct:
        return CalibrationResult(
            direction="sharpen",
            param_name="amount",
            param_value=0.0,
            baseline_sharpness=baseline,
            achieved_sharpness=baseline,
            achieved_delta_pct=0.0,
            target_delta_pct=target_delta_pct,
            metric=metric,
            kernel_sigma=kernel_sigma,
        )

    direction: Direction = "sharpen" if target_delta_pct > 0 else "soften"
    target_sharpness = baseline * (1.0 + target_delta_pct / 100.0)

    if direction == "soften":
        param_name = "strength"
        bracket = strength_bracket

        def apply(p: float) -> np.ndarray:
            return soften(img, p, sigma=kernel_sigma)
    else:
        param_name = "amount"
        bracket = amount_bracket

        def apply(p: float) -> np.ndarray:
            return sharpen(img, p, sigma=kernel_sigma)

    def _best_on_grid(grid: np.ndarray) -> tuple[float, float]:
        best_p = float(grid[0])
        best_s = _measure(apply(best_p), metric)
        best_err = abs(best_s - target_sharpness)
        for p in grid[1:]:
            s = _measure(apply(float(p)), metric)
            err = abs(s - target_sharpness)
            if err < best_err:
                best_p, best_s, best_err = float(p), s, err
                if err / baseline * 100 < tol_pct:
                    break
        return best_p, best_s

    coarse = np.linspace(bracket[0], bracket[1], coarse_grid)
    coarse_p, _ = _best_on_grid(coarse)

    step = (bracket[1] - bracket[0]) / (coarse_grid - 1)
    fine_lo = max(bracket[0], coarse_p - step)
    fine_hi = min(bracket[1], coarse_p + step)
    fine = np.linspace(fine_lo, fine_hi, fine_grid)
    chosen, achieved = _best_on_grid(fine)

    return CalibrationResult(
        direction=direction,
        param_name=param_name,
        param_value=float(chosen),
        baseline_sharpness=baseline,
        achieved_sharpness=achieved,
        achieved_delta_pct=(achieved - baseline) / baseline * 100.0,
        target_delta_pct=target_delta_pct,
        metric=metric,
        kernel_sigma=kernel_sigma,
    )


def apply_sharpness_change(
    img: np.ndarray,
    target_delta_pct: float,
    metric: str = "laplacian",
    kernel_sigma: float = 1.0,
) -> tuple[np.ndarray, CalibrationResult]:
    """Calibrate and apply the operator in one call. Returns (image, result)."""
    result = find_parameter_for_delta(
        img, target_delta_pct, metric=metric, kernel_sigma=kernel_sigma,
    )
    if result.direction == "soften":
        out = soften(img, result.param_value, sigma=kernel_sigma)
    else:
        out = sharpen(img, result.param_value, sigma=kernel_sigma)
    return out, result


# ----------------------------------------------------------------------------
# Production wrapper — precalibrated lookup table
# ----------------------------------------------------------------------------
#
# `CALIBRATION_LAPLACIAN` was fit against the ceiling-camera reference
# `6100002-2000-000008-2550.jpg` in this investigation directory. Each
# entry is (direction, operator_parameter) for a target Laplacian-variance
# delta in percent. Every entry lands within 0.5 pp of the requested delta
# on that reference; expect roughly-right but slightly-off deltas on
# dramatically different scenes. See `calibration.csv` for the full table
# with achieved percentages, and re-run the calibration pipeline in
# `sharpness_experiments.ipynb` if you need a different reference.
#
# Convention: `direction == "sharpen"` -> pass value to `sharpen(img, amount=...)`;
#             `direction == "soften"`  -> pass value to `soften(img, strength=...)`.

CALIBRATION_KERNEL_SIGMA: float = 1.0

CALIBRATION_LAPLACIAN: dict[int, tuple[Direction, float]] = {
    -15: ("soften",  0.10020),
    -14: ("soften",  0.09375),
    -13: ("soften",  0.08555),
    -12: ("soften",  0.07969),
    -11: ("soften",  0.07148),
    -10: ("soften",  0.06592),
     -9: ("soften",  0.06006),
     -8: ("soften",  0.05391),
     -7: ("soften",  0.04688),
     -6: ("soften",  0.04102),
     -5: ("soften",  0.03545),
     -4: ("soften",  0.02930),
     -3: ("soften",  0.02402),
     -2: ("soften",  0.01934),
     -1: ("soften",  0.01318),
      0: ("sharpen", 0.00000),
      1: ("sharpen", 0.01289),
      2: ("sharpen", 0.01875),
      3: ("sharpen", 0.02285),
      4: ("sharpen", 0.02783),
      5: ("sharpen", 0.03340),
      6: ("sharpen", 0.03867),
      7: ("sharpen", 0.04482),
      8: ("sharpen", 0.05010),
      9: ("sharpen", 0.05566),
     10: ("sharpen", 0.06270),
     11: ("sharpen", 0.06826),
     12: ("sharpen", 0.07354),
     13: ("sharpen", 0.07998),
     14: ("sharpen", 0.08350),
     15: ("sharpen", 0.09199),
}


def _interp_param(
    delta_pct: float,
    table: dict[int, tuple[Direction, float]],
) -> tuple[Direction, float]:
    """Look up the operator parameter for a signed delta.

    Integer deltas use the table directly. Non-integer deltas linearly
    interpolate the parameter value between the two flanking integer
    anchors, treating the zero anchor as param=0 (no-op) on both sides.
    `delta_pct == 0` returns the sharpen operator with amount=0, i.e. a
    no-op.
    """
    if abs(delta_pct) > 15:
        raise ValueError(f"delta_pct must be in [-15, 15]; got {delta_pct}")
    if delta_pct == 0:
        return ("sharpen", 0.0)

    direction: Direction = "sharpen" if delta_pct > 0 else "soften"
    lo = int(np.floor(delta_pct))
    hi = int(np.ceil(delta_pct))
    if lo == hi:
        return (direction, table[lo][1])

    # Anchor 0 is a no-op on both sides -> param value = 0.
    val_lo = 0.0 if lo == 0 else table[lo][1]
    val_hi = 0.0 if hi == 0 else table[hi][1]
    frac = delta_pct - lo
    return (direction, val_lo + frac * (val_hi - val_lo))


def adjust_sharpness(
    img: np.ndarray,
    delta_pct: float,
    table: dict[int, tuple[Direction, float]] = CALIBRATION_LAPLACIAN,
    kernel_sigma: float = CALIBRATION_KERNEL_SIGMA,
) -> np.ndarray:
    """Apply a pre-calibrated sharpness delta in one call.

    `delta_pct` is signed and clamped to [-15, 15]. Negative -> soften,
    positive -> sharpen, zero -> return a copy of `img`.

    Uses the `CALIBRATION_LAPLACIAN` lookup by default: no per-image
    optimization, so this is safe to call inside a hot production loop
    (single Gaussian blur + linear combine per invocation).

    Non-integer `delta_pct` values are linearly interpolated between the
    flanking integer anchors.
    """
    direction, param_value = _interp_param(delta_pct, table)
    if direction == "soften":
        return soften(img, param_value, sigma=kernel_sigma)
    return sharpen(img, param_value, sigma=kernel_sigma)


# ----------------------------------------------------------------------------
# Equalization: match src sharpness to ref sharpness
# ----------------------------------------------------------------------------

@dataclass
class EqualizationResult:
    metric: str
    ref_sharpness: float
    src_sharpness: float
    achieved_sharpness: float
    calibration: CalibrationResult


def equalize_sharpness(
    src: np.ndarray,
    ref: np.ndarray,
    metric: str = "laplacian",
    kernel_sigma: float = 1.0,
    tol_pct: float = 0.5,
) -> tuple[np.ndarray, EqualizationResult]:
    """Return a copy of `src` whose sharpness matches `ref` under `metric`."""
    ref_s = _measure(ref, metric)
    src_s = _measure(src, metric)
    if src_s <= 0:
        raise ValueError(f"Source {metric} sharpness is not positive ({src_s}).")

    target_delta_pct = (ref_s - src_s) / src_s * 100.0
    out, cal = apply_sharpness_change(
        src, target_delta_pct, metric=metric, kernel_sigma=kernel_sigma,
    )

    return out, EqualizationResult(
        metric=metric,
        ref_sharpness=ref_s,
        src_sharpness=src_s,
        achieved_sharpness=_measure(out, metric),
        calibration=cal,
    )


# ----------------------------------------------------------------------------
# Sweep helper — for plotting parameter-vs-sharpness curves
# ----------------------------------------------------------------------------

def sweep(
    img: np.ndarray,
    direction: Direction,
    values: list[float] | np.ndarray,
    metric: str = "laplacian",
    kernel_sigma: float = 1.0,
) -> dict:
    """Return per-parameter sharpness / delta_pct / noise for plotting.

    Keys: 'param', 'sharpness', 'delta_pct', 'noise', 'noise_delta_pct'.
    """
    baseline = _measure(img, metric)
    baseline_noise = noise_std(img)
    out = {"param": [], "sharpness": [], "delta_pct": [], "noise": [], "noise_delta_pct": []}
    for v in values:
        if direction == "soften":
            processed = soften(img, float(v), sigma=kernel_sigma)
        else:
            processed = sharpen(img, float(v), sigma=kernel_sigma)
        s = _measure(processed, metric)
        n = noise_std(processed)
        out["param"].append(float(v))
        out["sharpness"].append(s)
        out["delta_pct"].append((s - baseline) / baseline * 100.0)
        out["noise"].append(n)
        out["noise_delta_pct"].append(
            (n - baseline_noise) / baseline_noise * 100.0 if baseline_noise > 0 else 0.0
        )
    return out
