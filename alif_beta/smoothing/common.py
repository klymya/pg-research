#!/usr/bin/env python3
"""
smoothing.py

Collection of common 1D signal / time-series smoothing algorithms.

Included:
- Simple Moving Average (SMA)
- Weighted Moving Average (WMA)
- Exponential Moving Average (EMA)
- Gaussian smoothing (kernel convolution)
- Median filter (robust to outliers)
- Savitzky–Golay filter (optional SciPy)
- LOWESS / LOESS (optional statsmodels)

All functions accept 1D array-like input and return numpy arrays.
"""

from __future__ import annotations
from datetime import datetime
import numpy as np
from scipy.signal import savgol_filter
from statsmodels.nonparametric.smoothers_lowess import lowess as _lowess


ONE_PERSON_START = datetime.strptime("10:05", "%H:%M").time()
THREE_PERSONS_START = datetime.strptime("10:30", "%H:%M").time()
ONE_PERSON_CHAIRS_START = datetime.strptime("10:38", "%H:%M").time()


# -------------------------------------------------
# Helpers
# -------------------------------------------------

def _to_1d_float(x):
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("Input must be 1D")
    return x


# -------------------------------------------------
# 1) Simple Moving Average (SMA)
# -------------------------------------------------

def sma(x, window: int):
    """
    Simple moving average (box filter).

    window: number of samples
    """
    x = _to_1d_float(x)
    if window <= 0:
        raise ValueError("window must be > 0")

    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode="same")


# -------------------------------------------------
# 2) Weighted Moving Average (WMA)
# -------------------------------------------------

def wma(x, window: int):
    """
    Linearly weighted moving average.
    Recent samples have higher weight.
    """
    x = _to_1d_float(x)
    if window <= 0:
        raise ValueError("window must be > 0")

    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()
    return np.convolve(x, weights, mode="same")


# -------------------------------------------------
# 3) Exponential Moving Average (EMA)
# -------------------------------------------------

def ema(x, alpha: float):
    """
    Exponential moving average.

    alpha in (0, 1]
    """
    x = _to_1d_float(x)
    if not (0.0 < alpha <= 1.0):
        raise ValueError("alpha must be in (0, 1]")

    y = np.empty_like(x)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1.0 - alpha) * y[i - 1]
    return y


# -------------------------------------------------
# 4) Gaussian smoothing
# -------------------------------------------------

def gaussian_smooth(x, sigma: float, radius: int | None = None):
    """
    Gaussian kernel smoothing.

    sigma: standard deviation (in samples)
    radius: kernel radius; default = 3 * sigma
    """
    x = _to_1d_float(x)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")

    if radius is None:
        radius = int(3 * sigma)

    t = np.arange(-radius, radius + 1)
    kernel = np.exp(-(t ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()

    return np.convolve(x, kernel, mode="same")


# -------------------------------------------------
# 5) Median filter
# -------------------------------------------------

def median_filter(x, window: int):
    """
    Median filter (robust to spikes / outliers).
    """
    x = _to_1d_float(x)
    if window <= 0:
        raise ValueError("window must be > 0")

    r = window // 2
    y = np.empty_like(x)

    for i in range(len(x)):
        lo = max(0, i - r)
        hi = min(len(x), i + r + 1)
        y[i] = np.median(x[lo:hi])

    return y


# -------------------------------------------------
# 6) Savitzky–Golay filter (optional)
# -------------------------------------------------

def savgol(x, window: int, polyorder: int):
    """
    Savitzky–Golay smoothing.
    Requires scipy.
    """
    x = _to_1d_float(x)
    if window % 2 == 0:
        raise ValueError("window must be odd")

    return savgol_filter(x, window_length=window, polyorder=polyorder)


# -------------------------------------------------
# 7) LOWESS / LOESS (optional)
# -------------------------------------------------

def lowess(x, frac: float = 0.1):
    """
    LOWESS smoothing.
    Requires statsmodels.
    """
    x = _to_1d_float(x)

    t = np.arange(len(x))
    return _lowess(x, t, frac=frac, return_sorted=False)
