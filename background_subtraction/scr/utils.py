"""Small shared helpers for mask and geometry handling."""

from __future__ import annotations

import cv2
import numpy as np

from .structures import BoundingBox

FOREGROUND = 255
"""Value marking a foreground pixel in every mask produced by the pipeline."""


def ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Normalise an arbitrary mask to ``uint8`` values of ``0`` or ``255``.

    OpenCV subtractors encode shadows as ``127``; those pixels are treated as
    background so that shadow handling stays the responsibility of Layer 2.

    Args:
        mask: Single-channel mask of any numeric dtype.

    Returns:
        A ``uint8`` mask containing only ``0`` and ``255``.

    Raises:
        ValueError: If ``mask`` is not single-channel.
    """
    if mask.ndim == 3 and mask.shape[2] == 1:
        mask = mask[:, :, 0]
    if mask.ndim != 2:
        raise ValueError("mask must be single-channel")
    return np.where(mask >= FOREGROUND, FOREGROUND, 0).astype(np.uint8)


def ensure_three_channels(image: np.ndarray) -> np.ndarray:
    """Return ``image`` as a 3-channel ``uint8`` image.

    Args:
        image: Single- or three-channel image.

    Returns:
        The image itself when already 3-channel, otherwise a grayscale-to-BGR
        replication.

    Raises:
        ValueError: If the image has an unsupported channel count.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 1:
        return cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"unsupported image shape {image.shape}")


def iou(first: BoundingBox, second: BoundingBox) -> float:
    """Compute the intersection over union of two ``(x, y, w, h)`` boxes.

    Args:
        first: First bounding box.
        second: Second bounding box.

    Returns:
        The IoU in ``[0, 1]``; ``0.0`` when either box is degenerate.
    """
    ax, ay, aw, ah = first
    bx, by, bw, bh = second
    if aw <= 0 or ah <= 0 or bw <= 0 or bh <= 0:
        return 0.0

    left = max(ax, bx)
    top = max(ay, by)
    right = min(ax + aw, bx + bw)
    bottom = min(ay + ah, by + bh)
    if right <= left or bottom <= top:
        return 0.0

    intersection = (right - left) * (bottom - top)
    union = aw * ah + bw * bh - intersection
    return intersection / union
