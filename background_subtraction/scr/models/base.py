"""Abstract base class and shared helpers for the pure-Python BGS models.

Every model in :mod:`scr.models` is a self-contained reimplementation of a
classic background-subtraction algorithm using only NumPy (no OpenCV/pybgs
native code). They share one small interface so they stay interchangeable:

    subtractor = SomeSubtractor()
    mask = subtractor.apply(frame_bgr)   # uint8, values 0 / 255
    subtractor.reset()

Frames may be single-channel ``(H, W)`` or 3-channel ``(H, W, 3)`` ``uint8``
arrays; masks are always single-channel ``uint8`` with foreground marked 255.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

FOREGROUND = 255
"""Value marking a foreground pixel in every returned mask."""

BACKGROUND = 0
"""Value marking a background pixel in every returned mask."""


def prepare_frame(frame: np.ndarray) -> np.ndarray:
    """Validate a frame and coerce it to ``float32`` of shape ``(H, W, C)``.

    Args:
        frame: Single-channel ``(H, W)`` or 3-channel ``(H, W, 3)`` image.

    Returns:
        The frame as ``float32`` with an explicit channel axis.

    Raises:
        TypeError: If ``frame`` is not a NumPy array.
        ValueError: If the frame is neither single- nor three-channel.
    """
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy.ndarray")
    if frame.ndim == 2:
        work = frame[:, :, np.newaxis]
    elif frame.ndim == 3 and frame.shape[2] in (1, 3):
        work = frame
    else:
        raise ValueError(f"unsupported frame shape {frame.shape}")
    return work.astype(np.float32)


def to_grayscale(frame: np.ndarray) -> np.ndarray:
    """Reduce a validated ``(H, W, C)`` float frame to a ``(H, W)`` intensity map.

    Args:
        frame: Frame of shape ``(H, W, C)`` as returned by :func:`prepare_frame`.

    Returns:
        The per-pixel channel mean, ``float32`` of shape ``(H, W)``.
    """
    if frame.shape[2] == 1:
        return frame[:, :, 0]
    return frame.mean(axis=2)


def to_mask(foreground: np.ndarray) -> np.ndarray:
    """Convert a boolean foreground array to a ``uint8`` 0/255 mask.

    Args:
        foreground: Boolean array, ``True`` where foreground.

    Returns:
        A ``uint8`` mask with foreground pixels set to 255.
    """
    return foreground.astype(np.uint8) * FOREGROUND


class BackgroundSubtractor(ABC):
    """Common interface for the pure-Python background-subtraction models."""

    @abstractmethod
    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Update the model with ``frame`` and return its foreground mask.

        Args:
            frame: Single- or three-channel ``uint8`` image.

        Returns:
            A ``uint8`` foreground mask with values ``0``/``255``.
        """

    @abstractmethod
    def reset(self) -> None:
        """Discard the learned background model."""

    def get_background_image(self) -> np.ndarray | None:
        """Return the current background estimate, if the model exposes one.

        Returns:
            A ``uint8`` background image, or ``None`` when unavailable.
        """
        return None
