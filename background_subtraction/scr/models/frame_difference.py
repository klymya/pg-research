"""Frame-difference background subtraction.

Ports bgslibrary's ``FrameDifference`` (difference against the previous frame)
and ``StaticFrameDifference`` (difference against a fixed reference frame). Both
are the simplest possible detectors: a pixel is foreground when it differs from
the reference by more than a threshold.
"""

from __future__ import annotations

import numpy as np

from .base import BackgroundSubtractor, prepare_frame, to_mask


class FrameDifference(BackgroundSubtractor):
    """Temporal frame differencing against the previous frame.

    The reference is the previously observed frame, so only pixels that are
    *changing* register as foreground. bgslibrary default threshold is 15.
    """

    def __init__(self, threshold: float = 15.0) -> None:
        """Initialise the detector.

        Args:
            threshold: Per-channel absolute difference marking foreground.

        Raises:
            ValueError: If ``threshold`` is negative.
        """
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self._threshold = float(threshold)
        self._previous: np.ndarray | None = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Difference ``frame`` against the previous one and update the reference."""
        current = prepare_frame(frame)
        if self._previous is None or self._previous.shape != current.shape:
            self._previous = current
            return np.zeros(current.shape[:2], dtype=np.uint8)

        difference = np.abs(current - self._previous).max(axis=2)
        mask = to_mask(difference > self._threshold)
        self._previous = current
        return mask

    def reset(self) -> None:
        """Discard the stored previous frame."""
        self._previous = None

    def get_background_image(self) -> np.ndarray | None:
        """Return the previous frame as the background estimate, if any."""
        if self._previous is None:
            return None
        return np.clip(self._previous, 0, 255).astype(np.uint8).squeeze()


class StaticFrameDifference(BackgroundSubtractor):
    """Frame differencing against a fixed reference (the first frame).

    Unlike :class:`FrameDifference`, the reference never updates, so a newly
    introduced static object stays foreground indefinitely.
    """

    def __init__(self, threshold: float = 15.0) -> None:
        """Initialise the detector.

        Args:
            threshold: Per-channel absolute difference marking foreground.

        Raises:
            ValueError: If ``threshold`` is negative.
        """
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self._threshold = float(threshold)
        self._reference: np.ndarray | None = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Difference ``frame`` against the fixed reference frame."""
        current = prepare_frame(frame)
        if self._reference is None or self._reference.shape != current.shape:
            self._reference = current
            return np.zeros(current.shape[:2], dtype=np.uint8)

        difference = np.abs(current - self._reference).max(axis=2)
        return to_mask(difference > self._threshold)

    def reset(self) -> None:
        """Discard the stored reference frame."""
        self._reference = None

    def get_background_image(self) -> np.ndarray | None:
        """Return the fixed reference as the background estimate, if any."""
        if self._reference is None:
            return None
        return np.clip(self._reference, 0, 255).astype(np.uint8).squeeze()
