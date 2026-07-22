"""Adaptive-median background subtraction (McFarlane & Schofield).

Ports bgslibrary's dp ``AdaptiveMedianBGS``. The per-pixel background estimate
approximates the running median: every ``sampling_rate`` frames each channel of
the estimate is nudged by +/-1 toward the observed value, which converges to the
median without storing a frame window. bgslibrary defaults: threshold 40,
sampling rate 7, learning frames 30.
"""

from __future__ import annotations

import numpy as np

from .base import BackgroundSubtractor, prepare_frame, to_mask


class AdaptiveMedianSubtractor(BackgroundSubtractor):
    """Incremental-median background model.

    During the first ``learning_frames`` frames every pixel updates
    unconditionally; afterwards only background-classified pixels update, so
    foreground objects do not corrupt the estimate.
    """

    def __init__(
        self,
        threshold: float = 40.0,
        sampling_rate: int = 7,
        learning_frames: int = 30,
    ) -> None:
        """Initialise the model.

        Args:
            threshold: Per-channel absolute difference marking foreground.
            sampling_rate: The estimate is nudged once every this many frames.
            learning_frames: Frames during which every pixel updates.

        Raises:
            ValueError: If ``threshold`` < 0 or ``sampling_rate`` < 1.
        """
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        if sampling_rate < 1:
            raise ValueError("sampling_rate must be >= 1")
        self._threshold = float(threshold)
        self._sampling_rate = int(sampling_rate)
        self._learning_frames = int(learning_frames)
        self._median: np.ndarray | None = None
        self._frame_index = 0

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Classify ``frame`` against the median and nudge the estimate."""
        current = prepare_frame(frame)
        if self._median is None or self._median.shape != current.shape:
            self._median = current.copy()
            self._frame_index = 1
            return np.zeros(current.shape[:2], dtype=np.uint8)

        difference = np.abs(current - self._median)
        foreground = (difference > self._threshold).any(axis=2)
        mask = to_mask(foreground)

        self._frame_index += 1
        if self._frame_index % self._sampling_rate == 1:
            learning = self._frame_index < self._learning_frames
            updatable = (~foreground) | learning
            step = np.sign(current - self._median) * updatable[:, :, np.newaxis]
            self._median = self._median + step
        return mask

    def reset(self) -> None:
        """Discard the learned median and restart frame counting."""
        self._median = None
        self._frame_index = 0

    def get_background_image(self) -> np.ndarray | None:
        """Return the median background as a ``uint8`` image, if any."""
        if self._median is None:
            return None
        return np.clip(self._median, 0, 255).astype(np.uint8).squeeze()
