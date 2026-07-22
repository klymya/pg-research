"""Running Gaussian-average background subtraction (Wren et al., Pfinder).

Ports bgslibrary's dp ``WrenGA``. Each pixel is modelled by a single Gaussian
with a running mean per channel and one shared variance. A pixel is foreground
when its squared distance to the mean exceeds ``threshold`` times the variance.
bgslibrary defaults: alpha 0.005, threshold 12.25 (= 3.5^2), initial variance 36.
"""

from __future__ import annotations

import numpy as np

from .base import BackgroundSubtractor, prepare_frame, to_mask

_VARIANCE_FLOOR = 4.0
"""Lower clamp on the per-pixel variance, matching bgslibrary."""


class RunningGaussianAverage(BackgroundSubtractor):
    """Single-Gaussian-per-pixel background model.

    The mean tracks the scene by exponential smoothing and the variance adapts
    to the observed squared distance, clamped to ``[4, 5 * init_variance]``.
    During the first ``learning_frames`` frames every pixel updates; afterwards
    only background-classified pixels do.
    """

    def __init__(
        self,
        alpha: float = 0.005,
        threshold: float = 12.25,
        init_variance: float = 36.0,
        learning_frames: int = 30,
    ) -> None:
        """Initialise the model.

        Args:
            alpha: Learning rate in ``[0, 1]`` for the mean and variance.
            threshold: Mahalanobis-like multiplier; foreground when the squared
                distance exceeds ``threshold * variance``.
            init_variance: Initial per-pixel variance and basis of its ceiling.
            learning_frames: Frames during which every pixel updates.

        Raises:
            ValueError: If ``alpha`` is outside ``[0, 1]`` or a magnitude is
                non-positive.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if threshold <= 0 or init_variance <= 0:
            raise ValueError("threshold and init_variance must be positive")
        self._alpha = float(alpha)
        self._threshold = float(threshold)
        self._init_variance = float(init_variance)
        self._learning_frames = int(learning_frames)
        self._mean: np.ndarray | None = None
        self._variance: np.ndarray | None = None
        self._frame_index = 0

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Classify ``frame`` against the Gaussian and update the parameters."""
        current = prepare_frame(frame)
        if self._mean is None or self._mean.shape != current.shape:
            self._mean = current.copy()
            self._variance = np.full(current.shape[:2], self._init_variance, np.float32)
            self._frame_index = 1
            return np.zeros(current.shape[:2], dtype=np.uint8)

        assert self._variance is not None
        delta = current - self._mean
        squared_distance = np.sum(delta * delta, axis=2)
        foreground = squared_distance > self._threshold * self._variance
        mask = to_mask(foreground)

        self._frame_index += 1
        learning = self._frame_index < self._learning_frames
        updatable = (~foreground) | learning

        selector = updatable[:, :, np.newaxis]
        self._mean = self._mean + self._alpha * delta * selector

        new_variance = self._variance + self._alpha * (squared_distance - self._variance)
        new_variance = np.clip(
            new_variance, _VARIANCE_FLOOR, 5.0 * self._init_variance
        )
        self._variance = np.where(updatable, new_variance, self._variance)
        return mask

    def reset(self) -> None:
        """Discard the learned Gaussian parameters."""
        self._mean = None
        self._variance = None
        self._frame_index = 0

    def get_background_image(self) -> np.ndarray | None:
        """Return the Gaussian mean as a ``uint8`` background image, if any."""
        if self._mean is None:
            return None
        return np.clip(self._mean, 0, 255).astype(np.uint8).squeeze()
