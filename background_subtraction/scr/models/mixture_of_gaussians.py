"""Mixture-of-Gaussians background subtraction (Stauffer & Grimson, 1999).

Ports bgslibrary's dp ``GrimsonGMM``. Each pixel is modelled by ``K`` Gaussians
kept sorted by descending weight/standard-deviation. Incoming pixels are matched
to a mode; matched modes are reinforced and adapted, unmatched pixels replace the
weakest mode. The leading modes whose cumulative weight reaches
``background_ratio`` form the background; a pixel matched to a non-background
mode (or to no mode) is foreground.

The whole per-pixel mixture is updated with vectorised NumPy operations rather
than a Python loop over pixels. bgslibrary defaults: 3 Gaussians, alpha 0.01,
threshold 9.0 (= 3^2), background ratio 0.75, initial variance 36.
"""

from __future__ import annotations

import numpy as np

from .base import BackgroundSubtractor, prepare_frame, to_mask

_VARIANCE_FLOOR = 4.0
"""Lower clamp on a mode's variance, matching bgslibrary."""


class MixtureOfGaussians(BackgroundSubtractor):
    """Per-pixel Gaussian-mixture background model."""

    def __init__(
        self,
        n_gaussians: int = 3,
        alpha: float = 0.01,
        threshold: float = 9.0,
        background_ratio: float = 0.75,
        init_variance: float = 36.0,
    ) -> None:
        """Initialise the model.

        Args:
            n_gaussians: Number of Gaussian modes per pixel.
            alpha: Learning rate in ``[0, 1]``.
            threshold: Squared-distance match multiplier; a pixel matches a mode
                when its squared distance is below ``threshold * variance``.
            background_ratio: Cumulative-weight fraction ``T`` defining how many
                leading modes form the background.
            init_variance: Variance assigned to a freshly created mode.

        Raises:
            ValueError: If any argument is outside its valid range.
        """
        if n_gaussians < 1:
            raise ValueError("n_gaussians must be >= 1")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if threshold <= 0 or init_variance <= 0:
            raise ValueError("threshold and init_variance must be positive")
        if not 0.0 < background_ratio <= 1.0:
            raise ValueError("background_ratio must lie in (0, 1]")

        self._k = int(n_gaussians)
        self._alpha = float(alpha)
        self._threshold = float(threshold)
        self._background_ratio = float(background_ratio)
        self._init_variance = float(init_variance)

        self._weight: np.ndarray | None = None      # (H, W, K)
        self._mean: np.ndarray | None = None         # (H, W, K, C)
        self._variance: np.ndarray | None = None     # (H, W, K)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Update the mixture with ``frame`` and return its foreground mask."""
        current = prepare_frame(frame)
        if self._weight is None or self._mean.shape[:2] != current.shape[:2]:
            self._initialise(current)
            return np.zeros(current.shape[:2], dtype=np.uint8)

        foreground = self._update(current)
        return to_mask(foreground)

    def reset(self) -> None:
        """Discard the learned mixture."""
        self._weight = None
        self._mean = None
        self._variance = None

    def get_background_image(self) -> np.ndarray | None:
        """Return the most probable mode's mean as the background image, if any."""
        if self._mean is None:
            return None
        background = self._mean[:, :, 0, :]
        return np.clip(background, 0, 255).astype(np.uint8).squeeze()

    def _initialise(self, frame: np.ndarray) -> None:
        """Seed the mixture from the first frame with one dominant mode."""
        height, width, channels = frame.shape
        self._weight = np.zeros((height, width, self._k), dtype=np.float32)
        self._mean = np.zeros((height, width, self._k, channels), dtype=np.float32)
        self._variance = np.full(
            (height, width, self._k), self._init_variance, dtype=np.float32
        )
        self._weight[:, :, 0] = 1.0
        self._mean[:, :, 0, :] = frame

    def _update(self, frame: np.ndarray) -> np.ndarray:
        """Run one Stauffer-Grimson step and return the foreground boolean map."""
        assert self._weight is not None and self._mean is not None
        assert self._variance is not None

        # Squared distance of the pixel to each mode, and per-mode match test.
        delta = frame[:, :, np.newaxis, :] - self._mean       # (H, W, K, C)
        squared_distance = np.sum(delta * delta, axis=3)      # (H, W, K)
        matches = squared_distance < self._threshold * self._variance

        # Modes are kept sorted by descending priority, so the first matching
        # mode is the one the pixel is assigned to.
        matched_any = matches.any(axis=2)
        best = np.argmax(matches, axis=2)                     # (H, W)
        onehot = np.zeros_like(self._weight, dtype=bool)
        rows, cols = np.indices(best.shape)
        onehot[rows, cols, best] = matched_any

        # Which leading modes make up the background (cumulative weight < T).
        cumulative = np.cumsum(self._weight, axis=2)
        is_background_mode = (cumulative - self._weight) < self._background_ratio
        matched_bg = np.take_along_axis(
            is_background_mode, best[:, :, np.newaxis], axis=2
        )[:, :, 0]
        foreground = ~(matched_any & matched_bg)

        # Weight update: decay all modes, reinforce the matched one.
        self._weight = (1.0 - self._alpha) * self._weight + self._alpha * onehot

        # Mean/variance update of the matched mode (rho = alpha / new weight).
        safe_weight = np.where(onehot, self._weight, 1.0)
        rho = (self._alpha / safe_weight)[:, :, :, np.newaxis]
        mode_delta = frame[:, :, np.newaxis, :] - self._mean
        mean_update = self._mean + rho * mode_delta
        self._mean = np.where(onehot[:, :, :, np.newaxis], mean_update, self._mean)

        rho2 = self._alpha / safe_weight
        variance_update = self._variance + rho2 * (squared_distance - self._variance)
        variance_update = np.clip(
            variance_update, _VARIANCE_FLOOR, 5.0 * self._init_variance
        )
        self._variance = np.where(onehot, variance_update, self._variance)

        # Pixels matching no mode replace their weakest (last, lowest-priority) mode.
        self._replace_weakest_mode(frame, matched_any)

        # Renormalise weights and re-sort modes by descending priority.
        self._weight /= self._weight.sum(axis=2, keepdims=True)
        self._sort_modes()
        return foreground

    def _replace_weakest_mode(self, frame: np.ndarray, matched_any: np.ndarray) -> None:
        """Reseed the last mode of unmatched pixels with the current value."""
        assert self._weight is not None and self._mean is not None
        assert self._variance is not None
        unmatched = ~matched_any
        last = self._k - 1
        self._mean[unmatched, last, :] = frame[unmatched]
        self._variance[unmatched, last] = self._init_variance
        self._weight[unmatched, last] = self._alpha

    def _sort_modes(self) -> None:
        """Sort modes per pixel by descending weight / standard deviation."""
        assert self._weight is not None and self._mean is not None
        assert self._variance is not None
        priority = self._weight / np.sqrt(self._variance)
        order = np.argsort(-priority, axis=2)
        self._weight = np.take_along_axis(self._weight, order, axis=2)
        self._variance = np.take_along_axis(self._variance, order, axis=2)
        self._mean = np.take_along_axis(
            self._mean, order[:, :, :, np.newaxis], axis=2
        )
