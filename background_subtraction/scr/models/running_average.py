"""Running-average background subtraction.

Ports bgslibrary's ``AdaptiveBackgroundLearning`` and its selective variant
``AdaptiveSelectiveBackgroundLearning``. The background is an exponentially
weighted running average of past frames; a pixel is foreground when the current
frame deviates from it by more than a threshold.
"""

from __future__ import annotations

import numpy as np

from .base import BackgroundSubtractor, prepare_frame, to_mask


class RunningAverageSubtractor(BackgroundSubtractor):
    """Exponentially weighted running-average background model.

    The background updates as ``bg = alpha * frame + (1 - alpha) * bg``.

    With ``selective=True`` the update is applied only to pixels currently
    classified as background (bgslibrary's
    ``AdaptiveSelectiveBackgroundLearning``), which stops foreground objects
    from bleeding into the model at the cost of slower recovery.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        threshold: float = 15.0,
        selective: bool = False,
    ) -> None:
        """Initialise the model.

        Args:
            alpha: Learning rate in ``[0, 1]``; larger absorbs changes faster.
            threshold: Per-channel absolute difference marking foreground.
            selective: When ``True``, update only background-classified pixels.

        Raises:
            ValueError: If ``alpha`` is outside ``[0, 1]`` or ``threshold`` < 0.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")
        self._alpha = float(alpha)
        self._threshold = float(threshold)
        self._selective = bool(selective)
        self._background: np.ndarray | None = None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Classify ``frame`` against the running average and update the model."""
        current = prepare_frame(frame)
        if self._background is None or self._background.shape != current.shape:
            self._background = current.copy()
            return np.zeros(current.shape[:2], dtype=np.uint8)

        difference = np.abs(current - self._background).max(axis=2)
        foreground = difference > self._threshold
        mask = to_mask(foreground)

        updated = self._alpha * current + (1.0 - self._alpha) * self._background
        if self._selective:
            keep = foreground[:, :, np.newaxis]
            self._background = np.where(keep, self._background, updated)
        else:
            self._background = updated
        return mask

    def reset(self) -> None:
        """Discard the learned background model."""
        self._background = None

    def get_background_image(self) -> np.ndarray | None:
        """Return the running-average background as a ``uint8`` image, if any."""
        if self._background is None:
            return None
        return np.clip(self._background, 0, 255).astype(np.uint8).squeeze()
