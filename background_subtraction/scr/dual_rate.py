"""Layer 3a: the dual-rate background subtraction framework.

Two :class:`~scr.bgs_model.RateParameterizedBGS` instances observe the same
frame at different absorption rates.  A newly deposited static object is
absorbed quickly by the fast model - so it leaves the fast foreground - while
the slow model keeps reporting it.  The static-candidate cue is therefore
``slow AND NOT fast``.

Note:
    The task description phrases this the other way round ("present in the Fast
    mask but absent in the Slow mask"), which contradicts its own statement that
    the slow model "preserves newly static objects".  ``slow AND NOT fast`` is
    the physically correct rule and the one implemented here.
"""

from __future__ import annotations

import cv2
import numpy as np

from .bgs_model import RateParameterizedBGS
from .config import DualRateConfig, FeatureConfig, ShadowConfig
from .structures import DualRateMasks, Frame


class DualRateBGS:
    """Runs a fast and a slow background model and contrasts their masks."""

    def __init__(
        self,
        fast: RateParameterizedBGS,
        slow: RateParameterizedBGS,
        warmup_frames: int = 0,
    ) -> None:
        """Initialise the dual-rate stage.

        Args:
            fast: Model absorbing changes quickly.
            slow: Model absorbing changes slowly.
            warmup_frames: Frames during which the static mask is forced empty,
                giving the slow model time to converge.

        Raises:
            ValueError: If the fast model is slower than the slow one, or if
                ``warmup_frames`` is negative.
        """
        if fast.learning_rate < slow.learning_rate:
            raise ValueError("the fast model must have the higher learning rate")
        if warmup_frames < 0:
            raise ValueError("warmup_frames must be non-negative")

        self._fast = fast
        self._slow = slow
        self._warmup_frames = warmup_frames
        self._processed = 0

    @classmethod
    def from_config(
        cls,
        config: DualRateConfig | None = None,
        feature: FeatureConfig | None = None,
        shadow: ShadowConfig | None = None,
    ) -> DualRateBGS:
        """Build a dual-rate stage from configuration objects.

        Args:
            config: Fast/slow model parameters and warm-up length.
            feature: Layer 2 feature parameters shared by both models.
            shadow: Shadow veto parameters shared by both models.

        Returns:
            The configured :class:`DualRateBGS`.
        """
        config = config or DualRateConfig()
        return cls(
            fast=RateParameterizedBGS.from_config(config.fast, feature, shadow),
            slow=RateParameterizedBGS.from_config(config.slow, feature, shadow),
            warmup_frames=config.warmup_frames,
        )

    @property
    def fast(self) -> RateParameterizedBGS:
        """The fast background model."""
        return self._fast

    @property
    def slow(self) -> RateParameterizedBGS:
        """The slow background model."""
        return self._slow

    @property
    def in_warmup(self) -> bool:
        """Whether the slow model is still converging."""
        return self._processed < self._warmup_frames

    def apply(self, frame: Frame) -> DualRateMasks:
        """Update both models and derive the static-candidate mask.

        Args:
            frame: Preprocessed frame from Layer 0.

        Returns:
            The fast, slow and static masks for this frame.
        """
        fast_mask = self._fast.apply(frame)
        slow_mask = self._slow.apply(frame)

        if self.in_warmup:
            static_mask = np.zeros_like(slow_mask)
        else:
            static_mask = cv2.bitwise_and(slow_mask, cv2.bitwise_not(fast_mask))

        self._processed += 1
        return DualRateMasks(fast=fast_mask, slow=slow_mask, static=static_mask)

    def reset(self) -> None:
        """Reset both models and the warm-up counter."""
        self._fast.reset()
        self._slow.reset()
        self._processed = 0
