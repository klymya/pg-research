"""Layer 1: the rate-parameterized background model.

A :class:`RateParameterizedBGS` composes the Layer 2 feature transform, a
pluggable subtraction backend driven by an explicit learning rate, and the HSV
shadow veto.  The class is instantiated twice by Layer 3 - once fast, once slow
- which is the whole point of keeping the rate an input argument.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .backends import BGSBackend, create_backend
from .config import BGSConfig, FeatureConfig, ShadowConfig
from .features import FeatureExtractor, ShadowSuppressor, create_feature_extractor
from .structures import Frame

logger = logging.getLogger(__name__)


class RateParameterizedBGS:
    """Background model whose absorption rate is an explicit parameter.

    Alongside the backend's own model, an exponentially weighted BGR reference
    is maintained at the same learning rate.  It gives the shadow veto a
    background estimate for every backend, including those that do not expose
    their internal model.
    """

    def __init__(
        self,
        learning_rate: float,
        backend: BGSBackend,
        feature_extractor: FeatureExtractor,
        shadow_suppressor: ShadowSuppressor | None = None,
    ) -> None:
        """Initialise the model from already-built collaborators.

        Args:
            learning_rate: Absorption rate ``alpha`` in ``[0, 1]``.
            backend: Subtraction algorithm adapter.
            feature_extractor: Layer 2 differencing metric.
            shadow_suppressor: Optional HSV shadow veto.

        Raises:
            ValueError: If ``learning_rate`` is outside ``[0, 1]``.
        """
        if not 0.0 <= learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in [0, 1]")

        self._learning_rate = float(learning_rate)
        self._backend = backend
        self._feature_extractor = feature_extractor
        self._shadow_suppressor = shadow_suppressor or ShadowSuppressor()
        self._reference: np.ndarray | None = None

        if not backend.supports_learning_rate:
            logger.info(
                "Backend %s manages its own adaptation schedule; "
                "learning_rate=%.4f only drives the shadow reference model",
                type(backend).__name__,
                self._learning_rate,
            )

    @classmethod
    def from_config(
        cls,
        config: BGSConfig,
        feature: FeatureConfig | None = None,
        shadow: ShadowConfig | None = None,
    ) -> RateParameterizedBGS:
        """Build a model from configuration objects.

        Args:
            config: Layer 1 backend and learning-rate parameters.
            feature: Layer 2 feature parameters.
            shadow: Shadow veto parameters.

        Returns:
            The configured :class:`RateParameterizedBGS`.
        """
        return cls(
            learning_rate=config.learning_rate,
            backend=create_backend(config),
            feature_extractor=create_feature_extractor(feature),
            shadow_suppressor=ShadowSuppressor(shadow),
        )

    @property
    def learning_rate(self) -> float:
        """The absorption rate driving this model."""
        return self._learning_rate

    @property
    def backend(self) -> BGSBackend:
        """The underlying subtraction backend."""
        return self._backend

    @property
    def background_bgr(self) -> np.ndarray | None:
        """Exponentially weighted BGR background estimate, if initialised."""
        if self._reference is None:
            return None
        return np.clip(self._reference, 0, 255).astype(np.uint8)

    def apply(self, frame: Frame) -> np.ndarray:
        """Update the model with ``frame`` and return its foreground mask.

        Args:
            frame: Preprocessed frame from Layer 0.

        Returns:
            A ``uint8`` foreground mask with values ``0``/``255``.
        """
        features = self._feature_extractor.transform(frame)
        mask = self._backend.apply(features, self._learning_rate)
        mask = self._shadow_suppressor.suppress(mask, frame, self.background_bgr)
        self._update_reference(frame)
        return mask

    def reset(self) -> None:
        """Discard the backend model and the shadow reference."""
        self._backend.reset()
        self._reference = None

    def _update_reference(self, frame: Frame) -> None:
        """Absorb ``frame`` into the BGR reference used by the shadow veto."""
        current = frame.bgr.astype(np.float32)
        if self._reference is None or self._reference.shape != current.shape:
            self._reference = current.copy()
            return
        cv2.accumulateWeighted(current, self._reference, self._learning_rate)
