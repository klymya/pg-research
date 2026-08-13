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
from .utils import Q_ONE, quantise, round_shift as _round_shift

logger = logging.getLogger(__name__)


class RateParameterizedBGS:
    """Background model whose absorption rate is an explicit parameter.

    Alongside the backend's own model, an exponentially weighted BGR reference
    is maintained at the same learning rate.  It gives the shadow veto a
    background estimate for every backend, including those that do not expose
    their internal model.

    The reference exists solely to feed the veto, so it is only maintained while
    the veto is enabled (:attr:`ShadowConfig.enabled`).  With the veto off no
    reference is allocated and :attr:`background_bgr` stays ``None`` — the same
    contract as before the first frame.  This matters most for the embedded port,
    where the reference is 3 of the 5 persistent full-frame planes.
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
        # Absorption rate of the quantised reference, in Q8. Same rate as the
        # float path, so the two modes differ only in storage and rounding.
        self._alpha_q = min(quantise(self._learning_rate), Q_ONE)

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
        if self._shadow_suppressor.config.quantised:
            return self._reference  # already uint8; nothing to clip
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
        if self._shadow_suppressor.config.enabled:
            # Guarded rather than left to suppress()'s own early return: both
            # the uint8 view built by background_bgr and the float32 reference
            # behind it are full-frame allocations that nothing else reads.
            mask = self._shadow_suppressor.suppress(mask, frame, self.background_bgr)
            self._update_reference(frame)
        return mask

    def reset(self) -> None:
        """Discard the backend model and the shadow reference."""
        self._backend.reset()
        self._reference = None

    def _update_reference(self, frame: Frame) -> None:
        """Absorb ``frame`` into the BGR reference used by the shadow veto.

        With ``reference_dtype="uint8"`` the reference is 3 bytes per pixel
        instead of 12 and is absorbed with the same Q8 integer update the
        background model uses, so the result matches the C++ port exactly.
        """
        if self._shadow_suppressor.config.quantised:
            self._update_reference_uint8(frame)
            return
        current = frame.bgr.astype(np.float32)
        if self._reference is None or self._reference.shape != current.shape:
            self._reference = current.copy()
            return
        cv2.accumulateWeighted(current, self._reference, self._learning_rate)

    def _update_reference_uint8(self, frame: Frame) -> None:
        """Absorb ``frame`` into the quantised reference, in integer arithmetic.

        Quantising the reference introduces the same update dead zone as the
        quantised model mean: an absorption smaller than ``1 / (2 * alpha)``
        grey levels rounds to zero. Unlike the model there is no closed-form
        bound on how much error the veto tolerates -- it depends on the value
        ratio band and on hue stability, both of which vary with scene
        brightness -- so this mode is justified by measurement rather than by
        derivation. See ``shadow_veto_quantisation_roadmap.md``.
        """
        current = frame.bgr.astype(np.int32)
        if self._reference is None or self._reference.shape != current.shape:
            self._reference = frame.bgr.astype(np.uint8).copy()
            return
        reference = self._reference.astype(np.int32)
        step = _round_shift(self._alpha_q * (current - reference))
        self._reference = np.clip(reference + step, 0, 255).astype(np.uint8)
