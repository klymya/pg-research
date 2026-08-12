"""Running Gaussian-average background subtraction (Wren et al., Pfinder).

Ports bgslibrary's dp ``WrenGA``. Each pixel is modelled by a single Gaussian
with a running mean per channel and one shared variance. A pixel is foreground
when its squared distance to the mean exceeds ``threshold`` times the variance.
bgslibrary defaults: alpha 0.005, threshold 12.25 (= 3.5^2), initial variance 36.

The model state is stored as ``uint8`` by default -- 2 bytes per pixel at
``C = 1`` instead of the 8 bytes float32 costs -- which is what the Alif port
needs (600 KB -> 150 KB at QVGA). The arithmetic is integer Q8 fixed point and
every intermediate is ``int32``, so it is bit-identical to the C++ port in
``sign_of_life_cpp/src/BackgroundModel.cpp``. ``state_dtype="float32"``
restores the original floating-point state for reference comparisons.

Quantising the mean introduces an update dead zone: ``mean += alpha * delta``
rounds to zero whenever ``|delta| < 1 / (2 * alpha)``, so the mean stops
tracking below that distance. That is harmless only while the dead zone stays
under the smallest detectable distance ``sqrt(threshold * variance_floor)``
(7 grey levels at the defaults), which holds for the pipeline's ``alpha = 0.1``
but not for bgslibrary's ``alpha = 0.005``. :meth:`_warn_on_dead_zone` reports
the violation rather than silently degrading.
"""

from __future__ import annotations

import logging
import math

import numpy as np

from .base import BackgroundSubtractor, prepare_frame, to_mask

logger = logging.getLogger(__name__)

_VARIANCE_FLOOR = 4.0
"""Lower clamp on the per-pixel variance, matching bgslibrary."""

_Q_BITS = 8
"""Fractional bits of the fixed-point representation of alpha and threshold."""

_Q_ONE = 1 << _Q_BITS
"""Fixed-point representation of 1.0."""

_UINT8_MAX = 255
"""Largest value the quantised mean and variance can hold."""

_MAX_THRESHOLD_Q = 1 << 23
"""Ceiling on the quantised threshold.

``threshold_q * variance`` peaks at ``_MAX_THRESHOLD_Q * 255`` = 2.139e9, the
largest such product that still fits an ``int32``. Reaching it needs a
``threshold`` of 32,768 against a bgslibrary default of 12.25.
"""

STATE_DTYPES = ("uint8", "float32")
"""Accepted values of the ``state_dtype`` argument."""


def _round_shift(values: np.ndarray) -> np.ndarray:
    """Divide ``values`` by ``2 ** _Q_BITS``, rounding half away from zero.

    A plain right shift would round towards negative infinity and bias the mean
    downwards; truncation towards zero would widen the dead zone instead. Both
    cost accuracy, so the rounding is done on the magnitude and the sign
    reapplied.

    Args:
        values: ``int32`` array of Q8 fixed-point products.

    Returns:
        The rounded quotients as ``int32``.
    """
    return np.sign(values) * ((np.abs(values) + (_Q_ONE >> 1)) >> _Q_BITS)


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
        state_dtype: str = "uint8",
    ) -> None:
        """Initialise the model.

        Args:
            alpha: Learning rate in ``[0, 1]`` for the mean and variance.
            threshold: Mahalanobis-like multiplier; foreground when the squared
                distance exceeds ``threshold * variance``.
            init_variance: Initial per-pixel variance and basis of its ceiling.
            learning_frames: Frames during which every pixel updates.
            state_dtype: ``"uint8"`` for the quantised state (the default, and
                what the C++ port implements) or ``"float32"`` for the original
                floating-point state.

        Raises:
            ValueError: If ``alpha`` is outside ``[0, 1]``, a magnitude is
                non-positive, or ``state_dtype`` is not in :data:`STATE_DTYPES`.
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must lie in [0, 1]")
        if threshold <= 0 or init_variance <= 0:
            raise ValueError("threshold and init_variance must be positive")
        if state_dtype not in STATE_DTYPES:
            raise ValueError(
                f"unknown state_dtype {state_dtype!r}; expected one of {STATE_DTYPES}"
            )
        self._alpha = float(alpha)
        self._threshold = float(threshold)
        self._init_variance = float(init_variance)
        self._learning_frames = int(learning_frames)
        self._state_dtype = state_dtype
        self._mean: np.ndarray | None = None
        self._variance: np.ndarray | None = None
        self._frame_index = 0

        if state_dtype == "uint8":
            self._quantise_parameters()
            self._warn_on_dead_zone()

    def _quantise_parameters(self) -> None:
        """Derive the integer parameters driving the ``uint8`` state.

        The variance ceiling is what forces the clamping: ``5 * init_variance``
        is 180 at the default but overflows a ``uint8`` for any
        ``init_variance > 51``. Capping it keeps the state representable and the
        warning names the discrepancy rather than hiding it.
        """
        self._alpha_q = min(int(round(self._alpha * _Q_ONE)), _Q_ONE)
        self._threshold_q = min(int(round(self._threshold * _Q_ONE)), _MAX_THRESHOLD_Q)
        if int(round(self._threshold * _Q_ONE)) > _MAX_THRESHOLD_Q:
            logger.warning(
                "threshold=%.2f exceeds what the int32 comparison can hold; "
                "clamped to %.2f",
                self._threshold,
                _MAX_THRESHOLD_Q / _Q_ONE,
            )
        self._variance_floor_q = int(_VARIANCE_FLOOR)

        ceiling = int(round(5.0 * self._init_variance))
        self._variance_ceil_q = max(
            min(ceiling, _UINT8_MAX), self._variance_floor_q
        )
        if ceiling > _UINT8_MAX:
            logger.warning(
                "variance ceiling %d exceeds the uint8 state; clamped to %d "
                "(init_variance=%.1f; use state_dtype='float32' or "
                "init_variance <= 51 to avoid the cap)",
                ceiling,
                self._variance_ceil_q,
                self._init_variance,
            )

        self._init_variance_q = max(
            min(int(round(self._init_variance)), self._variance_ceil_q),
            self._variance_floor_q,
        )

    def _warn_on_dead_zone(self) -> None:
        """Warn when quantisation stops the mean from tracking the scene.

        With an integer mean the update ``alpha * delta`` rounds to zero for
        ``|delta| < 1 / (2 * alpha)``. The model stays usable while that dead
        zone is narrower than the smallest distance it can flag as foreground.
        """
        if self._alpha <= 0.0:
            dead_zone = math.inf
        else:
            dead_zone = 1.0 / (2.0 * self._alpha)
        detectable = math.sqrt(self._threshold * _VARIANCE_FLOOR)
        if dead_zone >= detectable:
            logger.warning(
                "alpha=%.4f gives a quantisation dead zone of %.1f grey levels, "
                "wider than the %.1f-level detection floor: the uint8 mean will "
                "stop tracking the scene. Use alpha > %.4f or "
                "state_dtype='float32'",
                self._alpha,
                dead_zone,
                detectable,
                1.0 / (2.0 * detectable),
            )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Classify ``frame`` against the Gaussian and update the parameters."""
        prepared = prepare_frame(frame)
        if self._state_dtype == "float32":
            return self._apply_float32(prepared)
        # The uint8 state can only hold whole grey levels in [0, 255]; frames
        # are uint8 by contract, so this is a no-op for every real input.
        return self._apply_uint8(
            np.clip(np.rint(prepared), 0, _UINT8_MAX).astype(np.int32)
        )

    def _apply_uint8(self, current: np.ndarray) -> np.ndarray:
        """Run the quantised-state update on an ``int32`` view of the frame.

        Every intermediate is computed in ``int32``: the deltas are signed and
        the squared distances reach 195,075 at ``C = 3``, so ``uint8`` (or even
        ``uint16``) arithmetic would wrap. The widest product below is
        ``alpha_q * (squared_distance - variance)`` at 49.9 M, which leaves a
        43x margin under ``INT32_MAX``.

        Args:
            current: Frame as ``int32`` of shape ``(H, W, C)``, values 0-255.

        Returns:
            A ``uint8`` foreground mask with values ``0``/``255``.
        """
        if self._mean is None or self._mean.shape != current.shape:
            self._mean = current.astype(np.uint8)
            self._variance = np.full(
                current.shape[:2], self._init_variance_q, np.uint8
            )
            self._frame_index = 1
            return np.zeros(current.shape[:2], dtype=np.uint8)

        assert self._variance is not None
        mean = self._mean.astype(np.int32)
        variance = self._variance.astype(np.int32)

        delta = current - mean
        squared_distance = np.sum(delta * delta, axis=2)
        # Both sides scaled by 2**_Q_BITS to compare against the Q8 threshold.
        foreground = (squared_distance << _Q_BITS) > self._threshold_q * variance
        mask = to_mask(foreground)

        self._frame_index += 1
        learning = self._frame_index < self._learning_frames
        updatable = (~foreground) | learning

        selector = updatable[:, :, np.newaxis]
        mean += _round_shift(self._alpha_q * delta) * selector
        # mean + alpha * (current - mean) cannot leave [0, 255] for alpha <= 1,
        # but the clamp also covers a rounded-up alpha_q of exactly _Q_ONE.
        self._mean = np.clip(mean, 0, _UINT8_MAX).astype(np.uint8)

        new_variance = variance + _round_shift(
            self._alpha_q * (squared_distance - variance)
        )
        # Clamped before the uint8 store: the raw update reaches 49.9 M.
        new_variance = np.clip(
            new_variance, self._variance_floor_q, self._variance_ceil_q
        )
        self._variance = np.where(updatable, new_variance, variance).astype(np.uint8)
        return mask

    def _apply_float32(self, current: np.ndarray) -> np.ndarray:
        """Run the original floating-point update, kept for reference runs."""
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
        if self._state_dtype == "uint8":
            return self._mean.squeeze()
        return np.clip(self._mean, 0, 255).astype(np.uint8).squeeze()
