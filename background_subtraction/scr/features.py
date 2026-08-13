"""Layer 2: illumination-invariant metrics and shadow suppression.

Two complementary mechanisms are provided:

* :class:`FeatureExtractor` implementations replace raw intensities with a
  representation that is stable under global illumination change; the result is
  what the Layer 1 background model actually models.
* :class:`ShadowSuppressor` is a post-filter vetoing foreground pixels whose
  HSV signature matches a cast shadow rather than a real object.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import cv2
import numpy as np

from .config import FeatureConfig, FeatureKind, Implementation, ShadowConfig
from .native_ops import bgr_to_hsv
from .structures import Frame
from .utils import Q_BITS as _Q_BITS, quantise

_EPSILON = 1e-6

_RATIO_BITS = 16
"""Fractional bits of the value-ratio bounds in the integer veto path."""

_LBP_NEIGHBOURS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
)
"""Clockwise 8-neighbour offsets, in units of the LBP radius."""


def compute_lbp(gray: np.ndarray, radius: int = 1) -> np.ndarray:
    """Compute the 8-neighbour local binary pattern code map.

    Each bit compares a neighbour against the centre pixel, so the code is
    invariant to any monotonic intensity transform - exactly the behaviour of a
    global illumination change.

    Args:
        gray: Single-channel ``uint8`` image.
        radius: Neighbourhood radius in pixels.

    Returns:
        A ``uint8`` code map of the same shape as ``gray``.

    Raises:
        ValueError: If ``gray`` is not single-channel or ``radius`` is < 1.
    """
    if gray.ndim != 2:
        raise ValueError("compute_lbp expects a single-channel image")
    if radius < 1:
        raise ValueError("radius must be >= 1")

    height, width = gray.shape
    padded = cv2.copyMakeBorder(
        gray, radius, radius, radius, radius, cv2.BORDER_REPLICATE
    )
    codes = np.zeros((height, width), dtype=np.uint8)
    for bit, (dy, dx) in enumerate(_LBP_NEIGHBOURS):
        top = radius + dy * radius
        left = radius + dx * radius
        neighbour = padded[top : top + height, left : left + width]
        codes |= (neighbour >= gray).astype(np.uint8) << bit
    return codes


class FeatureExtractor(ABC):
    """Turns a preprocessed frame into the image the BGS backend models."""

    @abstractmethod
    def transform(self, frame: Frame) -> np.ndarray:
        """Compute the differencing metric for ``frame``.

        Args:
            frame: Preprocessed frame.

        Returns:
            A ``uint8`` image, single- or three-channel.
        """


class GrayscaleExtractor(FeatureExtractor):
    """Baseline extractor passing raw grayscale intensities through."""

    def transform(self, frame: Frame) -> np.ndarray:
        """Return the grayscale view of ``frame``."""
        return frame.gray


class LocalBinaryPatternExtractor(FeatureExtractor):
    """Texture-based extractor robust to global illumination change."""

    def __init__(self, radius: int = 1, blur_ksize: int = 0) -> None:
        """Initialise the extractor.

        Args:
            radius: Neighbourhood radius of the LBP operator.
            blur_ksize: Odd kernel size of an optional median smoothing of the
                code map; ``0`` disables it.
        """
        self._radius = radius
        self._blur_ksize = blur_ksize

    def transform(self, frame: Frame) -> np.ndarray:
        """Return the LBP code map of ``frame``."""
        codes = compute_lbp(frame.gray, self._radius)
        if self._blur_ksize:
            codes = cv2.medianBlur(codes, self._blur_ksize)
        return codes


class ChromaticityExtractor(FeatureExtractor):
    """Normalised-RGB extractor discarding the intensity component.

    Dividing each channel by the pixel sum removes the multiplicative term of a
    global brightness change, leaving only the surface colour.
    """

    def transform(self, frame: Frame) -> np.ndarray:
        """Return the normalised-RGB image of ``frame``."""
        channels = frame.bgr.astype(np.float32)
        total = channels.sum(axis=2, keepdims=True) + _EPSILON
        normalised = channels / total * 255.0
        return normalised.astype(np.uint8)


def create_feature_extractor(config: FeatureConfig | None = None) -> FeatureExtractor:
    """Build the feature extractor described by ``config``.

    Args:
        config: Layer 2 feature parameters; defaults are used when omitted.

    Returns:
        The matching :class:`FeatureExtractor` instance.

    Raises:
        ValueError: If the configured feature kind is unknown.
    """
    config = config or FeatureConfig()
    if config.kind is FeatureKind.GRAY:
        return GrayscaleExtractor()
    if config.kind is FeatureKind.LBP:
        return LocalBinaryPatternExtractor(config.lbp_radius, config.lbp_blur_ksize)
    if config.kind is FeatureKind.CHROMATICITY:
        return ChromaticityExtractor()
    raise ValueError(f"unknown feature kind: {config.kind}")


class ShadowSuppressor:
    """Vetoes foreground pixels whose HSV signature matches a cast shadow.

    A shadowed surface keeps its hue and saturation while its brightness drops
    by a bounded factor.  Pixels satisfying all three conditions relative to the
    reference background are removed from the foreground mask.
    """

    def __init__(self, config: ShadowConfig | None = None) -> None:
        """Initialise the suppressor.

        Args:
            config: Shadow veto parameters; defaults are used when omitted.
        """
        self._config = config or ShadowConfig()

    @property
    def config(self) -> ShadowConfig:
        """The active shadow veto configuration."""
        return self._config

    def suppress(
        self,
        mask: np.ndarray,
        frame: Frame,
        background_bgr: np.ndarray | None,
    ) -> np.ndarray:
        """Remove shadow pixels from ``mask``.

        Args:
            mask: Foreground mask with values ``0``/``255``.
            frame: The frame the mask was computed on.
            background_bgr: Current background estimate as a ``uint8`` BGR
                image; the mask is returned unchanged when it is ``None``.

        Returns:
            The mask with shadow pixels set to ``0``.

        Raises:
            ValueError: If the background estimate does not match the frame size.
        """
        if not self._config.enabled or background_bgr is None:
            return mask
        if background_bgr.shape[:2] != frame.shape:
            raise ValueError("background estimate and frame sizes differ")

        return np.where(self._shadow_mask(frame, background_bgr), 0, mask).astype(
            np.uint8
        )

    def shadow_mask(
        self, frame: Frame, background_bgr: np.ndarray | None
    ) -> np.ndarray:
        """Return the shadow classification as a ``uint8`` ``0``/``255`` mask.

        Pixels set to ``255`` are the ones :meth:`suppress` would remove from a
        foreground mask. Useful for visualising and tuning the veto without
        rerunning the model. An all-zero mask is returned when the veto is
        disabled or no background estimate is available yet.

        Args:
            frame: The frame to classify.
            background_bgr: Current background estimate as a ``uint8`` BGR image.

        Returns:
            A ``uint8`` mask of the pixels classified as shadow.

        Raises:
            ValueError: If the background estimate does not match the frame size.
        """
        if not self._config.enabled or background_bgr is None:
            return np.zeros(frame.shape, dtype=np.uint8)
        if background_bgr.shape[:2] != frame.shape:
            raise ValueError("background estimate and frame sizes differ")

        return np.where(self._shadow_mask(frame, background_bgr), 255, 0).astype(
            np.uint8
        )

    def _to_hsv(self, image: np.ndarray) -> np.ndarray:
        """Convert a BGR image to HSV as ``uint8``.

        The conversion is integer fixed point in both implementations -- the
        native path uses reciprocal tables and OpenCV's is equivalent -- so
        neither mode of the veto pays for a floating-point conversion.
        """
        if self._config.implementation is Implementation.NATIVE:
            return bgr_to_hsv(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    def _shadow_mask(self, frame: Frame, background_bgr: np.ndarray) -> np.ndarray:
        """Return the boolean mask of pixels classified as shadow."""
        frame_hsv = self._to_hsv(frame.bgr)
        background_hsv = self._to_hsv(background_bgr)
        if self._config.quantised:
            return self._shadow_mask_integer(frame_hsv, background_hsv)
        return self._shadow_mask_float(frame_hsv, background_hsv)

    def _shadow_mask_float(
        self, frame_hsv: np.ndarray, background_hsv: np.ndarray
    ) -> np.ndarray:
        """Classify shadow pixels in floating point (the original path)."""
        cfg = self._config
        frame_hsv = frame_hsv.astype(np.float32)
        background_hsv = background_hsv.astype(np.float32)

        value_ratio = frame_hsv[:, :, 2] / (background_hsv[:, :, 2] + _EPSILON)
        saturation_delta = np.abs(frame_hsv[:, :, 1] - background_hsv[:, :, 1])
        hue_delta = np.abs(frame_hsv[:, :, 0] - background_hsv[:, :, 0])
        hue_delta = np.minimum(hue_delta, 180.0 - hue_delta)

        return (
            (value_ratio >= cfg.value_ratio_min)
            & (value_ratio <= cfg.value_ratio_max)
            & (saturation_delta <= cfg.saturation_delta_max)
            & (hue_delta <= cfg.hue_delta_max)
        )

    def _shadow_mask_integer(
        self, frame_hsv: np.ndarray, background_hsv: np.ndarray
    ) -> np.ndarray:
        """Classify shadow pixels in integer arithmetic, matching the C++ port.

        The division is turned into two cross-multiplications against the Q8
        bounds, which is where the one behavioural subtlety lives. The float
        form divides by ``V_background + 1e-6``, so at ``V_background = 0`` the
        ratio is either enormous or exactly zero and **both** fall outside
        ``[value_ratio_min, value_ratio_max]`` -- a black background pixel is
        never shadow. Cross-multiplied, ``V_frame * 256 >= min_q * 0`` and
        ``V_frame * 256 <= max_q * 0`` are both satisfied at
        ``V_frame = V_background = 0``, which would silently invert that. The
        explicit guard below keeps the two paths in agreement.

        The ratio bounds are scaled by ``2 ** _RATIO_BITS`` rather than the Q8
        used elsewhere: Q8 would round ``0.40`` to ``102/256 = 0.3984``, a 0.4 %
        shift of the threshold that disagrees with the float path on a
        measurable number of pixels. Q16 puts that error at 6e-6 for free --
        the products still fit ``int32`` with a 128x margin (``255 << 16`` =
        16.7 M against ``INT32_MAX``).
        """
        cfg = self._config
        frame_hsv = frame_hsv.astype(np.int32)
        background_hsv = background_hsv.astype(np.int32)

        value_frame = frame_hsv[:, :, 2]
        value_background = background_hsv[:, :, 2]
        scaled = value_frame << _RATIO_BITS
        ratio_min_q = int(round(cfg.value_ratio_min * (1 << _RATIO_BITS)))
        ratio_max_q = int(round(cfg.value_ratio_max * (1 << _RATIO_BITS)))

        saturation_delta = np.abs(frame_hsv[:, :, 1] - background_hsv[:, :, 1])
        hue_delta = np.abs(frame_hsv[:, :, 0] - background_hsv[:, :, 0])
        hue_delta = np.minimum(hue_delta, 180 - hue_delta)

        return (
            (value_background > 0)
            & (scaled >= ratio_min_q * value_background)
            & (scaled <= ratio_max_q * value_background)
            & (saturation_delta <= int(round(cfg.saturation_delta_max)))
            & (hue_delta <= int(round(cfg.hue_delta_max)))
        )
