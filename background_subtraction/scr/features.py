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

from .config import FeatureConfig, FeatureKind, ShadowConfig
from .structures import Frame

_EPSILON = 1e-6

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

    def _shadow_mask(self, frame: Frame, background_bgr: np.ndarray) -> np.ndarray:
        """Return the boolean mask of pixels classified as shadow."""
        cfg = self._config
        frame_hsv = frame.hsv.astype(np.float32)
        background_hsv = cv2.cvtColor(background_bgr, cv2.COLOR_BGR2HSV).astype(
            np.float32
        )

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
