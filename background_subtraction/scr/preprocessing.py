"""Layer 0: denoising and global frame registration.

The preprocessor removes sensor noise and compensates minor camera drift so
that the downstream background models observe a geometrically stable scene.
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .config import DenoiseMethod, PreprocessConfig, RegistrationMethod
from .structures import Frame

logger = logging.getLogger(__name__)


class Preprocessor:
    """Denoises frames and aligns them to the first observed frame.

    The first processed frame becomes the registration reference; every later
    frame is warped onto it with a translation-only model, which is the
    dominant component of tripod/mount drift.
    """

    def __init__(self, config: PreprocessConfig | None = None) -> None:
        """Initialise the preprocessor.

        Args:
            config: Layer 0 parameters; defaults are used when omitted.
        """
        self._config = config or PreprocessConfig()
        self._reference_gray: np.ndarray | None = None
        self._frame_index = 0

    @property
    def config(self) -> PreprocessConfig:
        """The active Layer 0 configuration."""
        return self._config

    def reset(self) -> None:
        """Drop the registration reference and restart frame numbering."""
        self._reference_gray = None
        self._frame_index = 0

    def process(self, image: np.ndarray) -> Frame:
        """Denoise and register a single BGR frame.

        Args:
            image: Raw ``uint8`` BGR frame of shape ``(H, W, 3)``.

        Returns:
            The preprocessed :class:`~scr.structures.Frame`.

        Raises:
            ValueError: If the image is not a 3-channel ``uint8`` array.
        """
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Preprocessor expects a 3-channel BGR image")
        if image.dtype != np.uint8:
            raise ValueError("Preprocessor expects a uint8 image")

        denoised = self._denoise(image)
        registered = self._register(denoised)

        frame = Frame(bgr=registered, index=self._frame_index)
        self._frame_index += 1
        return frame

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply the configured spatial filter to ``image``."""
        cfg = self._config
        if cfg.denoise is DenoiseMethod.GAUSSIAN:
            return cv2.GaussianBlur(image, (cfg.kernel_size, cfg.kernel_size), 0)
        if cfg.denoise is DenoiseMethod.BILATERAL:
            return cv2.bilateralFilter(
                image,
                cfg.kernel_size,
                cfg.bilateral_sigma_color,
                cfg.bilateral_sigma_space,
            )
        return image.copy()

    def _register(self, image: np.ndarray) -> np.ndarray:
        """Warp ``image`` onto the registration reference frame."""
        if self._config.registration is RegistrationMethod.NONE:
            return image

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self._reference_gray is None:
            self._reference_gray = gray
            return image

        shift = self._estimate_shift(gray)
        if shift is None:
            return image

        dx, dy = shift
        translation = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        return cv2.warpAffine(
            image,
            translation,
            (image.shape[1], image.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _estimate_shift(self, gray: np.ndarray) -> tuple[float, float] | None:
        """Estimate the translation aligning ``gray`` with the reference.

        Args:
            gray: Grayscale view of the current frame.

        Returns:
            The ``(dx, dy)`` translation to apply, or ``None`` when the
            estimate failed or exceeded ``max_shift_px``.
        """
        assert self._reference_gray is not None
        try:
            if self._config.registration is RegistrationMethod.PHASE_CORRELATION:
                (shift_x, shift_y), _ = cv2.phaseCorrelate(
                    self._reference_gray.astype(np.float32),
                    gray.astype(np.float32),
                )
                dx, dy = -shift_x, -shift_y
            else:
                warp = np.eye(2, 3, dtype=np.float32)
                criteria = (
                    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    self._config.ecc_iterations,
                    self._config.ecc_epsilon,
                )
                _, warp = cv2.findTransformECC(
                    self._reference_gray,
                    gray,
                    warp,
                    cv2.MOTION_TRANSLATION,
                    criteria,
                    None,
                    1,
                )
                # ECC returns the warp mapping the reference onto the current
                # frame; negating turns it into the forward correction applied
                # below, matching the phase-correlation branch.
                dx, dy = -float(warp[0, 2]), -float(warp[1, 2])
        except cv2.error:
            logger.debug("Frame registration failed; using the unwarped frame")
            return None

        if not np.isfinite([dx, dy]).all():
            return None
        if max(abs(dx), abs(dy)) > self._config.max_shift_px:
            logger.debug("Discarding implausible shift estimate (%.2f, %.2f)", dx, dy)
            return None
        return dx, dy
