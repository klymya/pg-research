"""Layer 3b: structural filtering of the static-candidate mask.

Morphological opening removes isolated speckle, closing bridges the gaps a
static object develops as the slow model erodes it, and area gating discards
components too small or too large to be a plausible left object.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import BlobConfig, Implementation
from .native_ops import MorphOp, StructuringShape, get_structuring_element, morphology_ex
from .structures import Blob
from .utils import FOREGROUND, ensure_binary_mask


class BlobFilter:
    """Cleans a binary mask and extracts its area-gated components."""

    def __init__(self, config: BlobConfig | None = None) -> None:
        """Initialise the filter.

        Args:
            config: Structural filtering parameters; defaults are used when
                omitted.
        """
        self._config = config or BlobConfig()

    @property
    def config(self) -> BlobConfig:
        """The active structural filtering configuration."""
        return self._config

    def filter(self, mask: np.ndarray) -> tuple[np.ndarray, tuple[Blob, ...]]:
        """Denoise ``mask`` and return its surviving blobs.

        Args:
            mask: Static-candidate mask with values ``0``/``255``.

        Returns:
            A tuple of the filtered mask - containing only surviving components
            - and those components as :class:`~scr.structures.Blob` objects,
            ordered by descending area.
        """
        cleaned = self._apply_morphology(ensure_binary_mask(mask))

        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            cleaned, connectivity=self._config.connectivity
        )

        output = np.zeros_like(cleaned)
        blobs: list[Blob] = []
        for label in range(1, count):  # Label 0 is the background.
            area = int(stats[label, cv2.CC_STAT_AREA])
            if not self._area_accepted(area):
                continue

            output[labels == label] = FOREGROUND
            blobs.append(
                Blob(
                    bbox=(
                        int(stats[label, cv2.CC_STAT_LEFT]),
                        int(stats[label, cv2.CC_STAT_TOP]),
                        int(stats[label, cv2.CC_STAT_WIDTH]),
                        int(stats[label, cv2.CC_STAT_HEIGHT]),
                    ),
                    area=area,
                    centroid=(float(centroids[label][0]), float(centroids[label][1])),
                )
            )

        blobs.sort(key=lambda blob: blob.area, reverse=True)
        return output, tuple(blobs)

    def _area_accepted(self, area: int) -> bool:
        """Whether ``area`` falls inside the configured gate."""
        if area < self._config.min_area:
            return False
        return self._config.max_area is None or area <= self._config.max_area

    def _apply_morphology(self, mask: np.ndarray) -> np.ndarray:
        """Run the configured opening and closing passes over ``mask``."""
        cfg = self._config
        if cfg.open_kernel:
            mask = self._morphology(mask, "open", cfg.open_kernel)
        if cfg.close_kernel:
            mask = self._morphology(mask, "close", cfg.close_kernel)
        return mask

    def _morphology(self, mask: np.ndarray, operation: str, ksize: int) -> np.ndarray:
        """Apply one elliptical opening/closing via the configured backend."""
        if self._config.implementation is Implementation.NATIVE:
            kernel = get_structuring_element(StructuringShape.ELLIPSE, (ksize, ksize))
            op = MorphOp.OPEN if operation == "open" else MorphOp.CLOSE
            return morphology_ex(mask, op, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        op = cv2.MORPH_OPEN if operation == "open" else cv2.MORPH_CLOSE
        return cv2.morphologyEx(mask, op, kernel)
