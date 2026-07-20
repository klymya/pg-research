"""Shared pytest fixtures and synthetic frame helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scr.structures import BoundingBox, Frame  # noqa: E402

FRAME_SIZE = (120, 160)
"""Synthetic frame size as ``(height, width)``."""


def make_background(
    size: tuple[int, int] = FRAME_SIZE, seed: int = 0
) -> np.ndarray:
    """Create a structured but temporally constant BGR background.

    The scene combines an illumination gradient, two high-contrast landmarks
    and mild sensor noise.  The landmarks matter: gradient-based registration
    cannot lock onto pure i.i.d. noise, so a realistic scene is required to
    exercise Layer 0 meaningfully.

    Args:
        size: Frame size as ``(height, width)``.
        seed: Seed of the noise generator, kept fixed across calls so that the
            same background can be regenerated deterministically.

    Returns:
        A ``uint8`` BGR image.
    """
    height, width = size
    rows, columns = np.mgrid[0:height, 0:width]
    gradient = 60.0 + 0.3 * columns + 0.2 * rows
    image = np.stack([gradient, gradient * 0.9, gradient * 1.1], axis=2)

    image[height // 6 : height // 2, width // 5 : width // 2] += 60.0
    image[int(height * 0.6) : int(height * 0.85), int(width * 0.55) :] += 45.0

    rng = np.random.default_rng(seed)
    image += rng.normal(0.0, 6.0, size=image.shape)
    return np.clip(image, 0, 255).astype(np.uint8)


def draw_rectangle(
    image: np.ndarray, bbox: BoundingBox, color: tuple[int, int, int] = (240, 240, 240)
) -> np.ndarray:
    """Return a copy of ``image`` with a filled rectangle drawn on it.

    Args:
        image: Source BGR image.
        bbox: Rectangle as ``(x, y, width, height)``.
        color: Fill colour in BGR.

    Returns:
        The modified copy.
    """
    x, y, width, height = bbox
    result = image.copy()
    result[y : y + height, x : x + width] = color
    return result


@pytest.fixture
def background() -> np.ndarray:
    """A constant textured BGR background frame."""
    return make_background()


@pytest.fixture
def frame(background: np.ndarray) -> Frame:
    """A :class:`~scr.structures.Frame` wrapping the background."""
    return Frame(bgr=background)
