"""ViBe background subtraction (Barnich & Van Droogenbroeck, 2011).

Ports bgslibrary's ``ViBe``. Each pixel keeps a set of ``n_samples`` past
intensity samples. A pixel is background when at least ``min_matches`` of its
samples lie within ``radius`` of the current value. Background pixels update
their own sample set and a random neighbour's, both with probability
``1 / subsampling`` — the random, memory-less update that gives ViBe its slow,
spatially-diffused adaptation. bgslibrary defaults: 20 samples, radius 20,
2 minimum matches, subsampling factor 16.

The model is intensity-based; colour frames are reduced to grayscale.
"""

from __future__ import annotations

import numpy as np

from .base import BackgroundSubtractor, prepare_frame, to_grayscale, to_mask

_INIT_NOISE = 10
"""Half-range of the uniform noise added when seeding the sample buffer."""


class ViBe(BackgroundSubtractor):
    """Sample-based non-parametric background model."""

    def __init__(
        self,
        n_samples: int = 20,
        radius: float = 20.0,
        min_matches: int = 2,
        subsampling: int = 16,
        seed: int | None = None,
    ) -> None:
        """Initialise the model.

        Args:
            n_samples: Number of intensity samples stored per pixel.
            radius: Match radius; a sample matches when within this distance.
            min_matches: Samples that must match for a pixel to be background.
            subsampling: Inverse update probability; ``16`` means 1 in 16
                background pixels updates per frame.
            seed: Optional seed for the internal RNG, for reproducibility.

        Raises:
            ValueError: If any argument is outside its valid range.
        """
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1")
        if radius <= 0:
            raise ValueError("radius must be positive")
        if not 1 <= min_matches <= n_samples:
            raise ValueError("min_matches must lie in [1, n_samples]")
        if subsampling < 1:
            raise ValueError("subsampling must be >= 1")

        self._n = int(n_samples)
        self._radius = float(radius)
        self._min_matches = int(min_matches)
        self._phi = int(subsampling)
        self._rng = np.random.default_rng(seed)
        self._samples: np.ndarray | None = None  # (H, W, n_samples)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Classify ``frame`` against the sample sets and update the model."""
        gray = to_grayscale(prepare_frame(frame))
        if self._samples is None or self._samples.shape[:2] != gray.shape:
            self._initialise(gray)
            return np.zeros(gray.shape, dtype=np.uint8)

        within = np.abs(gray[:, :, np.newaxis] - self._samples) <= self._radius
        matches = within.sum(axis=2)
        background = matches >= self._min_matches

        self._update(gray, background)
        return to_mask(~background)

    def reset(self) -> None:
        """Discard the sample sets."""
        self._samples = None

    def get_background_image(self) -> np.ndarray | None:
        """Return the mean of the samples as a ``uint8`` background image."""
        if self._samples is None:
            return None
        return np.clip(self._samples.mean(axis=2), 0, 255).astype(np.uint8)

    def _initialise(self, gray: np.ndarray) -> None:
        """Seed each pixel's sample set from the first frame plus noise."""
        height, width = gray.shape
        noise = self._rng.integers(
            -_INIT_NOISE, _INIT_NOISE, size=(height, width, self._n)
        ).astype(np.float32)
        self._samples = np.clip(gray[:, :, np.newaxis] + noise, 0.0, 255.0)

    def _update(self, gray: np.ndarray, background: np.ndarray) -> None:
        """Randomly refresh sample sets of background pixels and their neighbours."""
        assert self._samples is not None
        height, width = gray.shape

        # Conservative update: only background pixels, each with probability 1/phi.
        self_update = background & (self._rng.random((height, width)) < 1.0 / self._phi)
        rows, cols = np.nonzero(self_update)
        if rows.size:
            slots = self._rng.integers(0, self._n, size=rows.shape)
            self._samples[rows, cols, slots] = gray[rows, cols]

        # Spatial diffusion: propagate the value into a random 8-neighbour.
        diffuse = background & (self._rng.random((height, width)) < 1.0 / self._phi)
        rows, cols = np.nonzero(diffuse)
        if rows.size:
            d_rows = self._rng.integers(-1, 2, size=rows.shape)
            d_cols = self._rng.integers(-1, 2, size=cols.shape)
            n_rows = np.clip(rows + d_rows, 0, height - 1)
            n_cols = np.clip(cols + d_cols, 0, width - 1)
            slots = self._rng.integers(0, self._n, size=rows.shape)
            self._samples[n_rows, n_cols, slots] = gray[rows, cols]
