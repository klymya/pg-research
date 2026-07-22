"""Eigenbackground subtraction (Oliver, Rosario & Pentland, 2000).

Ports bgslibrary's dp ``Eigenbackground``. A subspace of the background is
learned by PCA over the first ``history_size`` frames. Each later frame is
projected onto the subspace and reconstructed; pixels whose reconstruction error
exceeds ``threshold`` are foreground, because the static background reconstructs
well while moving objects (absent from the training frames) do not.

PCA is computed here with :func:`numpy.linalg.svd`, which is mathematically
identical to :class:`sklearn.decomposition.PCA` — the class is a drop-in
replacement for the ``_fit``/``_project`` steps. NumPy is used directly because
the model needs no other scikit-learn machinery. bgslibrary defaults: 10
components, 20 history frames, threshold 225 (= 15^2).

The learned subspace is fixed once training completes, matching the original
non-adaptive formulation.
"""

from __future__ import annotations

import numpy as np

from .base import BackgroundSubtractor, prepare_frame, to_grayscale, to_mask


class EigenBackground(BackgroundSubtractor):
    """PCA-subspace background model over an intensity history."""

    def __init__(
        self,
        n_components: int = 10,
        history_size: int = 20,
        threshold: float = 225.0,
    ) -> None:
        """Initialise the model.

        Args:
            n_components: Number of principal components retained.
            history_size: Frames collected before the subspace is fitted.
            threshold: Squared per-pixel reconstruction error marking foreground.

        Raises:
            ValueError: If ``n_components`` or ``history_size`` < 1, if
                ``n_components`` exceeds ``history_size``, or ``threshold`` < 0.
        """
        if n_components < 1 or history_size < 1:
            raise ValueError("n_components and history_size must be >= 1")
        if n_components > history_size:
            raise ValueError("n_components must be <= history_size")
        if threshold < 0:
            raise ValueError("threshold must be non-negative")

        self._n_components = int(n_components)
        self._history_size = int(history_size)
        self._threshold = float(threshold)

        self._history: list[np.ndarray] = []
        self._shape: tuple[int, int] | None = None
        self._mean_vector: np.ndarray | None = None
        self._components: np.ndarray | None = None  # (k, n_features)

    @property
    def is_trained(self) -> bool:
        """Whether the PCA subspace has been fitted."""
        return self._components is not None

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """Collect a training frame or score ``frame`` against the subspace."""
        gray = to_grayscale(prepare_frame(frame))
        if self._shape is None:
            self._shape = gray.shape
        if gray.shape != self._shape:
            raise ValueError("frame size changed; call reset() before a new size")

        if not self.is_trained:
            self._history.append(gray.ravel())
            if len(self._history) >= self._history_size:
                self._fit()
            return np.zeros(self._shape, dtype=np.uint8)

        return to_mask(self._reconstruction_error(gray) > self._threshold)

    def reset(self) -> None:
        """Discard the history and the learned subspace."""
        self._history = []
        self._shape = None
        self._mean_vector = None
        self._components = None

    def get_background_image(self) -> np.ndarray | None:
        """Return the training mean as a ``uint8`` background image, if fitted."""
        if self._mean_vector is None or self._shape is None:
            return None
        return np.clip(self._mean_vector.reshape(self._shape), 0, 255).astype(np.uint8)

    def _fit(self) -> None:
        """Fit the PCA subspace from the collected history via SVD."""
        data = np.stack(self._history).astype(np.float32)  # (M, n_features)
        self._mean_vector = data.mean(axis=0)
        centered = data - self._mean_vector

        # Right singular vectors are the principal axes of the feature space.
        _, _, right = np.linalg.svd(centered, full_matrices=False)
        components = min(self._n_components, right.shape[0])
        self._components = right[:components]
        self._history = []

    def _reconstruction_error(self, gray: np.ndarray) -> np.ndarray:
        """Return the per-pixel squared reconstruction error for ``gray``."""
        assert self._components is not None and self._mean_vector is not None
        centered = gray.ravel() - self._mean_vector
        projected = self._components @ centered
        reconstructed = self._components.T @ projected + self._mean_vector
        error = gray.ravel() - reconstructed
        return (error * error).reshape(self._shape)
