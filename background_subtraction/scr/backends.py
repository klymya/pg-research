"""Background subtraction backends behind a single adapter interface.

Three families are exposed through one registry:

* native NumPy baselines (running average, temporal median) that honour the
  learning rate explicitly and match the 1-6 frame initialisation requirement;
* OpenCV's MOG2/KNN subtractors, which accept a per-call learning rate;
* ``pybgs`` algorithms such as ViBe and SuBSENSE, imported lazily because the
  package is an optional native dependency.

Backends that manage their own adaptation schedule report
``supports_learning_rate = False`` so that Layer 1 can surface the fact instead
of silently ignoring ``alpha``.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable

import cv2
import numpy as np

from .config import BGSConfig, Implementation
from .utils import FOREGROUND, ensure_binary_mask, ensure_three_channels

logger = logging.getLogger(__name__)


class BGSBackend(ABC):
    """Adapter over a concrete background subtraction algorithm."""

    #: Whether :meth:`apply` honours the ``learning_rate`` argument.
    supports_learning_rate: bool = True

    @abstractmethod
    def apply(self, image: np.ndarray, learning_rate: float) -> np.ndarray:
        """Update the model with ``image`` and return its foreground mask.

        Args:
            image: Feature image produced by Layer 2, ``uint8``.
            learning_rate: Absorption rate in ``[0, 1]``.

        Returns:
            A ``uint8`` mask with values ``0``/``255``.
        """

    @abstractmethod
    def reset(self) -> None:
        """Discard the learned background model."""


class RunningAverageBackend(BGSBackend):
    """Exponentially weighted running-average background model.

    The background is updated as ``bg = (1 - alpha) * bg + alpha * image`` and a
    pixel is foreground when any channel deviates by more than ``threshold``.
    """

    def __init__(self, threshold: float = 25.0) -> None:
        """Initialise the backend.

        Args:
            threshold: Per-channel absolute difference marking foreground.
        """
        self._threshold = float(threshold)
        self._background: np.ndarray | None = None

    def apply(self, image: np.ndarray, learning_rate: float) -> np.ndarray:
        """Update the running average and return the foreground mask."""
        current = image.astype(np.float32)
        if self._background is None or self._background.shape != current.shape:
            self._background = current.copy()
            return np.zeros(current.shape[:2], dtype=np.uint8)

        difference = np.abs(current - self._background)
        if difference.ndim == 3:
            difference = difference.max(axis=2)
        mask = np.where(difference > self._threshold, FOREGROUND, 0).astype(np.uint8)

        cv2.accumulateWeighted(current, self._background, learning_rate)
        return mask

    def reset(self) -> None:
        """Discard the learned background model."""
        self._background = None

    @property
    def background(self) -> np.ndarray | None:
        """Current background estimate as a ``uint8`` image, if any."""
        if self._background is None:
            return None
        return np.clip(self._background, 0, 255).astype(np.uint8)


class MedianBackend(BGSBackend):
    """Temporal median background model over a sliding frame window.

    The window replaces the learning rate as the adaptation control, so
    ``learning_rate`` is ignored.  A short history (1-6 frames) reproduces the
    fast-initialisation baseline required by the task.
    """

    supports_learning_rate = False

    def __init__(self, history: int = 6, threshold: float = 25.0) -> None:
        """Initialise the backend.

        Args:
            history: Number of frames kept in the sliding window.
            threshold: Per-channel absolute difference marking foreground.

        Raises:
            ValueError: If ``history`` is smaller than 1.
        """
        if history < 1:
            raise ValueError("history must be >= 1")
        self._history = history
        self._threshold = float(threshold)
        self._frames: deque[np.ndarray] = deque(maxlen=history)

    def apply(self, image: np.ndarray, learning_rate: float) -> np.ndarray:
        """Update the window and return the foreground mask."""
        del learning_rate  # The window length controls adaptation instead.
        current = image.astype(np.float32)
        if not self._frames:
            self._frames.append(current)
            return np.zeros(current.shape[:2], dtype=np.uint8)

        background = np.median(np.stack(self._frames), axis=0)
        difference = np.abs(current - background)
        if difference.ndim == 3:
            difference = difference.max(axis=2)
        mask = np.where(difference > self._threshold, FOREGROUND, 0).astype(np.uint8)

        self._frames.append(current)
        return mask

    def reset(self) -> None:
        """Discard the frame window."""
        self._frames.clear()

    @property
    def background(self) -> np.ndarray | None:
        """Median background estimate as a ``uint8`` image, if any."""
        if not self._frames:
            return None
        median = np.median(np.stack(self._frames), axis=0)
        return np.clip(median, 0, 255).astype(np.uint8)


class OpenCVBackend(BGSBackend):
    """Adapter over ``cv2.BackgroundSubtractorMOG2`` and ``...KNN``."""

    _FACTORIES: dict[str, Callable[..., Any]] = {
        "mog2": cv2.createBackgroundSubtractorMOG2,
        "knn": cv2.createBackgroundSubtractorKNN,
    }

    def __init__(self, algorithm: str = "mog2", **params: Any) -> None:
        """Initialise the backend.

        Args:
            algorithm: Either ``"mog2"`` or ``"knn"``.
            **params: Keyword arguments forwarded to the OpenCV factory, e.g.
                ``history``, ``varThreshold`` or ``detectShadows``.

        Raises:
            ValueError: If ``algorithm`` is not supported.
        """
        key = algorithm.lower()
        if key not in self._FACTORIES:
            raise ValueError(
                f"unknown OpenCV algorithm {algorithm!r}; "
                f"expected one of {sorted(self._FACTORIES)}"
            )
        self._algorithm = key
        self._params = dict(params)
        self._subtractor = self._FACTORIES[key](**self._params)

    def apply(self, image: np.ndarray, learning_rate: float) -> np.ndarray:
        """Update the subtractor and return the foreground mask.

        Shadow pixels reported by OpenCV (value ``127``) are folded into the
        background; shadow handling belongs to Layer 2.
        """
        mask = self._subtractor.apply(image, learningRate=learning_rate)
        return ensure_binary_mask(mask)

    def reset(self) -> None:
        """Recreate the underlying subtractor from scratch."""
        self._subtractor = self._FACTORIES[self._algorithm](**self._params)


class PybgsBackend(BGSBackend):
    """Adapter over an algorithm of the ``pybgs`` (bgslibrary) suite.

    ``pybgs`` algorithms manage their own update schedule and expose no
    per-call learning rate, hence ``supports_learning_rate`` is ``False``.
    """

    supports_learning_rate = False

    def __init__(self, algorithm: str = "ViBe", **params: Any) -> None:
        """Initialise the backend.

        Args:
            algorithm: Class name inside the ``pybgs`` module, e.g. ``"ViBe"``,
                ``"SuBSENSE"`` or ``"FrameDifference"``.
            **params: Keyword arguments forwarded to the algorithm constructor.

        Raises:
            ImportError: If ``pybgs`` is not installed.
            ValueError: If the algorithm name is unknown to ``pybgs``.
        """
        self._algorithm = algorithm
        self._params = dict(params)
        self._subtractor = self._create()

    def _create(self) -> Any:
        """Instantiate the underlying ``pybgs`` algorithm."""
        try:
            import pybgs
        except ImportError as error:  # pragma: no cover - environment dependent
            raise ImportError(
                "pybgs is not installed; install the OpenCV C++ development "
                "headers and run `pip install pybgs`"
            ) from error

        factory = getattr(pybgs, self._algorithm, None)
        if factory is None:
            raise ValueError(f"pybgs has no algorithm named {self._algorithm!r}")
        return factory(**self._params)

    def apply(self, image: np.ndarray, learning_rate: float) -> np.ndarray:
        """Update the algorithm and return the foreground mask.

        ``learning_rate`` is accepted for interface symmetry and ignored.
        """
        del learning_rate
        mask = self._subtractor.apply(ensure_three_channels(image))
        return ensure_binary_mask(mask)

    def reset(self) -> None:
        """Recreate the underlying algorithm from scratch."""
        self._subtractor = self._create()


class NativeModelBackend(BGSBackend):
    """Adapter exposing a pure-NumPy :mod:`scr.models` model as a backend.

    The native models set their absorption rate at construction and take no
    per-call learning rate, so this adapter reports
    ``supports_learning_rate = False`` and ignores the argument, exactly like
    :class:`PybgsBackend`. Fast/slow separation therefore comes from the
    constructor parameters (e.g. a different ``alpha``), not from the pipeline's
    per-call ``learning_rate``.
    """

    supports_learning_rate = False

    def __init__(self, model: Any) -> None:
        """Initialise the adapter.

        Args:
            model: A :class:`scr.models.base.BackgroundSubtractor` instance.
        """
        self._model = model

    def apply(self, image: np.ndarray, learning_rate: float) -> np.ndarray:
        """Run the wrapped model and return its foreground mask."""
        del learning_rate  # The model's rate is fixed at construction.
        return ensure_binary_mask(self._model.apply(image))

    def reset(self) -> None:
        """Reset the wrapped model."""
        self._model.reset()


def _make_running_average(config: BGSConfig) -> BGSBackend:
    """Build a :class:`RunningAverageBackend` from ``config``."""
    return RunningAverageBackend(threshold=config.threshold, **config.backend_params)


def _make_median(config: BGSConfig) -> BGSBackend:
    """Build a :class:`MedianBackend` from ``config``."""
    return MedianBackend(threshold=config.threshold, **config.backend_params)


def _make_opencv(algorithm: str) -> Callable[[BGSConfig], BGSBackend]:
    """Return a factory building an :class:`OpenCVBackend` for ``algorithm``."""

    def factory(config: BGSConfig) -> BGSBackend:
        return OpenCVBackend(algorithm, **config.backend_params)

    return factory


def _make_pybgs(algorithm: str) -> Callable[[BGSConfig], BGSBackend]:
    """Return a factory building a :class:`PybgsBackend` for ``algorithm``."""

    def factory(config: BGSConfig) -> BGSBackend:
        return PybgsBackend(algorithm, **config.backend_params)

    return factory


def _make_native(builder: Callable[[BGSConfig], Any]) -> Callable[[BGSConfig], BGSBackend]:
    """Wrap a native-model builder into a :class:`NativeModelBackend` factory.

    Native models live in :mod:`scr.models`, imported lazily so the pure-NumPy
    code is only loaded when a native backend is actually requested.

    Args:
        builder: Callable turning a :class:`BGSConfig` into a native model.

    Returns:
        A factory producing the adapted backend.
    """

    def factory(config: BGSConfig) -> BGSBackend:
        return NativeModelBackend(builder(config))

    return factory


def _native_running_average(config: BGSConfig) -> Any:
    from .models import RunningAverageSubtractor

    return RunningAverageSubtractor(
        alpha=config.learning_rate, threshold=config.threshold, **config.backend_params
    )


def _native_adaptive_median(config: BGSConfig) -> Any:
    from .models import AdaptiveMedianSubtractor

    return AdaptiveMedianSubtractor(threshold=config.threshold, **config.backend_params)


def _native_running_gaussian(config: BGSConfig) -> Any:
    from .models import RunningGaussianAverage

    return RunningGaussianAverage(alpha=config.learning_rate, **config.backend_params)


def _native_mixture_of_gaussians(config: BGSConfig) -> Any:
    from .models import MixtureOfGaussians

    return MixtureOfGaussians(alpha=config.learning_rate, **config.backend_params)


def _native_vibe(config: BGSConfig) -> Any:
    from .models import ViBe

    return ViBe(**config.backend_params)


def _native_frame_difference(config: BGSConfig) -> Any:
    from .models import FrameDifference

    return FrameDifference(threshold=config.threshold, **config.backend_params)


def _native_static_frame_difference(config: BGSConfig) -> Any:
    from .models import StaticFrameDifference

    return StaticFrameDifference(threshold=config.threshold, **config.backend_params)


def _native_eigenbackground(config: BGSConfig) -> Any:
    from .models import EigenBackground

    return EigenBackground(**config.backend_params)


BACKEND_REGISTRY: dict[str, Callable[[BGSConfig], BGSBackend]] = {
    "running_average": _make_running_average,
    "median": _make_median,
    "mog2": _make_opencv("mog2"),
    "knn": _make_opencv("knn"),
    "vibe": _make_pybgs("ViBe"),
    "subsense": _make_pybgs("SuBSENSE"),
    "lobster": _make_pybgs("LOBSTER"),
    "pawcs": _make_pybgs("PAWCS"),
}
"""Library-backed (OpenCV/pybgs) backend name to factory mapping."""

NATIVE_REGISTRY: dict[str, Callable[[BGSConfig], BGSBackend]] = {
    "running_average": _make_native(_native_running_average),
    "adaptive_median": _make_native(_native_adaptive_median),
    "median": _make_native(_native_adaptive_median),  # alias
    "running_gaussian": _make_native(_native_running_gaussian),
    "wren": _make_native(_native_running_gaussian),  # alias
    "mixture_of_gaussians": _make_native(_native_mixture_of_gaussians),
    "mog2": _make_native(_native_mixture_of_gaussians),  # alias
    "mog": _make_native(_native_mixture_of_gaussians),  # alias
    "vibe": _make_native(_native_vibe),
    "frame_difference": _make_native(_native_frame_difference),
    "static_frame_difference": _make_native(_native_static_frame_difference),
    "eigenbackground": _make_native(_native_eigenbackground),
    "eigen": _make_native(_native_eigenbackground),  # alias
}
"""Pure-NumPy native model name to factory mapping (see :mod:`scr.models`)."""


def _registry_for(implementation: Implementation) -> dict[str, Callable[[BGSConfig], BGSBackend]]:
    """Return the backend registry for an implementation."""
    if Implementation(implementation) is Implementation.NATIVE:
        return NATIVE_REGISTRY
    return BACKEND_REGISTRY


def available_backends(
    implementation: Implementation | str = Implementation.LIB,
) -> tuple[str, ...]:
    """List the registered backend names for an implementation.

    Args:
        implementation: ``LIB`` for the OpenCV/pybgs adapters (default), or
            ``NATIVE`` for the pure-NumPy models.

    Returns:
        The sorted backend names accepted by :class:`~scr.config.BGSConfig` for
        that implementation. Library names backed by ``pybgs`` only construct
        successfully when that optional package is installed.
    """
    return tuple(sorted(_registry_for(Implementation(implementation))))


def create_backend(config: BGSConfig) -> BGSBackend:
    """Build the backend described by ``config``.

    Dispatches to the library-backed or the native registry according to
    ``config.implementation``.

    Args:
        config: Layer 1 configuration naming the backend and its parameters.

    Returns:
        The instantiated backend adapter.

    Raises:
        ValueError: If the backend name is not registered for the selected
            implementation.
    """
    registry = _registry_for(config.implementation)
    factory = registry.get(config.backend.lower())
    if factory is None:
        raise ValueError(
            f"unknown {config.implementation.value} backend {config.backend!r}; "
            f"expected one of {available_backends(config.implementation)}"
        )
    return factory(config)
