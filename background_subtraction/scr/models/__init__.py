"""Pure-Python (NumPy) reimplementations of classic BGS algorithms.

These are standalone, dependency-light background-subtraction models ported from
the OpenCV and bgslibrary sources, sharing the
:class:`~scr.models.base.BackgroundSubtractor` interface
(``apply(frame) -> mask`` and ``reset()``). They are provided as reference
implementations; wiring them into the pipeline backends is a separate step.

Available models:

* :class:`FrameDifference`, :class:`StaticFrameDifference` - frame differencing.
* :class:`RunningAverageSubtractor` - exponential running average.
* :class:`AdaptiveMedianSubtractor` - incremental median.
* :class:`RunningGaussianAverage` - single Gaussian per pixel (Wren).
* :class:`MixtureOfGaussians` - Gaussian mixture (Stauffer-Grimson).
* :class:`ViBe` - sample-based non-parametric model.
* :class:`EigenBackground` - PCA-subspace model (Oliver).
"""

from .adaptive_median import AdaptiveMedianSubtractor
from .base import BACKGROUND, FOREGROUND, BackgroundSubtractor
from .eigenbackground import EigenBackground
from .frame_difference import FrameDifference, StaticFrameDifference
from .mixture_of_gaussians import MixtureOfGaussians
from .running_average import RunningAverageSubtractor
from .running_gaussian import RunningGaussianAverage
from .vibe import ViBe

__all__ = [
    "BACKGROUND",
    "FOREGROUND",
    "AdaptiveMedianSubtractor",
    "BackgroundSubtractor",
    "EigenBackground",
    "FrameDifference",
    "MixtureOfGaussians",
    "RunningAverageSubtractor",
    "RunningGaussianAverage",
    "StaticFrameDifference",
    "ViBe",
]
