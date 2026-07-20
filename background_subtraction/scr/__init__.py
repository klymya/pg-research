"""Prototype of a 4-layer non-NN background subtraction pipeline.

The package implements "Sign of Life" (left-object) detection as four layers:

* Layer 0 - :mod:`scr.preprocessing`: denoising and global frame registration.
* Layer 2 - :mod:`scr.features`: illumination-invariant metrics (LBP,
  chromaticity) plus an HSV shadow veto.
* Layer 1 - :mod:`scr.bgs_model`: a rate-parameterized background model built
  on a swappable backend from :mod:`scr.backends`.
* Layer 3 - :mod:`scr.dual_rate`, :mod:`scr.blobs`, :mod:`scr.fsm`: two models
  at different rates, structural filtering and a dwell-time state machine.

:class:`scr.pipeline.SignOfLifePipeline` chains them together.
"""

from .backends import (
    BGSBackend,
    MedianBackend,
    OpenCVBackend,
    PybgsBackend,
    RunningAverageBackend,
    available_backends,
    create_backend,
)
from .bgs_model import RateParameterizedBGS
from .blobs import BlobFilter
from .config import (
    BGSConfig,
    BlobConfig,
    DenoiseMethod,
    DualRateConfig,
    DwellConfig,
    FeatureConfig,
    FeatureKind,
    PipelineConfig,
    PreprocessConfig,
    RegistrationMethod,
    ShadowConfig,
)
from .dual_rate import DualRateBGS
from .features import (
    ChromaticityExtractor,
    FeatureExtractor,
    GrayscaleExtractor,
    LocalBinaryPatternExtractor,
    ShadowSuppressor,
    compute_lbp,
    create_feature_extractor,
)
from .fsm import StaticObjectFSM
from .pipeline import SignOfLifePipeline
from .preprocessing import Preprocessor
from .structures import (
    Blob,
    BoundingBox,
    DualRateMasks,
    Frame,
    ObjectState,
    PipelineResult,
    TrackedObject,
)

__all__ = [
    "BGSBackend",
    "BGSConfig",
    "Blob",
    "BlobConfig",
    "BlobFilter",
    "BoundingBox",
    "ChromaticityExtractor",
    "DenoiseMethod",
    "DualRateBGS",
    "DualRateConfig",
    "DualRateMasks",
    "DwellConfig",
    "FeatureConfig",
    "FeatureExtractor",
    "FeatureKind",
    "Frame",
    "GrayscaleExtractor",
    "LocalBinaryPatternExtractor",
    "MedianBackend",
    "ObjectState",
    "OpenCVBackend",
    "PipelineConfig",
    "PipelineResult",
    "PreprocessConfig",
    "Preprocessor",
    "PybgsBackend",
    "RateParameterizedBGS",
    "RegistrationMethod",
    "RunningAverageBackend",
    "ShadowConfig",
    "ShadowSuppressor",
    "SignOfLifePipeline",
    "StaticObjectFSM",
    "TrackedObject",
    "available_backends",
    "compute_lbp",
    "create_backend",
    "create_feature_extractor",
]
