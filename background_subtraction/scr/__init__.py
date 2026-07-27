"""Prototype of a 4-layer non-NN background subtraction pipeline.

The package implements "Sign of Life" (left-object) detection as four layers:

* Layer 0 - :mod:`scr.preprocessing`: denoising and global frame registration.
* Layer 2 - :mod:`scr.features`: illumination-invariant metrics (LBP,
  chromaticity) plus an HSV shadow veto.
* Layer 1 - :mod:`scr.bgs_model`: a rate-parameterized background model built
  on a swappable backend from :mod:`scr.backends`.
* Layer 3 - :mod:`scr.blobs`, :mod:`scr.fsm`: structural filtering of the
  foreground mask and a dwell-time state machine.

:class:`scr.pipeline.SignOfLifePipeline` chains them together.
"""

from .backends import (
    BGSBackend,
    MedianBackend,
    NativeModelBackend,
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
    DwellConfig,
    FeatureConfig,
    FeatureKind,
    Implementation,
    PipelineConfig,
    PreprocessConfig,
    RegistrationMethod,
    ShadowConfig,
)
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
from .native_ops import (
    MorphOp,
    StructuringShape,
    bgr_to_hsv,
    dilate,
    erode,
    get_structuring_element,
    hsv_to_bgr,
    morphology_ex,
)
from .pipeline import SignOfLifePipeline
from .preprocessing import Preprocessor
from .structures import (
    Blob,
    BoundingBox,
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
    "DwellConfig",
    "FeatureConfig",
    "FeatureExtractor",
    "FeatureKind",
    "Frame",
    "GrayscaleExtractor",
    "Implementation",
    "LocalBinaryPatternExtractor",
    "MedianBackend",
    "MorphOp",
    "NativeModelBackend",
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
    "StructuringShape",
    "TrackedObject",
    "available_backends",
    "bgr_to_hsv",
    "compute_lbp",
    "create_backend",
    "create_feature_extractor",
    "dilate",
    "erode",
    "get_structuring_element",
    "hsv_to_bgr",
    "morphology_ex",
]
