"""Configuration objects for the 4-layer background subtraction pipeline.

Every layer owns a frozen dataclass so that a pipeline is fully described by a
single :class:`PipelineConfig` instance.  Validation happens in ``__post_init__``
to fail fast at construction time instead of deep inside the frame loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class DenoiseMethod(str, Enum):
    """Spatial filter applied by Layer 0 before any modelling."""

    NONE = "none"
    GAUSSIAN = "gaussian"
    BILATERAL = "bilateral"


class RegistrationMethod(str, Enum):
    """Global frame alignment strategy used to compensate camera drift."""

    NONE = "none"
    PHASE_CORRELATION = "phase_correlation"
    ECC = "ecc"


class FeatureKind(str, Enum):
    """Differencing metric fed to the background model (Layer 2)."""

    GRAY = "gray"
    LBP = "lbp"
    CHROMATICITY = "chromaticity"


@dataclass(frozen=True)
class PreprocessConfig:
    """Layer 0 parameters: resizing, denoising and global frame registration.

    Attributes:
        resize: Target ``(width, height)`` the input is scaled to before any
            other step; ``None`` disables resizing. Defaults to QVGA
            ``(320, 240)``.
        denoise: Spatial filter used to attenuate sensor noise.
        kernel_size: Odd kernel size for the Gaussian/bilateral filter.
        bilateral_sigma_color: Colour sigma of the bilateral filter.
        bilateral_sigma_space: Spatial sigma of the bilateral filter.
        registration: Global alignment method compensating camera drift.
        ecc_iterations: Maximum iterations of the ECC solver.
        ecc_epsilon: Convergence threshold of the ECC solver.
        max_shift_px: Registration estimates larger than this are discarded as
            implausible; the frame is then passed through unwarped.
    """

    resize: tuple[int, int] | None = (320, 240)
    denoise: DenoiseMethod = DenoiseMethod.GAUSSIAN
    kernel_size: int = 5
    bilateral_sigma_color: float = 25.0
    bilateral_sigma_space: float = 7.0
    registration: RegistrationMethod = RegistrationMethod.NONE
    ecc_iterations: int = 50
    ecc_epsilon: float = 1e-4
    max_shift_px: float = 20.0

    def __post_init__(self) -> None:
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer >= 3")
        if self.max_shift_px <= 0:
            raise ValueError("max_shift_px must be positive")
        if self.resize is not None:
            width, height = self.resize
            if width < 1 or height < 1:
                raise ValueError("resize dimensions must be positive")


@dataclass(frozen=True)
class FeatureConfig:
    """Layer 2 parameters selecting the illumination-invariant metric.

    Attributes:
        kind: Feature transform applied before background modelling.
        lbp_radius: Neighbourhood radius of the local binary pattern operator.
        lbp_blur_ksize: Odd kernel size of an optional smoothing pass over the
            LBP code map; ``0`` disables it.
    """

    kind: FeatureKind = FeatureKind.GRAY
    lbp_radius: int = 1
    lbp_blur_ksize: int = 0

    def __post_init__(self) -> None:
        if self.lbp_radius < 1:
            raise ValueError("lbp_radius must be >= 1")
        if self.lbp_blur_ksize and (
            self.lbp_blur_ksize < 3 or self.lbp_blur_ksize % 2 == 0
        ):
            raise ValueError("lbp_blur_ksize must be 0 or an odd integer >= 3")


@dataclass(frozen=True)
class ShadowConfig:
    """HSV brightness-ratio veto suppressing moving-shadow foreground.

    A pixel is classified as shadow when it is darker than the background by a
    bounded factor while keeping a similar hue and saturation.

    Attributes:
        enabled: Whether the veto is applied at all.
        value_ratio_min: Lower bound of ``V_frame / V_background``.
        value_ratio_max: Upper bound of ``V_frame / V_background``.
        saturation_delta_max: Maximum absolute saturation difference (0-255).
        hue_delta_max: Maximum circular hue difference (OpenCV scale, 0-179).
    """

    enabled: bool = True
    value_ratio_min: float = 0.4
    value_ratio_max: float = 0.95
    saturation_delta_max: float = 40.0
    hue_delta_max: float = 12.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.value_ratio_min < self.value_ratio_max:
            raise ValueError("require 0 <= value_ratio_min < value_ratio_max")


class Implementation(str, Enum):
    """Which code path provides a background-subtraction backend.

    ``LIB`` uses the OpenCV/pybgs-backed adapters; ``NATIVE`` uses the
    pure-NumPy reference models in :mod:`scr.models`. The two share the
    :class:`~scr.backends.BGSBackend` interface, so a pipeline can be built with
    either without changing any other stage.
    """

    LIB = "lib"
    NATIVE = "native"


@dataclass(frozen=True)
class BGSConfig:
    """Layer 1 parameters of a single rate-parameterized background model.

    Attributes:
        backend: Registered backend name, see :func:`scr.backends.available_backends`.
        learning_rate: Absorption rate ``alpha`` in ``[0, 1]``.  Small values
            preserve newly static objects, large values absorb them quickly.
        threshold: Per-channel difference threshold used by the native
            running-average and median backends.
        backend_params: Extra keyword arguments forwarded to the backend.
        implementation: Whether to build the library-backed or the pure-NumPy
            native backend for ``backend``.
    """

    backend: str = "running_average"
    learning_rate: float = 0.01
    threshold: float = 25.0
    backend_params: Mapping[str, Any] = field(default_factory=dict)
    implementation: Implementation = Implementation.LIB

    def __post_init__(self) -> None:
        if not 0.0 <= self.learning_rate <= 1.0:
            raise ValueError("learning_rate must lie in [0, 1]")
        if self.threshold < 0:
            raise ValueError("threshold must be non-negative")
        object.__setattr__(self, "implementation", Implementation(self.implementation))
        object.__setattr__(
            self, "backend_params", MappingProxyType(dict(self.backend_params))
        )


@dataclass(frozen=True)
class DualRateConfig:
    """Layer 3 pair of background models running at different rates.

    Attributes:
        fast: Model absorbing changes quickly, tracking transient motion.
        slow: Model absorbing changes slowly, preserving newly static objects.
        warmup_frames: Number of initial frames during which the static mask is
            forced empty, letting the slow model converge.
    """

    fast: BGSConfig = field(
        default_factory=lambda: BGSConfig(learning_rate=0.1)
    )
    slow: BGSConfig = field(
        default_factory=lambda: BGSConfig(learning_rate=0.001)
    )
    warmup_frames: int = 30

    def __post_init__(self) -> None:
        if self.fast.learning_rate < self.slow.learning_rate:
            raise ValueError(
                "fast.learning_rate must be >= slow.learning_rate"
            )
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames must be non-negative")


@dataclass(frozen=True)
class BlobConfig:
    """Structural filtering of the static-candidate mask.

    Attributes:
        min_area: Blobs smaller than this pixel count are discarded.
        max_area: Blobs larger than this pixel count are discarded; ``None``
            disables the upper bound.
        open_kernel: Odd kernel size of the opening pass; ``0`` disables it.
        close_kernel: Odd kernel size of the closing pass; ``0`` disables it.
        connectivity: Connected-component connectivity, ``4`` or ``8``.
    """

    min_area: int = 150
    max_area: int | None = None
    open_kernel: int = 3
    close_kernel: int = 5
    connectivity: int = 8

    def __post_init__(self) -> None:
        if self.min_area < 0:
            raise ValueError("min_area must be non-negative")
        if self.max_area is not None and self.max_area < self.min_area:
            raise ValueError("max_area must be >= min_area")
        for name in ("open_kernel", "close_kernel"):
            value = getattr(self, name)
            if value and (value < 3 or value % 2 == 0):
                raise ValueError(f"{name} must be 0 or an odd integer >= 3")
        if self.connectivity not in (4, 8):
            raise ValueError("connectivity must be 4 or 8")


@dataclass(frozen=True)
class DwellConfig:
    """Finite state machine gating the "Sign of Life" decision.

    Attributes:
        dwell_frames: Consecutive-presence budget a candidate must accumulate
            before it is confirmed as a left object.
        max_missing_frames: Frames a candidate may stay unmatched before it is
            expired and forgotten.
        iou_match_threshold: Minimum IoU for associating a blob with a track.
        person_overlap_threshold: Minimum IoU with an active person box above
            which a candidate is vetoed as human-related.
    """

    dwell_frames: int = 60
    max_missing_frames: int = 15
    iou_match_threshold: float = 0.3
    person_overlap_threshold: float = 0.1

    def __post_init__(self) -> None:
        if self.dwell_frames < 1:
            raise ValueError("dwell_frames must be >= 1")
        if self.max_missing_frames < 0:
            raise ValueError("max_missing_frames must be non-negative")
        for name in ("iou_match_threshold", "person_overlap_threshold"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")


@dataclass(frozen=True)
class PipelineConfig:
    """Complete description of a "Sign of Life" pipeline instance."""

    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    shadow: ShadowConfig = field(default_factory=ShadowConfig)
    dual_rate: DualRateConfig = field(default_factory=DualRateConfig)
    blob: BlobConfig = field(default_factory=BlobConfig)
    dwell: DwellConfig = field(default_factory=DwellConfig)
