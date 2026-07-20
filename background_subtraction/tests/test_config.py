"""Tests for configuration validation."""

from __future__ import annotations

import pytest

from scr.config import (
    BGSConfig,
    BlobConfig,
    DualRateConfig,
    DwellConfig,
    FeatureConfig,
    PipelineConfig,
    PreprocessConfig,
    ShadowConfig,
)


class TestPreprocessConfig:
    """Validation of Layer 0 parameters."""

    @pytest.mark.parametrize("kernel_size", [1, 2, 4])
    def test_rejects_invalid_kernel(self, kernel_size: int) -> None:
        with pytest.raises(ValueError, match="kernel_size"):
            PreprocessConfig(kernel_size=kernel_size)

    def test_rejects_non_positive_shift_limit(self) -> None:
        with pytest.raises(ValueError, match="max_shift_px"):
            PreprocessConfig(max_shift_px=0.0)

    def test_accepts_defaults(self) -> None:
        assert PreprocessConfig().kernel_size == 5


class TestFeatureConfig:
    """Validation of Layer 2 feature parameters."""

    def test_rejects_zero_radius(self) -> None:
        with pytest.raises(ValueError, match="lbp_radius"):
            FeatureConfig(lbp_radius=0)

    def test_rejects_even_blur_kernel(self) -> None:
        with pytest.raises(ValueError, match="lbp_blur_ksize"):
            FeatureConfig(lbp_blur_ksize=4)

    def test_allows_disabled_blur(self) -> None:
        assert FeatureConfig(lbp_blur_ksize=0).lbp_blur_ksize == 0


class TestShadowConfig:
    """Validation of the HSV shadow veto parameters."""

    def test_rejects_inverted_ratio_bounds(self) -> None:
        with pytest.raises(ValueError, match="value_ratio_min"):
            ShadowConfig(value_ratio_min=0.9, value_ratio_max=0.5)


class TestBGSConfig:
    """Validation of Layer 1 parameters."""

    @pytest.mark.parametrize("learning_rate", [-0.1, 1.1])
    def test_rejects_out_of_range_learning_rate(self, learning_rate: float) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            BGSConfig(learning_rate=learning_rate)

    def test_rejects_negative_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            BGSConfig(threshold=-1.0)

    def test_backend_params_are_immutable(self) -> None:
        config = BGSConfig(backend_params={"history": 10})
        with pytest.raises(TypeError):
            config.backend_params["history"] = 20  # type: ignore[index]

    def test_backend_params_are_copied_from_source(self) -> None:
        source = {"history": 10}
        config = BGSConfig(backend_params=source)
        source["history"] = 99
        assert config.backend_params["history"] == 10


class TestDualRateConfig:
    """Validation of the dual-rate pairing."""

    def test_rejects_slower_fast_model(self) -> None:
        with pytest.raises(ValueError, match="fast.learning_rate"):
            DualRateConfig(
                fast=BGSConfig(learning_rate=0.001),
                slow=BGSConfig(learning_rate=0.1),
            )

    def test_rejects_negative_warmup(self) -> None:
        with pytest.raises(ValueError, match="warmup_frames"):
            DualRateConfig(warmup_frames=-1)

    def test_defaults_are_ordered(self) -> None:
        config = DualRateConfig()
        assert config.fast.learning_rate > config.slow.learning_rate


class TestBlobConfig:
    """Validation of the structural filter parameters."""

    def test_rejects_max_area_below_min(self) -> None:
        with pytest.raises(ValueError, match="max_area"):
            BlobConfig(min_area=100, max_area=50)

    def test_rejects_even_morphology_kernel(self) -> None:
        with pytest.raises(ValueError, match="open_kernel"):
            BlobConfig(open_kernel=4)

    def test_rejects_unsupported_connectivity(self) -> None:
        with pytest.raises(ValueError, match="connectivity"):
            BlobConfig(connectivity=6)


class TestDwellConfig:
    """Validation of the dwell-time parameters."""

    def test_rejects_zero_dwell(self) -> None:
        with pytest.raises(ValueError, match="dwell_frames"):
            DwellConfig(dwell_frames=0)

    @pytest.mark.parametrize("threshold", [-0.1, 1.5])
    def test_rejects_out_of_range_iou(self, threshold: float) -> None:
        with pytest.raises(ValueError, match="iou_match_threshold"):
            DwellConfig(iou_match_threshold=threshold)


class TestPipelineConfig:
    """Composition of the per-layer configurations."""

    def test_builds_with_defaults(self) -> None:
        config = PipelineConfig()
        assert config.preprocess.kernel_size == 5
        assert config.dual_rate.fast.learning_rate > 0

    def test_defaults_are_not_shared_between_instances(self) -> None:
        assert PipelineConfig().blob is not PipelineConfig().blob
