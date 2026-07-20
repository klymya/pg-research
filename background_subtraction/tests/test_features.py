"""Tests for Layer 2 feature extraction and shadow suppression."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FRAME_SIZE, make_background

from scr.config import FeatureConfig, FeatureKind, ShadowConfig
from scr.features import (
    ChromaticityExtractor,
    GrayscaleExtractor,
    LocalBinaryPatternExtractor,
    ShadowSuppressor,
    compute_lbp,
    create_feature_extractor,
)
from scr.structures import Frame
from scr.utils import FOREGROUND


class TestComputeLbp:
    """Properties of the local binary pattern operator."""

    def test_rejects_colour_input(self) -> None:
        with pytest.raises(ValueError, match="single-channel"):
            compute_lbp(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_rejects_zero_radius(self) -> None:
        with pytest.raises(ValueError, match="radius"):
            compute_lbp(np.zeros((4, 4), dtype=np.uint8), radius=0)

    def test_shape_and_dtype_are_preserved(self) -> None:
        gray = make_background()[:, :, 0]

        codes = compute_lbp(gray)

        assert codes.shape == gray.shape
        assert codes.dtype == np.uint8

    def test_uniform_region_yields_full_code(self) -> None:
        gray = np.full((8, 8), 100, dtype=np.uint8)

        codes = compute_lbp(gray)

        # Every neighbour equals the centre, so all eight bits are set.
        assert np.all(codes == 255)

    def test_invariant_to_global_brightness_gain(self) -> None:
        rng = np.random.default_rng(1)
        gray = rng.integers(20, 100, size=(32, 32), dtype=np.uint8)
        brightened = np.clip(gray.astype(np.float32) * 1.6, 0, 255).astype(np.uint8)

        np.testing.assert_array_equal(compute_lbp(gray), compute_lbp(brightened))

    @pytest.mark.parametrize("radius", [1, 2, 3])
    def test_supports_multiple_radii(self, radius: int) -> None:
        gray = make_background()[:, :, 0]

        codes = compute_lbp(gray, radius=radius)

        assert codes.shape == gray.shape


class TestExtractors:
    """Behaviour of the concrete feature extractors."""

    def test_grayscale_matches_frame_view(self) -> None:
        frame = Frame(bgr=make_background())

        np.testing.assert_array_equal(
            GrayscaleExtractor().transform(frame), frame.gray
        )

    def test_lbp_extractor_returns_code_map(self) -> None:
        frame = Frame(bgr=make_background())

        codes = LocalBinaryPatternExtractor().transform(frame)

        np.testing.assert_array_equal(codes, compute_lbp(frame.gray))

    def test_lbp_extractor_applies_optional_blur(self) -> None:
        frame = Frame(bgr=make_background())

        blurred = LocalBinaryPatternExtractor(blur_ksize=3).transform(frame)

        assert blurred.shape == frame.shape

    def test_chromaticity_is_invariant_to_brightness_gain(self) -> None:
        bgr = make_background()
        dimmed = (bgr.astype(np.float32) * 0.5).astype(np.uint8)
        extractor = ChromaticityExtractor()

        original = extractor.transform(Frame(bgr=bgr)).astype(int)
        scaled = extractor.transform(Frame(bgr=dimmed)).astype(int)

        # Quantisation of the halved image keeps the ratios only approximately.
        assert np.abs(original - scaled).mean() < 4.0

    def test_chromaticity_handles_black_pixels(self) -> None:
        frame = Frame(bgr=np.zeros((*FRAME_SIZE, 3), dtype=np.uint8))

        result = ChromaticityExtractor().transform(frame)

        assert np.all(result == 0)


class TestCreateFeatureExtractor:
    """Factory dispatch over :class:`FeatureConfig`."""

    @pytest.mark.parametrize(
        ("kind", "expected"),
        [
            (FeatureKind.GRAY, GrayscaleExtractor),
            (FeatureKind.LBP, LocalBinaryPatternExtractor),
            (FeatureKind.CHROMATICITY, ChromaticityExtractor),
        ],
    )
    def test_builds_requested_extractor(self, kind: FeatureKind, expected: type) -> None:
        assert isinstance(create_feature_extractor(FeatureConfig(kind=kind)), expected)

    def test_defaults_to_grayscale(self) -> None:
        assert isinstance(create_feature_extractor(), GrayscaleExtractor)

    def test_rejects_unknown_kind(self) -> None:
        config = FeatureConfig()
        object.__setattr__(config, "kind", "unknown")

        with pytest.raises(ValueError, match="unknown feature kind"):
            create_feature_extractor(config)


class TestShadowSuppressor:
    """Behaviour of the HSV brightness-ratio veto."""

    @staticmethod
    def _full_mask() -> np.ndarray:
        """A mask marking every pixel as foreground."""
        return np.full(FRAME_SIZE, FOREGROUND, dtype=np.uint8)

    def test_removes_uniformly_darkened_regions(self) -> None:
        background = make_background()
        shadowed = (background.astype(np.float32) * 0.7).astype(np.uint8)
        suppressor = ShadowSuppressor()

        result = suppressor.suppress(self._full_mask(), Frame(bgr=shadowed), background)

        assert result.mean() < 0.1 * FOREGROUND

    def test_keeps_genuine_objects(self) -> None:
        background = make_background()
        occluded = background.copy()
        occluded[:, :] = (10, 200, 250)
        suppressor = ShadowSuppressor()

        result = suppressor.suppress(self._full_mask(), Frame(bgr=occluded), background)

        assert np.all(result == FOREGROUND)

    def test_disabled_suppressor_is_a_no_op(self) -> None:
        background = make_background()
        shadowed = (background.astype(np.float32) * 0.7).astype(np.uint8)
        suppressor = ShadowSuppressor(ShadowConfig(enabled=False))
        mask = self._full_mask()

        result = suppressor.suppress(mask, Frame(bgr=shadowed), background)

        np.testing.assert_array_equal(result, mask)

    def test_missing_background_is_a_no_op(self) -> None:
        mask = self._full_mask()

        result = ShadowSuppressor().suppress(mask, Frame(bgr=make_background()), None)

        np.testing.assert_array_equal(result, mask)

    def test_rejects_mismatched_background_size(self) -> None:
        smaller = make_background(size=(60, 80))

        with pytest.raises(ValueError, match="sizes differ"):
            ShadowSuppressor().suppress(
                self._full_mask(), Frame(bgr=make_background()), smaller
            )
