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

    def test_shadow_mask_marks_shadow_pixels(self) -> None:
        # The shadow map (255 where shadow) is the complement of what survives
        # suppression, so the two must agree pixel for pixel.
        background = make_background()
        shadowed = (background.astype(np.float32) * 0.7).astype(np.uint8)
        suppressor = ShadowSuppressor()
        frame = Frame(bgr=shadowed)

        shadow = suppressor.shadow_mask(frame, background)
        suppressed = suppressor.suppress(self._full_mask(), frame, background)

        assert set(np.unique(shadow)).issubset({0, 255})
        np.testing.assert_array_equal(shadow > 0, suppressed == 0)

    def test_shadow_mask_is_empty_without_background(self) -> None:
        frame = Frame(bgr=make_background())

        shadow = ShadowSuppressor().shadow_mask(frame, None)

        assert shadow.shape == FRAME_SIZE
        assert not shadow.any()


QUANTISED = ShadowConfig(reference_dtype="uint8")
"""Veto with the quantised reference and integer comparisons."""


class TestQuantisedShadowVeto:
    """The integer comparison path used with a ``uint8`` reference.

    The arithmetic is the same veto with the division cross-multiplied, so the
    two paths must agree everywhere the ratio is well defined -- and must be
    checked explicitly where it is not.
    """

    @staticmethod
    def _full_mask() -> np.ndarray:
        return np.full(FRAME_SIZE, FOREGROUND, dtype=np.uint8)

    def test_rejects_unknown_reference_dtype(self) -> None:
        with pytest.raises(ValueError, match="reference_dtype"):
            ShadowConfig(reference_dtype="int8")

    def test_reference_cost_per_mode(self) -> None:
        assert ShadowConfig(enabled=False).reference_bytes_per_pixel == 0
        assert ShadowConfig(reference_dtype="float32").reference_bytes_per_pixel == 12
        assert QUANTISED.reference_bytes_per_pixel == 3
        assert QUANTISED.quantised
        assert not ShadowConfig(reference_dtype="float32").quantised

    def test_removes_uniformly_darkened_regions(self) -> None:
        background = make_background()
        shadowed = (background.astype(np.float32) * 0.7).astype(np.uint8)

        result = ShadowSuppressor(QUANTISED).suppress(
            self._full_mask(), Frame(bgr=shadowed), background
        )

        assert result.mean() < 0.1 * FOREGROUND

    def test_keeps_genuine_objects(self) -> None:
        background = make_background()
        occluded = background.copy()
        occluded[:, :] = (10, 200, 250)

        result = ShadowSuppressor(QUANTISED).suppress(
            self._full_mask(), Frame(bgr=occluded), background
        )

        assert np.all(result == FOREGROUND)

    def test_agrees_with_the_float_path_on_a_grey_ramp(self) -> None:
        # Grey keeps hue and saturation at zero for both frame and background,
        # isolating the value-ratio comparison that the integer path rewrites.
        # Exact agreement is not achievable and not the goal: the float path
        # divides by V_background + 1e-6, so at an exactly-on-the-bound ratio
        # (V_frame/V_background == 0.40) it answers "not shadow" while the
        # cross-multiplied form answers "shadow". Every disagreement must
        # therefore sit on a threshold, which is what this pins.
        values = np.arange(256, dtype=np.uint8)
        frame_grid, background_grid = np.meshgrid(values, values)
        frame = Frame(bgr=np.repeat(frame_grid[:, :, None], 3, axis=2).copy())
        background = np.repeat(background_grid[:, :, None], 3, axis=2).copy()
        config = ShadowConfig()

        integer = ShadowSuppressor(QUANTISED).shadow_mask(frame, background)
        floating = ShadowSuppressor(config).shadow_mask(frame, background)

        differ = integer != floating
        ratio = frame_grid.astype(np.float64) / np.maximum(background_grid, 1)
        on_bound = np.minimum(
            np.abs(ratio - config.value_ratio_min),
            np.abs(ratio - config.value_ratio_max),
        )
        assert differ.mean() < 0.001
        assert np.all(on_bound[differ] < 1e-9)

    def test_black_background_is_never_shadow(self) -> None:
        # The float form divides by V_background + 1e-6, so V_background = 0
        # always falls outside the ratio band. Cross-multiplied, 0 >= 0 and
        # 0 <= 0 would both hold at V_frame = 0 and invert the decision.
        black = np.zeros((*FRAME_SIZE, 3), dtype=np.uint8)
        frame = Frame(bgr=black.copy())

        shadow = ShadowSuppressor(QUANTISED).shadow_mask(frame, black)

        assert not shadow.any()
