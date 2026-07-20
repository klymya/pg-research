"""Tests for the shared mask and geometry helpers."""

from __future__ import annotations

import numpy as np
import pytest

from scr.utils import FOREGROUND, ensure_binary_mask, ensure_three_channels, iou


class TestEnsureBinaryMask:
    """Normalisation of masks to strict ``0``/``255`` values."""

    def test_folds_shadow_pixels_into_background(self) -> None:
        mask = np.array([[0, 127, 255]], dtype=np.uint8)

        np.testing.assert_array_equal(
            ensure_binary_mask(mask), np.array([[0, 0, FOREGROUND]], dtype=np.uint8)
        )

    def test_squeezes_a_trailing_singleton_axis(self) -> None:
        mask = np.full((4, 4, 1), FOREGROUND, dtype=np.uint8)

        assert ensure_binary_mask(mask).shape == (4, 4)

    def test_rejects_multichannel_masks(self) -> None:
        with pytest.raises(ValueError, match="single-channel"):
            ensure_binary_mask(np.zeros((4, 4, 3), dtype=np.uint8))

    def test_output_dtype_is_uint8(self) -> None:
        mask = np.ones((4, 4), dtype=np.float32) * 255

        assert ensure_binary_mask(mask).dtype == np.uint8


class TestEnsureThreeChannels:
    """Channel coercion used before handing images to ``pybgs``."""

    def test_replicates_grayscale(self) -> None:
        gray = np.arange(16, dtype=np.uint8).reshape(4, 4)

        result = ensure_three_channels(gray)

        assert result.shape == (4, 4, 3)
        np.testing.assert_array_equal(result[:, :, 0], gray)

    def test_expands_a_singleton_channel(self) -> None:
        image = np.zeros((4, 4, 1), dtype=np.uint8)

        assert ensure_three_channels(image).shape == (4, 4, 3)

    def test_passes_bgr_through_unchanged(self) -> None:
        image = np.zeros((4, 4, 3), dtype=np.uint8)

        assert ensure_three_channels(image) is image

    def test_rejects_unsupported_shapes(self) -> None:
        with pytest.raises(ValueError, match="unsupported image shape"):
            ensure_three_channels(np.zeros((4, 4, 5), dtype=np.uint8))


class TestIou:
    """Intersection over union of ``(x, y, w, h)`` boxes."""

    def test_identical_boxes_score_one(self) -> None:
        assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)

    def test_disjoint_boxes_score_zero(self) -> None:
        assert iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0

    def test_touching_boxes_score_zero(self) -> None:
        assert iou((0, 0, 10, 10), (10, 0, 10, 10)) == 0.0

    def test_half_overlap(self) -> None:
        # Intersection 50, union 150.
        assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)

    def test_degenerate_box_scores_zero(self) -> None:
        assert iou((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0

    def test_is_symmetric(self) -> None:
        first, second = (0, 0, 10, 10), (5, 5, 10, 10)

        assert iou(first, second) == pytest.approx(iou(second, first))
