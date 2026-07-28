"""Tests for the Layer 3 structural filter."""

from __future__ import annotations

import numpy as np
import pytest

from scr.blobs import BlobFilter
from scr.config import BlobConfig, Implementation
from scr.utils import FOREGROUND

MASK_SIZE = (120, 160)
"""Size of the synthetic masks used in these tests."""


def _mask_with(*boxes: tuple[int, int, int, int]) -> np.ndarray:
    """Create a mask with the given ``(x, y, w, h)`` rectangles set to foreground."""
    mask = np.zeros(MASK_SIZE, dtype=np.uint8)
    for x, y, width, height in boxes:
        mask[y : y + height, x : x + width] = FOREGROUND
    return mask


class TestAreaGating:
    """Rejection of components outside the configured area range."""

    def test_keeps_a_large_component(self) -> None:
        mask = _mask_with((40, 30, 30, 20))
        blob_filter = BlobFilter(BlobConfig(min_area=100, open_kernel=0, close_kernel=0))

        filtered, blobs = blob_filter.filter(mask)

        assert len(blobs) == 1
        assert blobs[0].bbox == (40, 30, 30, 20)
        assert blobs[0].area == 600
        assert filtered.any()

    def test_drops_a_small_component(self) -> None:
        mask = _mask_with((40, 30, 3, 3))
        blob_filter = BlobFilter(BlobConfig(min_area=100, open_kernel=0, close_kernel=0))

        filtered, blobs = blob_filter.filter(mask)

        assert blobs == ()
        assert not filtered.any()

    def test_drops_a_component_above_max_area(self) -> None:
        mask = _mask_with((10, 10, 100, 90))
        blob_filter = BlobFilter(
            BlobConfig(min_area=10, max_area=500, open_kernel=0, close_kernel=0)
        )

        _, blobs = blob_filter.filter(mask)

        assert blobs == ()

    def test_blobs_are_sorted_by_descending_area(self) -> None:
        mask = _mask_with((10, 10, 20, 20), (80, 60, 40, 30))
        blob_filter = BlobFilter(BlobConfig(min_area=100, open_kernel=0, close_kernel=0))

        _, blobs = blob_filter.filter(mask)

        assert [blob.area for blob in blobs] == sorted(
            (blob.area for blob in blobs), reverse=True
        )


class TestMorphology:
    """Effect of the opening and closing passes."""

    def test_opening_removes_speckle(self) -> None:
        rng = np.random.default_rng(0)
        mask = (rng.random(MASK_SIZE) < 0.02).astype(np.uint8) * FOREGROUND
        blob_filter = BlobFilter(BlobConfig(min_area=1, open_kernel=3, close_kernel=0))

        filtered, _ = blob_filter.filter(mask)

        assert filtered.sum() < mask.sum()

    def test_closing_bridges_a_gap(self) -> None:
        mask = _mask_with((40, 30, 14, 20), (56, 30, 14, 20))
        blob_filter = BlobFilter(BlobConfig(min_area=50, open_kernel=0, close_kernel=5))

        _, blobs = blob_filter.filter(mask)

        assert len(blobs) == 1

    def test_disabled_morphology_keeps_components_separate(self) -> None:
        mask = _mask_with((40, 30, 14, 20), (56, 30, 14, 20))
        blob_filter = BlobFilter(BlobConfig(min_area=50, open_kernel=0, close_kernel=0))

        _, blobs = blob_filter.filter(mask)

        assert len(blobs) == 2


class TestOutputContract:
    """Shape, dtype and value guarantees of the filtered mask."""

    def test_output_is_binary(self) -> None:
        mask = _mask_with((40, 30, 30, 20))

        filtered, _ = BlobFilter().filter(mask)

        assert filtered.dtype == np.uint8
        assert set(np.unique(filtered)).issubset({0, FOREGROUND})

    def test_empty_mask_yields_no_blobs(self) -> None:
        filtered, blobs = BlobFilter().filter(np.zeros(MASK_SIZE, dtype=np.uint8))

        assert blobs == ()
        assert not filtered.any()

    def test_centroid_matches_the_rectangle_centre(self) -> None:
        mask = _mask_with((40, 30, 20, 20))
        blob_filter = BlobFilter(BlobConfig(min_area=100, open_kernel=0, close_kernel=0))

        _, blobs = blob_filter.filter(mask)

        assert blobs[0].centroid == pytest.approx((49.5, 39.5))

    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_supports_both_connectivities(self, connectivity: int) -> None:
        mask = _mask_with((40, 30, 30, 20))
        blob_filter = BlobFilter(
            BlobConfig(
                min_area=100,
                open_kernel=0,
                close_kernel=0,
                connectivity=connectivity,
            )
        )

        _, blobs = blob_filter.filter(mask)

        assert len(blobs) == 1


class TestContour:
    """Per-blob outer contour extraction."""

    def test_blob_carries_a_contour_around_its_bbox(self) -> None:
        mask = _mask_with((40, 30, 30, 20))
        blob_filter = BlobFilter(BlobConfig(min_area=100, open_kernel=0, close_kernel=0))

        _, blobs = blob_filter.filter(mask)
        contour = blobs[0].contour

        assert contour is not None and contour.shape[1] == 2
        x, y, width, height = blobs[0].bbox
        assert contour[:, 0].min() == x
        assert contour[:, 0].max() == x + width - 1
        assert contour[:, 1].min() == y
        assert contour[:, 1].max() == y + height - 1


class TestNativeBackend:
    """The native connected-components path matches the OpenCV path."""

    @staticmethod
    def _config(implementation: Implementation) -> BlobConfig:
        return BlobConfig(
            min_area=100,
            open_kernel=0,
            close_kernel=0,
            implementation=implementation,
        )

    def test_native_and_library_blobs_agree(self) -> None:
        mask = _mask_with((10, 10, 20, 20), (80, 60, 40, 30), (120, 20, 25, 25))

        _, lib_blobs = BlobFilter(self._config(Implementation.LIB)).filter(mask)
        native_filtered, native_blobs = BlobFilter(
            self._config(Implementation.NATIVE)
        ).filter(mask)

        assert [(b.bbox, b.area, b.centroid) for b in native_blobs] == [
            (b.bbox, b.area, b.centroid) for b in lib_blobs
        ]
        assert native_filtered.any()
