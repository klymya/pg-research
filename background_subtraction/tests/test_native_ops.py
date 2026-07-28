"""Tests for the pure-NumPy OpenCV-equivalent operations.

OpenCV is the baseline: every native operation must reproduce the corresponding
``cv2`` call bit-for-bit, so the tests assert exact array equality.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from scr.native_ops import (
    CC_STAT_AREA,
    MorphOp,
    StructuringShape,
    bgr_to_hsv,
    connected_components,
    dilate,
    erode,
    find_contour,
    get_structuring_element,
    hsv_to_bgr,
    morphology_ex,
)

RNG = np.random.default_rng(20240724)


class TestBgrToHsv:
    """``bgr_to_hsv`` versus ``cv2.cvtColor(..., COLOR_BGR2HSV)``."""

    def test_matches_opencv_on_random_image(self) -> None:
        image = RNG.integers(0, 256, (240, 320, 3), dtype=np.uint8)

        np.testing.assert_array_equal(
            bgr_to_hsv(image), cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        )

    def test_matches_opencv_on_full_value_range(self) -> None:
        # Every (B, G, R) triple below 32^3, tiled into an image, exercises the
        # whole conversion table exhaustively.
        values = np.arange(0, 256, 8, dtype=np.uint8)
        grid = np.array(np.meshgrid(values, values, values)).reshape(3, -1).T
        image = grid.astype(np.uint8).reshape(-1, 1, 3)

        np.testing.assert_array_equal(
            bgr_to_hsv(image), cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        )

    @pytest.mark.parametrize("value", [0, 1, 127, 254, 255])
    def test_matches_opencv_on_gray_frames(self, value: int) -> None:
        image = np.full((8, 8, 3), value, dtype=np.uint8)

        np.testing.assert_array_equal(
            bgr_to_hsv(image), cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        )

    def test_rejects_non_three_channel(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            bgr_to_hsv(np.zeros((4, 4), dtype=np.uint8))

    def test_rejects_non_uint8(self) -> None:
        with pytest.raises(ValueError, match="uint8"):
            bgr_to_hsv(np.zeros((4, 4, 3), dtype=np.float32))


class TestHsvToBgr:
    """``hsv_to_bgr`` versus ``cv2.cvtColor(..., COLOR_HSV2BGR)``.

    The forward path is fixed-point (exact); the inverse is a float conversion,
    so OpenCV is matched to within one LSB per channel -- the standard 8-bit
    HSV<->RGB round-trip tolerance -- not bit-for-bit.
    """

    def test_matches_opencv_within_one_lsb(self) -> None:
        bgr = RNG.integers(0, 256, (240, 320, 3), dtype=np.uint8)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        diff = np.abs(
            hsv_to_bgr(hsv).astype(int)
            - cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(int)
        )

        assert diff.max() <= 1

    def test_hue_wraparound_matches_opencv_within_one_lsb(self) -> None:
        # Arbitrary HSV, including hue above 180, exercises the sextant wrap.
        hsv = RNG.integers(0, 256, (200, 200, 3), dtype=np.uint8)

        diff = np.abs(
            hsv_to_bgr(hsv).astype(int)
            - cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).astype(int)
        )

        assert diff.max() <= 1

    def test_achromatic_pixels_are_exact(self) -> None:
        # Saturation 0 -> gray; this collapses to an exact integer result.
        hue = RNG.integers(0, 180, 400, dtype=np.uint8)
        value = RNG.integers(0, 256, 400, dtype=np.uint8)
        hsv = np.stack([hue, np.zeros(400, np.uint8), value], axis=1).reshape(-1, 1, 3)

        np.testing.assert_array_equal(
            hsv_to_bgr(hsv), cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        )

    def test_round_trip_tracks_opencv_round_trip(self) -> None:
        # 8-bit HSV quantises hue, so a BGR->HSV->BGR round-trip is inherently
        # lossy; what matters is that the native round-trip is as faithful as
        # OpenCV's own -- to within one LSB per channel.
        bgr = RNG.integers(0, 256, (64, 64, 3), dtype=np.uint8)

        native = hsv_to_bgr(bgr_to_hsv(bgr))
        opencv = cv2.cvtColor(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), cv2.COLOR_HSV2BGR
        )

        assert np.abs(native.astype(int) - opencv.astype(int)).max() <= 1

    def test_rejects_non_three_channel(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            hsv_to_bgr(np.zeros((4, 4), dtype=np.uint8))

    def test_rejects_non_uint8(self) -> None:
        with pytest.raises(ValueError, match="uint8"):
            hsv_to_bgr(np.zeros((4, 4, 3), dtype=np.float32))


class TestStructuringElement:
    """``get_structuring_element`` versus ``cv2.getStructuringElement``."""

    @pytest.mark.parametrize(
        ("shape", "cv_shape"),
        [
            (StructuringShape.RECT, cv2.MORPH_RECT),
            (StructuringShape.CROSS, cv2.MORPH_CROSS),
            (StructuringShape.ELLIPSE, cv2.MORPH_ELLIPSE),
        ],
    )
    @pytest.mark.parametrize("size", [(3, 3), (5, 5), (7, 7), (5, 3), (9, 11)])
    def test_matches_opencv(
        self, shape: StructuringShape, cv_shape: int, size: tuple[int, int]
    ) -> None:
        np.testing.assert_array_equal(
            get_structuring_element(shape, size),
            cv2.getStructuringElement(cv_shape, size),
        )

    def test_rejects_degenerate_size(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            get_structuring_element(StructuringShape.ELLIPSE, (0, 3))


class TestMorphology:
    """Erode / dilate / open / close versus their ``cv2`` counterparts."""

    KERNEL_SIZES = [(3, 3), (5, 5), (7, 7)]

    @staticmethod
    def _kernels(size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        native = get_structuring_element(StructuringShape.ELLIPSE, size)
        reference = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)
        return native, reference

    @staticmethod
    def _binary_mask() -> np.ndarray:
        return (RNG.integers(0, 2, (120, 160)) * 255).astype(np.uint8)

    @staticmethod
    def _grayscale() -> np.ndarray:
        return RNG.integers(0, 256, (120, 160), dtype=np.uint8)

    @pytest.mark.parametrize("size", KERNEL_SIZES)
    def test_erode_matches_opencv(self, size: tuple[int, int]) -> None:
        native, reference = self._kernels(size)
        for image in (self._binary_mask(), self._grayscale()):
            np.testing.assert_array_equal(
                erode(image, native), cv2.erode(image, reference)
            )

    @pytest.mark.parametrize("size", KERNEL_SIZES)
    def test_dilate_matches_opencv(self, size: tuple[int, int]) -> None:
        native, reference = self._kernels(size)
        for image in (self._binary_mask(), self._grayscale()):
            np.testing.assert_array_equal(
                dilate(image, native), cv2.dilate(image, reference)
            )

    @pytest.mark.parametrize("size", KERNEL_SIZES)
    @pytest.mark.parametrize(
        ("op", "cv_op"),
        [(MorphOp.OPEN, cv2.MORPH_OPEN), (MorphOp.CLOSE, cv2.MORPH_CLOSE)],
    )
    def test_open_close_match_opencv(
        self, size: tuple[int, int], op: MorphOp, cv_op: int
    ) -> None:
        native, reference = self._kernels(size)
        for image in (self._binary_mask(), self._grayscale()):
            np.testing.assert_array_equal(
                morphology_ex(image, op, native),
                cv2.morphologyEx(image, cv_op, reference),
            )

    def test_border_pixels_match_opencv(self) -> None:
        # A single foreground pixel in the corner stresses the border handling.
        image = np.zeros((20, 20), dtype=np.uint8)
        image[0, 0] = 255
        native, reference = self._kernels((5, 5))

        np.testing.assert_array_equal(
            dilate(image, native), cv2.dilate(image, reference)
        )

    def test_rejects_colour_image(self) -> None:
        native = get_structuring_element(StructuringShape.ELLIPSE, (3, 3))
        with pytest.raises(ValueError, match="single-channel"):
            erode(np.zeros((4, 4, 3), dtype=np.uint8), native)


def _blob_mask() -> np.ndarray:
    """A mask with three well-separated rectangular blobs."""
    mask = np.zeros((60, 80), dtype=np.uint8)
    mask[5:25, 8:33] = 255
    mask[30:40, 50:60] = 255
    mask[45:47, 10:14] = 255
    return mask


class TestConnectedComponents:
    """``connected_components`` versus ``cv2.connectedComponentsWithStats``."""

    @staticmethod
    def _foreground_records(count, stats, centroids):
        """Per-label (stats, rounded centroid) tuples, order-independent."""
        return sorted(
            tuple(int(v) for v in stats[i])
            + (round(float(centroids[i, 0]), 3), round(float(centroids[i, 1]), 3))
            for i in range(1, count)
        )

    @pytest.mark.parametrize("connectivity", [4, 8])
    def test_matches_opencv(self, connectivity: int) -> None:
        mask = _blob_mask()

        count, _, stats, centroids = connected_components(mask, connectivity)
        cv_count, _, cv_stats, cv_centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=connectivity
        )

        assert count == cv_count
        assert self._foreground_records(
            count, stats, centroids
        ) == self._foreground_records(cv_count, cv_stats, cv_centroids)

    def test_matches_opencv_on_random_masks(self) -> None:
        for _ in range(5):
            mask = (RNG.integers(0, 2, (48, 64)) * 255).astype(np.uint8)
            count, _, stats, centroids = connected_components(mask, 8)
            cv_count, _, cv_stats, cv_centroids = cv2.connectedComponentsWithStats(
                mask, connectivity=8
            )
            assert count == cv_count
            assert self._foreground_records(
                count, stats, centroids
            ) == self._foreground_records(cv_count, cv_stats, cv_centroids)

    def test_four_and_eight_connectivity_differ_on_diagonals(self) -> None:
        mask = np.zeros((5, 5), dtype=np.uint8)
        mask[1, 1] = mask[2, 2] = 255  # diagonally touching

        assert connected_components(mask, 8)[0] == 2  # 1 component + background
        assert connected_components(mask, 4)[0] == 3  # 2 components + background

    def test_empty_mask_has_only_background(self) -> None:
        count, labels, _, _ = connected_components(np.zeros((8, 8), np.uint8), 8)
        assert count == 1
        assert not labels.any()

    def test_rejects_colour_image(self) -> None:
        with pytest.raises(ValueError, match="single-channel"):
            connected_components(np.zeros((4, 4, 3), np.uint8))

    def test_rejects_bad_connectivity(self) -> None:
        with pytest.raises(ValueError, match="connectivity"):
            connected_components(np.zeros((4, 4), np.uint8), connectivity=6)


class TestFindContour:
    """``find_contour`` versus ``cv2.findContours`` (external, no approximation)."""

    @staticmethod
    def _rectangle() -> np.ndarray:
        mask = np.zeros((40, 45), np.uint8)
        mask[5:25, 8:33] = 255
        return mask

    @staticmethod
    def _l_shape() -> np.ndarray:
        mask = np.zeros((40, 40), np.uint8)
        mask[5:30, 5:15] = 255  # vertical bar
        mask[20:30, 5:30] = 255  # horizontal foot
        return mask

    @pytest.mark.parametrize("shape", ["rectangle", "l_shape"])
    def test_point_set_matches_opencv(self, shape: str) -> None:
        mask = self._rectangle() if shape == "rectangle" else self._l_shape()
        contour = find_contour(mask)
        cv_contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        cv_points = {(int(p[0][0]), int(p[0][1])) for p in cv_contours[0]}

        assert {(int(x), int(y)) for x, y in contour} == cv_points

    def test_contour_is_a_closed_eight_connected_loop(self) -> None:
        mask = np.zeros((50, 50), np.uint8)
        yy, xx = np.ogrid[:50, :50]
        mask[(yy - 25) ** 2 + (xx - 25) ** 2 <= 15**2] = 255

        contour = find_contour(mask)

        assert len(contour) > 1
        for i in range(len(contour)):
            a, b = contour[i], contour[(i + 1) % len(contour)]
            assert max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1]))) <= 1

    def test_bounding_box_matches_the_region(self) -> None:
        mask = _blob_mask()
        # Trace the first (largest) component in isolation.
        _, labels, _, _ = connected_components(mask, 8)
        component = (labels == 1).astype(np.uint8)

        contour = find_contour(component)
        ys, xs = np.nonzero(component)

        assert contour[:, 0].min() == xs.min()
        assert contour[:, 0].max() == xs.max()
        assert contour[:, 1].min() == ys.min()
        assert contour[:, 1].max() == ys.max()

    def test_single_pixel_returns_one_point(self) -> None:
        mask = np.zeros((10, 10), np.uint8)
        mask[4, 6] = 255

        contour = find_contour(mask)

        np.testing.assert_array_equal(contour, np.array([[6, 4]], dtype=np.int32))

    def test_empty_mask_returns_no_points(self) -> None:
        assert find_contour(np.zeros((6, 6), np.uint8)).shape == (0, 2)

    def test_rejects_colour_image(self) -> None:
        with pytest.raises(ValueError, match="single-channel"):
            find_contour(np.zeros((4, 4, 3), np.uint8))
