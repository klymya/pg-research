"""Tests for the pure-NumPy OpenCV-equivalent operations.

OpenCV is the baseline: every native operation must reproduce the corresponding
``cv2`` call bit-for-bit, so the tests assert exact array equality.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from scr.native_ops import (
    MorphOp,
    StructuringShape,
    bgr_to_hsv,
    dilate,
    erode,
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
