"""Tests for the shared data structures."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FRAME_SIZE, make_background

from scr.structures import (
    Blob,
    Frame,
    ObjectState,
    PipelineResult,
    TrackedObject,
)


class TestFrame:
    """Validation and lazy colour-space views."""

    def test_rejects_grayscale_input(self) -> None:
        with pytest.raises(ValueError, match="3-channel"):
            Frame(bgr=np.zeros(FRAME_SIZE, dtype=np.uint8))

    def test_rejects_float_input(self) -> None:
        with pytest.raises(ValueError, match="uint8"):
            Frame(bgr=np.zeros((*FRAME_SIZE, 3), dtype=np.float32))

    def test_gray_view_shape(self) -> None:
        frame = Frame(bgr=make_background())

        assert frame.gray.shape == FRAME_SIZE
        assert frame.gray.dtype == np.uint8

    def test_hsv_view_shape(self) -> None:
        frame = Frame(bgr=make_background())

        assert frame.hsv.shape == (*FRAME_SIZE, 3)

    def test_views_are_cached(self) -> None:
        frame = Frame(bgr=make_background())

        assert frame.gray is frame.gray
        assert frame.hsv is frame.hsv

    def test_shape_property(self) -> None:
        assert Frame(bgr=make_background()).shape == FRAME_SIZE


class TestPipelineResult:
    """Convenience accessors on the per-frame result."""

    @staticmethod
    def _result(events: tuple[TrackedObject, ...]) -> PipelineResult:
        """Build a result carrying ``events``."""
        empty = np.zeros(FRAME_SIZE, dtype=np.uint8)
        return PipelineResult(
            frame=Frame(bgr=make_background()),
            foreground=empty,
            filtered_mask=empty,
            events=events,
        )

    def test_no_events_means_no_signal(self) -> None:
        assert self._result(()).has_sign_of_life is False

    def test_sign_of_life_defaults_to_false(self) -> None:
        assert self._result(()).sign_of_life is False

    def test_events_mean_a_signal(self) -> None:
        track = TrackedObject(
            track_id=0, bbox=(0, 0, 4, 4), area=16, state=ObjectState.CONFIRMED
        )

        assert self._result((track,)).has_sign_of_life is True

    def test_blobs_default_to_empty(self) -> None:
        assert self._result(()).blobs == ()


class TestBlob:
    """Immutability of the blob record."""

    def test_is_frozen(self) -> None:
        blob = Blob(bbox=(0, 0, 4, 4), area=16, centroid=(2.0, 2.0))

        with pytest.raises(AttributeError):
            blob.area = 32  # type: ignore[misc]

    def test_contour_defaults_to_none(self) -> None:
        assert Blob(bbox=(0, 0, 4, 4), area=16, centroid=(2.0, 2.0)).contour is None

    def test_equality_ignores_the_contour(self) -> None:
        # The contour is metadata; two blobs with the same geometry are equal
        # even with different (or array-valued) contours, which also keeps the
        # frozen dataclass hashable.
        base = Blob(bbox=(0, 0, 4, 4), area=16, centroid=(2.0, 2.0))
        with_contour = Blob(
            bbox=(0, 0, 4, 4),
            area=16,
            centroid=(2.0, 2.0),
            contour=np.array([[0, 0], [3, 0], [3, 3], [0, 3]], dtype=np.int32),
        )

        assert base == with_contour
        assert hash(base) == hash(with_contour)
