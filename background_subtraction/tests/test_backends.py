"""Tests for the background subtraction backend adapters."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FRAME_SIZE, draw_rectangle, make_background

from scr.backends import (
    BACKEND_REGISTRY,
    MedianBackend,
    OpenCVBackend,
    PybgsBackend,
    RunningAverageBackend,
    available_backends,
    create_backend,
)
from scr.config import BGSConfig
from scr.utils import FOREGROUND

OBJECT_BBOX = (60, 40, 30, 25)
"""Bounding box of the synthetic object drawn onto the background."""


def _has_pybgs() -> bool:
    """Whether the optional ``pybgs`` package is importable."""
    try:
        import pybgs  # noqa: F401
    except ImportError:
        return False
    return True


requires_pybgs = pytest.mark.skipif(
    not _has_pybgs(), reason="pybgs is an optional native dependency"
)


class TestRunningAverageBackend:
    """Behaviour of the exponentially weighted running average."""

    def test_first_frame_is_all_background(self) -> None:
        backend = RunningAverageBackend()

        mask = backend.apply(make_background()[:, :, 0], learning_rate=0.1)

        assert mask.shape == FRAME_SIZE
        assert not mask.any()

    def test_detects_a_new_object(self) -> None:
        background = make_background()
        backend = RunningAverageBackend(threshold=25.0)
        backend.apply(background, learning_rate=0.01)

        mask = backend.apply(draw_rectangle(background, OBJECT_BBOX), 0.01)

        x, y, width, height = OBJECT_BBOX
        assert mask[y : y + height, x : x + width].mean() > 0.9 * FOREGROUND

    def test_object_is_absorbed_at_a_high_rate(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        backend = RunningAverageBackend(threshold=25.0)
        backend.apply(background, learning_rate=0.9)

        for _ in range(20):
            mask = backend.apply(occupied, learning_rate=0.9)

        assert not mask.any()

    def test_object_persists_at_a_low_rate(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        backend = RunningAverageBackend(threshold=25.0)
        backend.apply(background, learning_rate=0.001)

        for _ in range(20):
            mask = backend.apply(occupied, learning_rate=0.001)

        assert mask.any()

    def test_reset_clears_the_model(self) -> None:
        backend = RunningAverageBackend()
        backend.apply(make_background(), 0.1)

        backend.reset()

        assert backend.background is None

    def test_background_estimate_is_uint8(self) -> None:
        backend = RunningAverageBackend()
        backend.apply(make_background(), 0.1)

        assert backend.background is not None
        assert backend.background.dtype == np.uint8

    def test_shape_change_reinitialises_the_model(self) -> None:
        backend = RunningAverageBackend()
        backend.apply(make_background(), 0.1)

        mask = backend.apply(make_background(size=(60, 80)), 0.1)

        assert mask.shape == (60, 80)


class TestMedianBackend:
    """Behaviour of the sliding-window temporal median."""

    def test_rejects_zero_history(self) -> None:
        with pytest.raises(ValueError, match="history"):
            MedianBackend(history=0)

    def test_does_not_support_learning_rate(self) -> None:
        assert MedianBackend().supports_learning_rate is False

    def test_detects_a_new_object_after_short_init(self) -> None:
        background = make_background()
        backend = MedianBackend(history=6, threshold=25.0)
        for _ in range(6):
            backend.apply(background, learning_rate=0.0)

        mask = backend.apply(draw_rectangle(background, OBJECT_BBOX), 0.0)

        x, y, width, height = OBJECT_BBOX
        assert mask[y : y + height, x : x + width].mean() > 0.9 * FOREGROUND

    def test_reset_clears_the_window(self) -> None:
        backend = MedianBackend()
        backend.apply(make_background(), 0.0)

        backend.reset()

        assert backend.background is None


class TestOpenCVBackend:
    """Adapter over the OpenCV subtractors."""

    def test_rejects_unknown_algorithm(self) -> None:
        with pytest.raises(ValueError, match="unknown OpenCV algorithm"):
            OpenCVBackend("gmg")

    @pytest.mark.parametrize("algorithm", ["mog2", "knn"])
    def test_detects_a_new_object(self, algorithm: str) -> None:
        background = make_background()
        backend = OpenCVBackend(algorithm, detectShadows=False)
        for _ in range(10):
            backend.apply(background, learning_rate=0.5)

        mask = backend.apply(draw_rectangle(background, OBJECT_BBOX), 0.5)

        x, y, width, height = OBJECT_BBOX
        assert mask[y : y + height, x : x + width].mean() > 0.5 * FOREGROUND

    def test_mask_is_binary(self) -> None:
        background = make_background()
        backend = OpenCVBackend("mog2", detectShadows=True)
        for _ in range(5):
            backend.apply(background, learning_rate=0.5)

        mask = backend.apply((background * 0.6).astype(np.uint8), 0.5)

        assert set(np.unique(mask)).issubset({0, FOREGROUND})

    def test_reset_rebuilds_the_subtractor(self) -> None:
        backend = OpenCVBackend("mog2")
        background = make_background()
        for _ in range(5):
            backend.apply(background, learning_rate=0.5)

        backend.reset()
        mask = backend.apply(draw_rectangle(background, OBJECT_BBOX), 0.5)

        # A freshly reset model treats its first frame as background.
        assert not mask.any()


class TestPybgsBackend:
    """Adapter over the optional ``pybgs`` suite."""

    def test_raises_a_clear_error_when_unavailable(self) -> None:
        if _has_pybgs():
            pytest.skip("pybgs is installed in this environment")
        with pytest.raises(ImportError, match="pybgs is not installed"):
            PybgsBackend("ViBe")

    @requires_pybgs
    def test_rejects_unknown_algorithm(self) -> None:
        with pytest.raises(ValueError, match="no algorithm named"):
            PybgsBackend("NotAnAlgorithm")

    @requires_pybgs
    def test_produces_a_binary_mask(self) -> None:
        background = make_background()
        backend = PybgsBackend("FrameDifference")
        backend.apply(background, learning_rate=0.0)

        mask = backend.apply(draw_rectangle(background, OBJECT_BBOX), 0.0)

        assert set(np.unique(mask)).issubset({0, FOREGROUND})

    def test_does_not_support_learning_rate(self) -> None:
        assert PybgsBackend.supports_learning_rate is False


class TestBackendRegistry:
    """Factory dispatch over :class:`BGSConfig`."""

    def test_lists_registered_names(self) -> None:
        names = available_backends()

        assert "running_average" in names
        assert "mog2" in names
        assert names == tuple(sorted(BACKEND_REGISTRY))

    @pytest.mark.parametrize(
        ("backend", "expected"),
        [
            ("running_average", RunningAverageBackend),
            ("median", MedianBackend),
            ("mog2", OpenCVBackend),
            ("KNN", OpenCVBackend),
        ],
    )
    def test_builds_requested_backend(self, backend: str, expected: type) -> None:
        instance = create_backend(BGSConfig(backend=backend))

        assert isinstance(instance, expected)

    def test_rejects_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="unknown backend"):
            create_backend(BGSConfig(backend="magic"))

    def test_forwards_backend_params(self) -> None:
        config = BGSConfig(backend="median", backend_params={"history": 3})

        backend = create_backend(config)

        assert isinstance(backend, MedianBackend)
