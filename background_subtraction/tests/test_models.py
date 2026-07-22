"""Tests for the pure-Python background-subtraction models."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from scr.models import (
    AdaptiveMedianSubtractor,
    BackgroundSubtractor,
    EigenBackground,
    FrameDifference,
    MixtureOfGaussians,
    RunningAverageSubtractor,
    RunningGaussianAverage,
    StaticFrameDifference,
    ViBe,
)
from scr.models.base import prepare_frame, to_grayscale, to_mask

SIZE = (40, 50)
"""Small synthetic frame size keeping the heavier models fast in tests."""

OBJECT = (slice(15, 25), slice(20, 35))
"""Region occupied by the injected synthetic object (150 pixels)."""


def scene(seed: int = 0) -> np.ndarray:
    """Return a static textured BGR background frame."""
    rng = np.random.default_rng(seed)
    return rng.integers(60, 120, (*SIZE, 3), dtype=np.uint8)


def with_object(frame: np.ndarray, value: int = 240) -> np.ndarray:
    """Return a copy of ``frame`` with a bright rectangle drawn on it."""
    result = frame.copy()
    result[OBJECT] = value
    return result


def region_foreground(mask: np.ndarray) -> int:
    """Count foreground pixels inside the object region."""
    return int((mask[OBJECT] > 0).sum())


def outside_foreground(mask: np.ndarray) -> int:
    """Count foreground pixels outside the object region."""
    outside = mask.copy()
    outside[OBJECT] = 0
    return int((outside > 0).sum())


# Factories producing fresh models with fast-converging test parameters.
MODEL_FACTORIES: dict[str, Callable[[], BackgroundSubtractor]] = {
    "frame_difference": lambda: FrameDifference(),
    "static_frame_difference": lambda: StaticFrameDifference(),
    "running_average": lambda: RunningAverageSubtractor(alpha=0.1),
    "adaptive_median": lambda: AdaptiveMedianSubtractor(sampling_rate=1, learning_frames=5),
    "running_gaussian": lambda: RunningGaussianAverage(alpha=0.05, learning_frames=5),
    "mog": lambda: MixtureOfGaussians(alpha=0.05),
    "vibe": lambda: ViBe(seed=0),
    "eigen": lambda: EigenBackground(n_components=3, history_size=5),
}


class TestBaseHelpers:
    """The shared frame/mask coercion helpers."""

    def test_prepare_frame_adds_channel_axis(self) -> None:
        frame = np.zeros(SIZE, dtype=np.uint8)

        prepared = prepare_frame(frame)

        assert prepared.shape == (*SIZE, 1)
        assert prepared.dtype == np.float32

    def test_prepare_frame_rejects_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="unsupported frame shape"):
            prepare_frame(np.zeros((4, 4, 2), dtype=np.uint8))

    def test_prepare_frame_rejects_non_array(self) -> None:
        with pytest.raises(TypeError, match="numpy.ndarray"):
            prepare_frame([[0, 0], [0, 0]])  # type: ignore[arg-type]

    def test_to_grayscale_averages_channels(self) -> None:
        frame = np.dstack(
            [np.full(SIZE, 30.0), np.full(SIZE, 60.0), np.full(SIZE, 90.0)]
        ).astype(np.float32)

        assert np.allclose(to_grayscale(frame), 60.0)

    def test_to_mask_is_zero_or_255(self) -> None:
        mask = to_mask(np.array([[True, False]]))

        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})


@pytest.mark.parametrize("name", sorted(MODEL_FACTORIES))
class TestCommonContract:
    """Contract that every model must satisfy."""

    def test_first_frame_is_all_background(self, name: str) -> None:
        model = MODEL_FACTORIES[name]()

        mask = model.apply(scene())

        assert mask.shape == SIZE
        assert not mask.any()

    def test_output_is_binary_uint8(self, name: str) -> None:
        model = MODEL_FACTORIES[name]()
        background = scene()

        for _ in range(8):
            mask = model.apply(background)
        mask = model.apply(with_object(background))

        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

    def test_static_background_stays_mostly_empty(self, name: str) -> None:
        model = MODEL_FACTORIES[name]()
        background = scene()

        for _ in range(20):
            mask = model.apply(background)

        assert (mask > 0).mean() < 0.05

    def test_detects_a_new_object(self, name: str) -> None:
        model = MODEL_FACTORIES[name]()
        background = scene()
        for _ in range(20):
            model.apply(background)

        mask = model.apply(with_object(background))

        assert region_foreground(mask) > 75  # majority of the 150-pixel region
        assert outside_foreground(mask) < 0.05 * mask.size

    def test_grayscale_input_supported(self, name: str) -> None:
        model = MODEL_FACTORIES[name]()
        gray = scene()[:, :, 0]

        mask = model.apply(gray)

        assert mask.shape == SIZE

    def test_reset_returns_to_initial_state(self, name: str) -> None:
        model = MODEL_FACTORIES[name]()
        background = scene()
        for _ in range(10):
            model.apply(background)

        model.reset()
        mask = model.apply(background)

        assert not mask.any()


class TestFrameDifference:
    """Motion-based semantics of frame differencing."""

    def test_detects_object_on_appearance(self) -> None:
        model = FrameDifference()
        background = scene()
        model.apply(background)

        mask = model.apply(with_object(background))

        assert region_foreground(mask) > 100

    def test_static_object_fades_after_appearance(self) -> None:
        model = FrameDifference()
        background = scene()
        occupied = with_object(background)
        model.apply(background)
        model.apply(occupied)

        mask = model.apply(occupied)  # object no longer moving

        assert region_foreground(mask) == 0

    def test_static_variant_keeps_detecting(self) -> None:
        model = StaticFrameDifference()
        background = scene()
        occupied = with_object(background)
        model.apply(background)

        model.apply(occupied)
        mask = model.apply(occupied)

        assert region_foreground(mask) > 100

    def test_rejects_negative_threshold(self) -> None:
        with pytest.raises(ValueError, match="threshold"):
            FrameDifference(threshold=-1.0)


class TestRunningAverage:
    """Running-average specifics."""

    def test_fast_rate_absorbs_object(self) -> None:
        model = RunningAverageSubtractor(alpha=0.6)
        background = scene()
        occupied = with_object(background)
        model.apply(background)

        for _ in range(25):
            mask = model.apply(occupied)

        assert region_foreground(mask) < 30

    def test_selective_update_preserves_object_longer(self) -> None:
        background = scene()
        occupied = with_object(background)
        plain = RunningAverageSubtractor(alpha=0.3, selective=False)
        selective = RunningAverageSubtractor(alpha=0.3, selective=True)
        for model in (plain, selective):
            model.apply(background)

        for _ in range(15):
            plain_mask = plain.apply(occupied)
            selective_mask = selective.apply(occupied)

        assert region_foreground(selective_mask) >= region_foreground(plain_mask)

    def test_background_image_available(self) -> None:
        model = RunningAverageSubtractor()
        model.apply(scene())

        background = model.get_background_image()

        assert background is not None
        assert background.shape == (*SIZE, 3)

    @pytest.mark.parametrize("alpha", [-0.1, 1.1])
    def test_rejects_out_of_range_alpha(self, alpha: float) -> None:
        with pytest.raises(ValueError, match="alpha"):
            RunningAverageSubtractor(alpha=alpha)


class TestAdaptiveMedian:
    """Adaptive-median specifics."""

    def test_rejects_zero_sampling_rate(self) -> None:
        with pytest.raises(ValueError, match="sampling_rate"):
            AdaptiveMedianSubtractor(sampling_rate=0)

    def test_median_converges_toward_background(self) -> None:
        model = AdaptiveMedianSubtractor(sampling_rate=1)
        background = scene()
        for _ in range(60):
            model.apply(background)

        estimate = model.get_background_image()

        assert estimate is not None
        assert np.abs(estimate.astype(int) - background.astype(int)).mean() < 5


class TestRunningGaussian:
    """Single-Gaussian specifics."""

    def test_rejects_out_of_range_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha"):
            RunningGaussianAverage(alpha=2.0)

    def test_rejects_non_positive_variance(self) -> None:
        with pytest.raises(ValueError, match="init_variance"):
            RunningGaussianAverage(init_variance=0.0)


class TestMixtureOfGaussians:
    """Gaussian-mixture specifics."""

    def test_rejects_zero_modes(self) -> None:
        with pytest.raises(ValueError, match="n_gaussians"):
            MixtureOfGaussians(n_gaussians=0)

    def test_rejects_out_of_range_background_ratio(self) -> None:
        with pytest.raises(ValueError, match="background_ratio"):
            MixtureOfGaussians(background_ratio=1.5)

    def test_handles_bimodal_background(self) -> None:
        # A background that flips between two appearances should be learned as
        # background by both modes and not flagged as foreground.
        model = MixtureOfGaussians(n_gaussians=3, alpha=0.1)
        scene_a = scene(seed=1)
        scene_b = scene(seed=2)
        for index in range(60):
            mask = model.apply(scene_a if index % 2 == 0 else scene_b)

        assert (mask > 0).mean() < 0.15

    def test_weights_stay_normalised(self) -> None:
        model = MixtureOfGaussians()
        background = scene()
        for _ in range(10):
            model.apply(background)

        total = model._weight.sum(axis=2)  # type: ignore[union-attr]
        assert np.allclose(total, 1.0, atol=1e-4)


class TestViBe:
    """ViBe specifics."""

    def test_is_reproducible_with_a_seed(self) -> None:
        background = scene()
        occupied = with_object(background)
        masks = []
        for _ in range(2):
            model = ViBe(seed=42)
            for _ in range(10):
                model.apply(background)
            masks.append(model.apply(occupied))

        np.testing.assert_array_equal(masks[0], masks[1])

    def test_rejects_min_matches_above_samples(self) -> None:
        with pytest.raises(ValueError, match="min_matches"):
            ViBe(n_samples=5, min_matches=6)


class TestEigenBackground:
    """Eigenbackground specifics."""

    def test_trains_after_history_is_full(self) -> None:
        model = EigenBackground(n_components=3, history_size=5)
        background = scene()

        for index in range(5):
            assert model.is_trained is (index >= 5)
            model.apply(background)

        assert model.is_trained is True

    def test_returns_background_until_trained(self) -> None:
        model = EigenBackground(n_components=3, history_size=5)
        background = scene()

        for _ in range(4):
            mask = model.apply(background)
            assert not mask.any()

    def test_rejects_more_components_than_history(self) -> None:
        with pytest.raises(ValueError, match="n_components"):
            EigenBackground(n_components=30, history_size=10)

    def test_rejects_frame_size_change(self) -> None:
        model = EigenBackground(n_components=2, history_size=3)
        model.apply(scene())

        with pytest.raises(ValueError, match="frame size changed"):
            model.apply(np.zeros((10, 10, 3), dtype=np.uint8))
