"""Tests for the Layer 3 dual-rate framework."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import draw_rectangle, make_background

from scr.backends import RunningAverageBackend
from scr.bgs_model import RateParameterizedBGS
from scr.config import BGSConfig, DualRateConfig, ShadowConfig
from scr.dual_rate import DualRateBGS
from scr.features import GrayscaleExtractor, ShadowSuppressor
from scr.structures import Frame

OBJECT_BBOX = (60, 40, 30, 25)
"""Bounding box of the synthetic object deposited into the scene."""


def _model(learning_rate: float) -> RateParameterizedBGS:
    """Build a shadow-free running-average model at ``learning_rate``."""
    return RateParameterizedBGS(
        learning_rate=learning_rate,
        backend=RunningAverageBackend(threshold=25.0),
        feature_extractor=GrayscaleExtractor(),
        shadow_suppressor=ShadowSuppressor(ShadowConfig(enabled=False)),
    )


def _stage(warmup_frames: int = 0) -> DualRateBGS:
    """Build a dual-rate stage with well-separated rates."""
    return DualRateBGS(
        fast=_model(0.6), slow=_model(0.0005), warmup_frames=warmup_frames
    )


class TestConstruction:
    """Construction constraints of the dual-rate stage."""

    def test_rejects_inverted_rates(self) -> None:
        with pytest.raises(ValueError, match="fast model"):
            DualRateBGS(fast=_model(0.001), slow=_model(0.1))

    def test_rejects_negative_warmup(self) -> None:
        with pytest.raises(ValueError, match="warmup_frames"):
            DualRateBGS(fast=_model(0.5), slow=_model(0.01), warmup_frames=-1)

    def test_from_config_builds_both_models(self) -> None:
        stage = DualRateBGS.from_config(
            DualRateConfig(
                fast=BGSConfig(learning_rate=0.2), slow=BGSConfig(learning_rate=0.002)
            )
        )

        assert stage.fast.learning_rate == 0.2
        assert stage.slow.learning_rate == 0.002


class TestStaticCue:
    """The ``slow AND NOT fast`` static-candidate rule."""

    def test_deposited_object_appears_in_the_static_mask(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        stage = _stage()
        for _ in range(5):
            stage.apply(Frame(bgr=background))

        for _ in range(15):
            masks = stage.apply(Frame(bgr=occupied))

        x, y, width, height = OBJECT_BBOX
        region = masks.static[y : y + height, x : x + width]
        assert region.mean() > 200
        assert masks.slow[y : y + height, x : x + width].mean() > 200
        assert masks.fast[y : y + height, x : x + width].mean() < 50

    def test_static_region_outside_the_object_stays_empty(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        stage = _stage()
        for _ in range(5):
            stage.apply(Frame(bgr=background))

        for _ in range(15):
            masks = stage.apply(Frame(bgr=occupied))

        outside = masks.static.copy()
        x, y, width, height = OBJECT_BBOX
        outside[y : y + height, x : x + width] = 0
        assert not outside.any()

    def test_static_cue_is_the_and_not_of_the_masks(self) -> None:
        background = make_background()
        stage = _stage()
        stage.apply(Frame(bgr=background))

        masks = stage.apply(Frame(bgr=draw_rectangle(background, OBJECT_BBOX)))

        expected = np.where((masks.slow > 0) & (masks.fast == 0), 255, 0)
        np.testing.assert_array_equal(masks.static, expected.astype(np.uint8))

    def test_static_scene_produces_no_candidates(self) -> None:
        background = make_background()
        stage = _stage()

        for _ in range(10):
            masks = stage.apply(Frame(bgr=background))

        assert not masks.static.any()


class TestWarmup:
    """Suppression of the static cue while the slow model converges."""

    def test_static_mask_is_empty_during_warmup(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        stage = _stage(warmup_frames=5)
        stage.apply(Frame(bgr=background))

        for _ in range(4):
            masks = stage.apply(Frame(bgr=occupied))

        assert stage.in_warmup is False
        assert not masks.static.any()

    def test_static_mask_is_live_after_warmup(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        stage = _stage(warmup_frames=3)
        stage.apply(Frame(bgr=background))

        for _ in range(10):
            masks = stage.apply(Frame(bgr=occupied))

        assert masks.static.any()


class TestReset:
    """Reset semantics of the dual-rate stage."""

    def test_reset_clears_models_and_warmup(self) -> None:
        stage = _stage(warmup_frames=2)
        stage.apply(Frame(bgr=make_background()))

        stage.reset()

        assert stage.in_warmup is True
        assert stage.fast.background_bgr is None
        assert stage.slow.background_bgr is None
