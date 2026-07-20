"""Tests for the Layer 1 rate-parameterized background model."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import draw_rectangle, make_background

from scr.backends import RunningAverageBackend
from scr.bgs_model import RateParameterizedBGS
from scr.config import BGSConfig, FeatureConfig, FeatureKind, ShadowConfig
from scr.features import GrayscaleExtractor, ShadowSuppressor
from scr.structures import Frame

OBJECT_BBOX = (60, 40, 30, 25)
"""Bounding box of the synthetic object drawn onto the background."""


def _build(learning_rate: float, shadow_enabled: bool = False) -> RateParameterizedBGS:
    """Build a model over the running-average backend.

    Args:
        learning_rate: Absorption rate handed to the model.
        shadow_enabled: Whether the HSV shadow veto is active.

    Returns:
        The assembled model.
    """
    return RateParameterizedBGS(
        learning_rate=learning_rate,
        backend=RunningAverageBackend(threshold=25.0),
        feature_extractor=GrayscaleExtractor(),
        shadow_suppressor=ShadowSuppressor(ShadowConfig(enabled=shadow_enabled)),
    )


class TestConstruction:
    """Construction and configuration of the model."""

    @pytest.mark.parametrize("learning_rate", [-0.01, 1.01])
    def test_rejects_out_of_range_rate(self, learning_rate: float) -> None:
        with pytest.raises(ValueError, match="learning_rate"):
            _build(learning_rate)

    def test_from_config_wires_every_collaborator(self) -> None:
        model = RateParameterizedBGS.from_config(
            BGSConfig(backend="running_average", learning_rate=0.05),
            FeatureConfig(kind=FeatureKind.LBP),
            ShadowConfig(enabled=False),
        )

        assert model.learning_rate == 0.05
        assert isinstance(model.backend, RunningAverageBackend)

    def test_logs_when_backend_ignores_the_rate(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level("INFO"):
            RateParameterizedBGS.from_config(
                BGSConfig(backend="median", learning_rate=0.05)
            )

        assert "adaptation schedule" in caplog.text


class TestApply:
    """Foreground extraction and background absorption."""

    def test_first_frame_yields_empty_mask(self) -> None:
        model = _build(0.01)

        mask = model.apply(Frame(bgr=make_background()))

        assert not mask.any()

    def test_detects_a_new_object(self) -> None:
        background = make_background()
        model = _build(0.01)
        model.apply(Frame(bgr=background))

        mask = model.apply(Frame(bgr=draw_rectangle(background, OBJECT_BBOX)))

        x, y, width, height = OBJECT_BBOX
        assert mask[y : y + height, x : x + width].mean() > 200

    def test_fast_rate_absorbs_the_object(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        model = _build(0.9)
        model.apply(Frame(bgr=background))

        for _ in range(20):
            mask = model.apply(Frame(bgr=occupied))

        assert not mask.any()

    def test_slow_rate_preserves_the_object(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        model = _build(0.001)
        model.apply(Frame(bgr=background))

        for _ in range(20):
            mask = model.apply(Frame(bgr=occupied))

        assert mask.any()

    def test_shadow_veto_removes_darkened_regions(self) -> None:
        background = make_background()
        shadowed = (background.astype(np.float32) * 0.7).astype(np.uint8)
        model = RateParameterizedBGS.from_config(
            BGSConfig(learning_rate=0.001, threshold=10.0),
            FeatureConfig(),
            ShadowConfig(enabled=True),
        )
        model.apply(Frame(bgr=background))

        vetoed = model.apply(Frame(bgr=shadowed))
        model_without_veto = RateParameterizedBGS.from_config(
            BGSConfig(learning_rate=0.001, threshold=10.0),
            FeatureConfig(),
            ShadowConfig(enabled=False),
        )
        model_without_veto.apply(Frame(bgr=background))
        unfiltered = model_without_veto.apply(Frame(bgr=shadowed))

        assert vetoed.sum() < unfiltered.sum()


class TestBackgroundReference:
    """The BGR reference maintained for the shadow veto."""

    def test_is_none_before_the_first_frame(self) -> None:
        assert _build(0.1).background_bgr is None

    def test_tracks_the_scene(self) -> None:
        background = make_background()
        model = _build(1.0)

        model.apply(Frame(bgr=background))
        model.apply(Frame(bgr=background))

        assert model.background_bgr is not None
        np.testing.assert_allclose(model.background_bgr, background, atol=1)

    def test_reset_clears_backend_and_reference(self) -> None:
        model = _build(0.1)
        model.apply(Frame(bgr=make_background()))

        model.reset()

        assert model.background_bgr is None
