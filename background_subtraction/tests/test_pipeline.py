"""End-to-end tests of the "Sign of Life" pipeline."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import draw_rectangle, make_background

from scr.config import (
    BGSConfig,
    BlobConfig,
    DenoiseMethod,
    DwellConfig,
    PipelineConfig,
    PreprocessConfig,
    ShadowConfig,
    SignOfLifeConfig,
)
from scr.pipeline import SignOfLifePipeline
from scr.structures import ObjectState, PipelineResult

OBJECT_BBOX = (60, 40, 30, 25)
"""Bounding box of the object deposited into the synthetic scene."""

DWELL_FRAMES = 5
"""Dwell budget used by the end-to-end tests."""


def _config(dwell_frames: int = DWELL_FRAMES) -> PipelineConfig:
    """Build a pipeline configuration tuned for the synthetic scene.

    A single background model absorbs a deposited object at its learning rate,
    so the rate is kept low enough (0.02) that the object stays in the
    foreground long enough for the dwell FSM to confirm it.
    """
    return PipelineConfig(
        preprocess=PreprocessConfig(resize=None, denoise=DenoiseMethod.NONE),
        shadow=ShadowConfig(enabled=False),
        bgs=BGSConfig(learning_rate=0.02, threshold=25.0),
        blob=BlobConfig(min_area=100, open_kernel=3, close_kernel=5),
        dwell=DwellConfig(dwell_frames=dwell_frames, max_missing_frames=3),
    )


def _run(
    pipeline: SignOfLifePipeline,
    frames: list[np.ndarray],
    person_boxes: list[tuple[int, int, int, int]] | None = None,
) -> list[PipelineResult]:
    """Feed ``frames`` through ``pipeline`` and collect the results."""
    boxes = person_boxes or []
    return [pipeline.process(frame, boxes) for frame in frames]


class TestLeftObjectDetection:
    """The nominal abandoned-object scenario."""

    def test_signals_after_the_dwell_time(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        pipeline = SignOfLifePipeline.from_config(_config())

        results = _run(pipeline, [background] * 6 + [occupied] * 25)

        assert any(result.has_sign_of_life for result in results)

    def test_signal_is_raised_once(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        pipeline = SignOfLifePipeline.from_config(_config())

        results = _run(pipeline, [background] * 6 + [occupied] * 30)

        assert sum(len(result.events) for result in results) == 1

    def test_confirmed_track_covers_the_object(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        pipeline = SignOfLifePipeline.from_config(_config())

        results = _run(pipeline, [background] * 6 + [occupied] * 25)
        events = [event for result in results for event in result.events]

        x, y, width, height = OBJECT_BBOX
        detected_x, detected_y, detected_w, detected_h = events[0].bbox
        assert abs(detected_x - x) <= 5
        assert abs(detected_y - y) <= 5
        assert abs(detected_w - width) <= 10
        assert abs(detected_h - height) <= 10

    def test_static_scene_produces_no_signal(self) -> None:
        background = make_background()
        pipeline = SignOfLifePipeline.from_config(_config())

        results = _run(pipeline, [background] * 40)

        assert not any(result.has_sign_of_life for result in results)


class TestSignOfLife:
    """The per-frame area-based Sign-of-Life flag (independent of the FSM)."""

    @staticmethod
    def _with_threshold(area_threshold: int) -> PipelineConfig:
        base = _config()
        return PipelineConfig(
            preprocess=base.preprocess,
            shadow=base.shadow,
            bgs=base.bgs,
            blob=base.blob,
            dwell=base.dwell,
            sign_of_life=SignOfLifeConfig(area_threshold=area_threshold),
        )

    def test_flag_raised_when_a_large_region_appears(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)  # 30x25 = 750 px
        pipeline = SignOfLifePipeline.from_config(self._with_threshold(400))

        results = _run(pipeline, [background] * 3 + [occupied] * 3)

        assert any(result.sign_of_life for result in results)

    def test_flag_stays_false_on_an_empty_scene(self) -> None:
        background = make_background()
        pipeline = SignOfLifePipeline.from_config(self._with_threshold(400))

        results = _run(pipeline, [background] * 8)

        assert not any(result.sign_of_life for result in results)

    def test_threshold_above_the_region_area_suppresses_the_flag(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        pipeline = SignOfLifePipeline.from_config(self._with_threshold(100_000))

        results = _run(pipeline, [background] * 3 + [occupied] * 3)

        assert not any(result.sign_of_life for result in results)
        # ... yet the regions are still detected as blobs.
        assert any(result.blobs for result in results)


class TestPersonSuppression:
    """The human-tracking veto applied at the pipeline level."""

    def test_object_under_a_person_box_is_suppressed(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        pipeline = SignOfLifePipeline.from_config(_config())
        person_box = (55, 35, 40, 35)

        results = _run(pipeline, [background] * 6 + [occupied] * 25, [person_box])

        assert not any(result.has_sign_of_life for result in results)


class TestResize:
    """Layer-0 resizing applied through the full pipeline."""

    @staticmethod
    def _resized_config() -> PipelineConfig:
        """Config that resizes the 120x160 scene to a smaller frame."""
        base = _config()
        return PipelineConfig(
            preprocess=PreprocessConfig(resize=(80, 60), denoise=DenoiseMethod.NONE),
            shadow=base.shadow,
            bgs=base.bgs,
            blob=base.blob,
            dwell=base.dwell,
        )

    def test_results_are_in_resized_coordinates(self) -> None:
        pipeline = SignOfLifePipeline.from_config(self._resized_config())

        result = pipeline.process(make_background())

        assert result.frame.shape == (60, 80)
        assert result.foreground.shape == (60, 80)

    def test_person_box_veto_survives_resize(self) -> None:
        # The person box is given in original 120x160 coordinates; the pipeline
        # must scale it to the 80x60 frame for the veto to still cover the object.
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        pipeline = SignOfLifePipeline.from_config(self._resized_config())
        person_box = (55, 35, 40, 35)  # original-resolution box over the object

        results = _run(pipeline, [background] * 6 + [occupied] * 25, [person_box])

        assert not any(result.has_sign_of_life for result in results)


class TestResultContract:
    """Shape and content guarantees of :class:`PipelineResult`."""

    def test_result_exposes_every_stage(self) -> None:
        background = make_background()
        pipeline = SignOfLifePipeline.from_config(_config())

        result = pipeline.process(background)

        assert result.frame.index == 0
        assert result.foreground.shape == background.shape[:2]
        assert result.filtered_mask.shape == background.shape[:2]
        assert result.blobs == ()
        assert result.has_sign_of_life is False

    def test_frame_indices_increment(self) -> None:
        background = make_background()
        pipeline = SignOfLifePipeline.from_config(_config())

        results = _run(pipeline, [background] * 3)

        assert [result.frame.index for result in results] == [0, 1, 2]

    def test_filtered_mask_is_binary(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        pipeline = SignOfLifePipeline.from_config(_config())

        results = _run(pipeline, [background] * 6 + [occupied] * 10)

        for result in results:
            assert set(np.unique(result.filtered_mask)).issubset({0, 255})

    def test_default_configuration_runs(self) -> None:
        pipeline = SignOfLifePipeline.from_config()

        result = pipeline.process(make_background())

        assert isinstance(result, PipelineResult)


class TestReset:
    """Reset semantics of the whole pipeline."""

    def test_reset_returns_every_stage_to_its_initial_state(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        pipeline = SignOfLifePipeline.from_config(_config())
        _run(pipeline, [background] * 6 + [occupied] * 25)

        pipeline.reset()

        assert pipeline.model.background_bgr is None
        result = pipeline.process(background)

        assert result.frame.index == 0
        assert result.tracked == ()

    def test_signal_can_be_reproduced_after_reset(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        pipeline = SignOfLifePipeline.from_config(_config())
        frames = [background] * 6 + [occupied] * 25
        _run(pipeline, frames)

        pipeline.reset()
        results = _run(pipeline, frames)

        assert sum(len(result.events) for result in results) == 1


class TestStateTransitions:
    """Track lifecycle observed through the pipeline."""

    def test_candidate_precedes_confirmation(self) -> None:
        background = make_background()
        occupied = draw_rectangle(background, OBJECT_BBOX)
        pipeline = SignOfLifePipeline.from_config(_config(dwell_frames=8))

        results = _run(pipeline, [background] * 6 + [occupied] * 25)
        states = [
            {track.state for track in result.tracked}
            for result in results
            if result.tracked
        ]

        assert states[0] == {ObjectState.CANDIDATE}
        assert ObjectState.CONFIRMED in states[-1]


@pytest.mark.parametrize("feature_kind", ["gray", "lbp", "chromaticity"])
def test_pipeline_runs_with_every_feature(feature_kind: str) -> None:
    """Every Layer 2 metric can drive the full pipeline without error."""
    from scr.config import FeatureConfig, FeatureKind

    config = _config()
    config = PipelineConfig(
        preprocess=config.preprocess,
        feature=FeatureConfig(kind=FeatureKind(feature_kind)),
        shadow=config.shadow,
        bgs=config.bgs,
        blob=config.blob,
        dwell=config.dwell,
    )
    pipeline = SignOfLifePipeline.from_config(config)
    background = make_background()

    results = _run(pipeline, [background] * 5 + [draw_rectangle(background, OBJECT_BBOX)] * 10)

    assert len(results) == 15
