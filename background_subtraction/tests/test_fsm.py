"""Tests for the Layer 3 dwell-time finite state machine."""

from __future__ import annotations

from scr.config import DwellConfig
from scr.fsm import StaticObjectFSM
from scr.structures import Blob, BoundingBox, ObjectState


def _blob(bbox: BoundingBox) -> Blob:
    """Build a blob covering ``bbox`` entirely."""
    x, y, width, height = bbox
    return Blob(
        bbox=bbox,
        area=width * height,
        centroid=(x + width / 2, y + height / 2),
    )


BOX = (40, 30, 20, 20)
"""Bounding box of the synthetic static object."""


class TestDwellPromotion:
    """Promotion of a candidate to the confirmed state."""

    def test_confirms_after_the_dwell_budget(self) -> None:
        fsm = StaticObjectFSM(DwellConfig(dwell_frames=3))

        events = [fsm.update([_blob(BOX)], index)[:] for index in range(3)]

        assert events[0] == () and events[1] == ()
        assert len(events[2]) == 1
        assert events[2][0].state is ObjectState.CONFIRMED

    def test_confirms_only_once(self) -> None:
        fsm = StaticObjectFSM(DwellConfig(dwell_frames=2))
        for index in range(2):
            fsm.update([_blob(BOX)], index)

        later = fsm.update([_blob(BOX)], 2)

        assert later == ()
        assert fsm.tracks[0].state is ObjectState.CONFIRMED

    def test_does_not_confirm_before_the_budget(self) -> None:
        fsm = StaticObjectFSM(DwellConfig(dwell_frames=10))

        for index in range(9):
            assert fsm.update([_blob(BOX)], index) == ()

        assert fsm.tracks[0].state is ObjectState.CANDIDATE


class TestAssociation:
    """Matching of blobs to existing tracks."""

    def test_slightly_moved_blob_keeps_the_track(self) -> None:
        fsm = StaticObjectFSM(DwellConfig(dwell_frames=5, iou_match_threshold=0.3))
        fsm.update([_blob(BOX)], 0)

        fsm.update([_blob((42, 31, 20, 20))], 1)

        assert len(fsm.tracks) == 1
        assert fsm.tracks[0].dwell == 2

    def test_disjoint_blob_spawns_a_new_track(self) -> None:
        fsm = StaticObjectFSM(DwellConfig(dwell_frames=5))
        fsm.update([_blob(BOX)], 0)

        fsm.update([_blob((120, 90, 20, 20))], 1)

        assert len(fsm.tracks) == 2

    def test_each_blob_is_claimed_at_most_once(self) -> None:
        fsm = StaticObjectFSM(DwellConfig(dwell_frames=5, iou_match_threshold=0.1))
        fsm.update([_blob(BOX), _blob((44, 34, 20, 20))], 0)

        fsm.update([_blob(BOX)], 1)

        dwells = sorted(track.dwell for track in fsm.tracks)
        assert dwells == [1, 2]

    def test_track_ids_are_unique(self) -> None:
        fsm = StaticObjectFSM()
        fsm.update([_blob(BOX), _blob((120, 90, 20, 20))], 0)

        ids = [track.track_id for track in fsm.tracks]
        assert len(set(ids)) == len(ids)


class TestExpiry:
    """Eviction of candidates that stop being observed."""

    def test_track_survives_a_short_gap(self) -> None:
        fsm = StaticObjectFSM(DwellConfig(dwell_frames=5, max_missing_frames=3))
        fsm.update([_blob(BOX)], 0)

        for index in range(1, 4):
            fsm.update([], index)

        assert len(fsm.tracks) == 1

    def test_track_is_evicted_after_the_gap_budget(self) -> None:
        fsm = StaticObjectFSM(DwellConfig(dwell_frames=5, max_missing_frames=2))
        fsm.update([_blob(BOX)], 0)

        for index in range(1, 5):
            fsm.update([], index)

        assert fsm.tracks == ()

    def test_missing_counter_resets_on_rematch(self) -> None:
        fsm = StaticObjectFSM(DwellConfig(dwell_frames=5, max_missing_frames=3))
        fsm.update([_blob(BOX)], 0)
        fsm.update([], 1)

        fsm.update([_blob(BOX)], 2)

        assert fsm.tracks[0].missing == 0


class TestPersonVeto:
    """Suppression of candidates that belong to a person."""

    def test_overlapping_blob_does_not_spawn_a_track(self) -> None:
        fsm = StaticObjectFSM(DwellConfig(dwell_frames=2))

        fsm.update([_blob(BOX)], 0, person_boxes=[(38, 28, 24, 24)])

        assert fsm.tracks == ()

    def test_appearing_person_resets_the_dwell(self) -> None:
        fsm = StaticObjectFSM(DwellConfig(dwell_frames=5))
        fsm.update([_blob(BOX)], 0)
        fsm.update([_blob(BOX)], 1)

        fsm.update([_blob(BOX)], 2, person_boxes=[(38, 28, 24, 24)])

        assert fsm.tracks[0].dwell == 0
        assert fsm.tracks[0].vetoed is True

    def test_distant_person_does_not_veto(self) -> None:
        fsm = StaticObjectFSM(DwellConfig(dwell_frames=2))

        fsm.update([_blob(BOX)], 0, person_boxes=[(120, 90, 20, 20)])
        events = fsm.update([_blob(BOX)], 1, person_boxes=[(120, 90, 20, 20)])

        assert len(events) == 1

    def test_zero_threshold_vetoes_only_on_real_overlap(self) -> None:
        fsm = StaticObjectFSM(DwellConfig(dwell_frames=2, person_overlap_threshold=0.0))

        fsm.update([_blob(BOX)], 0, person_boxes=[(120, 90, 20, 20)])

        assert len(fsm.tracks) == 1


class TestReset:
    """Reset semantics of the state machine."""

    def test_reset_drops_tracks_and_ids(self) -> None:
        fsm = StaticObjectFSM()
        fsm.update([_blob(BOX)], 0)

        fsm.reset()
        fsm.update([_blob(BOX)], 0)

        assert len(fsm.tracks) == 1
        assert fsm.tracks[0].track_id == 0
