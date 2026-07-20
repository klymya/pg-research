"""Layer 3c: the dwell-time finite state machine.

Static-object candidates are associated across frames by bounding-box IoU.  A
candidate accumulates dwell while it keeps being observed and is promoted to
``CONFIRMED`` - the "Sign of Life" signal - once the dwell budget is reached.
Candidates overlapping an active person box are vetoed: a person standing still
must not raise a left-object alarm.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import replace

from .config import DwellConfig
from .structures import Blob, BoundingBox, ObjectState, TrackedObject
from .utils import iou


class StaticObjectFSM:
    """Tracks static-object candidates and applies the dwell-time rule."""

    def __init__(self, config: DwellConfig | None = None) -> None:
        """Initialise the state machine.

        Args:
            config: Dwell-time and association parameters; defaults are used
                when omitted.
        """
        self._config = config or DwellConfig()
        self._tracks: list[TrackedObject] = []
        self._next_id = 0

    @property
    def config(self) -> DwellConfig:
        """The active dwell-time configuration."""
        return self._config

    @property
    def tracks(self) -> tuple[TrackedObject, ...]:
        """The live, mutable tracks currently known to the FSM.

        The objects keep being updated in place on every :meth:`update` call;
        use :meth:`snapshot` to retain per-frame state.
        """
        return tuple(self._tracks)

    def snapshot(self) -> tuple[TrackedObject, ...]:
        """Copy the live tracks so they can be retained across frames.

        Returns:
            Detached copies of the current tracks, unaffected by later updates.
        """
        return tuple(replace(track) for track in self._tracks)

    def reset(self) -> None:
        """Drop every track and restart identifier assignment."""
        self._tracks.clear()
        self._next_id = 0

    def update(
        self,
        blobs: Sequence[Blob],
        frame_index: int,
        person_boxes: Iterable[BoundingBox] = (),
    ) -> tuple[TrackedObject, ...]:
        """Advance the state machine by one frame.

        Args:
            blobs: Static-object candidates detected on this frame.
            frame_index: Zero-based index of the current frame.
            person_boxes: Bounding boxes of actively tracked humans; candidates
                overlapping one of them are vetoed.

        Returns:
            Detached copies of the tracks promoted to ``CONFIRMED`` on this
            frame.  A non-empty result is the "Sign of Life" signal.
        """
        boxes = tuple(person_boxes)
        matched = self._associate(blobs)

        events: list[TrackedObject] = []
        for track in self._tracks:
            blob_index = matched.get(track.track_id)
            if blob_index is None:
                track.missing += 1
                continue

            blob = blobs[blob_index]
            track.bbox = blob.bbox
            track.area = blob.area
            track.missing = 0
            track.last_seen = frame_index
            track.vetoed = self._overlaps_person(blob.bbox, boxes)

            if track.vetoed:
                track.dwell = 0
                track.state = ObjectState.CANDIDATE
                continue

            track.dwell += 1
            if (
                track.state is ObjectState.CANDIDATE
                and track.dwell >= self._config.dwell_frames
            ):
                track.state = ObjectState.CONFIRMED
                events.append(track)

        self._spawn_unmatched(blobs, matched, frame_index, boxes)
        self._evict_expired()
        return tuple(replace(track) for track in events)

    def _associate(self, blobs: Sequence[Blob]) -> dict[int, int]:
        """Greedily match tracks to blobs by descending IoU.

        Args:
            blobs: Candidates detected on the current frame.

        Returns:
            A mapping of track identifier to the index of its matched blob.
        """
        pairs = [
            (iou(track.bbox, blob.bbox), track.track_id, index)
            for track in self._tracks
            for index, blob in enumerate(blobs)
        ]
        pairs = [pair for pair in pairs if pair[0] >= self._config.iou_match_threshold]
        pairs.sort(key=lambda pair: pair[0], reverse=True)

        matched: dict[int, int] = {}
        used_blobs: set[int] = set()
        for _, track_id, blob_index in pairs:
            if track_id in matched or blob_index in used_blobs:
                continue
            matched[track_id] = blob_index
            used_blobs.add(blob_index)
        return matched

    def _spawn_unmatched(
        self,
        blobs: Sequence[Blob],
        matched: dict[int, int],
        frame_index: int,
        person_boxes: Sequence[BoundingBox],
    ) -> None:
        """Create candidate tracks for blobs left unassociated."""
        claimed = set(matched.values())
        for index, blob in enumerate(blobs):
            if index in claimed:
                continue
            if self._overlaps_person(blob.bbox, person_boxes):
                continue
            self._tracks.append(
                TrackedObject(
                    track_id=self._next_id,
                    bbox=blob.bbox,
                    area=blob.area,
                    dwell=1,
                    first_seen=frame_index,
                    last_seen=frame_index,
                )
            )
            self._next_id += 1

    def _evict_expired(self) -> None:
        """Remove tracks unseen for longer than the configured budget."""
        self._tracks = [
            track
            for track in self._tracks
            if track.missing <= self._config.max_missing_frames
        ]

    def _overlaps_person(
        self, bbox: BoundingBox, person_boxes: Sequence[BoundingBox]
    ) -> bool:
        """Whether ``bbox`` overlaps any person box beyond the veto threshold.

        A threshold of ``0`` vetoes on any non-zero overlap rather than on every
        box, which is what a zero-tolerance setting is meant to express.
        """
        threshold = self._config.person_overlap_threshold
        return any(
            (overlap := iou(bbox, person)) > 0 and overlap >= threshold
            for person in person_boxes
        )
