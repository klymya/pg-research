"""Data structures exchanged between the pipeline layers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property

import cv2
import numpy as np

BoundingBox = tuple[int, int, int, int]
"""Axis-aligned box in ``(x, y, width, height)`` pixel coordinates."""


@dataclass
class Frame:
    """A preprocessed frame with lazily derived colour-space views.

    Attributes:
        bgr: Denoised and registered frame, ``uint8`` of shape ``(H, W, 3)``.
        index: Zero-based position of the frame in the stream.
    """

    bgr: np.ndarray
    index: int = 0

    def __post_init__(self) -> None:
        if self.bgr.ndim != 3 or self.bgr.shape[2] != 3:
            raise ValueError("Frame.bgr must be a 3-channel image")
        if self.bgr.dtype != np.uint8:
            raise ValueError("Frame.bgr must be of dtype uint8")

    @cached_property
    def gray(self) -> np.ndarray:
        """Grayscale view of the frame."""
        return cv2.cvtColor(self.bgr, cv2.COLOR_BGR2GRAY)

    @cached_property
    def hsv(self) -> np.ndarray:
        """HSV view of the frame (OpenCV ranges: H 0-179, S/V 0-255)."""
        return cv2.cvtColor(self.bgr, cv2.COLOR_BGR2HSV)

    @property
    def shape(self) -> tuple[int, int]:
        """Frame size as ``(height, width)``."""
        return self.bgr.shape[:2]


@dataclass(frozen=True)
class Blob:
    """A connected foreground region surviving the structural filters.

    Attributes:
        bbox: Bounding box in ``(x, y, width, height)``.
        area: Number of foreground pixels in the component.
        centroid: Sub-pixel centroid as ``(x, y)``.
    """

    bbox: BoundingBox
    area: int
    centroid: tuple[float, float]


class ObjectState(str, Enum):
    """Lifecycle of a static-object candidate inside the dwell-time FSM."""

    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    EXPIRED = "expired"


@dataclass
class TrackedObject:
    """A static-object candidate followed across frames by the FSM.

    Attributes:
        track_id: Stable identifier assigned on creation.
        bbox: Most recent bounding box.
        area: Most recent blob area in pixels.
        state: Current FSM state.
        dwell: Accumulated presence counter, in frames.
        missing: Consecutive frames without a matching blob.
        first_seen: Frame index at which the candidate appeared.
        last_seen: Frame index of the most recent match.
        vetoed: Whether the candidate currently overlaps a person box.
    """

    track_id: int
    bbox: BoundingBox
    area: int
    state: ObjectState = ObjectState.CANDIDATE
    dwell: int = 0
    missing: int = 0
    first_seen: int = 0
    last_seen: int = 0
    vetoed: bool = False


@dataclass(frozen=True)
class PipelineResult:
    """Everything the pipeline produced for a single frame.

    Attributes:
        frame: The preprocessed frame the result refers to.
        foreground: Foreground mask of the single background model, ``uint8``
            with values ``0``/``255``.
        filtered_mask: Foreground mask after morphology and area gating.
        blobs: Structurally valid static-object candidates.
        tracked: All live tracks known to the FSM after this frame.
        events: Tracks that reached the confirmed state on this frame; a
            non-empty list is the "Sign of Life" signal.
    """

    frame: Frame
    foreground: np.ndarray
    filtered_mask: np.ndarray
    blobs: tuple[Blob, ...] = ()
    tracked: tuple[TrackedObject, ...] = ()
    events: tuple[TrackedObject, ...] = field(default=())

    @property
    def has_sign_of_life(self) -> bool:
        """Whether a left object was confirmed on this frame."""
        return bool(self.events)
