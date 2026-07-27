"""End-to-end orchestration of the 4-layer "Sign of Life" pipeline.

The pipeline owns no IO: it consumes decoded frames and optional person boxes
supplied by the caller, which keeps video reading and evaluation out of the
prototype's core.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .bgs_model import RateParameterizedBGS
from .blobs import BlobFilter
from .config import PipelineConfig
from .fsm import StaticObjectFSM
from .preprocessing import Preprocessor
from .structures import BoundingBox, PipelineResult


def _scale_boxes(
    boxes: Iterable[BoundingBox],
    from_shape: tuple[int, int],
    to_shape: tuple[int, int],
) -> tuple[BoundingBox, ...]:
    """Scale ``(x, y, w, h)`` boxes from one frame size to another.

    Args:
        boxes: Boxes in ``from_shape`` coordinates.
        from_shape: Source frame size as ``(height, width)``.
        to_shape: Target frame size as ``(height, width)``.

    Returns:
        The boxes in ``to_shape`` coordinates; returned unchanged when the two
        sizes match.
    """
    boxes = tuple(boxes)
    if from_shape == to_shape or not boxes:
        return boxes
    scale_x = to_shape[1] / from_shape[1]
    scale_y = to_shape[0] / from_shape[0]
    return tuple(
        (
            round(x * scale_x),
            round(y * scale_y),
            round(w * scale_x),
            round(h * scale_y),
        )
        for x, y, w, h in boxes
    )


class SignOfLifePipeline:
    """Chains Layer 0, the single Layer 1/2 background model and Layer 3."""

    def __init__(
        self,
        preprocessor: Preprocessor,
        model: RateParameterizedBGS,
        blob_filter: BlobFilter,
        fsm: StaticObjectFSM,
    ) -> None:
        """Initialise the pipeline from already-built stages.

        Args:
            preprocessor: Layer 0 denoising and registration stage.
            model: Layer 1/2 rate-parameterized background subtraction model.
            blob_filter: Structural filter over the foreground mask.
            fsm: Dwell-time state machine raising the left-object signal.
        """
        self._preprocessor = preprocessor
        self._model = model
        self._blob_filter = blob_filter
        self._fsm = fsm

    @classmethod
    def from_config(cls, config: PipelineConfig | None = None) -> SignOfLifePipeline:
        """Build a fully configured pipeline.

        Args:
            config: Complete pipeline description; defaults are used when
                omitted.

        Returns:
            The configured :class:`SignOfLifePipeline`.
        """
        config = config or PipelineConfig()
        return cls(
            preprocessor=Preprocessor(config.preprocess),
            model=RateParameterizedBGS.from_config(
                config.bgs, config.feature, config.shadow
            ),
            blob_filter=BlobFilter(config.blob),
            fsm=StaticObjectFSM(config.dwell),
        )

    @property
    def model(self) -> RateParameterizedBGS:
        """The background subtraction model."""
        return self._model

    @property
    def fsm(self) -> StaticObjectFSM:
        """The dwell-time state machine."""
        return self._fsm

    def process(
        self,
        image: np.ndarray,
        person_boxes: Iterable[BoundingBox] = (),
    ) -> PipelineResult:
        """Run one frame through every layer.

        Args:
            image: Raw ``uint8`` BGR frame of shape ``(H, W, 3)``.
            person_boxes: Bounding boxes of actively tracked humans in the
                *original* image coordinates, used to veto candidates that
                belong to a person. They are scaled internally to match any
                resize applied by Layer 0.

        Returns:
            The per-frame :class:`~scr.structures.PipelineResult`; its
            ``events`` field carries the "Sign of Life" signal. Masks, blobs and
            track boxes are in the (possibly resized) frame coordinates.
        """
        original_shape = image.shape[:2]
        frame = self._preprocessor.process(image)
        scaled_boxes = _scale_boxes(person_boxes, original_shape, frame.shape)

        foreground = self._model.apply(frame)
        filtered_mask, blobs = self._blob_filter.filter(foreground)
        events = self._fsm.update(blobs, frame.index, scaled_boxes)

        return PipelineResult(
            frame=frame,
            foreground=foreground,
            filtered_mask=filtered_mask,
            blobs=blobs,
            tracked=self._fsm.snapshot(),
            events=events,
        )

    def reset(self) -> None:
        """Return every stage to its initial state."""
        self._preprocessor.reset()
        self._model.reset()
        self._fsm.reset()
