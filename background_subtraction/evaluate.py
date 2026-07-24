"""Pixel-level F-Measure evaluation of background-subtraction results.

Scores a folder of predicted binary masks against a folder of CDNET-14
groundtruth masks and reports Precision, Recall and F-Measure (with IoU as a
bonus). Usable both from the terminal and as an importable function:

    # terminal
    python evaluate.py data/canoe/groundtruth results/canoe

    # imported
    from evaluate import evaluate
    score = evaluate("data/canoe/groundtruth", "results/canoe").f_measure

Metric definition
-----------------
For each pixel the prediction is compared to the groundtruth to accumulate a
confusion matrix over the whole sequence (micro-averaged over pixels, the
CDNET per-video convention), from which:

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F-Measure = 2 * Precision * Recall / (Precision + Recall)

F-Measure is the standard single-number score in the change-detection
literature. It is the Dice coefficient of the foreground class and is
monotonically equivalent to IoU (Jaccard): ``IoU = F / (2 - F)``.

CDNET groundtruth encoding and masking
--------------------------------------
CDNET masks use five grayscale values, and only two of them are scored — the
other three encode regions that must be excluded from the metric:

    0   background          -> counted (TN / FP)
    255 foreground          -> counted (TP / FN)
    50  hard shadow         -> excluded (don't-care)
    170 unknown / boundary  -> excluded (don't-care)
    85  outside ROI         -> excluded (don't-care)

Restricting the confusion matrix to pixels where ``gt in {0, 255}`` therefore
applies every CDNET masking gate at once: frames outside the temporal ROI are
all-85 and drop out entirely, the spatial ROI is encoded as 85, and shadow and
ambiguous pixels are never penalised. No separate ``ROI.bmp`` or
``temporalROI.txt`` is needed.

Predictions are treated as foreground where their value exceeds
``--fg-threshold`` (default 127), so both 0/255 binary masks and probability
maps are accepted. Groundtruth and prediction are matched by the frame number
embedded in their filenames (``gt000123.png`` <-> ``bin000123.png``), so the
two folders may use different prefixes.

Resolution
----------
The pipeline may emit masks at a different (usually smaller, e.g. QVGA)
resolution than the native-resolution groundtruth. When the two differ, both
are resized to the smaller of the two sizes with nearest-neighbour
interpolation before the confusion matrix is accumulated, preserving the exact
groundtruth label values and the binary prediction.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger("evaluate")

IMAGE_SUFFIXES = (".png", ".bmp", ".jpg", ".jpeg")
"""Mask extensions recognised in the groundtruth and result folders."""

FRAME_NUMBER = re.compile(r"(\d+)")
"""Extracts the frame number from a stem such as ``gt000123`` or ``bin000123``."""

GT_FOREGROUND = 255
"""Groundtruth value marking a foreground pixel."""

GT_BACKGROUND = 0
"""Groundtruth value marking a background pixel; all other values are excluded."""

DEFAULT_FG_THRESHOLD = 127
"""Predictions above this value are treated as foreground."""


def _safe_div(numerator: float, denominator: float) -> float:
    """Divide, returning 0.0 when the denominator is zero.

    Args:
        numerator: Dividend.
        denominator: Divisor.

    Returns:
        ``numerator / denominator``, or ``0.0`` if ``denominator`` is zero.
    """
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class EvaluationResult:
    """Confusion-matrix counts and the metrics derived from them.

    Attributes:
        tp: True-positive pixel count (predicted FG, groundtruth FG).
        fp: False-positive pixel count (predicted FG, groundtruth BG).
        fn: False-negative pixel count (predicted BG, groundtruth FG).
        tn: True-negative pixel count (predicted BG, groundtruth BG).
        frames: Number of matched frames that contributed to the counts.
        precision: ``TP / (TP + FP)``.
        recall: ``TP / (TP + FN)``.
        f_measure: ``2 * P * R / (P + R)`` — the headline score.
        iou: ``TP / (TP + FP + FN)`` — Jaccard index of the foreground class.
    """

    tp: int
    fp: int
    fn: int
    tn: int
    frames: int
    precision: float
    recall: float
    f_measure: float
    iou: float

    @classmethod
    def from_counts(
        cls, tp: int, fp: int, fn: int, tn: int, frames: int
    ) -> EvaluationResult:
        """Build a result, computing the derived metrics from raw counts.

        Args:
            tp: True-positive pixel count.
            fp: False-positive pixel count.
            fn: False-negative pixel count.
            tn: True-negative pixel count.
            frames: Number of matched frames evaluated.

        Returns:
            The populated :class:`EvaluationResult`.
        """
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f_measure = _safe_div(2 * precision * recall, precision + recall)
        iou = _safe_div(tp, tp + fp + fn)
        return cls(
            tp=tp,
            fp=fp,
            fn=fn,
            tn=tn,
            frames=frames,
            precision=precision,
            recall=recall,
            f_measure=f_measure,
            iou=iou,
        )

    def summary(self) -> str:
        """Return a human-readable one-block summary of the result."""
        return (
            f"frames={self.frames}  "
            f"F-Measure={self.f_measure:.4f}  "
            f"Precision={self.precision:.4f}  "
            f"Recall={self.recall:.4f}  "
            f"IoU={self.iou:.4f}\n"
            f"TP={self.tp} FP={self.fp} FN={self.fn} TN={self.tn}"
        )


def frame_number(path: Path) -> int | None:
    """Extract the frame number embedded in a filename.

    Args:
        path: Mask path such as ``gt000123.png``.

    Returns:
        The parsed frame number, or ``None`` when the name carries no digits.
    """
    match = FRAME_NUMBER.search(path.stem)
    return int(match.group(1)) if match else None


def index_masks(directory: Path) -> dict[int, Path]:
    """Index the mask files of a directory by frame number.

    Args:
        directory: Directory holding mask images.

    Returns:
        A mapping of frame number to mask path.

    Raises:
        FileNotFoundError: If the directory does not exist or holds no masks.
        NotADirectoryError: If ``directory`` is not a directory.
    """
    if not directory.exists():
        raise FileNotFoundError(f"directory does not exist: {directory}")
    if not directory.is_dir():
        raise NotADirectoryError(f"not a directory: {directory}")

    index: dict[int, Path] = {}
    for path in sorted(directory.iterdir()):
        if not (path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES):
            continue
        number = frame_number(path)
        if number is not None:
            index[number] = path

    if not index:
        raise FileNotFoundError(f"no mask images found in {directory}")
    return index


def resize_to_smallest(
    groundtruth: np.ndarray, prediction: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Resize both masks to the smaller of the two resolutions.

    The pipeline may emit masks at a different (usually smaller, e.g. QVGA)
    resolution than the native-resolution CDNET groundtruth. Both are brought to
    the smaller of the two sizes so the confusion matrix is well defined.
    Nearest-neighbour interpolation is mandatory: it preserves the exact
    groundtruth label values (0/50/85/170/255) and the binary prediction, which
    any smoothing interpolation would corrupt.

    Args:
        groundtruth: Single-channel groundtruth mask.
        prediction: Single-channel predicted mask.

    Returns:
        The two masks, both at the smaller ``(height, width)`` size.
    """
    if groundtruth.shape[:2] == prediction.shape[:2]:
        return groundtruth, prediction

    gt_area = groundtruth.shape[0] * groundtruth.shape[1]
    pred_area = prediction.shape[0] * prediction.shape[1]
    height, width = (
        groundtruth.shape[:2] if gt_area <= pred_area else prediction.shape[:2]
    )

    def fit(mask: np.ndarray) -> np.ndarray:
        if mask.shape[:2] == (height, width):
            return mask
        return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

    return fit(groundtruth), fit(prediction)


def accumulate_counts(
    groundtruth: np.ndarray, prediction: np.ndarray, fg_threshold: int
) -> tuple[int, int, int, int]:
    """Accumulate the confusion matrix for one groundtruth/prediction pair.

    When the two masks differ in size they are first resized to the smaller
    resolution via :func:`resize_to_smallest`. Only pixels whose groundtruth
    value is exactly ``0`` or ``255`` are counted; shadow (50), unknown (170)
    and outside-ROI (85) pixels are ignored.

    Args:
        groundtruth: Single-channel CDNET groundtruth mask.
        prediction: Single-channel predicted mask.
        fg_threshold: Prediction values above this are treated as foreground.

    Returns:
        The ``(tp, fp, fn, tn)`` pixel counts for this pair.
    """
    groundtruth, prediction = resize_to_smallest(groundtruth, prediction)

    gt_fg = groundtruth == GT_FOREGROUND
    gt_bg = groundtruth == GT_BACKGROUND
    pred_fg = prediction > fg_threshold

    tp = int(np.count_nonzero(pred_fg & gt_fg))
    fp = int(np.count_nonzero(pred_fg & gt_bg))
    fn = int(np.count_nonzero(~pred_fg & gt_fg))
    tn = int(np.count_nonzero(~pred_fg & gt_bg))
    return tp, fp, fn, tn


def evaluate(
    groundtruth_dir: Path | str,
    result_dir: Path | str,
    fg_threshold: int = DEFAULT_FG_THRESHOLD,
) -> EvaluationResult:
    """Evaluate predicted masks against CDNET groundtruth.

    This is the importable entry point. Groundtruth and result masks are paired
    by the frame number in their filenames; the metric is micro-averaged over
    all pixels of the matched frames (the CDNET per-video convention).

    Args:
        groundtruth_dir: Folder of CDNET groundtruth masks (``gt*.png``).
        result_dir: Folder of predicted masks (e.g. ``bin*.png``).
        fg_threshold: Prediction values above this are treated as foreground.

    Returns:
        The :class:`EvaluationResult`; ``result.f_measure`` is the F-score.

    Raises:
        FileNotFoundError: If either folder is missing, empty, or the two share
            no common frame numbers.
    """
    groundtruth_index = index_masks(Path(groundtruth_dir))
    result_index = index_masks(Path(result_dir))

    common = sorted(set(groundtruth_index) & set(result_index))
    if not common:
        raise FileNotFoundError(
            "no common frame numbers between the groundtruth and result folders"
        )

    only_gt = len(groundtruth_index) - len(common)
    only_result = len(result_index) - len(common)
    if only_gt or only_result:
        logger.warning(
            "Evaluating %d matched frames (%d groundtruth-only, %d result-only ignored)",
            len(common),
            only_gt,
            only_result,
        )

    total_tp = total_fp = total_fn = total_tn = 0
    evaluated = 0
    for number in common:
        groundtruth = cv2.imread(str(groundtruth_index[number]), cv2.IMREAD_GRAYSCALE)
        prediction = cv2.imread(str(result_index[number]), cv2.IMREAD_GRAYSCALE)
        if groundtruth is None or prediction is None:
            logger.warning("Skipping unreadable frame %d", number)
            continue

        tp, fp, fn, tn = accumulate_counts(groundtruth, prediction, fg_threshold)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_tn += tn
        evaluated += 1

    return EvaluationResult.from_counts(
        total_tp, total_fp, total_fn, total_tn, evaluated
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command line interface.

    Args:
        argv: Argument list to parse; ``sys.argv[1:]`` when omitted.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Compute the pixel-level F-Measure of background-subtraction "
            "results against CDNET-14 groundtruth."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "groundtruth_dir", type=Path, help="folder of CDNET groundtruth masks"
    )
    parser.add_argument(
        "result_dir", type=Path, help="folder of predicted binary masks"
    )
    parser.add_argument(
        "--fg-threshold",
        type=int,
        default=DEFAULT_FG_THRESHOLD,
        help="prediction values above this are treated as foreground",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="print only the F-Measure value (for scripting/piping)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument list to parse; ``sys.argv[1:]`` when omitted.

    Returns:
        A process exit code: ``0`` on success, ``1`` on a handled error.
    """
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    try:
        result = evaluate(args.groundtruth_dir, args.result_dir, args.fg_threshold)
    except (FileNotFoundError, NotADirectoryError, ValueError) as error:
        logger.error("%s", error)
        return 1

    if args.score_only:
        print(f"{result.f_measure:.6f}")
    else:
        print(result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
