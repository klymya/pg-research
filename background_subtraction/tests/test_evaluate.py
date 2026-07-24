"""Tests for the pixel-level F-Measure evaluator."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

import evaluate
from evaluate import (
    EvaluationResult,
    accumulate_counts,
    frame_number,
    index_masks,
    resize_to_smallest,
)


def _write_mask(path: Path, mask: np.ndarray) -> None:
    """Write a single-channel mask to ``path``, creating parents."""
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), mask)


class TestEvaluationResult:
    """Derivation of metrics from confusion-matrix counts."""

    def test_perfect_prediction(self) -> None:
        result = EvaluationResult.from_counts(tp=100, fp=0, fn=0, tn=50, frames=1)

        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f_measure == 1.0
        assert result.iou == 1.0

    def test_precision_and_recall(self) -> None:
        # TP=60, FP=20, FN=40 -> P=0.75, R=0.60.
        result = EvaluationResult.from_counts(tp=60, fp=20, fn=40, tn=0, frames=1)

        assert result.precision == pytest.approx(0.75)
        assert result.recall == pytest.approx(0.60)
        assert result.f_measure == pytest.approx(2 * 0.75 * 0.60 / (0.75 + 0.60))

    def test_f_measure_matches_iou_identity(self) -> None:
        result = EvaluationResult.from_counts(tp=123, fp=45, fn=67, tn=999, frames=1)

        # F = 2 * IoU / (1 + IoU) for the binary foreground class.
        assert result.f_measure == pytest.approx(
            2 * result.iou / (1 + result.iou)
        )

    def test_no_predictions_and_no_groundtruth_is_zero(self) -> None:
        result = EvaluationResult.from_counts(tp=0, fp=0, fn=0, tn=100, frames=1)

        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f_measure == 0.0
        assert result.iou == 0.0

    def test_summary_contains_headline_score(self) -> None:
        result = EvaluationResult.from_counts(tp=10, fp=0, fn=0, tn=5, frames=1)

        assert "F-Measure=1.0000" in result.summary()


class TestFrameNumber:
    """Extraction of the frame number from a filename."""

    @pytest.mark.parametrize(
        ("name", "expected"),
        [("gt000123.png", 123), ("bin000007.png", 7), ("frame.png", None)],
    )
    def test_parses_number(self, name: str, expected: int | None) -> None:
        assert frame_number(Path(name)) == expected


class TestAccumulateCounts:
    """Per-pair confusion-matrix accumulation and CDNET masking."""

    def test_excludes_shadow_unknown_and_non_roi(self) -> None:
        # Values 50, 170, 85 must be ignored; only 0 and 255 are scored.
        groundtruth = np.array([[0, 255, 50, 85, 170]], dtype=np.uint8)
        prediction = np.full((1, 5), 255, dtype=np.uint8)

        tp, fp, fn, tn = accumulate_counts(groundtruth, prediction, fg_threshold=127)

        assert (tp, fp, fn, tn) == (1, 1, 0, 0)

    def test_counts_all_four_cells(self) -> None:
        groundtruth = np.array([[255, 255, 0, 0]], dtype=np.uint8)
        prediction = np.array([[255, 0, 255, 0]], dtype=np.uint8)

        tp, fp, fn, tn = accumulate_counts(groundtruth, prediction, fg_threshold=127)

        assert (tp, fp, fn, tn) == (1, 1, 1, 1)

    def test_threshold_controls_foreground(self) -> None:
        groundtruth = np.array([[255, 255]], dtype=np.uint8)
        prediction = np.array([[100, 200]], dtype=np.uint8)

        tp, _, fn, _ = accumulate_counts(groundtruth, prediction, fg_threshold=127)

        assert (tp, fn) == (1, 1)  # only the 200 pixel clears the threshold

    def test_mismatched_shapes_are_resized_to_smallest(self) -> None:
        # A perfect 8x8 prediction against a 4x4 groundtruth: after resizing the
        # prediction down to 4x4 they agree exactly.
        groundtruth = np.full((4, 4), 255, dtype=np.uint8)
        prediction = np.full((8, 8), 255, dtype=np.uint8)

        tp, fp, fn, tn = accumulate_counts(groundtruth, prediction, fg_threshold=127)

        assert (tp, fp, fn, tn) == (16, 0, 0, 0)


class TestResizeToSmallest:
    """Bringing mismatched masks to a common resolution."""

    def test_same_shape_is_a_no_op(self) -> None:
        a = np.zeros((4, 4), np.uint8)
        b = np.ones((4, 4), np.uint8)

        out_a, out_b = resize_to_smallest(a, b)

        assert out_a is a and out_b is b

    def test_larger_is_downscaled_to_smaller(self) -> None:
        groundtruth = np.zeros((4, 6), np.uint8)  # 24 px
        prediction = np.zeros((4, 5), np.uint8)  # 20 px -> target

        out_gt, out_pred = resize_to_smallest(groundtruth, prediction)

        assert out_gt.shape == (4, 5)
        assert out_pred.shape == (4, 5)

    def test_preserves_label_values_via_nearest(self) -> None:
        # Downscaling must not invent intermediate labels (e.g. by averaging).
        groundtruth = np.array([[0, 50, 85, 170, 255, 255]], dtype=np.uint8)
        prediction = np.zeros((1, 3), np.uint8)

        out_gt, _ = resize_to_smallest(groundtruth, prediction)

        assert set(np.unique(out_gt)).issubset({0, 50, 85, 170, 255})


class TestIndexMasks:
    """Indexing of a mask folder by frame number."""

    def test_indexes_by_number(self, tmp_path: Path) -> None:
        for number in (1, 2, 10):
            _write_mask(tmp_path / f"gt{number:06d}.png", np.zeros((2, 2), np.uint8))

        index = index_masks(tmp_path)

        assert set(index) == {1, 2, 10}
        assert index[10].name == "gt000010.png"

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            index_masks(tmp_path / "absent")

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no mask images"):
            index_masks(tmp_path)


class TestEvaluate:
    """End-to-end folder evaluation."""

    def test_perfect_match(self, tmp_path: Path) -> None:
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:6, 2:6] = 255
        _write_mask(tmp_path / "gt" / "gt000001.png", mask)
        _write_mask(tmp_path / "res" / "bin000001.png", mask)

        result = evaluate.evaluate(tmp_path / "gt", tmp_path / "res")

        assert result.frames == 1
        assert result.f_measure == 1.0

    def test_matches_across_prefixes_and_numbers(self, tmp_path: Path) -> None:
        for number in (1, 6, 11):
            mask = np.zeros((8, 8), dtype=np.uint8)
            mask[0:4, 0:4] = 255
            _write_mask(tmp_path / "gt" / f"gt{number:06d}.png", mask)
            _write_mask(tmp_path / "res" / f"bin{number:06d}.png", mask)

        result = evaluate.evaluate(tmp_path / "gt", tmp_path / "res")

        assert result.frames == 3
        assert result.f_measure == 1.0

    def test_unmatched_frames_are_ignored(self, tmp_path: Path) -> None:
        mask = np.full((4, 4), 255, dtype=np.uint8)
        for number in (1, 2, 3):
            _write_mask(tmp_path / "gt" / f"gt{number:06d}.png", mask)
        _write_mask(tmp_path / "res" / "bin000002.png", mask)  # only frame 2

        result = evaluate.evaluate(tmp_path / "gt", tmp_path / "res")

        assert result.frames == 1

    def test_no_common_frames_raises(self, tmp_path: Path) -> None:
        _write_mask(tmp_path / "gt" / "gt000001.png", np.zeros((4, 4), np.uint8))
        _write_mask(tmp_path / "res" / "bin000009.png", np.zeros((4, 4), np.uint8))

        with pytest.raises(FileNotFoundError, match="no common frame"):
            evaluate.evaluate(tmp_path / "gt", tmp_path / "res")

    def test_accepts_string_paths(self, tmp_path: Path) -> None:
        mask = np.full((4, 4), 255, dtype=np.uint8)
        _write_mask(tmp_path / "gt" / "gt000001.png", mask)
        _write_mask(tmp_path / "res" / "bin000001.png", mask)

        result = evaluate.evaluate(str(tmp_path / "gt"), str(tmp_path / "res"))

        assert result.f_measure == 1.0


class TestMain:
    """Command line entry point."""

    def _make_pair(self, tmp_path: Path) -> tuple[Path, Path]:
        """Write a matching groundtruth/result pair and return the two dirs."""
        mask = np.zeros((6, 6), dtype=np.uint8)
        mask[1:4, 1:4] = 255
        _write_mask(tmp_path / "gt" / "gt000001.png", mask)
        _write_mask(tmp_path / "res" / "bin000001.png", mask)
        return tmp_path / "gt", tmp_path / "res"

    def test_score_only_prints_a_float(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        gt_dir, result_dir = self._make_pair(tmp_path)

        exit_code = evaluate.main([str(gt_dir), str(result_dir), "--score-only"])

        assert exit_code == 0
        assert float(capsys.readouterr().out.strip()) == pytest.approx(1.0)

    def test_summary_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        gt_dir, result_dir = self._make_pair(tmp_path)

        exit_code = evaluate.main([str(gt_dir), str(result_dir)])

        assert exit_code == 0
        assert "F-Measure=1.0000" in capsys.readouterr().out

    def test_missing_folder_returns_one(self, tmp_path: Path) -> None:
        assert evaluate.main([str(tmp_path / "absent"), str(tmp_path)]) == 1
