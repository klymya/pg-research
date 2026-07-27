"""Tests for the command line runner."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import pytest
from conftest import draw_rectangle, make_background

import main
from scr.config import DenoiseMethod, FeatureKind, RegistrationMethod
from scr.structures import Frame, PipelineResult

OBJECT_BBOX = (60, 40, 30, 25)
"""Bounding box of the object deposited into the synthetic sequence."""


def _write_sequence(directory: Path, count: int, deposit_from: int | None = None) -> None:
    """Write a synthetic CDNET-style sequence to ``directory``.

    Args:
        directory: Destination directory, created if missing.
        count: Number of frames to write.
        deposit_from: Index from which the static object is present; ``None``
            keeps the scene empty throughout.
    """
    directory.mkdir(parents=True, exist_ok=True)
    background = make_background()
    occupied = draw_rectangle(background, OBJECT_BBOX)
    for index in range(count):
        image = (
            occupied
            if deposit_from is not None and index >= deposit_from
            else background
        )
        cv2.imwrite(str(directory / f"in{index + 1:06d}.jpg"), image)


class TestDiscoverFrames:
    """Sequence discovery and ordering."""

    def test_lists_images_in_order(self, tmp_path: Path) -> None:
        _write_sequence(tmp_path, 3)

        frames = main.discover_frames(tmp_path)

        assert [path.name for path in frames] == [
            "in000001.jpg",
            "in000002.jpg",
            "in000003.jpg",
        ]

    def test_prefers_a_nested_input_directory(self, tmp_path: Path) -> None:
        _write_sequence(tmp_path / "input", 2)
        (tmp_path / "groundtruth").mkdir()

        frames = main.discover_frames(tmp_path)

        assert len(frames) == 2
        assert frames[0].parent.name == "input"

    def test_ignores_non_image_files(self, tmp_path: Path) -> None:
        _write_sequence(tmp_path, 2)
        (tmp_path / "temporalROI.txt").write_text("1 2")

        assert len(main.discover_frames(tmp_path)) == 2

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            main.discover_frames(tmp_path / "absent")

    def test_file_input_raises(self, tmp_path: Path) -> None:
        target = tmp_path / "frame.jpg"
        target.write_bytes(b"")

        with pytest.raises(NotADirectoryError, match="not a directory"):
            main.discover_frames(target)

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="no images found"):
            main.discover_frames(tmp_path)


class TestOutputName:
    """Derivation of the CDNET result filename."""

    def test_mirrors_the_source_frame_number(self) -> None:
        assert main.output_name(Path("in000123.jpg"), 0) == "bin000123.png"

    def test_falls_back_to_the_sequence_index(self) -> None:
        assert main.output_name(Path("frame.jpg"), 7) == "bin000008.png"

    def test_pads_short_numbers(self) -> None:
        assert main.output_name(Path("in7.png"), 0) == "bin000007.png"


class TestSelectMask:
    """Selection of the persisted pipeline output."""

    @staticmethod
    def _result() -> PipelineResult:
        """Build a result whose masks are distinguishable by value."""
        shape = (8, 8)
        return PipelineResult(
            frame=Frame(bgr=np.zeros((*shape, 3), dtype=np.uint8)),
            foreground=np.full(shape, 1, dtype=np.uint8),
            filtered_mask=np.full(shape, 4, dtype=np.uint8),
        )

    @pytest.mark.parametrize(
        ("source", "expected"),
        [("foreground", 1), ("filtered", 4)],
    )
    def test_returns_the_requested_mask(self, source: str, expected: int) -> None:
        assert main.select_mask(self._result(), source)[0, 0] == expected

    def test_rejects_an_unknown_source(self) -> None:
        with pytest.raises(ValueError, match="unknown mask source"):
            main.select_mask(self._result(), "bogus")


class TestBuildConfig:
    """Mapping of CLI arguments onto the pipeline configuration."""

    def test_maps_every_layer(self, tmp_path: Path) -> None:
        args = main.parse_args(
            [
                "-i", str(tmp_path), "-o", str(tmp_path / "out"),
                "--denoise", "bilateral",
                "--registration", "ecc",
                "--feature", "lbp",
                "--backend", "mog2",
                "--alpha", "0.2",
                "--min-area", "42",
                "--dwell-frames", "7",
                "--no-shadow-veto",
            ]
        )

        config = main.build_config(args)

        assert config.preprocess.denoise is DenoiseMethod.BILATERAL
        assert config.preprocess.registration is RegistrationMethod.ECC
        assert config.feature.kind is FeatureKind.LBP
        assert config.shadow.enabled is False
        assert config.bgs.backend == "mog2"
        assert config.bgs.learning_rate == 0.2
        assert config.blob.min_area == 42
        assert config.dwell.dwell_frames == 7

    def test_defaults_are_valid(self, tmp_path: Path) -> None:
        args = main.parse_args(["-i", str(tmp_path), "-o", str(tmp_path / "out")])

        config = main.build_config(args)

        assert config.bgs.backend == "running_gaussian"
        assert 0.0 < config.bgs.learning_rate <= 1.0

    def test_invalid_learning_rate_is_rejected(self, tmp_path: Path) -> None:
        args = main.parse_args(
            [
                "-i", str(tmp_path), "-o", str(tmp_path / "out"),
                "--alpha", "1.5",
            ]
        )

        with pytest.raises(ValueError, match="learning_rate"):
            main.build_config(args)


class TestRun:
    """End-to-end behaviour of the runner."""

    def test_writes_one_binary_mask_per_frame(self, tmp_path: Path) -> None:
        sequence = tmp_path / "seq"
        output = tmp_path / "out"
        _write_sequence(sequence, 12, deposit_from=6)
        args = main.parse_args(["-i", str(sequence), "-o", str(output)])

        written = main.run(args)

        masks = sorted(output.glob("*.png"))
        assert written == 12
        assert len(masks) == 12
        assert masks[0].name == "bin000001.png"

        image = cv2.imread(str(masks[-1]), cv2.IMREAD_UNCHANGED)
        assert image.ndim == 2
        assert set(np.unique(image)).issubset({0, 255})

    def test_creates_a_missing_output_directory(self, tmp_path: Path) -> None:
        sequence = tmp_path / "seq"
        _write_sequence(sequence, 2)
        output = tmp_path / "nested" / "out"
        args = main.parse_args(["-i", str(sequence), "-o", str(output)])

        main.run(args)

        assert output.is_dir()

    def test_limit_truncates_the_sequence(self, tmp_path: Path) -> None:
        sequence = tmp_path / "seq"
        output = tmp_path / "out"
        _write_sequence(sequence, 10)
        args = main.parse_args(
            ["-i", str(sequence), "-o", str(output), "--limit", "4"]
        )

        assert main.run(args) == 4

    def test_detects_a_deposited_object(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        sequence = tmp_path / "seq"
        output = tmp_path / "out"
        _write_sequence(sequence, 30, deposit_from=8)
        args = main.parse_args(
            [
                "-i", str(sequence), "-o", str(output),
                "--denoise", "none",
                "--no-shadow-veto",
                "--backend", "running_average",
                "--alpha", "0.02",
                "--dwell-frames", "5",
                "--min-area", "100",
            ]
        )

        with caplog.at_level(logging.INFO):
            main.run(args)

        assert "Sign of life" in caplog.text

    def test_unreadable_frames_are_skipped(self, tmp_path: Path) -> None:
        sequence = tmp_path / "seq"
        output = tmp_path / "out"
        _write_sequence(sequence, 3)
        (sequence / "in000004.jpg").write_bytes(b"not an image")
        args = main.parse_args(["-i", str(sequence), "-o", str(output)])

        assert main.run(args) == 3

    def test_dump_background_writes_a_subfolder(self, tmp_path: Path) -> None:
        sequence = tmp_path / "seq"
        output = tmp_path / "out"
        _write_sequence(sequence, 5)
        args = main.parse_args(
            ["-i", str(sequence), "-o", str(output), "--mask", "foreground", "--dump-background"]
        )

        main.run(args)

        masks = sorted(output.glob("bin*.png"))
        backgrounds = sorted((output / "background").glob("bg*.png"))
        assert len(masks) == 5
        assert len(backgrounds) == 5
        # A background image is a 3-channel scene, not a binary mask.
        image = cv2.imread(str(backgrounds[-1]), cv2.IMREAD_UNCHANGED)
        assert image.ndim == 3
        assert len(np.unique(image)) > 2

    def test_no_background_subfolder_without_flag(self, tmp_path: Path) -> None:
        sequence = tmp_path / "seq"
        output = tmp_path / "out"
        _write_sequence(sequence, 3)
        args = main.parse_args(["-i", str(sequence), "-o", str(output)])

        main.run(args)

        assert not (output / "background").exists()


class TestMain:
    """Exit codes of the process entry point."""

    def test_returns_zero_on_success(self, tmp_path: Path) -> None:
        sequence = tmp_path / "seq"
        _write_sequence(sequence, 2)

        assert main.main(["-i", str(sequence), "-o", str(tmp_path / "out")]) == 0

    def test_returns_one_on_a_missing_sequence(self, tmp_path: Path) -> None:
        assert main.main(["-i", str(tmp_path / "absent"), "-o", str(tmp_path)]) == 1

    def test_returns_one_on_an_invalid_configuration(self, tmp_path: Path) -> None:
        sequence = tmp_path / "seq"
        _write_sequence(sequence, 2)

        exit_code = main.main(
            [
                "-i", str(sequence), "-o", str(tmp_path / "out"),
                "--alpha", "1.5",
            ]
        )

        assert exit_code == 1
