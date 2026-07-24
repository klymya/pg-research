"""Tests for Layer 0 denoising and registration."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import FRAME_SIZE, make_background

from scr.config import DenoiseMethod, PreprocessConfig, RegistrationMethod
from scr.preprocessing import Preprocessor


class TestFrameValidation:
    """Input contract of :meth:`Preprocessor.process`."""

    def test_rejects_grayscale_input(self) -> None:
        preprocessor = Preprocessor()
        with pytest.raises(ValueError, match="3-channel"):
            preprocessor.process(np.zeros(FRAME_SIZE, dtype=np.uint8))

    def test_rejects_float_input(self) -> None:
        preprocessor = Preprocessor()
        with pytest.raises(ValueError, match="uint8"):
            preprocessor.process(np.zeros((*FRAME_SIZE, 3), dtype=np.float32))


class TestDenoising:
    """Behaviour of the configured spatial filters."""

    @pytest.mark.parametrize(
        "method",
        [DenoiseMethod.GAUSSIAN, DenoiseMethod.BILATERAL],
    )
    def test_filters_reduce_noise_variance(self, method: DenoiseMethod) -> None:
        image = make_background()
        preprocessor = Preprocessor(PreprocessConfig(resize=None, denoise=method))

        result = preprocessor.process(image)

        assert result.bgr.shape == image.shape
        assert result.bgr.var() < image.var()

    def test_none_preserves_pixels_but_copies(self) -> None:
        image = make_background()
        preprocessor = Preprocessor(
            PreprocessConfig(resize=None, denoise=DenoiseMethod.NONE)
        )

        result = preprocessor.process(image)

        np.testing.assert_array_equal(result.bgr, image)
        assert result.bgr is not image


class TestFrameIndexing:
    """Frame numbering and reset semantics."""

    def test_index_increments(self) -> None:
        preprocessor = Preprocessor()
        image = make_background()

        indices = [preprocessor.process(image).index for _ in range(3)]

        assert indices == [0, 1, 2]

    def test_reset_restarts_numbering(self) -> None:
        preprocessor = Preprocessor()
        image = make_background()
        preprocessor.process(image)

        preprocessor.reset()

        assert preprocessor.process(image).index == 0


class TestRegistration:
    """Global frame alignment against the first observed frame."""

    @staticmethod
    def _shifted(image: np.ndarray, dx: int, dy: int) -> np.ndarray:
        """Translate ``image`` by ``(dx, dy)`` with replicated borders."""
        return np.roll(np.roll(image, dy, axis=0), dx, axis=1)

    @pytest.mark.parametrize(
        "method",
        [RegistrationMethod.PHASE_CORRELATION, RegistrationMethod.ECC],
    )
    def test_registration_reduces_misalignment(
        self, method: RegistrationMethod
    ) -> None:
        reference = make_background()
        drifted = self._shifted(reference, dx=3, dy=2)
        config = PreprocessConfig(
            resize=None, denoise=DenoiseMethod.NONE, registration=method
        )
        preprocessor = Preprocessor(config)

        preprocessor.process(reference)
        registered = preprocessor.process(drifted)

        before = np.abs(drifted.astype(int) - reference.astype(int)).mean()
        after = np.abs(registered.bgr.astype(int) - reference.astype(int)).mean()
        assert after < before

    def test_none_leaves_frame_untouched(self) -> None:
        reference = make_background()
        drifted = self._shifted(reference, dx=3, dy=2)
        config = PreprocessConfig(
            resize=None,
            denoise=DenoiseMethod.NONE,
            registration=RegistrationMethod.NONE,
        )
        preprocessor = Preprocessor(config)

        preprocessor.process(reference)
        registered = preprocessor.process(drifted)

        np.testing.assert_array_equal(registered.bgr, drifted)

    def test_implausible_shift_is_discarded(self) -> None:
        reference = make_background()
        drifted = self._shifted(reference, dx=40, dy=0)
        config = PreprocessConfig(
            resize=None,
            denoise=DenoiseMethod.NONE,
            registration=RegistrationMethod.PHASE_CORRELATION,
            max_shift_px=5.0,
        )
        preprocessor = Preprocessor(config)

        preprocessor.process(reference)
        registered = preprocessor.process(drifted)

        np.testing.assert_array_equal(registered.bgr, drifted)

    def test_flat_frames_do_not_raise(self) -> None:
        flat = np.full((*FRAME_SIZE, 3), 128, dtype=np.uint8)
        config = PreprocessConfig(
            resize=None,
            denoise=DenoiseMethod.NONE,
            registration=RegistrationMethod.ECC,
        )
        preprocessor = Preprocessor(config)

        preprocessor.process(flat)
        registered = preprocessor.process(flat)

        assert registered.bgr.shape == flat.shape


class TestResize:
    """Input resizing applied before every other Layer 0 step."""

    def test_defaults_to_qvga(self) -> None:
        preprocessor = Preprocessor()
        image = make_background(size=(240, 360))  # non-QVGA input

        result = preprocessor.process(image)

        assert result.shape == (240, 320)  # (height, width) of QVGA

    def test_custom_size(self) -> None:
        preprocessor = Preprocessor(PreprocessConfig(resize=(80, 60)))

        result = preprocessor.process(make_background())

        assert result.shape == (60, 80)

    def test_none_keeps_native_resolution(self) -> None:
        preprocessor = Preprocessor(PreprocessConfig(resize=None))
        image = make_background(size=(90, 110))

        result = preprocessor.process(image)

        assert result.shape == (90, 110)

    def test_no_op_when_already_target_size(self) -> None:
        preprocessor = Preprocessor(
            PreprocessConfig(resize=(160, 120), denoise=DenoiseMethod.NONE)
        )
        image = make_background(size=(120, 160))

        result = preprocessor.process(image)

        np.testing.assert_array_equal(result.bgr, image)
