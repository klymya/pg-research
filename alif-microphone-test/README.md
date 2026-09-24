# alif-microphone-test

Python prototype of a production acceptance test for the sensor's microphone.

**Status:** ARCHIVED — frozen 2024-07 · 3.0 MB

## Purpose

A microphone that passes assembly can still be deaf at some frequencies. This prototypes the
acceptance test: play synthetic tones at **5, 10 and 15 kHz**, record them back, and check the
response — the audio counterpart to what `camera_qc/` does for camera modules.

`synthetic_test.ipynb` both generates the test tones and evaluates recordings.
`records/` holds reference captures made with an iPhone 11; `samples/` the synthetic tones.

It remained a prototype: no CLI, no pass/fail thresholds, no report. If microphone QC becomes a
requirement, `camera_qc/` is the model to follow — it is the same problem solved properly.

## Connections

- Audio analogue of `camera_qc/` (camera-module incoming QC).
- Targets the same Alif sensor hardware as `alif_ensemble_data_collection`.

## Dependencies

Notebook-only; standard scientific Python for signal generation and analysis
(`numpy`, `scipy`, `matplotlib`). No declared requirements.

---

## Original notes

Python prototype of microphone test. The test uses multiple audio samples with different frequencies: 5, 10 and 15 kHz.

Folder structure:
- records - recorded samples using iPhone 11;
- samples - synthetic test samples;
- synthetic_test.ipynb - generates samples and test records.