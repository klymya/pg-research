# Socionext

Evaluation of a Socionext 60 GHz FMCW radar as a people-sensing alternative to cameras.

**Status:** ARCHIVED — frozen 2023-05 · 4 tracked files, 24 KB

## Purpose

Radar promises presence and counting without capturing images — attractive for privacy-
sensitive installations. This is the parameter-feasibility study: can the chirp be configured
to cover a room at useful resolution?

`main.py` (`config_parameters_socionext`) works the FMCW trade-off directly:

| Parameter | Value |
|---|---|
| Centre frequency | 60.5 GHz |
| Bandwidth | 3.4 GHz |
| Chirp time | 1100 µs |
| Max range | 6 m (0.022 m resolution) |

The note in the code records the constraint: at 12 m range the bandwidth must be halved, or the
Nyquist rate is not met — so range and range-resolution trade directly against each other.
`DoCSV.py` exports the swept parameters; `C/sample_app_multi/` is the vendor sample app and
`Python/Parameters tuning and csv creation/` the tuning scripts.

## Outcome

Not selected. Cameras remained the sensing modality. No written conclusion exists — the
parameter study is the whole record.

## Connections

- Parallel investigation to `vayyar/`, the other 60 GHz radar evaluated around the same time.
- Both were alternatives to the camera pipeline that `people_detection_pg2` continues.

## Dependencies

`numpy`, `pandas`. No declared requirements; the C sample is vendor-supplied.
