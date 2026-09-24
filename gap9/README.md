# gap9

Evaluation of the GreenWaves **GAP9** microcontroller as the inference target for PG2.

**Status:** ARCHIVED — frozen 2023-12 · 91 MB. **Hardware not selected — Alif Ensemble was chosen.**

## Purpose

Before committing to a sensor SoC, the candidates had to be measured on the actual workload:
can this chip run a people detector at acceptable accuracy, size and speed? This project ports
YOLOv8 and YOLOX to GAP9's toolchain and evaluates the result.

The work is mostly quantisation with GreenWaves' `nntool`, which is where the difficulty lay —
the same class of problem later faced on the Alif side with int8 export.

| Path | What |
|---|---|
| `yolov8.ipynb`, `yolov8_fp.ipynb`, `yolov8_coco.ipynb`, `eval_yolov8.ipynb` | YOLOv8 quantisation and evaluation |
| `yolox.ipynb`, `yolox_pg.ipynb`, `yolox_no_quant.ipynb` | YOLOX variants, PG data, and an unquantised baseline |
| `notebooks/` | Second set including `gap9_yolo_example.ipynb`, `model_eval.ipynb` |
| `nntool/`, `sdk/` | Local fixes to GreenWaves' tooling and private SDK |
| `weights/`, `yolox_precalc.pickle` | Model artefacts and precalculated data |

## Outcome — and a warning

**GAP9 was not chosen; the Alif Ensemble was.** The rationale was never written down.

The notebook outputs — accuracy tables, quantisation error plots, timing figures — are the
**only surviving evidence** of why. That is why every notebook here is deliberately exempt from
`nbstripout`: stripping them would erase the basis of a hardware decision.

If you need that rationale, read the notebook outputs directly. If you can reconstruct it,
write it into `PG2.wiki` and this file.

## Connections

- Alternative to the Alif Ensemble path that `people_detection_pg2`, `alif_perf` and
  `alif_ensemble_data_collection` all continue.
- The quantisation problem it wrestles with is the same one solved differently in
  `people_detection_pg2` (DeGirum fork + separated head + Vela).

## Dependencies

`torch`, `yolox`, `fiftyone` (dataset browsing), plus GreenWaves' `nntool` and the GAP9 SDK —
both vendored here with local fixes. No `requirements.txt`; the SDK is not public.

---

## Original notes

- nntool - bug fix for nntol 
- sdk - bug fix and updates for GAP9 private SDK