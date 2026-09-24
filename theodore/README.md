# theodore

Evaluation of the TU-Chemnitz **THEODORE** synthetic overhead-fisheye dataset as a source of
training data.

**Status:** ARCHIVED — frozen 2024-06 · 289 MB

## Purpose

PG's cameras look straight down through a fisheye lens, a viewpoint almost absent from public
datasets. THEODORE is synthetic but has the right geometry, so the question was whether it
could substitute for — or augment — scarce in-house footage.

The work covers extracting the TFRecord subsets, converting them to YOLO and U-Net training
formats, and running both detection and segmentation pipelines (`pipeline/` in Python,
`pipeline_cpp/` in C++).

It was superseded by in-house cam4 data, which has the real lens, the real noise and real
people. The dataset details below are the original notes.

## Connections

- Alternative data source to the in-house datasets in `object-detection-v3/` and
  `people_detection_pg2/cam4_revB/`.
- `extract_mask.ipynb` / U-Net work parallels `leg_position_detectors/unet/`.
- Dataset is external — see the download link below; `data/` is gitignored.

## Dependencies

`torch`, `opencv-python`, `numpy`, `segmentation_models_pytorch` (U-Net),
`albumentations`, `tensorflow` (TFRecord reading), `ultralytics`. No `requirements.txt`.

---

## Dataset

# [THEODORE dataset](https://www.tu-chemnitz.de/etit/dst/forschung/comp_vision/datasets/theodore/index.php.en)

<b> Please, download `training_25k_1.tfrecord`, `training_25k_4.tfrecord`, `training_25k_sample_4_mask.tfrecord` files to `data` from https://cloud.dst.etit.tu-chemnitz.de/index.php/s/4C4C4CeyRS2RaeB </b>

According to the paper the best object detection results was achieved using 25K dataset. So 25k_1, 25k_4, 25k_mask subsets were downloaded and analyzed. The 1 and 4 have 6250 intersected images.

- 25k_samples_4 - as train and validation for object detection and 25k_4_masks - for segmentation
- takes 2.5 random images from 25k_samples_1 - as holdout