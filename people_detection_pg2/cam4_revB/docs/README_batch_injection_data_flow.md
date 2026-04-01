# Batch Injection Data Flow

This document describes the logic behind the pipeline listed in [batch_injection_pipeline.md](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/batch_injection_pipeline.md).

It focuses on what each step does, which artifacts it reads and writes, and which preprocessing algorithms are actually applied. The pipeline mixes three kinds of work:

- annotation/bootstrap setup
- dataset split and conversion
- true image preprocessing

Only Step 5 applies pixel-level image preprocessing. Steps 0 to 4 and 6 are dataset-management or annotation-processing stages.

## End-to-End Flow

1. Initialize annotation folders for the new batch and review them visually.
2. Detect overlap between the new batch and the old holdout set to avoid data leakage.
3. Split the new batch into `train`, `val`, and `holdout`.
4. Convert CVAT annotations into a YOLO dataset with one class (`person`) and one keypoint per person.
5. Remove weak edge boxes using an inner-region intersection rule.
6. Apply image preprocessing:
   white balance only, or white balance followed by CLAHE.
7. Merge the processed new batch into the main accumulated dataset.

## Step-by-Step

### 0. Annotation bootstrap and manual review

Files:

- [init_fix_batches_8_10.py](/home/klymy/projects/researches/annotation_poc/init_fix_batches_8_10.py)
- [annotation_viz_cam4_revB.ipynb](/home/klymy/projects/researches/annotation_poc/annotation_viz_cam4_revB.ipynb)

Purpose:

- create a per-batch corrected annotation folder
- copy the base CVAT exports into working filenames
- visually inspect merged annotations on sample images

Logic:

- `init_fix_batches_8_10.py` copies:
  - `instances_default.json` -> `batch{n}_corrected.json`
  - `annotations.xml` -> `batch{n}_corrected.xml`
- `annotation_viz_cam4_revB.ipynb` parses the XML and overlays boxes plus points on sample images.

Algorithms used:

- no image preprocessing
- XML parsing and ID-based merge of CVAT `box` and `points` objects

Notes:

- This is a dataset preparation and QA step, not a pixel-processing step.

### 1. Find holdout overlap with old data

File:

- [find_intersection_batch1_3_vs_4_5.ipynb](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/find_intersection_batch1_3_vs_4_5.ipynb)

Purpose:

- prevent leakage from previously reserved holdout scenes into the new batch

Inputs:

- previous split file, for example `train_val_holdout_subsets_batches_1_3.json`
- images from the new batch

Outputs:

- intersection JSON, for example `batches1_3_vs_12_holdout_intersection.json`

Logic:

- image filenames are reduced to a scene/location token using `utils_cam4_revb.get_data()`
- tokens from the old `holdout` split are intersected with tokens from the new batch
- the shared tokens are saved for exclusion from train/val

Algorithm / heuristic:

- string-based grouping by scene token extracted from filename

Notes:

- This is split hygiene, not image preprocessing.

### 2. Split new batch into train / val / holdout

File:

- [subsets_splitting_batch4_5.ipynb](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/subsets_splitting_batch4_5.ipynb)

Purpose:

- create a deterministic split for the new batch

Inputs:

- all images from the new batch
- holdout-overlap JSON from Step 1

Outputs:

- split JSON, for example `train_val_holdout_subsets_batches_12.json`

Logic:

- images whose parsed scene token matches the overlap list go to `holdout`
- remaining images become the working pool
- the working pool is shuffled with `np.random.seed(42)`
- first `10%` goes to `val`, the rest to `train`

Algorithm / heuristic:

- deterministic random split with fixed seed
- holdout assignment by scene-token membership

Notes:

- This is dataset partitioning, not image preprocessing.

### 3. Convert CVAT annotations to YOLO dataset

File:

- [cvat2yolo_convert.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/cvat2yolo_convert.py)
- [cvat2yolo.ipynb](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/cvat2yolo.ipynb)

Helpers:

- [annotation_poc/utils.py](/home/klymy/projects/researches/annotation_poc/utils.py)
- [yolov8_dataset_util.py](/home/klymy/projects/researches/object-detection-v3/utils/yolov8_dataset_util.py)

Purpose:

- convert corrected CVAT annotations into a trainable YOLO dataset

Inputs:

- corrected CVAT XML
- split JSON from Step 2
- source images from the batch directory

Outputs:

- dataset directory under `.../yolo_dataset/<dataset_name>`
- `images/train`, `images/val`, `images/holdout`
- `labels/train`, `labels/val`, `labels/holdout`
- `project.yaml` and `project_holdout.yaml`
- `conversion_summary.json`
- validation CSVs for skipped images and invalid annotations

Logic:

- CVAT XML is parsed per image
- `box` and `points` entries are matched by annotation `ID`
- each object becomes:
  - class id `0` (`person`)
  - normalized bbox corners
  - normalized point `(x, y)`
- labels are written in YOLO pose-style text format:
  `cls x_center y_center width height keypoint_x keypoint_y`
- images are copied and converted from `.tif` to `.jpg`

Algorithms used:

- ID-based fusion of bounding boxes and keypoints
- coordinate normalization by image width and height
- bbox conversion from corners to center-width-height
- no photometric or geometric image enhancement
- explicit validation of missing images, split mismatches, invalid bbox geometry, and normalized coordinates

Notes:

- `cvat2yolo_convert.py` is the current production implementation.
- `cvat2yolo.ipynb` is the notebook reference.
- This is annotation-format conversion plus file export.
- It establishes the data representation used by later preprocessing.

### 4. Filter boxes near the image border

File:

- [filter_small_bboxes.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/filter_small_bboxes.py)
- [filter_small_bboxes_v2.ipynb](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/filter_small_bboxes_v2.ipynb)

Purpose:

- remove annotations that are mostly outside the trusted inner image region

Inputs:

- YOLO dataset from Step 3

Outputs:

- copied dataset with filtered label files
- regenerated `project.yaml` and `project_holdout.yaml`

Key parameters:

- `MARGIN = 0.10`
- `MIN_INSIDE_RATIO = 0.01`

Logic:

- an inner valid box is defined in normalized coordinates:
  `[0.10, 0.10, 0.90, 0.90]`
- for each YOLO bbox, the code computes:
  - bbox area
  - intersection area with the inner box
  - `inside_ratio = intersection_area / bbox_area`
- the annotation is removed when `inside_ratio < 0.01`
- invalid boxes with non-positive width or height are also removed and reported separately

Algorithm used:

- explicit rectangle intersection test in normalized image coordinates

Why this matters:

- this removes partial edge objects and annotations that are mostly outside the reliable image center
- it is label filtering, not image pixel modification

Important note:

- `filter_small_bboxes.py` is the current production implementation
- `filter_small_bboxes_v2.ipynb` is the notebook reference for the same step
- the older `filter_small_bboxes.ipynb` used a `min_margin` heuristic; the v2/script version uses true overlap area, which is more correct

### 5. Apply image preprocessing

File:

- [preprocess_dataset.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/preprocess_dataset.py)
- [white_balance_alif.ipynb](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/white_balance_alif.ipynb)

Helpers:

- [alif_awb.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/alif_awb.py)
- [utils_cam4_revb.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/utils_cam4_revb.py)

Purpose:

- create processed dataset variants with improved color constancy and optionally improved local contrast

Input dataset:

- filtered dataset from Step 4, for example `batch_H270_16-44-09_margin_10`

Output datasets:

- `<base>_wb_alif`
- `<base>_wb_alif_clahe`

Actual image-processing flow:

1. Read image with OpenCV.
2. Convert `BGR -> RGB`.
3. Apply ALIF white balance (`alif_awb.white_balance_vectorized`).
4. Optionally apply CLAHE (`utils_cam4_revb.clahe`).
5. Convert `RGB -> BGR`.
6. Overwrite the copied image on disk.

#### 5.1 ALIF white balance

Function:

- `white_balance_vectorized(image)`

Operations:

- apply a fixed `3x3` color-correction matrix to RGB values
- clip corrected values to `[0, 255]`
- apply an sRGB gamma LUT built from the sRGB OETF
- compute channel means over the whole image
- compute gray-world scaling:
  - `mean_gray = mean(mean_rgb)`
  - `scale = mean_gray / mean_rgb`
- rescale each channel by that factor
- clip back to `uint8`

Algorithms used:

- linear color correction using a fixed calibration matrix
- gamma mapping with an sRGB LUT
- global gray-world white balance

Interpretation:

- the matrix compensates sensor/color-channel bias
- the gray-world step forces average R/G/B intensities toward the same gray target
- this is a global method, not a local/adaptive white-balance model

#### 5.2 CLAHE variant

Function:

- `utils_cam4_revb.clahe(img)`

Operations:

- convert `RGB -> LAB`
- split LAB channels
- run CLAHE only on `L`
  - `clipLimit = 3.0`
  - `tileGridSize = (8, 8)`
- merge channels
- convert `LAB -> RGB`

Algorithm used:

- Contrast Limited Adaptive Histogram Equalization on luminance only

Interpretation:

- improves local contrast while preserving chroma better than per-channel equalization
- useful in low-contrast indoor frames

#### 5.3 Blurring branch

There is also an experimental `process_file_blurred()` branch:

- ALIF white balance
- Gaussian blur with kernel `(3, 3)`

But it is not used in the executed export loop. The notebook writes only:

- white balance only
- white balance + CLAHE

Notes:

- `preprocess_dataset.py` is the current production implementation
- `white_balance_alif.ipynb` is the notebook reference

### 6. Merge the processed batch into the accumulated dataset

File:

- [merge_yolo_datasets.py](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/merge_yolo_datasets.py)
- [update_dataset.ipynb](/home/klymy/projects/researches/people_detection_pg2/cam4_revB/update_dataset.ipynb)

Purpose:

- append the newly processed batch to an existing master dataset

Inputs:

- existing base dataset
- processed new-batch dataset from Step 5

Outputs:

- merged dataset, for example `batch1_12_margin_15_bg_wb_alif`
- updated `project_holdout_v2.yaml`

Logic:

- copy the base dataset into a new result folder
- copy new `train` and `val` images and labels into the result dataset
- clone old holdout into `holdout_v2`
- append the new batch holdout into `holdout_v2`

Algorithms used:

- file-level dataset merge only

Notes:

- `merge_yolo_datasets.py` is the current production implementation
- no new image preprocessing is applied here
- this step defines the final dataset artifact used for training/evaluation

## What Counts As Image Preprocessing Here

True image preprocessing in this pipeline is limited to Step 5:

- ALIF color correction + gray-world white balance
- optional luminance-only CLAHE

Everything else is one of:

- annotation preparation
- split generation
- annotation conversion
- label filtering
- dataset merging

## Practical Summary For Engineers And Agents

- If the goal is to reproduce the exact dataset content, execute all steps in order.
- If the goal is to reproduce only pixel preprocessing, Step 5 is the critical stage.
- If the goal is to reproduce training labels correctly, Step 4 is also critical because it removes border boxes.
- If the goal is to avoid evaluation leakage, Steps 1 and 2 are critical because holdout is scene-token based, not purely random.

## Minimal Data Lineage

`raw batch images + CVAT export`
-> annotation bootstrap / QA
-> holdout-overlap detection
-> train/val/holdout split
-> CVAT XML -> YOLO labels + JPG export
-> border-box filtering
-> white balance or white balance + CLAHE
-> merge into master dataset
