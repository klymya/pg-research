# annotation_poc

Quality assurance for CVAT annotations: validate them, visualise the failures, and fix the
recurring batch problems (wrong filename prefixes, mismatched IDs, misnumbered minibatches)
before the data reaches a training set.

**Status:** ACTIVE · last change 2026-05 · 78 MB

## Purpose

Annotation batches arrive from CVAT — some in-house, some outsourced — and defects in them are
expensive: a mislabelled or misaligned box silently degrades every model trained on it
afterwards. This project catches those defects while they are still cheap to fix.

The PG2 annotation schema pairs each **box** with a **points** annotation carrying the same ID
(the box marks the person, the points mark the ground contact used for px→cm projection). Most
validation here is about that pairing holding.

`Pilot/` holds the vendor evaluation: the same frames annotated in-house and by an outsourced
provider, kept for comparison.

## Working pipeline

### 1. Unpack — `unzip_annotations.py`
Unzips CVAT exports into separate per-batch directories.

### 2. Validate — `validate_cvat_annotations.py`
Checks that **each CVAT box has exactly one matching points annotation by ID**, and writes a
validation report.

### 3. Look at what failed — `visualize_incorrect_annotations.py`
Renders the offending frames from that report so a human can judge whether it is a real
annotation error or a schema quirk.

### 4. Fix recurring batch problems
One-off repair scripts, each written for a specific batch incident and kept as a record:

| Script | Fixes |
|---|---|
| `fix_prefixes_b1.py` | Wrong filename prefixes in batch 1 |
| `fix_minibatch2_filenames.py`, `fix_minibatch2_annotations.py` | Minibatch 2 filename/annotation mismatch (`minibatch2_mistakes/`) |
| `init_fix_batches_8_10.py` | Batch 8–10 initial corrections |
| `merge_b11_annotations.py` | Merges batch 11 annotations |

### 5. Review visually — the notebooks
`annotation_viz.ipynb`, `annotation_viz_cam4_revB.ipynb`, `annotation_viz_person.ipynb` overlay
annotations on frames for sampled review. `utils.py` holds the shared CVAT XML parsing and
drawing helpers (`normalize_annotation_id`, box/point loaders).

## Layout

| Path | What |
|---|---|
| `utils.py` | CVAT XML parsing, ID normalisation, overlay drawing |
| `validate_cvat_annotations.py` | The validator |
| `fix_*.py`, `merge_*.py`, `init_fix_*.py` | Per-batch repair scripts (historical record) |
| `Pilot/` | Outsourced-vendor comparison set |
| `3images/`, `batch_visualized/`, `minibatch2_mistakes/` | Sample sets and captured failures |

## Connections

- **Upstream of `people_detection_pg2/cam4_revB/`** — annotations validated here feed the
  CVAT→YOLO conversion and dataset builds there.
- Same CVAT schema is consumed by `object-detection-v3/read_verify_data.ipynb` for the
  furniture datasets.
- Nothing outside this repo depends on it.

## Dependencies

Core: `pillow`, `matplotlib`; standard library `xml.etree` for CVAT parsing. No
`requirements.txt`.

```bash
pip install pillow matplotlib
```

> The repair scripts are **not** general tools — each was written against one batch's specific
> defect and hard-codes paths and prefixes. Read one before reusing it.
