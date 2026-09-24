# data-exploration

Archaeology on the PG1-era video archive: find every recording, deduplicate it, parse its
annotations into a queryable database, then build training datasets from it.

**Status:** ARCHIVED — frozen 2024-10 · 354 MB

## Purpose

Years of recordings had accumulated on the fileserver with no index — no reliable way to answer
"what footage do we have, how much of it is annotated, and which of it is already used by the
CI tests". This project built that index and then used it to assemble datasets.

The end product is a SQLite database (`people_detection.db`, gitignored) of recordings and
their boxes, plus COCO/YOLOX dataset builders on top of it. `video_similarity/` and
`embeddings_full/` support near-duplicate detection, so the same scene does not appear on both
sides of a train/val split.

The numbered pipeline below is the original author's runbook and remains the entry point.

## Connections

- Reads the fileserver archives (`/mnt/pg-fs01`, `/mnt/pg-fs02`); those paths may no longer
  resolve.
- Imports `PG_PyTraining` for `.seq` parsing.
- `people_detection_pg2` imports `data-exploration/video_splitting` directly.
- Duplicate-detection approach was reused in `object-detection-v3/`.

## Dependencies

`pandas`, `numpy`, `opencv-python`, `tqdm`, `transformers` (embeddings), `torch`,
`torchvision`, `pillow`, `pyyaml`, plus `sqlite3` from the standard library. No
`requirements.txt`.

---

## Original runbook

Steps:
1. Gather paths to seq files:
```bash
python collect_avi_paths.py
``` 

2. Concatenate results of `pg01.py` and `pg02.py` into `data.csv`, the file shoul have columns `seq_path` and `is_algo_test` - is file used for Algo CI test:
```bash
python combine_data.py
```

3. Check path to avi files:
```bash
python is_video_exist.py
```

4. Remove duplicates from the datalist - use `deduplication.ipynb`

5. Read annotations from seq files:
```bash
python parse_seq_files.py
```

6. Create sqlite3 database
```bash
sqlite3 people_detection.db
```

7. Add table to the db using `create_boxes_table.sql`:
```bash
sqlite3 people_detection.db < create_boxes_table.sq
```

8. Load data to db:
```bash
python import.py
```