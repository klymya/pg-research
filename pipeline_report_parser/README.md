# pipeline_report_parser

Aggregates tracking-analysis results across PG1 benchmark runs into a single comparison table.

**Status:** ARCHIVED — frozen 2025-04 · 1 notebook, 8 KB

## Purpose

The PG1 benchmark pipeline wrote one `TrackingAnalysisResults.csv` per run per dataset, on the
fileserver. Comparing releases meant opening many files; this notebook collects them into one
frame.

`aggregate_tracking_reports.ipynb` reads from `/mnt/pg-fs02/benchmarks-analysis/<run_id>/
TrackingAnalysis/TrackingAnalysisResults.csv` across four datasets — `PG1_AlgoDev_DLtest02_RV9`,
`PG1_AlgoDev_DLtest_RV59`, `PG1_AlgoDev_VLC_RV5_V2` and `10fps`.

> **Run IDs are hard-coded** in the notebook (e.g.
> `20250408_020344_5_3_0+gitv6_2_0-rc_1-1-g9ec5a865_...`), and the notebook itself flags the
> list as `# Requre setup`. To use it for other runs you must edit that list by hand.

## Connections

- Reads the fileserver output of the PG1 benchmark pipeline; those paths may no longer resolve.
- PG1-era. The PG2 equivalent is
  `people_detection_pg2/extended_evaluation_metrics/`, which generates its own reports rather
  than scraping a pipeline's output.

## Dependencies

`pandas` only, plus access to `/mnt/pg-fs02`.
