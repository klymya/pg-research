# camera_qc

Compares a Device-Under-Test camera module against a reference camera and produces a one-page
**PDF verdict**: PASS, MARGINAL or FAIL. Built for QA — one command, two folders in, one PDF out.

**Status:** ACTIVE · last change 2026-09 · 995 MB on disk (capture batches, untracked)

## Purpose

Incoming camera modules vary. Some are optically defective; others merely differ in colour
balance, which is correctable downstream. This tool separates those two cases so QA can reject
the first and accept the second, with a paper trail.

**Only sharpness can fail a unit.** Colour, exposure and noise are measured and reported but
never change the verdict: colour is correctable by CCM/white balance, and a raised noise floor
is a matter of degree rather than an optical defect.

It is a delivered tool rather than research — an installable package with a CLI, a config file
and tests.

## Connections

- **Metrics ported from `people_detection_pg2/investigations/rotated_cam_color_dfiff/`**
  (`sensor_metrics.py`), where they were first developed for the colour-difference study.
- Measures the same rev-C hardware that `camera_calibration/` calibrates; a module that fails
  here should never reach calibration.
- Standalone otherwise — nothing imports it, and it imports nothing from sibling projects.

## Dependencies

Declared in `pyproject.toml`: `numpy`, `opencv-python`, `pandas`, `matplotlib`, `pyyaml`,
`markdown`, `weasyprint` (PDF rendering), `tqdm`. Installs a `camera-qc` console script.

```bash
cd camera_qc && pip install -e .
sudo apt install -y libpango-1.0-0 libpangoft2-1.0-0   # WeasyPrint needs these on Linux
```

## Data

Capture batches (`batch_*/`, `revc/`, `captures/`) are **deliberately untracked** — they are
~1 GB of QA photographs with no value in version control. Reports are generated, not committed.

---

## Quickstart (3 steps)

### 1. Install (once)

```bash
cd camera_qc
pip install -e .
```

WeasyPrint needs a few system libraries on Linux:

```bash
sudo apt install -y libpango-1.0-0 libpangoft2-1.0-0
```

### 2. Capture images

In the **same controlled environment**, capture:

- One image with the **reference** camera → save in a folder named `ref/`
- One or more images with the **DUT** camera → save in a folder named `dut/`

```
captures/
├── ref/
│   └── reference.jpg          # exactly one image
└── dut/
    ├── 001.jpg
    ├── 002.jpg
    └── 003.jpg                # one or more images
```

Important:
- Use the **same framing**, **same lighting**, and **same orientation** for ref and DUT.
- All images must have the **same resolution**.

### 3. Run the tool

```bash
camera-qc \
  --ref ./captures/ref \
  --dut ./captures/dut \
  --scene chessboard \
  --out ./report.pdf
```

Open `report.pdf`. The first banner shows the verdict.

---

## Report modes

The **simplified report is the default**. It measures sharpness alone and is
built from three parts:

1. **Summary** — total DUT images, and how many passed / failed.
2. **Per-image table** — one row per DUT: image name, reference sharpness, the
   image's sharpness, % of reference, and a PASS/FAIL flag.
3. **Comparisons** — the ref (left) vs DUT (right) image pair for every unit.

DUT images are ordered **sharpest first** (ties broken on filename) in both
report modes and in the terminal output, so `#1` means the same image
everywhere. The summary also reports how many images are **at or above the
reference sharpness** — stricter than PASS, which tolerates a drop of up to
the tolerance band.

The two modes can disagree. Both `sharpness` and `sharpness_fft` are
*blocking*, but the simplified report measures only `sharpness`, so a DUT
that passes `sharpness` and fails `sharpness_fft` reads PASS in the simplified
report and FAIL in the full one — the full report is the stricter of the two.
Beyond that it adds the advisory colour / exposure / noise numbers, the
per-image narratives and the RGB histograms.

```bash
camera-qc --ref ./ref --dut ./dut --scene chessboard   # simplified (default)
camera-qc --ref ./ref --dut ./dut --scene chessboard --no-simplified   # full
```

| Mode | Metrics measured | Report contents | 76 images |
|---|---|---|---:|
| `--simplified` *(default)* | `sharpness` | summary, per-image table, ref\|DUT pairs | ~41 s |
| `--no-simplified` | all 9 | the above + narratives, per-metric tables, RGB histograms | ~92 s |

Most of the saving comes from skipping the 77 histogram figures, not from the
metrics themselves — measuring all nine on 76 images costs only ~0.5 s.

---

## Scene presets

| `--scene` | When to use | Min DUT images | Tolerance band |
|---|---|---|---|
| `chessboard` | Controlled target (color chart / chessboard) | 1 | Tight |
| `ceiling` | Ceiling-mounted in the real install location | 3 | Loose |

---

## What can fail a unit

**Only sharpness can fail a DUT.** Colour, exposure and noise differences are
reported and highlighted, but never change the verdict: colour is correctable
downstream (CCM / white balance) and a raised noise floor is a matter of degree
rather than an optical defect, so neither is grounds for rejecting a module.

| Role | Metrics | Effect when outside tolerance |
|---|---|---|
| **blocking** | `sharpness`, `sharpness_fft` | **FAIL** |
| **advisory** | `L`, `a*`, `b*`, `R`, `G`, `B`, `noise` | Warning only — amber highlight in the PDF, `!` in the console, verdict unchanged |

## Verdict meaning

| Verdict | What it means | Exit code |
|---|---|---|
| **PASS** | No blocking metric outside tolerance. May still carry colour/noise warnings. | 0 |
| **MARGINAL** | Not reachable at the default thresholds (see below). | 1 |
| **FAIL** | One or more **blocking** metrics outside tolerance. | 2 |

A colour-shifted, slightly noisier but optically sharp module reads:

```
Batch verdict: PASS  (0/2 blocking metrics outside tolerance ...)
Advisory: 5/7 colour/noise metrics outside tolerance (a*, b*, R, B, noise) —
reported only, does not affect the verdict.
```

To soften the ladder, set `fail_if_n_blocking_outside_tol: 2` in `config.yaml` —
one blocking metric off then reads MARGINAL and two or more read FAIL.

To move a metric between the two groups, edit `metric_severity` in
`config.yaml`; anything not listed defaults to `blocking`.

---

## Metrics evaluated

Ported from the validated `sensor_metrics.py` in
`people_detection_pg2/investigations/rotated_cam_color_dfiff/`. Each metric is one of:

- **Closer to reference is better** — `L` (luminance), `a*`, `b*` (chromaticity), `R`, `G`, `B` (per-channel mean) — *advisory*
- **Higher is better** — `sharpness` (Laplacian variance), `sharpness_fft` (high-freq energy) — *blocking*
- **Lower is better** — `noise` (high-frequency residual std) — *advisory*

The simplified report (the default) measures `sharpness` only. The advisory
metrics can never fail a unit, so they are omitted rather than computed and
ignored; `sharpness_fft` is a second blocking metric and is measured only in
the full report.

---

## Merging capture batches

Captures arrive as one `batch_<n>/` folder per run, each with its own `ref/`
and `dut/`. `build_batch_all.py` merges them into a single batch to evaluate
in one pass.

```bash
python -m camera_qc.build_batch_all --src ~/Pictures --out ./batch_all
```

A unit that failed in `batch_n` is re-worked and re-shot in a later batch, so
when the same filename appears in several batches the **highest-numbered batch
wins** (numeric order — `batch_10` is newer than `batch_9`). Finder/Explorer
duplicates (`… copy.jpg`, `…_copy.jpg`, `… (1).jpg`) collapse onto the
original; non-images are ignored. Every run writes `manifest.csv` recording,
per output file, its unit, source batch, absolute source path, how many
captures of that unit exist, which batches they were in, and an md5.

| Flag | Effect |
|---|---|
| `--ref-from BATCH` | Which batch's `ref/` image becomes the global reference (default `batch_0`). The chosen reference module is excluded from the DUT set. |
| `--include-ref-images` | Also treat each batch's `ref/` image as a DUT capture. Needed when `--ref-from` names a different module than the other batches used, so that module's captures are not lost. |
| `--keep-duplicates` | Keep **every** capture instead of one per unit, naming each `<unit>_batch<nn>.jpg`. Byte-identical clones inside a single batch are still dropped. |
| `--symlink` | Link instead of copy. |
| `--dry-run` | Print the selection without writing. |
| `--force` | Rebuild over an existing output folder. |

`--keep-duplicates` is the mode for auditing the rework itself — every capture
of a unit sits together in the report, so you can read whether a re-shoot
actually improved it:

```bash
python -m camera_qc.build_batch_all --keep-duplicates --out ./batch_all_full
```

### Curated acceptance set

`build_batch_all.py` is mechanical: a repeated serial means the same module
was re-shot, so the latest capture wins. That is right for most units but not
all — sometimes the *module itself* was swapped, and the two captures are two
different physical modules sharing a serial label. Only the test log knows
which is which, so those exceptions are declared in
`build_acceptance_set.py`:

```bash
python -m camera_qc.build_acceptance_set --out ./batch_acceptance
```

- `MODULE_REPLACEMENTS` — units whose repeated captures are distinct modules.
  Each declared batch contributes its own DUT, named `<unit>_batch<nn>.jpg`;
  captures in batches *not* listed are earlier states of the first listed
  module and are superseded as usual.
- `UNIT_NOTES` — facts the optical metrics cannot express (e.g. a module
  failed on electronics, not optics). Carried into `manifest.csv`.
- `EXPECTED_CAPTURES` / `EXPECTED_TOTAL_DUT` — the declared headcount. The
  build **fails** rather than emit a set that quietly disagrees with it, so
  re-running over changed source folders cannot silently move the numbers.

---

## Common errors

| Error | Fix |
|---|---|
| `Expected exactly one reference image in ref/, found 3` | Put **one** image in `ref/`. |
| `No DUT images found in dut/` | Add `.jpg`/`.png` files to `dut/`. |
| `DUT image shape mismatch` | Re-capture so ref and DUT have the same resolution. |
| `Scene preset 'ceiling' requires at least 3 DUT image(s); got 1` | Capture more DUT images for the ceiling scene. |

---

## Optional: capture-pipeline preprocessing

The images used to build the training datasets go through **ALIF white balance
→ CLAHE** before they are stored. If your reference image comes from a raw
source while the DUT captures are already preprocessed (or vice versa), apply
the same pipeline here so the two sides are measured in the same space:

```bash
camera-qc --ref ./captures/ref --dut ./captures/dut --scene ceiling \
  --preprocess-ref \
  --save-preprocessed ./preprocessed \
  --out ./report.pdf
```

| Flag | Effect |
|---|---|
| `--preprocess-ref` | Apply ALIF WB → CLAHE to the reference image. |
| `--preprocess-dut` | Apply the same pipeline to every DUT image. |
| `--save-preprocessed [DIR]` | Write the processed images to `DIR/ref` and `DIR/dut` for inspection. `DIR` is optional — the bare flag writes to `./preprocessed`. |

Processing happens **in memory after loading — the `ref/` and `dut/` folders are
never modified.**

> **Process both sides or neither.** WB + CLAHE shifts `L`, `a*`, `b*`, `R/G/B`
> and sharpness substantially. If only one side is preprocessed, the tool prints
> a warning and the PDF carries a warning box: the deltas then mix the
> processing difference with the sensor difference and are not a sensor-only
> comparison.

The implementation lives in `camera_qc/preprocess.py` and is a **verbatim copy**
of `people_detection_pg2/cam4_revB/alif_awb.py` (`white_balance_vectorized`) and
the `apply_clahe` in that folder's `preprocess_dataset.py`, so this tool stays
usable standalone. Output is bit-identical to
`preprocess_dataset.py --mode wb_alif_clahe`. The sharpness augmentation stage
of that script is deliberately not copied — it is a random training-time
augmentation, not part of the capture pipeline. **If the dataset pipeline
changes, update `preprocess.py` to match.**

---

## Advanced

### Python API

```python
from camera_qc import evaluate

result = evaluate(ref_dir="./ref", dut_dir="./dut", scene="chessboard")
print(result.verdict.verdict)         # "pass" | "marginal" | "fail"
result.write_report("./report.pdf")

# Full metric pack + long-form report:
result = evaluate(ref_dir="./ref", dut_dir="./dut", scene="chessboard",
                  simplified=False)
```

With preprocessing, or on a single array:

```python
result = evaluate(ref_dir="./ref", dut_dir="./dut", scene="chessboard",
                  preprocess_ref=True, save_preprocessed="./preprocessed")

from camera_qc import preprocess_rgb   # RGB uint8 in, RGB uint8 out
processed = preprocess_rgb(image_rgb)
```

### Override tolerances

Copy `camera_qc/config.yaml`, edit, and pass it in:

```bash
camera-qc --ref ./ref --dut ./dut --scene chessboard \
  --config ./my_tolerances.yaml --out ./report.pdf
```

### Debug the report

`--keep-md` keeps the intermediate Markdown + PNGs next to the PDF:

```bash
camera-qc --ref ./ref --dut ./dut --scene chessboard --keep-md
```

---

## Tolerances — current state and tuning

> **Heads up: the default tolerances in `config.yaml` are a v1 seed, not a
> validated production threshold. Treat current verdicts as *indicative* and
> re-tune the config against a real batch of known-good DUTs before relying on
> the pass/fail outcome for acceptance decisions.**

### Where the chessboard preset comes from

The chessboard tolerances are seeded from
`people_detection_pg2/investigations/rotated_cam_color_dfiff/sensor_color_comparison_postccm.ipynb`
section 9 — the `|Δ| after CCM` between the previously accepted CCM-corrected
"new" sensor and the "old" reference, averaged over 4 paired scenes and
expressed as % of the reference mean. The argument: a DUT that deviates by no
more than this already-accepted sensor did is acceptable.

The measured numbers, and what `config.yaml` ships (the colour bands are the
measurements rounded; the sharpness bands are deliberately tightened below
them):

| Metric | Measured \|Δ\| after CCM | In config |
|---|---:|---:|
| L | 9.6% | 10 |
| a* | 6.8% | 7 |
| b* | 12.0% | 12 |
| R | 10.1% | 10 |
| G | 9.7% | 10 |
| B | 6.1% | 7 |
| sharpness | 15.8% | 10 (tightened — DUT must reach 90% of ref) |
| noise | 9.5% | 15 (widened — see note 4) |
| sharpness_fft | 10.8% | 10 (tightened) |

### Known limitations to fix during tuning

1. **N = 4 scenes.** The empirical basis is small; one outlier scene can swing
   a tolerance by several percentage points.
2. **Ceiling preset is a guess.** It's set to `chessboard × 2` — not measured
   on ceiling-mounted captures. Replace it with real data.
3. **The blocking group is sharpness only.** `sharpness` and `sharpness_fft`
   measure the same physical property two ways and count the same toward the
   tally, so a borderline-soft module trips both blocking metrics at once —
   the verdict effectively rests on a single property measured twice.
4. **`noise` is advisory,** so its 15% band only decides when the warning
   appears, not whether a unit is rejected. The band is looser than the 9.5%
   measured after CCM; confirm it against real known-good batches so the
   warning fires on genuinely noisy modules rather than on normal spread.
5. **Pass/fail thresholds (`fail_if_n_blocking_outside_tol: 1`).** This too is a
   reasonable default, not a tuned value.

### How to retune

1. Collect ≥10 known-good DUT batches per scene preset.
2. Run the tool against each; record per-metric `|Δ|` values.
3. Set each `tolerances_pct` value to roughly the 95th percentile of the
   observed `|Δ|` for accepted devices (or whatever pass-rate target you have).
4. Optionally add weights or hard-fail metrics.
5. Re-run on a known-bad device to confirm it now fails for the right reasons.

See [SPEC.md M6](../camera_quality_eval_tool_SPEC.md) for the planned tuning
milestone.
