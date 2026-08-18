# camera-qc

Compare a Device-Under-Test (DUT) camera module against a reference camera and produce a one-page **PDF verdict**: PASS, MARGINAL, or FAIL.

Built for QA: one command, two folders in, one PDF out.

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

## Scene presets

| `--scene` | When to use | Min DUT images | Tolerance band |
|---|---|---|---|
| `chessboard` | Controlled target (color chart / chessboard) | 1 | Tight |
| `ceiling` | Ceiling-mounted in the real install location | 3 | Loose |

---

## What can fail a unit

**Only optics — sharpness and noise — can fail a DUT.** Colour and exposure
differences are reported and highlighted, but never change the verdict: they are
correctable downstream (CCM / white balance), so they are not grounds for
rejecting a module.

| Role | Metrics | Effect when outside tolerance |
|---|---|---|
| **blocking** | `sharpness`, `sharpness_fft`, `noise` | **FAIL** |
| **advisory** | `L`, `a*`, `b*`, `R`, `G`, `B` | Warning only — amber highlight in the PDF, `!` in the console, verdict unchanged |

## Verdict meaning

| Verdict | What it means | Exit code |
|---|---|---|
| **PASS** | No blocking metric outside tolerance. May still carry colour warnings. | 0 |
| **MARGINAL** | Not reachable at the default thresholds (see below). | 1 |
| **FAIL** | One or more **blocking** metrics outside tolerance. | 2 |

A colour-shifted but optically fine module reads:

```
Batch verdict: PASS  (0/3 blocking metrics outside tolerance ...)
Advisory: 4/6 colour metrics outside tolerance (a*, b*, R, B) — reported only,
does not affect the verdict.
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
- **Lower is better** — `noise` (high-frequency residual std) — *blocking*

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

The measured numbers (rounded into `config.yaml`):

| Metric | Measured \|Δ\| after CCM | In config |
|---|---:|---:|
| L | 9.6% | 10 |
| a* | 6.8% | 7 |
| b* | 12.0% | 12 |
| R | 10.1% | 10 |
| G | 9.7% | 10 |
| B | 6.1% | 7 |
| sharpness | 15.8% | 16 |
| noise | 9.5% | 10 |
| sharpness_fft | 10.8% | 11 |

### Known limitations to fix during tuning

1. **N = 4 scenes.** The empirical basis is small; one outlier scene can swing
   a tolerance by several percentage points.
2. **Ceiling preset is a guess.** It's set to `chessboard × 2` — not measured
   on ceiling-mounted captures. Replace it with real data.
3. **Equal weighting within the blocking group.** `sharpness`, `sharpness_fft`
   and `noise` count the same toward the tally, and any one of them failing
   fails the unit. Note `sharpness` and `sharpness_fft` measure the same
   physical property two ways, so a borderline-soft module trips two of the
   three blocking metrics at once.
4. **`noise` tolerance is 15%,** not the 9.5% measured after CCM. It is a
   deliberately looser band, chosen so that a unit is not failed for a small
   noise increase over the reference; confirm it against real known-good
   batches.
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
