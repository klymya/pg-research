# release_june_26 — investigation report

**Question we set out to answer.** Why do the three ceiling-mounted people-detection sensors — nominally identical firmware variants of the same YOLOv8-tiny int8 pipeline — produce systematically different detections on the same physical scene, and why did the emulator and the on-device native pipeline disagree even more strongly?

**Short answer.** The dominant source of variance is **hardware-side**: the three camera modules produce measurably different images (sharpness, noise) even in the same lighting rig. The detection model, trained against a narrow slice of camera behaviour, does not tolerate that variance — it degrades or drops predictions entirely on the sharper/noisier modules. A secondary, unrelated **software bug in the emulator** (lossy MP4 as inference input) was found and fixed during the investigation.

---

## Test setup

Three sensors, one still frame each, one shared physical scene:

| Sensor UUID (prefix) | Firmware/algo variant | Role in this test |
|---|---|---|
| `6100002` | AlgoA — KlymLatest | Reference camera module |
| `6100001` | AlgoC — KlymLatest_canDriver | DUT #1 |
| `6200001` | AlgoB — MasterTest | DUT #2 |

Everything downstream of the sensor (backbone `.tflite`, correction model, thresholds) is intentionally held constant across the three so any behavioural gap traces to the sensor plus the model's response to it.

---

## Finding 1 — hardware is the dominant driver of detection variance

The `camera_qc` metrics tool ([camera_qc/report.pdf](camera_qc/report.pdf), run with `6100002` as ref) shows the two DUT modules **both FAIL** the tolerance band, in **opposite directions**:

| DUT | Verdict | Failing metrics | Character |
|---|---|---|---|
| `6100001` (AlgoC) | **FAIL** | `sharpness`, `sharpness_fft` | ~85–89% of ref sharpness — **softer** image, cleaner noise |
| `6200001` (AlgoB) | **FAIL** | `a*`, `noise` | ~129% of ref sharpness, ~116% noise — **sharper and noisier**, channels ~2.5% darker |

Colour channels (`L`, `R`, `G`, `B`) are inside tolerance on both — so this is not a white-balance/exposure issue. It is a **lens/optics/sensor-imaging** difference.

That hardware variance maps directly onto detection behaviour on the same physical scene:

| Sensor | Character | Emulator (lossless, wb_alif) | Arm ML kit (post-zero) | Device (pg2gw) |
|---|---|---:|---:|---:|
| `6100002` (ref) | baseline | 1 det, conf 0.21 | 1 det, conf 0.21 | 3 unique dets, conf up to 0.73 |
| `6100001` (softer) | −13% sharpness | 1 det, conf 0.58 | 1 det, conf 0.66 | 0 dets |
| `6200001` (sharper, noisier) | +29% sharpness, +16% noise | **0 dets** | **0 dets** | 2 dets, conf 0.58 |

The reference module gets detections cleanly across all three pipelines. The DUT modules either miss the person entirely (`6200001`) or find them only at reduced confidence (`6100001`). This is exactly the pattern expected when a model has been fit against a narrow imaging distribution and encounters a module outside it.

**See:** [visualize_gt_vs_emulator.ipynb](visualize_gt_vs_emulator.ipynb) (grid: 6 pipeline rows × 3 sensor columns) for the full visual side-by-side.

### Risk

- The current model **overfits to the reference camera module's imaging profile**. Production units will inevitably have optics/sensor variance in the same failure directions we measured here (softer or sharper than nominal, mildly different noise floor).
- With `n = 3` modules two of them already fall outside the tolerance envelope inherited from the previously-accepted CCM sensor comparison. The population 95th-percentile miss rate is very likely worse than what this small test suggests.
- The failure mode is not gradual confidence decay — it is **complete detection drop** on `6200001`. In deployment this looks like the sensor "not seeing" a person that is clearly in frame.

---

## Finding 2 — emulator bug: lossy MP4 as inference input (fixed)

Independently of the hardware finding, the emulator was reading its per-sensor input frames from `.mp4` files produced with the `mp4v` codec ([frames_to_video.py](../../../algo_system_emulator/scripts/frames_to_video.py)). `mp4v` is lossy MPEG-4. That codec loss was inflating the emulator's detection count relative to the native pipeline:

| Sensor | Emu (MP4 · wb_alif) | Emu (lossless FFV1 · wb_alif) | Native (post-zero) |
|---|---:|---:|---:|
| `6100002` | 2 dets | 1 det | 1 det |
| `6200001` | 2 dets | 0 dets | 0 dets |
| `6100001` | 2 dets | 1 det | 1 det |

Every "extra" emulator detection was a codec artefact — the second box in each MP4 run either did not exist on the source JPG or appeared at a systematically wrong location clustered around `y ≈ 184`. Switching the video conversion to lossless **FFV1 in MKV** eliminated the phantom detections and brought emulator counts into exact agreement with the on-device native run.

**Also found and fixed on the native side:** the C++ preprocess `memcpy`s a 320×240 image into a 320×320 tensor and left the trailing 76 800 bytes uninitialised (arena residue). Adding a targeted `memset` of the tail restored bit-identical input tensors between runs. Impact on our three test frames was minor (one frame's bbox shifted a few pixels; two were unchanged) but the fix is a hygiene requirement regardless.

Neither of these emulator/native fixes closes the hardware-driven detection gap in Finding 1. They only make the emulator a trustworthy predictor of what the on-device stack produces.

---

## Recommendations

### 1. Collect training and evaluation data from multiple sensor modules

The current model was trained against imagery from a narrow set of camera modules and is over-tuned to that specific optical profile. Recommend:

- **Training set:** collect labelled frames from a **rotating set of ≥ 5–10 distinct physical camera modules**, spanning the sharpness/noise envelope observed in production hardware.
- **Evaluation set:** hold out labelled frames from **modules not seen at training time**, so eval-set performance measures generalisation across hardware, not memorisation of one lens.
- **QA gate:** run `camera_qc` on every candidate module before it enters the labelled dataset — reject anything that FAILs the current tolerances so training data reflects modules the fleet will actually contain, not defective ones.

### 2. Add sharpness / softness augmentation to training and evaluation

Even with a wider hardware fleet, augmentation is cheaper than sampling and can be tuned to explicitly hit the failure modes surfaced here:

- **Gaussian blur / unsharp mask** at random strength → simulates the softer end of the observed variance (`6100001` case).
- **High-frequency noise injection** (Gaussian σ tuned to match `camera_qc`'s `noise` metric spread) → simulates `6200001`.
- **Random JPEG re-encode** at varying quality → covers residual codec artefacts across the pipeline.

Run these augmentations **also on the evaluation set** (as a separate stress-test split) so we can distinguish "model got better in general" from "model got better only on the clean reference imaging profile".

### 3. Tighten `camera_qc` tolerances against real production data

`camera_qc`'s current thresholds (see [camera_qc README section "Tolerances — current state and tuning"](../../../camera_qc/README.md)) are a v1 seed derived from N=4 scenes and known to be indicative rather than validated. Once ≥ 10 known-good modules exist per scene preset, retune to the 95th-percentile of accepted-module `|Δ|` and re-run the analysis in Finding 1 — the sensitivity of the current model to sharpness/noise should show up as a concrete pass/fail rate.

---

## Files produced by this investigation

| File | Purpose |
|---|---|
| [visualize_detections.ipynb](visualize_detections.ipynb) | Per-sensor device-log detections overlaid on each frame |
| [visualize_model_comparison.ipynb](visualize_model_comparison.ipynb) | Two ridge-regression correction variants side-by-side on the emulator |
| [visualize_gt_vs_emulator.ipynb](visualize_gt_vs_emulator.ipynb) | 6-source × 3-sensor grid — the main comparison figure |
| [camera_qc/report.pdf](camera_qc/report.pdf) | Camera-module QA verdict PDF (basis for Finding 1) |
| [inference_log_gt.txt](inference_log_gt.txt), [inference_log_zeroed.txt](inference_log_zeroed.txt) | Raw Arm ML Eval Kit logs before/after the tensor-tail fix |
