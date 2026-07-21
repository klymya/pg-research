# 4-Layer Non-NN Background Subtraction Prototype

Prototype of the confirmed 4-layer background subtraction (BGS) architecture for
**"Sign of Life"** detection: isolating newly introduced *static* objects while
suppressing dynamic office noise (shadows, lighting shifts, camera drift).

The package is pure algorithm code. It performs **no video IO, no dataset
reading and no evaluation** — the caller supplies decoded frames and optional
person bounding boxes, and receives per-frame masks, blobs and events.

## Architecture

```txt
[ Frame Input ]
       │
       ▼
 ┌───────────┐
 │  Layer 0  │ preprocessing.py — denoise, colour space, registration
 └─────┬─────┘
       │  (Frame: bgr + lazy gray/hsv views)
       ▼
 ┌───────────┐     ┌───────────┐
 │  Layer 2  │ ──> │  Layer 1  │ features.py -> bgs_model.py + backends.py
 └───────────┘     └─────┬─────┘
                         │  (instantiated twice: fast + slow)
                         ▼
                   ┌───────────┐
                   │  Layer 3  │ dual_rate.py + blobs.py + fsm.py
                   └─────┬─────┘
                         ▼
               [ "Sign of Life" Signal ]
```

| Module | Layer | Responsibility |
| --- | --- | --- |
| `config.py` | — | Frozen, self-validating dataclasses for every layer |
| `structures.py` | — | `Frame`, `Blob`, `TrackedObject`, `DualRateMasks`, `PipelineResult` |
| `utils.py` | — | Mask binarisation, channel coercion, IoU |
| `preprocessing.py` | 0 | Gaussian/bilateral denoising, phase-correlation or ECC registration |
| `features.py` | 2 | Grayscale / LBP / chromaticity metrics + HSV shadow veto |
| `backends.py` | 1 | Swappable subtraction algorithms behind one adapter interface |
| `bgs_model.py` | 1 | `RateParameterizedBGS` — feature + backend(α) + shadow veto |
| `dual_rate.py` | 3 | Fast/slow model pair and the static-candidate cue |
| `blobs.py` | 3 | Morphology and area-gated connected components |
| `fsm.py` | 3 | IoU association, dwell-time promotion, person veto |
| `pipeline.py` | — | `SignOfLifePipeline` chaining all of the above |
| `../main.py` | — | CLI runner: reads a sequence, writes binary PNG masks |

### Design notes

**Layer 2 is a strategy, not a stage.** "Use LBP or chromaticity instead of raw
intensity" is a *feature transform* injected into Layer 1, plus a *post-filter*
on its output mask. Modelling it as `FeatureExtractor` + `ShadowSuppressor`
rather than a pipeline stage keeps the data flow honest and makes the metric
swappable in one config field.

**The learning rate is not universal.** OpenCV's MOG2/KNN accept a per-call
`learningRate`; `pybgs` algorithms such as ViBe and SuBSENSE manage their own
adaptation schedule and expose none. Backends therefore declare
`supports_learning_rate`, and Layer 1 logs when α cannot be honoured instead of
silently ignoring it. See
[pybgs cannot form the dual-rate pair alone](#pybgs-cannot-form-the-dual-rate-pair-alone)
for the consequence.

**Every model keeps its own BGR reference.** The HSV shadow veto needs a
background estimate, but not every backend exposes its internal model.
`RateParameterizedBGS` maintains an exponentially weighted BGR reference at the
same α, so the veto works with any backend.

### ⚠️ The static-object rule is inverted in the task description

The task states the signal should fire when an item is *"present in the Fast
mask but absent in the Slow mask"*, but the same paragraph specifies that the
slow model *"preserves newly static objects"*. These contradict each other.

Physically: a deposited object is absorbed quickly by the **fast** model, so it
**disappears from the fast foreground**, while the **slow** model keeps
reporting it. The abandoned-object cue is therefore:

```python
static_candidate = slow_mask AND NOT fast_mask
```

This is the standard dual-background formulation and what `dual_rate.py`
implements. The task's phrasing describes transient motion, not a left object.

### ⚠️ `pybgs` cannot form the dual-rate pair alone

Verified empirically: `pybgs` algorithms expose **no constructor arguments**
(`ViBe.__init__(self) -> None`) and no per-call learning rate. Two instances of
the same algorithm are therefore identical and produce **bit-identical masks**,
so `slow AND NOT fast` is always empty and no signal can ever fire:

```text
fast=vibe, slow=vibe  ->  fast == slow (identical), static = 0 px
```

Use a **mixed pair** instead — a rate-controlled fast model against a `pybgs`
slow model. All three combinations below detect the deposited object correctly:

| fast | slow | first event |
| --- | --- | --- |
| `running_average` (α=0.6) | `vibe` | frame 21, bbox exact |
| `mog2` (α=0.6) | `subsense` | frame 19 |
| `running_average` (α=0.6) | `subsense` | frame 21 |

Consequently `backend_params` is unusable for `pybgs` backends: passing any
keyword raises `TypeError` from the pybind11 constructor.

## Pipeline walkthrough

`SignOfLifePipeline.process(frame_bgr, person_boxes)` runs one frame through
the layers below and returns a `PipelineResult`. State (background models,
warm-up counter, tracks) persists across calls, so frames must be fed in
temporal order.

```
raw BGR frame
  │
  ├─▶ L0  denoise ─▶ (optional) register to first frame        → Frame(bgr, gray, hsv)
  │
  ├─▶ L2  feature transform (gray / LBP / chromaticity)        → feature image
  │
  ├─▶ L1  fast model.apply(feature, α_fast) ─▶ shadow veto     → fast mask
  │       slow model.apply(feature, α_slow) ─▶ shadow veto     → slow mask
  │
  ├─▶ L3a static = slow AND NOT fast   (empty during warm-up)  → static mask
  │
  ├─▶ L3b morphology (open→close) ─▶ connected components ─▶ area gate → blobs
  │
  └─▶ L3c IoU-associate blobs to tracks ─▶ dwell counter ─▶ person veto
                                                       → events (Sign of Life)
```

### Layer 0 — preprocessing (`preprocessing.py`)

1. **Denoise.** `gaussian` (default, `kernel_size=5`) or `bilateral`
   (edge-preserving, `sigma_color=25`, `sigma_space=7`), or `none`.
2. **Colour views.** The result is wrapped in a `Frame` that lazily derives
   `gray` and `hsv` on first access and caches them.
3. **Registration (optional, default `none`).** The first frame becomes the
   reference; each later frame is aligned to it by a **translation-only** model
   — `phase_correlation` (FFT, fast) or `ecc` (`cv2.findTransformECC`,
   `MOTION_TRANSLATION`, 50 iters). Estimates above `max_shift_px=20` are
   rejected as implausible and the frame passes through unwarped. Rotation and
   scale drift are out of scope.

### Layers 1 & 2 — rate-parameterized model + illumination invariance

Layer 2 is applied **inside** Layer 1, not as a separate stage:

- **Feature transform** (`features.py`) replaces raw intensity with an
  illumination-robust metric before modelling:
  - `gray` — raw grayscale (baseline).
  - `lbp` — 8-neighbour local binary pattern; invariant to any monotonic
    intensity change (global lighting shifts), `radius=1`.
  - `chromaticity` — normalised RGB (`c / (r+g+b)`), discards the intensity
    component.
- **Background model** (`bgs_model.py` + `backends.py`) — a
  `RateParameterizedBGS` wrapping a swappable backend whose learning rate α is
  an explicit argument. The default `running_average` backend maintains
  `bg = (1−α)·bg + α·frame` and flags a pixel as foreground when any channel
  deviates by more than `threshold=25`.
- **Shadow veto** (`ShadowSuppressor`, on by default) — a post-filter that
  drops foreground pixels whose HSV signature matches a cast shadow: brightness
  ratio `V_frame/V_bg ∈ [0.4, 0.95]` with hue and saturation roughly unchanged
  (`|ΔH| ≤ 12`, `|ΔS| ≤ 40`). The veto compares against an exponentially
  weighted BGR reference each model keeps at its own α, so it works even for
  backends that hide their internal model.

### Layer 3 — dual-rate framework, structural filter, FSM

- **L3a dual rate** (`dual_rate.py`) — two Layer-1 models observe every frame:
  a **fast** model (high α, absorbs a deposited object quickly, so it leaves
  the fast foreground) and a **slow** model (low α, keeps reporting it). The
  static-object cue is `static = slow AND NOT fast`
  (see [the caveat above](#️-the-static-object-rule-is-inverted-in-the-task-description)).
  During the first `warmup_frames` the static mask is forced empty so the slow
  model can converge.
- **L3b structural filter** (`blobs.py`) — morphological **opening**
  (`open_kernel=3`, removes speckle) then **closing** (`close_kernel=5`,
  bridges gaps), connected components, and an **area gate**
  (`min_area=150`, `max_area=None`). Survivors become `Blob`s.
- **L3c dwell-time FSM** (`fsm.py`) — blobs are associated to existing tracks
  by bounding-box IoU (`iou_match_threshold=0.3`). A track accumulates *dwell*
  while it keeps being seen and is promoted to **confirmed** — the Sign-of-Life
  event — once `dwell ≥ dwell_frames`. A track unseen for more than
  `max_missing_frames=15` is evicted. Any candidate overlapping an active
  person box beyond `person_overlap_threshold=0.1` IoU is vetoed and its dwell
  reset, so a stationary person raises no alarm.

### Background-model initialization & convergence time

The model is **cold-started on frame 0** — the first frame is copied straight
into the background, so a single frame is enough to begin producing masks (this
matches the task's "1–6 frame initialisation" for the `median` backend, whose
window fills in ≤ 6 frames). What takes longer is *convergence*: how fast the
running-average background absorbs a change, governed entirely by α.

For `bg = (1−α)·bg + α·frame`, the old estimate decays by `(1−α)` per frame:

| Role | Backend | α (default) | Half-life | ~95% converged | Meaning |
| --- | --- | --- | --- | --- | --- |
| Fast | `running_average` | `0.1` | ~7 frames | ~28 frames | Absorbs a stopped object in seconds |
| Slow | `running_average` | `0.001` | ~690 frames | ~3000 frames | Retains it for minutes |
| — | `median` | window=6 | — | ≤ 6 frames | No α; adaptation set by window length |

The **fast/slow separation is the whole mechanism**: the object must vanish
from the fast mask (fast α high enough to absorb it within the dwell window)
while surviving in the slow mask (slow α low enough to keep it for at least
`dwell_frames`). A useful rule of thumb: `1/α_fast ≪ dwell_frames ≪ 1/α_slow`.
`warmup_frames` should exceed the slow model's rise time or the whole scene
reads as a static candidate on the first post-warm-up frames.

### Main parameters at a glance

Library defaults (`PipelineConfig()`); the `main.py` CLI exposes the same
values under the flags in brackets.

| Parameter | Default | Layer | Effect |
| --- | --- | --- | --- |
| `denoise` `[--denoise]` | `gaussian` | 0 | Sensor-noise filter |
| `registration` `[--registration]` | `none` | 0 | Camera-drift compensation |
| `feature.kind` `[--feature]` | `gray` | 2 | Differencing metric (`gray`/`lbp`/`chromaticity`) |
| `shadow.enabled` `[--no-shadow-veto]` | `True` | 2 | HSV moving-shadow veto |
| `fast.backend` / `slow.backend` `[--fast/slow-backend]` | `running_average` | 1 | Subtraction algorithm |
| `fast.learning_rate` `[--fast-alpha]` | `0.1` | 1/3 | Fast absorption rate |
| `slow.learning_rate` `[--slow-alpha]` | `0.001` | 1/3 | Slow absorption rate |
| `threshold` `[--threshold]` | `25.0` | 1 | Native-backend foreground threshold |
| `warmup_frames` `[--warmup-frames]` | `30` | 3 | Static cue suppressed until slow model settles |
| `blob.min_area` `[--min-area]` | `150` | 3 | Smallest accepted blob (px) |
| `dwell_frames` `[--dwell-frames]` | `60` | 3 | Presence budget before a Sign-of-Life event |
| `max_missing_frames` `[--max-missing-frames]` | `15` | 3 | Frames a track survives unmatched |
| `iou_match_threshold` | `0.3` | 3 | Track↔blob association |
| `person_overlap_threshold` | `0.1` | 3 | IoU with a person box that vetoes a candidate |

Every config validates in `__post_init__`, so an illegal combination (e.g. a
fast model slower than the slow one) fails at construction, not mid-stream.

## Installation

Uses the `pgpytrain` pyenv environment:

```bash
pyenv activate pgpytrain
cd background_subtraction
python -m pytest tests -q
```

### Verified versions

| Component | Version |
| --- | --- |
| Python | 3.10.13 (`pyenv` env `pgpytrain`) |
| `opencv-python` | 4.13.0.92 (`cv2.__version__` → `4.13.0`) |
| `pybgs` | 3.3.1 (dist metadata; `pybgs.__version__` reports `3.3.0`) |
| `numpy` | 2.2.6 |
| `pytest` | 9.0.1 |

`pybgs` reports a different internal version than its distribution metadata —
upstream does not bump the compiled constant. Trust `pip show pybgs` (3.3.1).

Note that `opencv-contrib-python` 4.7.0.72 is also present in the environment.
Both packages provide the `cv2` module; `opencv-python` 4.13.0.92 currently
wins on `sys.path`. This is a latent hazard — a reinstall of either package can
silently change the effective OpenCV version.

### `pybgs` (bgslibrary)

`pybgs` is a source distribution that compiles against the OpenCV C++ headers,
so it requires `libopencv-dev` **before** `pip install`:

```bash
sudo apt install libopencv-dev cmake
pip install pybgs
```

Both are installed in this environment and all 26 bgslibrary algorithms are
available (`ViBe`, `SuBSENSE`, `LOBSTER`, `PAWCS`, `KDE`, `CodeBook`, …).

The import is still lazy, so the pipeline runs without `pybgs`; only the
`vibe`, `subsense`, `lobster` and `pawcs` backends become unavailable, and the
`pybgs`-dependent tests skip automatically.

Two behaviours to be aware of: `PAWCS` and `LOBSTER` look for `./config/*.xml`
relative to the **current working directory** and log a harmless failure when
absent, and `PAWCS` warns when a grayscale image arrives as 3-channel — which
it does today, since `PybgsBackend` replicates single-channel features to BGR.

## Running on a dataset

`main.py` runs the pipeline over an image sequence and writes one binary PNG
per frame, following the CDNET result convention (`bin%06d.png`, mirroring the
input frame number):

```bash
python main.py --input data/busStation --output results/busStation
```

The `--input` path may be a CDNET sequence directory or the `input/`
subdirectory itself — a nested `input/` is detected and used automatically.
The output directory is created if missing.

Masks are single-channel `uint8` containing only `0` and `255`.

### Selecting what to save

`--mask` chooses which pipeline output is persisted:

| Value | Content |
| --- | --- |
| `filtered` (default) | Static candidates after morphology and area gating — the pipeline's final product |
| `static` | Raw `slow AND NOT fast` cue, before structural filtering |
| `slow` / `fast` | Individual model foregrounds, useful for tuning α and diagnosing an empty result |

An empty `filtered` output is often correct rather than a failure — on a
dynamic-background sequence such as `canoe` the static cue is water ripple,
which opening and the 150 px area gate legitimately remove. Re-run with
`--mask slow` to confirm the earlier layers are producing foreground.

### Useful options

```bash
# texture metric, mixed pybgs pair, shorter dwell
python main.py -i data/busStation -o results/bus \
    --feature lbp --fast-backend running_average --slow-backend subsense \
    --fast-alpha 0.6 --slow-alpha 0.0005 --dwell-frames 30 --warmup-frames 50

# quick smoke run over the first 200 frames
python main.py -i data/traffic -o results/traffic --limit 200 --mask slow
```

`--help` lists every option, grouped by layer. Confirmed events are logged as
they fire:

```text
INFO Sign of life: frame 298 (in000299.jpg), track 0, bbox (36, 57, 19, 36)
INFO Wrote 700 masks, 2 sign-of-life events
```

The person-suppression veto is not exercised by the CLI — no human detector is
wired in, so `person_boxes` is always empty. Supply it through the Python API
when a tracker is available.

## Python API

```python
import numpy as np

from scr import PipelineConfig, SignOfLifePipeline
from scr.config import BGSConfig, DualRateConfig, DwellConfig, FeatureConfig, FeatureKind

config = PipelineConfig(
    feature=FeatureConfig(kind=FeatureKind.LBP),
    dual_rate=DualRateConfig(
        fast=BGSConfig(backend="mog2", learning_rate=0.05),
        slow=BGSConfig(backend="mog2", learning_rate=0.0005),
        warmup_frames=60,
    ),
    dwell=DwellConfig(dwell_frames=90),
)
pipeline = SignOfLifePipeline.from_config(config)

for frame_bgr, person_boxes in stream:          # caller-supplied frames
    result = pipeline.process(frame_bgr, person_boxes)
    if result.has_sign_of_life:
        for event in result.events:
            print(f"left object {event.track_id} at {event.bbox}")
```

`result` also carries `masks.fast`, `masks.slow`, `masks.static`,
`filtered_mask`, `blobs` and `tracked` for inspection and visualisation.

Individual layers are usable standalone — each has a `from_config` constructor
and a `reset()`:

```python
from scr import DualRateBGS, Preprocessor, RateParameterizedBGS

model = RateParameterizedBGS.from_config(BGSConfig(backend="knn", learning_rate=0.01))
mask = model.apply(Preprocessor().process(frame_bgr))
```

## Backends

`available_backends()` lists the registered names:

| Name | Source | Honours α | Notes |
| --- | --- | --- | --- |
| `running_average` | native | ✅ | Primary baseline; `bg = (1-α)·bg + α·frame` |
| `median` | native | ❌ | Sliding-window median; 1–6 frame initialisation |
| `mog2` | OpenCV | ✅ | Shadow pixels (127) folded into background |
| `knn` | OpenCV | ✅ | Local evaluation baseline |
| `vibe`, `subsense`, `lobster`, `pawcs` | `pybgs` | ❌ | Slow model only — see the [dual-rate caveat](#️-pybgs-cannot-form-the-dual-rate-pair-alone) |

Extra constructor arguments are passed through `BGSConfig.backend_params`, e.g.
`BGSConfig(backend="mog2", backend_params={"history": 500, "detectShadows": False})`.
`pybgs` backends accept no arguments, so leave `backend_params` empty for them.

## Configuration reference

| Config | Key parameters |
| --- | --- |
| `PreprocessConfig` | `denoise`, `kernel_size`, `registration`, `max_shift_px` |
| `FeatureConfig` | `kind` (`gray`/`lbp`/`chromaticity`), `lbp_radius`, `lbp_blur_ksize` |
| `ShadowConfig` | `value_ratio_min/max`, `saturation_delta_max`, `hue_delta_max` |
| `BGSConfig` | `backend`, `learning_rate`, `threshold`, `backend_params` |
| `DualRateConfig` | `fast`, `slow`, `warmup_frames` |
| `BlobConfig` | `min_area`, `max_area`, `open_kernel`, `close_kernel`, `connectivity` |
| `DwellConfig` | `dwell_frames`, `max_missing_frames`, `iou_match_threshold`, `person_overlap_threshold` |

All configs validate in `__post_init__`, so an invalid combination fails at
construction rather than mid-stream. `DualRateConfig` additionally rejects a
fast model that is slower than the slow one.

## Tests

```bash
pyenv activate pgpytrain
cd background_subtraction
python -m pytest tests -q
```

203 tests over synthetic scenes — **202 pass, 1 skips** (the skip asserts the
error message raised when `pybgs` is *absent*, so it is inactive here). Coverage
spans config validation, LBP illumination invariance, shadow veto behaviour,
backend absorption dynamics at different α, blob gating, FSM dwell/veto/expiry
transitions, end-to-end left-object detection including the person-suppression
path, and the CLI runner (sequence discovery, result naming, mask selection,
argument mapping and exit codes).

## Scope

The `scr/` package is **algorithm only** — it performs no IO and consumes
decoded frames handed to it by the caller. All file access lives in `main.py`,
which reads a sequence and writes masks. **Evaluation is still not
implemented**: no metric is computed against the groundtruth.

The CDNET-14 sequences referenced by the task are already present under
`data/` (`boats`, `busStation`, `canoe`, `traffic`, `tramCrossroad_1fps`,
`turnpike_0_5fps`), each with `input/`, `groundtruth/`, `ROI.bmp` and
`temporalROI.txt`. The natural next step is a thin frame reader plus an
evaluator scoring `filtered_mask` against `groundtruth/` over the temporal ROI
(recall / precision / F-measure per the CDNET protocol). That is deliberately
not implemented here.

## Known limitations

* **`pybgs` backends cannot be paired with themselves** — see the
  [dual-rate caveat](#️-pybgs-cannot-form-the-dual-rate-pair-alone). Use a
  mixed fast/slow pair.
* `PybgsBackend` replicates single-channel features to 3 channels, which is
  wasteful for grayscale/LBP metrics and triggers a `PAWCS` performance
  warning. All four registered algorithms accept `CV_8UC1` directly, so the
  coercion could be dropped.
* Registration is translation-only. Rotation and scale drift are out of scope
  for the prototype; ECC can be extended to `MOTION_EUCLIDEAN` if needed.
* The FSM associates by bounding-box IoU alone. Objects deposited within one
  IoU threshold of each other may be merged into a single track.
* `warmup_frames` must exceed the slow model's convergence time, otherwise the
  entire scene registers as a static candidate on the first post-warmup frames.
