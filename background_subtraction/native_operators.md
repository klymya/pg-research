# Native operators, algorithms & methods — pipeline ∩ notebook

Catalog of the **native** (pure-NumPy, self-implemented, no OpenCV) operators,
algorithms and methods that the current single-model pipeline uses **and** that
`pipeline_experiments.ipynb` exercises.

## Scope & method

- **"Native"** = implemented in this repository in pure NumPy/Python with no
  OpenCV (`cv2`) or third-party CV call. OpenCV-backed steps are listed
  separately in the [appendix](#appendix--still-opencv-backed-in-the-same-path).
- **"Used in the pipeline"** = on the per-frame data path of
  `SignOfLifePipeline.process()`.
- **"Used in the notebook"** = exercised by `pipeline_experiments.ipynb`, which
  builds the pipeline with `Implementation.NATIVE` (`running_gaussian`, α = 0.1,
  QVGA, gray feature, HSV shadow veto, elliptical open/close).
- Verified by reading the source (`scr/`) and the notebook, not inferred.

## Native building blocks (used by both the pipeline and the notebook)

### Background model — Layer 1

| Operator / method | What it does | Algorithm / source | Location |
| --- | --- | --- | --- |
| `RunningGaussianAverage` | Single Gaussian per pixel (running mean + shared variance); foreground when the squared distance to the mean exceeds `threshold · variance` | Wren et al. *Pfinder* running Gaussian average; ports bgslibrary `WrenGA` | `scr/models/running_gaussian.py` |
| `prepare_frame` | Coerce a frame to `float32`, shape `(H, W, C)` | NumPy dtype/axis handling | `scr/models/base.py` |
| `to_mask` | Boolean foreground → `uint8` `0`/`255` mask | NumPy | `scr/models/base.py` |
| `NativeModelBackend` | Adapts a pure-NumPy model to the `BGSBackend` interface (fixed rate at construction) | Adapter pattern | `scr/backends.py` |

### Colour conversion — Layer 2 (shadow veto)

| Operator / method | What it does | Algorithm / source | Location |
| --- | --- | --- | --- |
| `bgr_to_hsv` | BGR → HSV (8-bit), **bit-exact** with `cv2.COLOR_BGR2HSV` | Fixed-point integer path, ported from OpenCV `RGB2HSV_b` (`color_hsv.simd.hpp`, `hsv_shift = 12`, `sdiv`/`hdiv` LUTs) | `scr/native_ops.py` |

### Shadow suppression — Layer 2

| Operator / method | What it does | Algorithm / source | Location |
| --- | --- | --- | --- |
| `ShadowSuppressor` (veto logic) | Flags a foreground pixel as cast shadow when `V_frame/V_bg ∈ [0.4, 0.95]`, `|ΔS| ≤ 40` and circular `|ΔH| ≤ 12` | HSV brightness-ratio shadow test (NumPy comparisons; HSV via `bgr_to_hsv`) | `scr/features.py` |

### Morphology — Layer 3 (structural filter)

| Operator / method | What it does | Algorithm / source | Location |
| --- | --- | --- | --- |
| `get_structuring_element` | Build the elliptical kernel (3×3 open, 5×5 close), **bit-exact** with `cv2.getStructuringElement(MORPH_ELLIPSE)` | Ported from OpenCV `getStructuringElement` (`morph.dispatch.cpp`) | `scr/native_ops.py` |
| `erode` | Min filter over the structuring element (default border 255) | OpenCV `morph.simd.hpp` erosion | `scr/native_ops.py` |
| `dilate` | Max filter over the structuring element (default border 0) | OpenCV `morph.simd.hpp` dilation | `scr/native_ops.py` |
| `morphology_ex` (`OPEN`, `CLOSE`) | Opening = erode→dilate, closing = dilate→erode; **bit-exact** with `cv2.morphologyEx` | Compound morphology | `scr/native_ops.py` |

### Connected components & contour — Layer 3

| Operator / method | What it does | Algorithm / source | Location |
| --- | --- | --- | --- |
| `connected_components` | Label the foreground regions and return per-label bbox / area / centroid; **matches** `cv2.connectedComponentsWithStats` (stats + centroids, 4-/8-connectivity) | Two-pass union-find labelling (Rosenfeld & Pfaltz; He et al.) | `scr/native_ops.py` |
| `find_contour` | Trace an object's outer boundary → ordered `(x, y)` points; point set **matches** `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_NONE)` | Moore-neighbour boundary tracing (Jacob's stopping criterion) | `scr/native_ops.py` |

> `BlobFilter` (Layer 3) uses `connected_components` on the native path and
> attaches each surviving blob's `find_contour` result to `Blob.contour`
> (with the bounding box from the component stats). The notebook draws these —
> contour and bbox — in the 5th grid panel.

### Mask & geometry helpers — Layers 1/3

| Operator / method | What it does | Algorithm / source | Location |
| --- | --- | --- | --- |
| `ensure_binary_mask` | Normalise any mask to `uint8` `0`/`255` (folds OpenCV shadow value 127 to background) | NumPy threshold | `scr/utils.py` |
| `iou` | Intersection-over-union of two `(x, y, w, h)` boxes | Axis-aligned box IoU | `scr/utils.py` |
| `StaticObjectFSM` | Dwell-time tracking: IoU association, dwell counter, person-overlap veto → confirmed events | Native finite-state machine | `scr/fsm.py` |

> The FSM runs every frame in the notebook but raises no sign-of-life event at
> 0.1 fps (too few frames to accumulate dwell); it is on the path and native, so
> it is listed here for completeness.

## Note — `hsv_to_bgr` (notebook only)

`scr/native_ops.py::hsv_to_bgr` (HSV → BGR, float path, matches `cv2` **within
1 LSB**) is the inverse of `bgr_to_hsv`. It is **not on the pipeline data
path** — the pipeline only needs the forward conversion. The notebook uses it
in the *"HSV shadow filter"* grid panel (`bgr_to_hsv` → tint detected shadows in
HSV → `hsv_to_bgr`), so it is native and used in the notebook, but **not** in the
pipeline itself.

## Appendix — still OpenCV-backed in the same path

For the C / Alif port, these steps on the notebook's pipeline path are **not yet
native** (they still call `cv2`):

| Step | OpenCV call | Location |
| --- | --- | --- |
| Resize to QVGA (Layer 0) | `cv2.resize` (`INTER_AREA`) | `scr/preprocessing.py` |
| Gaussian denoise (Layer 0) | `cv2.GaussianBlur` | `scr/preprocessing.py` |
| Grayscale feature (Layer 2) | `cv2.cvtColor(BGR2GRAY)` via `Frame.gray` | `scr/structures.py`, `scr/features.py` |
| Shadow-veto background reference (Layer 1) | `cv2.accumulateWeighted` | `scr/bgs_model.py` |
| Frame decode (notebook IO) | `cv2.imread` | `pipeline_experiments.ipynb` |

Connected components is now native (`connected_components`); on the library path
`BlobFilter` still calls `cv2.connectedComponentsWithStats`, selected by
`BlobConfig.implementation`.

## Porting to C/C++ for Alif

This section is the working spec for the C implementation and its integration
into the Alif ML embedded evaluation kit. It maps every operator above (plus the
Layer-0 steps still done with OpenCV in Python) onto a concrete C/C++ approach:
what to reuse from the kit, what to write natively, the data type to use, and
how to keep numeric parity with this Python reference.

Sources: `alif_syntaxis_standart.md` (standards/toolchain, Eigen assessment),
`readme.md` (pipeline), and the kit itself
(`~/projects/rtos_sensor/projects/alif_ml-embedded-evaluation-kit_pg/`,
`~/projects/rtos_sensor/dependencies/eigen`).

### Channel order & frame size — the C port is RGB + QVGA

Verified from the kit's inference path: **the Alif frames are RGB, QVGA**, and no
channel swap happens before the tensor.

- Camera → RGB888 (`image_processing.c`, *"Modified for RGB888"*;
  `hal_get_image_data`); embedded test images → `gen_rgb_cpp.py`
  (`Image.open(...).convert("RGB")`); `DetectorPreProcess::DoPreProcess` is a
  plain `memcpy` into the tensor (only an optional uint8→int8 `−128` offset). A
  tree-wide search finds **no `BGR`** in `source/`.
- Frames are **QVGA** already, so no resize-for-model is needed on device.

**Therefore the C port is RGB-native and QVGA-fixed** — implement `rgb_to_hsv`
(not `bgr_to_hsv`) and treat channel 0 as R, channel 2 as B throughout; **no
per-operator R↔B swap is required on Alif.**

This diverges from the Python reference (BGR, because it decodes CDNET with
`cv2.imread`), but the two are **behavior-equivalent**: HSV, grayscale and the
running-Gaussian distance are computed from the real R/G/B *values*, not their
storage order. So `rgb_to_hsv(rgb_frame)` on Alif equals `bgr_to_hsv(bgr_frame)`
in Python for the same scene, and the F-scores carry over. Wherever this spec
says "RGB", the Python column still shows the BGR-named reference function — same
algorithm, R and B roles swapped.

### Target constraints (recap)

- **C99 / C++14, strict ISO** — no GNU extensions, **no C++17**.
- **`-fno-exceptions -fno-rtti`** — no `throw` / `try` / `dynamic_cast` / `typeid`.
  Report failures by return code/status; pre-size buffers instead of relying on
  container growth that could throw.
- **Cortex-M55/M85 + Helium/MVE**, hardware **single-precision** FP (`f32`),
  `fp16`/`int8` MVE available for later optimisation.
- **Style:** Apache-2.0 SPDX header; `#ifndef` guards (no `#pragma once`);
  `arm::app` nested namespaces; `PascalCase` types *and* methods, `m_` members;
  `.hpp/.cpp` (C++) and `.h/.c` (C); LLVM `.clang-format`, 4-space, 100-col,
  Allman braces for functions.

### Reuse what the kit already provides

Prefer these vetted, already-compiling facilities over new code:

| Python operator | Reuse in the kit | Path | Caveat |
| --- | --- | --- | --- |
| Grayscale feature (`Frame.gray`) | `arm::app::image::RgbToGrayscale` | `source/application/api/common/include/ImageUtils.hpp` | Already **RGB** — matches the camera frame directly (no swap). Confirm its coefficients match cv2 BT.601 (`0.299R+0.587G+0.114B`) before assuming bit-parity with the Python gray |
| Layer-0 resize | `interpolate()` / `crop_and_interpolate()` / `frame_crop()` | `source/hal/.../image/.../image_processing.h` | Frames are already **QVGA**, so no model-resize is needed; use only if the sensor size differs. **Bilinear**, not `INTER_AREA` — expect small drift if used |
| FSM IoU + boxes (Layer 3) | `arm::app::image::Box`, `CalculateBoxIOU`, `CalculateBoxIntersect/Union`, `CalculateNMS` | `ImageUtils.hpp` | `Box` is centre `(x,y,w,h)`; our `Blob.bbox` is corner `(x,y,w,h)` — convert |
| Scalar math (variance, stddev, dot) | `arm::app::math::MathUtils::SqrtF32 / MeanF32 / StdDevF32 / DotProductF32` (CMSIS-DSP-backed) | `source/math/include/PlatformMath.hpp` | The running-Gaussian model needs **no `sqrt`** (it compares squared distance to `threshold·variance`) |
| Small dense linear algebra | **Eigen 3.3.9** (header-only) | `dependencies/eigen`, pattern in `pg_object_detection/src/kalmanFilter.cc` | **Not needed** for the `running_gaussian` pipeline; only `EigenBackground` (SVD) would use it — see the Eigen assessment in `alif_syntaxis_standart.md` |

### Operators to implement natively in C

Ordered by porting difficulty. "Parity" = whether the C port can be made
**bit-identical** to this Python reference.

| Operator (Python) | C approach | Type | Parity | Notes |
| --- | --- | --- | --- | --- |
| `bgr_to_hsv` → **`rgb_to_hsv`** | Same integer path as the Python `bgr_to_hsv`, but with **R = channel 0, B = channel 2** (`blueIdx = 2`): two `const int32_t[256]` LUTs (`sdiv`, `hdiv`), `hsv_shift=12`, per-pixel min/max/select | `uint8`(RGB)→`uint8`(HSV), `int32` temps | **Bit-exact** vs `cv2.COLOR_RGB2HSV` | No float. ~2 KB const LUTs. Feed the camera RGB frame directly — no swap |
| `erode` / `dilate` / `morphology_ex` | Min/max over a precomputed structuring-element offset list; border = 255 (erode) / 0 (dilate) | `uint8` | **Bit-exact** | Precompute `get_structuring_element` (ellipse) offsets once. MVE `vminq/vmaxq` later |
| `connected_components` | Two-pass union-find: pass 1 assigns provisional labels from already-scanned neighbours and records equivalences; pass 2 flattens/renumbers the table in one ascending sweep and accumulates stats | `uint8` in, `int32`/`uint16` labels | Matches cv2 stats/centroids | **No pixel stack.** Pre-size the union-find table to `(w·h+1)/2` provisional labels; ascending root order = raster order of first appearance, so numbering matches the flood fill it replaced |
| `find_contour` | Moore-neighbour tracing: fixed 8-neighbour CW offset table, Jacob's stopping criterion | `uint8` in, `int32` points | Matches cv2 point set | Pre-size the output point buffer to the component perimeter bound |
| `RunningGaussianAverage` | Per-pixel `f32`: `delta`, squared distance (sum over C), compare `threshold·var`, update mean/var by `alpha`, clip variance to `[4, 5·init]` | `uint8` frame → `f32` state → `uint8` mask | Parity within f32 rounding | FPU/MVE `f32`; **no sqrt**. State = `mean` + `variance` (see budget) |
| Shadow veto (`ShadowSuppressor`) | Per-pixel from two HSV pixels: `V_f/V_b` ratio, `|ΔS|`, circular `|ΔH|` (`min(d,180-d)`); AND the three bounds | `uint8` HSV → `f32` (or fixed-point) | Parity within f32/fixed rounding | Only the ratio needs division; consider fixed-point to avoid per-pixel FP divide |
| Blob area gate + stats | Integer compare against `min_area`/`max_area`; bbox/centroid from CC stats | integer/`f32` | Exact | Trivial control code |
| Dwell-time FSM (`StaticObjectFSM`) | Track list + IoU association + dwell counter + person-overlap veto | integer/`f32` | Exact (logic) | Reuse `CalculateBoxIOU`; small fixed track pool (no dynamic growth) |
| `hsv_to_bgr` → **`hsv_to_rgb`** | Sector table + `f32` (`v·255`, rounding); emit **R,G,B** order | `uint8`↔`uint8` | ≤ 1 LSB vs `cv2.COLOR_HSV2RGB` | **Optional** — only needed for on-device overlay drawing; skip if not visualising |

### Layer-0 steps to reimplement (OpenCV in Python, none on Alif)

For a complete on-device pipeline these must exist in C too (they are in the
[appendix](#appendix--still-opencv-backed-in-the-same-path) as still-cv2):

| Step | C approach |
| --- | --- |
| Resize → QVGA | **Not needed** — frames are already QVGA; only reuse HAL `interpolate()` (bilinear) if the sensor size differs |
| Gaussian denoise (5×5) | Separable Gaussian (two 1-D passes) in C, or a HAL/CMSIS-DSP filter; a 5×5 box blur is an acceptable cheaper substitute |
| Shadow-veto background reference (`accumulateWeighted`) | `ref = (1-α)·ref + α·frame` as an `f32` (or `fp16`) per-pixel loop, on the **RGB** frame |
| Grayscale feature | `RgbToGrayscale` (RGB in — reuse table) |

### Numeric-parity strategy

1. **Integer ops first** (`rgb_to_hsv`, morphology, connected components,
   contour, grayscale): these are bit-exact by construction. Port the LUTs and
   offset tables verbatim; validate on the kit's **native-gcc host build**
   against golden vectors exported from `tests/test_native_ops.py` before ever
   flashing hardware. To reuse those BGR-based Python vectors, either swap the
   test frame's R/B once when generating them or compare `rgb_to_hsv(rgb)`
   against `bgr_to_hsv(bgr)` on the same scene (identical by construction).
2. **Float ops** (`running_gaussian`, veto ratio, optional `hsv_to_rgb`): use
   IEEE single-precision and **preserve the operation order** of the Python
   reference; expect parity within f32 rounding, validate to a small tolerance.
3. Only after parity is locked, consider `fp16`/fixed-point/MVE for speed
   (Helium supports `fp16` and `int8` vector paths).
4. **Channel order is settled: RGB end-to-end on Alif.** The frames are RGB
   (verified above), so the C ops are RGB-native (`rgb_to_hsv`, `RgbToGrayscale`,
   RGB shadow reference) and **no R↔B swap is needed anywhere**. The only place
   the Python↔C difference surfaces is when reusing the BGR-named Python golden
   vectors (see point 1).

### Suggested module layout & integration

Mirror `pg_object_detection` (PreProcess → model → PostProcess → tracker):

- New use case `pg_sign_of_life` (or a `bgs/` API library under
  `source/application/api/`):
  - **PreProcess** — (QVGA already) denoise → `RgbToGrayscale`; keep the RGB
    frame for the shadow veto.
  - **Model** — `RunningGaussian` (Layer 1) + shadow veto (Layer 2, `rgb_to_hsv`).
  - **PostProcess** — morphology → connected components → contour → area gate.
  - **Tracker** — dwell FSM reusing `Box`/`CalculateBoxIOU`.
- Data flow on device: Ensemble camera **RGB QVGA** frame → PreProcess → model →
  PostProcess → blobs/FSM → draw contour+bbox to LCD with `ImageUtils` colours.
- Suggested native C signatures to target (host-testable):

  ```c
  void RgbToHsv(const uint8_t* rgb, uint8_t* hsv, int w, int h);           /* HWC, 3ch RGB */
  void MorphologyEx(const uint8_t* src, uint8_t* dst, int w, int h,
                    const uint8_t* se, int kw, int kh, MorphOp op);        /* 0/255 */
  int  ConnectedComponents(const uint8_t* mask, int w, int h, int conn,
                           int32_t* labels, CcStat* stats, float* centroids,
                           int maxLabels);                                 /* ret count */
  int  FindContour(const uint8_t* mask, int w, int h,
                   Point* out, int maxPoints);                            /* ret npts */
  void RunningGaussianApply(GaussianState* st, const uint8_t* gray,
                            uint8_t* fgMask, int w, int h);
  ```

### Memory budget (QVGA 320×240 = 76 800 px, gray feature)

| Buffer | Type | Size |
| --- | --- | --- |
| Camera frame (RGB, QVGA) | `uint8` ×3 | ~225 KB |
| Grayscale feature | `uint8` | ~75 KB |
| Model mean (gray) | `f32` | ~300 KB |
| Model variance | `f32` | ~300 KB |
| Shadow BG reference (RGB) | `f32` ×3 | ~900 KB ← largest; use `fp16` (~450 KB) or `uint8` |
| Foreground / morph / filtered masks | `uint8` ×3 | ~225 KB |
| CC label buffer | `int32` (or `int16`) | ~300 KB (~150 KB) |

Rough working set ≈ **2.3–2.6 MB** `f32` (≈ 1.6–1.9 MB with `fp16` reference and
`int16` labels) — feasible on Ensemble SRAM but budget deliberately; the shadow
reference and the `int32` label buffer are the two levers.

### Porting milestones

1. Host-side C library of the **integer** ops (HSV, morphology, CC, contour,
   gray) with golden-vector tests vs the Python reference (native-gcc build).
2. **Float** model (`running_gaussian`) + `accumulateWeighted` + shadow veto;
   verify masks vs Python within tolerance.
3. Layer-0 (HAL `interpolate` + denoise) and Layer-3 FSM (reuse `Box`/IoU).
4. Assemble the `pg_sign_of_life` use case; validate end-to-end on host, then on
   device (Cortex-M55 + camera + LCD).
5. Optimise hot loops with MVE / `fp16` only after parity is established.

## Source references

- **Running Gaussian average**: C. Wren et al., *Pfinder: Real-Time Tracking of
  the Human Body* (1997); bgslibrary `WrenGA`.
- **HSV & morphology**: OpenCV `modules/imgproc/src/` — `color_hsv.simd.hpp`
  (`RGB2HSV_b`, `HSV2RGB_native`), `morph.dispatch.cpp` (`getStructuringElement`),
  `morph.simd.hpp` (erode/dilate).
- **Contour tracing**: Moore-neighbour (a.k.a. radial-sweep) boundary following
  with Jacob's stopping criterion; the reference is
  `cv2.findContours` (Suzuki–Abe, 1985), matched on the traced point set.
- **Connected components**: two-pass union-find labelling (Rosenfeld & Pfaltz,
  1966; He et al., 2008); the reference is `cv2.connectedComponentsWithStats`,
  which uses the same family of algorithm.

Equivalence to `cv2` for every native operation is asserted in
`tests/test_native_ops.py`.
