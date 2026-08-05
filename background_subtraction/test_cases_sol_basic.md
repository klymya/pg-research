# Sign-of-Life / BGS — Basic Test Cases (hot-desk suite)

| Date | Author | Action |
|---|---|---|
| 2026-08-05 | Klym Yamkovyi | Creation — basic 7-case probe for SoL / background-subtraction evaluation and tuning |

Data-collection specification for a **basic** test suite used to evaluate, tune and improve the
Sign-of-Life (SoL) and background-subtraction (BGS) features implemented in
[`background_subtraction/`](readme.md).

**Basic means:** one mounting height, desk at frame centre, one desk, one person, no boundary or
crowding stress. Corner cases (DA edge, multiple people, corner mounting, coat on chair) are out of
scope — see [Not covered](#not-covered-and-why).

Scenario: a **hot desk** — an office / co-working position that starts genuinely unoccupied, is used
by one person, and is then vacated. This is the natural shape for BGS: the model gets a clean empty
scene to converge on, then one well-defined change to hold.

---

## 1. What this suite measures

The pipeline produces two signals (`readme.md` §Layer 3): a **per-frame SoL flag** (any region
≥ `sol_area` survives the area gate) and a **dwell-confirmed left-object event** from the FSM. The
suite is built to measure both, plus the failure modes around them.

| # | Measurement | Derived from |
|---|---|---|
| M1 | **Entry latency** — frames from the person entering to the first SoL flag | t_enter vs first flag |
| M2 | **Static-person hold time** — how long a still person keeps raising the flag | annotated stillness intervals in phase B |
| M3 | **Re-detection** — does the person reappear in the mask once they move again | stillness → movement transitions |
| M4 | **Left-object hold time** — frames the SoL flag survives after the person leaves, per object | tail phase |
| M5 | **Steady-state false-positive rate** on a converged empty scene | empty prefix + TC07 |
| M6 | **Illumination-step FP burst** — peak foreground area and frames to recovery | TC06, TC07 |
| M7 | **Chair contribution** — blob area from a displaced chair alone | TC01 (nothing left on desk) |
| M8 | **Minimum reliably detectable object size, per resolution** — the smallest area gate that still holds M5, at QVGA / VGA-crop / VGA-full | TC03 + TC07, replayed offline at each resolution |

### The primary tuning question

With the current defaults (`learning_rate = 0.1`, `dwell_frames = 60`, `threshold = 25`) the numbers
do not agree with each other. For the running-average form `bg ← (1−α)·bg + α·frame`, an object whose
initial contrast against the desk is `Δ₀` stops being foreground after
`n = ln(25/Δ₀) / ln(0.9)` frames:

| Left object | Assumed Δ₀ | Frames held | At 0.1 fps | At 1 fps |
|---|---|---|---|---|
| Laptop on a light desk | ~80 | ≈ 11 | ≈ 1.8 min | ≈ 11 s |
| Mid-contrast hoodie | ~40 | ≈ 4.5 | ≈ 45 s | ≈ 5 s |

`dwell_frames = 60` requires a candidate to persist for **60** frames. A deposited object survives
**≈ 5–11**. **The dwell FSM therefore cannot fire at the current defaults** — consistent with the
empty Sign-of-Life cue already observed in the low-fps CDNET runs, but here it would be a property of
the settings rather than of the dataset. This suite is designed to confirm that quantitatively and to
supply the data for the fix: either lower α (α ≤ 1/`dwell_frames` ≈ 0.017 for a 60-frame dwell), or
reduce `dwell_frames` below the measured hold time, or decouple the two by keeping a separate
slow-decay reference for the left-object path. M4 is the measurement that decides it.

---

## 2. Fixed parameters (identical in every case)

| Parameter | Value | Rationale |
|---|---|---|
| Sensor height | **2.6 m ± 0.15 m** (record the actual value per case) | Fleet mode; ~96 % of installs are 2.5–3.0 m |
| View | Strict top-down, no tilt | Project standard |
| Sensor | PG2 Cam4.2 revB on Maor's standard data-collection build | Comparable with the people-detection eval suite |
| Frame position | **Desk centred in frame** | Minimum fisheye distortion, maximum px/m — the "easy" region by construction |
| **Desk surface** | **Bare in the baseline state** — no monitor, keyboard, dock, papers, plants or personal items | Simulates a **shared** desk, which belongs to nobody and is empty between users. Also removes any ambiguity about which object caused a detection: the only things ever on the desk are what the person brings and what they leave |
| Environment | Open space / co-working, single hot desk | |
| Background complexity | Medium (chair and nearby furniture; the desk surface itself is empty) | |
| People | **1** person, seated desk work | |
| Chair | Ordinary office chair, **moves naturally** — position annotated, not restored | See [§7](#7-the-chair) |
| Objects | One laptop, one mug, one hoodie — the same physical items across all cases | Removes item-to-item variance |

---

## 3. Geometry and the detectability budget

PG2 Cam4.2 revB: `FX = 294.5 px` at VGA 640×480, FOV 122.5° × 92.9°
(`PG2.wiki/Algo/Reference/Camera-Parameters-Reference.md`). The BGS pipeline works at **QVGA**
(half linear scale). Near frame centre the fisheye is close to linear, so px/m ≈ FX/d — and the desk
is *at* frame centre, where that approximation is at its best.

At a 2.6 m mount with a 0.75 m desk (camera → desktop = 1.85 m): **≈ 80 px/m on the desktop**,
**≈ 57 px/m on the floor**.

Projected areas against the current gates (`min_area = 150`, `sol_area = 500`):

| Object | Footprint | Area @ QVGA | vs `min_area` | vs `sol_area` |
|---|---|---|---|---|
| Mug | ⌀ 0.09 m | **≈ 40 px** | **3.7× below** | **12× below** |
| Laptop | 0.33 × 0.25 m | ≈ 525 px | passes | **borderline** |
| Laptop + mug | — | ≈ 565 px | passes | borderline |
| Hoodie, crumpled | 0.40 × 0.35 m | ≈ 890 px | passes | passes |
| Displaced chair | 0.45 × 0.45 m | ≈ 950 px | passes | passes |
| Seated person | — | ≈ 2 300 px | passes | passes |

> These are computed, not measured — a small-angle projection at frame centre. Verify against the
> first recorded case before trusting them for tuning.

**Consequences that shape the expected results:**

1. **A left mug is ≈ 40 px at QVGA and ≈ 160 px at VGA.** This is the suite's **resolution decision
   probe** — see [§3.1](#31-the-resolution-question-qvga-vs-vga).
2. **A left laptop sits on the `sol_area` threshold** (525 vs 500). The most tuning-sensitive case in
   the suite, and the one most affected by exactly how the laptop is placed.
3. **A displaced chair (≈ 950 px) outscores a left laptop.** It is the largest single change in most
   cases. Annotating it is mandatory, and TC01 exists to quantify it.

### 3.1 The resolution question: QVGA vs VGA

TC-SOL-03 (mug only) exists to answer **"is QVGA enough, or do we switch to VGA — spend more memory,
get better accuracy?"** The two halves of that trade have very different statuses.

**Accuracy side — open, and this suite answers it for free.** The sensor is native VGA 640×480; the
pipeline downsamples to QVGA. Every case recorded here can therefore be replayed offline at **both**
resolutions from the *same* frames, with no additional capture. That is the whole reason
[§6](#6-capture-requirements) requires frames stored at full VGA.

The question is not "can a 160 px mug be gated" — trivially yes — but **at what area gate does the
empty-scene false-positive rate become unacceptable, at each resolution?** Resolution does not change
what is physically detectable; it changes the signal-to-noise ratio at which a given *physical* size
can be gated. The gates are absolute pixel counts, so preserving the same physical gate at VGA means
scaling them 4× (`min_area` 150 → 600, `sol_area` 500 → 2000) — at which point the mug's 160 px is
*still* below `min_area`. Detecting a mug requires lowering the **physical** size gate; resolution
only decides whether that gate can be held above the noise floor. TC-SOL-03 supplies the signal,
TC-SOL-07 supplies the noise, and the output is a **minimum reliably detectable object size per
resolution** (M8).

**Memory side — already answered, and the answer is not "more memory".** From the measured linker
figures (`report_optimization.html`): free SRAM0 = 1,144,304 B (persistent), free SRAM1 = 1,076,023 B
(scratch), total **2,220,327 B**. Both arenas are statically sized and scale linearly with processed
pixels; VGA is 4× QVGA.

| Arena | QVGA (measured) | VGA (computed) | Free | Verdict |
|---|---|---|---|---|
| Persistent (model mean + variance, f32) | 614,656 B | 2,457,600 B | 1,144,304 B | **2.1× over** |
| Scratch (peak) | 977,666 B | ≈ 4,683,264 B | 1,076,023 B | **4.4× over** |
| **Total** | **1,592,322 B** | **≈ 7,140,864 B** | **2,220,327 B** | **3.2× over** |

Scratch is worse than a plain 4× because the connected-components label type breaks: worst-case
provisional labels are `(w·h+1)/2` = 38,400 at QVGA (fits `uint16`) but **153,600 at VGA, which
overflows `uint16`**. Labels and the union-find table must both widen to `uint32`, doubling those two
buffers on top of the 4×.

**≈ 7.14 MB exceeds the 6,815,744 B total of SRAM0 + SRAM1 combined** — whole-frame VGA does not fit
even if the 2,857,600 B NN tensor arena were deleted outright. It is not a memory *cost*, it is a
memory *impossibility*. `fp16` model planes close the persistent gap to within ~84 KB but leave
scratch ~3.6 MB over, so quantisation does not change the conclusion.

**So the real decision is not QVGA vs VGA:**

| Option | Memory | Scale over the desk | Note |
|---|---|---|---|
| **A** — QVGA full frame (current) | 1.59 MB, fits | 80 px/m | Accept the size floor |
| **B** — **VGA on a fixed ROI crop** | Scales with the *crop*, not the sensor: a 320×240 crop at native VGA costs **exactly today's memory** | **160 px/m** — 2× linear | Halves the covered area; viable for a fixed desk, not for whole-room presence |
| **C** — intermediate (e.g. 480×360) | ≈ 2.25× QVGA ⇒ persistent 1.38 MB, 238 KB over | 120 px/m | Does not fit without quantisation |
| **D** — strip/tile processing at VGA | Cuts scratch, but the persistent model planes are whole-frame (2.46 MB) and still do not fit | 160 px/m | Needs cross-strip label merging; substantial rework |

**Option B is what this suite can genuinely settle.** The desk is fixed and frame-centred by design,
so a VGA crop around it is directly testable offline on the same recordings at zero memory cost. If
the mug clears an area gate that still holds the TC-SOL-07 false-positive rate, the answer is a
concrete trade — 2× resolution over the desk, at the price of full-frame coverage — instead of a
resolution wish that cannot be built.

---

## 4. Timeline template

Every populated case follows the same script. Times are from the start of the recording.

| From | To | Phase | Content |
|---|---|---|---|
| 00:00 | 05:00 | **A — empty** | Desk **bare and unoccupied**, nobody in frame. Background-model convergence |
| 05:00 | 20:00 | **B — occupied** | Person enters (`t_enter`), sits, works at the desk, then packs per the case's leave condition and walks out of frame (`t_out`) |
| 20:00 | 35:00 | **C — empty** | Desk unattended for **15 min** |

**Total 35 min per case → ≈ 2 100 frames at 1 FPS. Seven cases ≈ 4.1 h of capture.**

Each phase is sized against the model's own time constant. At 0.1 fps sampling, α = 0.1 converges in
≈ 28 frames ≈ 4.7 min:

- **Phase A (5 min)** is the convergence budget — 30 frames at 0.1 fps. It must not be shortened, or
  the person arrives while the background is still settling and every downstream measurement is
  contaminated.
- **Phase B (15 min)** is ≈ 3× the absorption time, long enough to see a seated person absorbed into
  the background and re-detected when they move again.
- **Phase C (15 min)** is ≈ 3× the absorption time, so the left object's full decay curve *and* a
  stable post-absorption baseline are both captured.

Phase B is **natural desk work** — typing, reading, phone, posture shifts — not a script. Whether
ordinary desk activity generates enough change at one frame per 10 s to keep a person detected is one
of the questions the suite exists to answer, and scripting it would destroy the answer. Periods where
the person happens to sit still are **annotated afterwards from the frames** (§8) and are what M2 and
M3 are computed over.

---

## 5. The 7 cases

| ID | Lighting | Leave condition | Primary purpose |
|---|---|---|---|
| **TC-SOL-01** | Artificial | **Takes everything** | Negative control for objects; isolates the chair (M7) |
| **TC-SOL-02** | Artificial | **Laptop only** | Threshold-boundary case (M4) |
| **TC-SOL-03** | Artificial | **Mug only** | **Resolution decision probe**: is QVGA enough, or is a VGA crop needed (M8) |
| **TC-SOL-04** | Artificial | **Laptop + mug** | Nominal positive left-object case (M4) |
| **TC-SOL-05** | Artificial | **Hoodie on the table** | Large soft object, deformable, contrast-sensitive (M4) |
| **TC-SOL-06** | **Artificial → dark**, switched at 25:00 | Laptop + mug | Does an illumination step destroy the left-object signal? (M6) |
| **TC-SOL-07** | **Artificial → dark**, switched at 20:00 | *No person at all* | Pure FP baseline and pure FP burst (M5, M6) |

TC-SOL-01 through 06 use the [timeline template](#4-timeline-template). TC-SOL-07 keeps the same
35-minute length but has no person at all: the desk stays empty throughout, with the lights switched
off at 20:00 — the same clock position at which the other cases become unattended.

### Per-case detail

#### TC-SOL-01 — Artificial light, takes everything

- **A. Environment** — open-space hot desk, medium background complexity, desk + chair + nearby
  furniture. Lighting: ceiling artificial only, blinds closed, constant throughout.
- **B. People setup** — 1 person, seated desk work per the template, full visibility, normal
  clothing contrast.
- **C. Sensor** — 2.6 m ± 0.15, centre of AoI, desk centred in frame.
- **D. Leave condition** — laptop, mug and all personal items removed. Desk surface returns to its
  Phase-A state. The chair is left wherever it naturally ends up.
- **Expected** — after departure the only persistent change is the chair (≈ 950 px). Prediction: the
  SoL flag **stays raised** on the chair alone. This is the measurement of M7 and the baseline every
  other case's left-object area is read against.

#### TC-SOL-02 — Artificial light, laptop only

- As TC-SOL-01, except: mug and personal items removed, **laptop left closed** on the desk in a
  marked position.
- **Expected** — ≈ 525 px against a 500 px threshold; flag held ≈ 11 frames (≈ 1.8 min at 0.1 fps)
  before absorption. The most likely case to flip under tuning.

#### TC-SOL-03 — Artificial light, mug only *(resolution decision probe)*

- As TC-SOL-01, except: **mug left**, everything else removed.
- **Mug must be high-contrast against the desk** (e.g. a dark matte mug on a light desk) and its
  colour recorded. A white mug on a white desk would measure the *contrast* limit, not the *size*
  limit, and would answer the wrong question. If effort allows, place **two mugs** — one
  high-contrast, one low-contrast — in marked positions; that adds the contrast axis at no extra
  capture cost.
- **Expected at QVGA** — ≈ 40 px, below `min_area` (150): **not detected**. After the chair blob is
  accounted for, the desk reads as empty.
- **Expected at VGA** — ≈ 160 px, which clears today's *absolute* gate but not a physically
  equivalent one. Whether it is *reliably* detectable depends on the empty-scene FP rate at the gate
  required to catch it — which is exactly M8, computed against TC-SOL-07.
- This case is what decides between options A and B in [§3.1](#31-the-resolution-question-qvga-vs-vga).

#### TC-SOL-04 — Artificial light, laptop + mug

- As TC-SOL-01, except: **laptop (closed) and mug both left**, in marked positions.
- **Expected** — the nominal positive case. Area ≈ 565 px, dominated by the laptop; the mug
  contributes little. Hold time similar to TC-SOL-02.

#### TC-SOL-05 — Artificial light, hoodie on the table

- As TC-SOL-01, except: a **hoodie (or coat) left crumpled on the desk surface**, never on the chair.
  Use a **mid-to-high contrast** garment against the desk (record garment and desk colours).
- **Expected** — largest left object (≈ 890 px), so the highest chance of clearing the gate; but
  soft, low-relief and contrast-dependent. If a mid-contrast garment (Δ₀ ≈ 40) is used the hold time
  drops to ≈ 4–5 frames, which is the shortest of the ladder despite the largest area. Area and
  contrast pull in opposite directions here — that is the point of the case.

#### TC-SOL-06 — Illumination step during the tail, laptop + mug

- Identical to TC-SOL-04 through departure. At **25:00 (5 min into phase C)** the ceiling lights are
  switched **off** (blinds already closed) → dark room for the remaining 10 min.
- **Expected** — a global illumination step makes the entire frame foreground for ≈ 1/α frames.
  Prediction: a full-frame FP burst, the left-object signal is buried, and the model reconverges to
  the dark scene in ≈ 28 frames (≈ 4.7 min at 0.1 fps). Measures whether any left-object memory
  survives a light change — the answer, at the current α, is expected to be *no*.

#### TC-SOL-07 — Empty control with illumination step, no person

- **35 minutes, nobody enters the frame.** Ceiling lights on for 00:00–20:00, switched **off** at
  20:00, dark for 20:00–35:00.
- **Expected** — Phase 1 gives the steady-state false-positive rate on a converged empty scene (M5);
  the step gives the FP burst magnitude and recovery time with nothing else in the scene (M6); the
  dark tail gives the FP rate in the low-light, high-sensor-noise regime, which is where the
  threshold (25) is most at risk.

---

## 6. Capture requirements

Inherited from `Model-Evaluation-Test-Cases.md` so the same recordings stay comparable with the
people-detection suite and can be resampled to any cadence.

- **Frames MUST be stored at full sensor resolution, VGA 640×480 — no downscaling at capture.**
  The QVGA-vs-VGA decision ([§3.1](#31-the-resolution-question-qvga-vs-vga)) is answered by replaying
  the *same* recordings at both resolutions. Capturing at QVGA would destroy that possibility and
  force a re-collection.
- **1 FPS** target (0.5 FPS acceptable fallback; below that not acceptable). The system runs at
  1 frame / 10 s — capturing 10× faster is what allows offline resampling to 0.1 fps and to the
  6 / 4 / 3 frames-per-minute variants without re-collection.
- **Raw frames: AWB off, CLAHE off.** Same reason as the detection suite — the data must remain
  usable for preprocessing experiments.
- **Auto-exposure: record the setting, and prefer fixed exposure.** With AE enabled, a light switch
  triggers an exposure ramp that reaches the background model as a *second* global illumination
  change, on a different time constant from the physical one. TC-SOL-06 and TC-SOL-07 are
  uninterpretable unless this is known. If AE cannot be fixed, log it and expect the burst to have
  two phases.
- **Mark reference positions with floor tape**: the desk footprint, the chair's Phase-A position, and
  the laptop / mug / hoodie placement spots. Chair displacement must be measurable from the frames.
- **Output structure** — one root folder, one sub-folder per case, frames in temporal order:

```text
SOL_basic_suite_v1/
├── TC-SOL-01_artificial_takes_all/
│   ├── 0001.png
│   └── ...
├── TC-SOL-02_artificial_laptop/
└── TC-SOL-07_empty_control_transition/
```

---

## 7. The chair

The chair moves in every populated case, and at ≈ 950 px it is the **largest persistent change** —
larger than any object left on the desk. It is therefore recorded and annotated, not controlled away:
that is what a real hot desk does, and suppressing it would make the suite measure something that
never happens in deployment.

The consequence must be carried into the analysis: **the raw SoL flag cannot distinguish "laptop left"
from "chair moved".** TC-SOL-01 (nothing left) is what makes the ladder readable — subtract its
chair-only response from every other case to get the object's contribution. Concretely, for each case
report both:

- **Flag-level result** — did SoL fire, and for how long (what the deployed system sees)
- **Blob-level result** — the surviving blobs with area and centroid, each attributed to *chair* or
  *object* using the taped reference positions

If the chair alone reliably holds the flag, that is a genuine finding about the feature's semantics
("someone was here" is arguably the right answer), not a defect — but it has to be stated rather than
discovered later inside an aggregate score.

---

## 8. Annotation / ground truth

The parent evaluation suite does not specify a GT format; this one must, or nothing above is
scorable. Minimum per case:

**Per-frame** (`frames.csv`, one row per captured frame):

| Column | Values |
|---|---|
| `frame` | filename |
| `t` | seconds from recording start |
| `phase` | `A` / `B` / `C` |
| `person_present` | 0 / 1 |
| `person_still` | 0 / 1 — annotated after capture; the person is seated with no visible body movement. Source for M2 / M3 |
| `person_bbox` | `x,y,w,h` or empty — *optional but recommended*, so the same capture serves people-detection evaluation |
| `objects_on_desk` | subset of `{laptop, mug, hoodie}` |
| `chair_displaced` | 0 / 1 |
| `light_state` | `artificial` / `dark` |

**Per-case** (`case.yaml`): sensor height, actual desk-to-camera distance, exposure mode, lux at desk
level in each light state, garment and desk colours, `t_enter`, `t_out`, `t_light_change`, objects
left, and the measured chair displacement in cm.

Lux at desk level in both light states is required — "a bit dark" is not reproducible, and the dark
segments of TC-SOL-06/07 are exactly where the sensor-noise regime decides the result.

---

## 9. Analysis plan

Run each case through `main.py` / `SignOfLifePipeline` at the deployment configuration (native
`running_gaussian`, α = 0.1, QVGA, shadow veto **off** — the current build default), resampled to
0.1 fps, and produce per case:

1. **Foreground area vs time**, with phase boundaries and `t_light_change` marked.
2. **SoL flag vs time**, against `person_present` and `objects_on_desk` ground truth.
3. **Surviving blobs in the tail**, attributed to chair vs object.

Then across cases: the M1–M7 table, and an α sweep on the recorded data (α ∈ {0.1, 0.05, 0.02, 0.01})
against left-object hold time and empty-scene FP rate. That sweep, plus M4, is what settles the
α / `dwell_frames` inconsistency in [§1](#the-primary-tuning-question) — and it needs no additional
capture, which is the main reason for recording at 1 FPS.

### Resolution sweep (M8)

Replay **the same recordings** at three working resolutions and, for each, sweep the area gate to
find the smallest value that keeps the TC-SOL-07 false-positive rate acceptable:

| Configuration | Processed pixels | Scale on the desktop | Fits the Alif arenas? |
|---|---|---|---|
| QVGA 320×240 full frame (current) | 76,800 | 80 px/m | yes — 1.59 MB of 2.22 MB |
| **VGA 320×240 crop** around the desk | 76,800 | **160 px/m** | **yes — identical footprint** |
| VGA 640×480 full frame | 307,200 | 160 px/m | **no** — 3.2× over ([§3.1](#31-the-resolution-question-qvga-vs-vga)) |

The third row is included as the *accuracy ceiling* reference only — it cannot be deployed, but it
bounds what any resolution increase could ever buy. Report, per configuration: minimum area gate at
an acceptable FP rate, the corresponding **physical** object size in cm², and whether the mug,
laptop and hoodie clear it. That table is the QVGA-vs-VGA-crop decision, with both the accuracy gain
and its cost (halved coverage) stated in the same place.

---

## Not covered, and why

Stated explicitly so the suite's limits are not mistaken for coverage:

| Not covered | Reason |
|---|---|
| **Mixed lighting (daylight + artificial)** | Requires window-dependent scheduling; it is also where the HSV shadow veto would matter, and the veto is **off** in the current build. Deferred to a second pass, together with the veto-on comparison |
| **Dark room as a converged steady state** | Only reached post-transition in TC-SOL-06/07. The 20-minute dark tail of TC-SOL-07 gives ≈ 15 min of steady state, which is indicative but not a dedicated case |
| **Desk away from frame centre** | px/m falls toward the DA edge; a laptop that clears `min_area` at centre may not at the border. Deliberately excluded as a corner case |
| **Multiple people, crowding, occlusion** | Corner cases |
| **Coat on chair** | Explicitly excluded — hard case |
| **Person returns after leaving** | Corner case; would confound M4 |
| **Mounting heights other than 2.6 m** | Basic suite fixes scale |

The single most valuable addition to a second pass is **mixed lighting with the shadow veto on and
off**, since that is the only configuration decision from the memory-optimisation step that has not
been tested against indoor data.
