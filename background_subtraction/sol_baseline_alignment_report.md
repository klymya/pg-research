# Alignment Report — SoL Arbitration Baseline vs. the Firmware as Built

Evaluates `sol_people_detection_baseline.md` against the code and infrastructure
described in `alif_image_processing_flow.md`, checked directly in
`rtos_sensor/projects/alif_ml-embedded-evaluation-kit_pg` rather than against
the flow document's summary.

Every claim below cites `file:line`. "Exists" means the symbol is in the linked
firmware; "missing" means it exists only in the Python prototype
(`background_subtraction/scr/`) or nowhere.

---

## 1. Verdict

| # | Baseline requirement | Status | Cost |
| --- | --- | --- | --- |
| R1 | Detection runs every frame, before SoL | **fits, needs reorder** | ~10 lines |
| R2 | `n_persons` from detections with `conf ≥ conf_person` | **exists** | 3 lines |
| R3 | Skip SoL entirely while occupied | **fits as designed** | 1 branch |
| R4 | State machine (`WARMUP`/`OCCUPIED`/`WATCH`) in `sensing.cc` | **new code** | ~60 lines |
| R5 | `BackgroundModel::Apply` unchanged | **confirmed correct** | 0 |
| R6 | Learn-in counts empty frames only | **works unmodified** | 0 |
| R7 | Global-change guard | **fits, no library change** | ~15 lines |
| R8 | Model reset on global change | **exists** — `SignOfLifeDetector::Reset()` | 0 |
| R9 | Dwell FSM with IoU association | **missing on target** | ~200 lines, port |
| R10 | `Rebaseline(region, gray)` | **missing, and harder than stated** | API change |
| R11 | SoL result reaches the application | **missing — no accessor at all** | ~20 lines + proto |
| R12 | State survives across wakes | **partly solvable** (scalars yes, model no) | see §5.2 |
| R13 | Frame cadence supports N / D / learn_frames | **fits** — 10 s wake (§5.1) | 0, but retune `learn_frames` |

Eight of thirteen are free or nearly free — the arbitration layer itself fits
the existing structure well. **One item blocks: R12, persistence** (§5.2).
R13 was reported as a second blocker in an earlier revision of this report and
is withdrawn — see §5.1.

---

## 2. What already fits

### 2.1 The call site is one function, and the reorder is trivial

`sensing_task` (`source/use_case/pg_object_detection/src/sensing.cc:89`) already
holds everything the state machine needs, in one place, on one thread. Today the
order inside `EVT_INFERENCE_START` is:

```
hal_get_image_data(320,240)  →  sol_detector.Process(...)  →  run_inference(...)
                                        (sensing.cc:119)          (sensing.cc:125)
```

The baseline needs `run_inference` first, then the gate, then `Process`. Both
calls take the same `const uint8_t* image` and neither mutates it
(`SignOfLifeDetector.hpp:64`, "Read only; the caller's buffer is never
modified"), so swapping them is safe. No data dependency is introduced —
`run_inference` fills `detections`, which the gate then reads.

### 2.2 `Apply` really is unchanged — verified, not assumed

The baseline claims the learn-in "counts empty frames, not elapsed frames" if
the caller simply declines to call the pipeline. Confirmed:
`m_frameIndex` is a member incremented *inside* `Apply`
(`BackgroundModel.cc:62`), and `learning` is `m_frameIndex < m_learnFrames`
(`:63`). Skipping the call skips the increment. The selective update
(`updatable = !foreground || learning`, `:71`) and the variance clamps behave
identically.

`Pipeline::Process` resets the scratch arena on entry (`SolPipeline.cc:40`), so
skipping frames leaves nothing stale to clean up.

**This is the strongest alignment result in the report: the central mechanism of
the baseline requires zero changes to the SoL library.**

### 2.3 Model reset already exists

`SignOfLifeDetector::Reset()` (`SignOfLifeDetector.hpp:73`) →
`Pipeline::Reset()` → `BackgroundModel::Reset()` (`BackgroundModel.cc:29`),
which clears `m_mean`/`m_variance` and zeroes `m_frameIndex` while keeping arena
capacity — exactly the semantics §5.2 of the baseline needs for the
global-change path, and exactly the "reset, not counter-reset" semantics §4
requires.

### 2.4 The global-change guard needs no library split

The baseline's flowchart puts the guard between layers 0–1 and layers 3+, and
§8 implies `Process()` must be split to allow it. It does not.
`FrameResult.rawMask` is returned by `Process` and is valid until the next call
(`SolTypes.hpp:29`), so the caller can count foreground pixels *after* the fact
and discard the result. The cost is one wasted morphology + CCL pass on the rare
global-change frame; the model is not corrupted in the meantime, because on such
a frame nearly every pixel is foreground and the selective update leaves it
alone.

**Correction to the baseline, §8:** `sol::Pipeline` needs no split, for the
guard or for anything else.

### 2.5 Confidence threshold is reachable

`conf_person = 0.3` sits above the post-process gate of `0.2`
(`DetectorPostProcessing`, `PostProcessParams::confidenceThreshold`) and below
the `0.5` used by `PresentInferenceResult` for the sleep policy
(`UseCaseHandler.cc:57`), so detections at 0.3 are present in the returned
vector. `numClasses == 1`, so every detection is a person — no class filter
needed.

---

## 3. What needs new code

### 3.1 R9 — there is no dwell FSM on the target

`StaticObjectFSM`, `TrackedObject`, `ObjectState` and `DwellConfig` exist only
in `background_subtraction/scr/fsm.py` and `scr/config.py`. The Alif tree has
**no equivalent**: `sol::PipelineConfig` (`SolTypes.hpp:80–110`) has no
`dwell_frames`, no `iou_match_threshold`, no `max_missing_frames`, and
`FrameResult` (`SolTypes.hpp:64`) carries no tracks. `signOfLife` on target is a
**pure per-frame area test** (`SolPipeline.cc:114`).

**Correction to the baseline, §8.** Two rows are written against the Python
prototype and are wrong for the firmware:

* *"`DwellConfig`: `dwell_frames 60 → 3`"* — there is no `DwellConfig` to edit.
* *"`StaticObjectFSM` unchanged"* — it does not exist here.

The port is mechanical (~200 lines: IoU, greedy association, dwell counters,
eviction) and `std::vector<TrackedObject>` is consistent with the existing
`std::vector<Blob>` on the DTCM heap. But it is new work, and the baseline's
`D`-frame persistence rule cannot be implemented without it.

Note there *is* a tracker in the tree — `BYTETracker`, `STrack`,
`kalmanFilter`, `lapjv` — compiled into `libpg_object_detection.a` and pulling
in Eigen, but instantiated by nothing (`alif_image_processing_flow.md` findings,
and confirmed here). It is built for multi-object detection tracking with motion
models, not for static-blob dwell; repurposing it would cost more than porting
the 200-line FSM.

### 3.2 R10 — `Rebaseline` cannot be a caller-side call

The baseline specifies `Rebaseline(region, gray)` on `BackgroundModel`, called
from `sensing.cc` after the report. The gray image is not reachable from there:
it is a local inside `Pipeline::Process` (`SolPipeline.cc:48`) backed by the
scratch arena and never returned — `FrameResult` exposes only `rawMask`,
`filteredMask` and `blobs`.

Three options, in increasing intrusiveness:

1. **Re-baseline inside `Process`**, driven by a `pendingRebaseline` region set
   by the caller before the next call. No API surface beyond one setter; the
   region is applied against the frame that produced it only if the caller acts
   on the same frame's result, which it does.
2. **Add `Image gray` to `FrameResult`.** Cheap in bytes (the buffer already
   exists in scratch), but widens the struct's lifetime contract.
3. **Recompute gray** in `Rebaseline` from the RGB frame over the blob's bbox
   only. Costs a few thousand pixels of BT.601, needs no new state, and keeps
   the model's API self-contained. **Recommended.**

### 3.3 R11 — the SoL result never leaves `sensing.cc`

`sensing.h:29` exposes `sensing_get_detections()`; there is **no
`sensing_get_sol_result()`**. `sol_present_result` (`sensing.cc:53`) prints the
flag and blob list to UART and drops them. Nothing in `application.cc` reads a
SoL result, and `app_save_detections()` handles detections only.

For telemetry there is a plausible landing site already in the protobuf:
`Telemetrypayload__MotionNotification { bool motion_detected }`
(`pb/telemetry_payload.pb-c.h:219`), which no firmware code populates —
`motion_detected` appears elsewhere only in the unrelated PIR test
(`TestPir.cc`). Whether the SoL flag should ride that message or get its own is
a gateway-side decision; `Production.h:66` explicitly warns that Alif and GW
definitions drifting apart stalls provisioning, so this is a coordinated change,
not a local one.

The baseline's telemetry row (`state`, `empty_run`, `sign_of_life`, per-blob
`dwell`) therefore costs an accessor, an FSM wiring change in `application.cc`,
**and** a schema negotiation.

---

## 4. What the baseline gets right that the firmware makes cheap

Three of the baseline's simplifications are worth more against the real code
than the document claims:

* **No person mask** means no dependency on detection geometry — and the boxes
  really are in 320×320 coordinates while the frame is 320×240
  (`m_scaleFactor` fixed at 1.0, `DetectionModel.cc:70`; the geometry mismatch
  is documented in `alif_image_processing_flow.md` §5.7). Any masked design
  would have had to solve that first.
* **Skipping SoL while occupied** removes 9 full-frame passes on exactly the
  frames where the NN path is doing the most work.
* **`Apply` untouched** keeps the Python ↔ host-C++ ↔ Alif parity net intact
  (`CONTEXT.md` §1.4: 22/22 masks, 374 tests). A `skip`-mask parameter would
  have needed the whole net re-validated.

---

## 5. Blockers

### 5.1 R13 — the deployed cadence is 10 s (≈ 0.1 fps), and it works

> **Revised.** An earlier version of this report read the deployed cadence off
> the validation bounds `WAKE_PERIOD_MIN_SEC 120` … `MAX 3600`
> (`Production.h:69–72`) and concluded the frame rate was 2 minutes to 1 hour.
> That was wrong. The deployed setting is **`wake_period_sec = 10`**, and 10 is
> explicitly carved out of those bounds — `is_valid_wake_period` returns true
> for `WAKE_PERIOD_DEFAULT_SEC` before it range-checks anything
> (`Production.h:75`). The conclusions below replace the earlier ones.

**How the cadence is actually derived** (`sleep_policy.cc:26`):

```c
target = (wake_sec >= 30u) ? (wake_sec / 30u) : 1u;
p->wake_source_is_wd = (target > 1u) ? 1u : 0u;
```

The 30 in the divisor is the TPL5010 watchdog pulse period. For
`wake_period_sec ≥ 30` the device sleeps through `target − 1` watchdog pulses
and runs a full cycle on the `target`-th. For `wake_period_sec = 10` the
quotient branch is skipped, `target = 1`, `wake_source_is_wd = 0`, and **every
LPTIMER wake is a full cycle** — one inference per 10 s.

At 10 s every frame-count parameter is comfortable:

| Parameter | Frames | Wall-clock @ 10 s |
| --- | --- | --- |
| `N` (resume hysteresis) | 3 | 30 s |
| `D` (dwell) | 3 | 30 s |
| `learn_frames = 5` | 5 empty | 50 s (80 s including `N`) |
| `learn_frames = 30` | 30 empty | 5 min (5.5 min including `N`) |
| α = 0.1 time constant | ~10 | 100 s |
| EWMA memory `(2−α)/α` | 19 | ~3.2 min |

**R13 is not a blocker.** Three consequences, all favourable:

1. **The learn-in is practical.** Under a minute of empty scene at
   `learn_frames = 5`, five and a half minutes even at 30. Both are realistic in
   an office.
2. **`N` works as a dropout filter again.** Frames 10 s apart are the same
   scene, so a single missed detection really is detector flicker and the
   hysteresis argument in §3 of the baseline holds as written.
3. **The research cadence matches deployment exactly.** 0.1 fps is
   `data_0_1fps*`, which is what the BGS evaluation and
   `pipeline_experiments.ipynb` already run on. That alignment was accidental —
   nothing in the research documents states the deployment cadence — but it is
   real, and it means the measured F-scores are at the right rate. (The thinness
   of CDNET at 0.1 fps, 5–6 frames per sequence, remains a separate problem —
   `CONTEXT.md` §1.6.)
   `test_cases_sol_basic.md` is likewise fine: its 35-minute timeline yields
   **210 frames** per case at 10 s, and §1 of that document already reasons
   about 0.1 fps explicitly.

**What this does change: `learn_frames` should go back up.** The reduction to 5
in the baseline (§4.1) was argued partly on the clean-scene assumption and
partly on cadence pain that turns out not to exist. Only the first argument
survives, and against it stands sensitivity: a 5-frame model classifies
foreground at `|Δ| > 16.7` gray levels versus `10.1` at 19 frames and `7.4`
asymptotically (baseline §4.3). With the mug / low-contrast case (TC-SOL-03, M8)
as the known sensitivity risk, paying **3 minutes instead of 50 seconds** to be
1.65× more sensitive is the better trade. **Recommend `learn_frames = 19`** —
the EWMA's own effective sample size, past which nothing accumulates anyway.

**Correction to the baseline, §10.2:** the cadence is neither undefined nor
hostile. It is 10 s, the frame-count parameters are sound as written, and the
"specify in seconds" recommendation is unnecessary — though still worth doing
defensively, since the gateway *can* push 120–3600 s and the design should
degrade predictably if it ever does.

**The burst-capture idea is no longer needed for cadence** — but see 5.2, where
it remains one of the two candidate answers to persistence.

### 5.2 R12 — persistence splits cleanly into a solvable half and a blocked half

`alif_image_processing_flow.md` and the baseline both state that
`.bss.sram0` is re-zeroed on every wake, so the model dies. Checking the retained
region shows the problem is *two* problems with different answers:

| What must survive | Size | Home available? |
| --- | --- | --- |
| FSM scalars — `state`, `empty_run`, learn-in count, flag | ~16 B | **yes** |
| Background model — `mean` + `variance` | **614,656 B** | **no** |

The retained region is `struct backup_4k_t` at `0x4902C000`, 4096 B, magic +
CRC32 protected (`backup4k_sram.c:5–10`). It already carries
`sleep_policy_t.had_detection` — *precedent for exactly this kind of
cross-wake boolean* — plus the GW cookie, network-transition state and rollback
counters, in well under 100 bytes. Adding four `uint32_t` for the SoL state
machine is a two-line change to the struct.

(Constraint to respect: every field in that region must be `uint32_t` or
`uint64_t` — `backup4k_sram.h:28` documents an M55 GCC read-modify-write that
corrupts adjacent bytes on sub-word writes.)

So the arbitration layer can be made wake-persistent today. The **background
model cannot** — 614,656 B against a 4 KB region is 150× over. Until either a
larger retained region is identified or the model is rebuilt per wake, the
machine reaches `WARMUP` and stops there.

**Correction to the baseline, §12 Q4:** the question is now sharper. Retention
exists and is in use; the open item is a *large* retained region, and the
fallback (rebuild the model each wake) is only viable in combination with the
burst capture of §5.1 — and even then it cannot detect an object left between
wakes, which is the product's core case.

### 5.3 Installation mode never reaches SoL

`st_inference` (`application.cc:634`) sets `EVT_INFERENCE_START` only when
`mode == MODE_OPERATIONAL`. In `MODE_INSTALLATION` the process survives
(`tx_thread_sleep` rather than STOP) — the one mode where a background model
*could* accumulate — but `sensing_task` is never signalled, so neither the
detector nor SoL runs. The baseline does not mention installation mode; as
written it cannot run there either.

This is worth revisiting deliberately: installation mode is currently the only
context in which the baseline's state machine could be exercised on real
hardware before 5.2 is solved.

---

## 6. Interactions the baseline does not cover

| Interaction | Current behaviour | Question |
| --- | --- | --- |
| **Sleep policy** | `PresentInferenceResult` sets `had_detection` at conf ≥ 0.5 (`UseCaseHandler.cc:57`), which shortens the wake interval | Should `sign_of_life = true` also shorten it? An unattended object arguably warrants a faster cadence — and would partly mitigate §5.1 |
| **`EVT_CAPTURE_IMAGE`** | Same frame, JPEG-encoded and sent | Should a SoL event attach a JPEG? The transport already exists |
| **Watchdog** | `sensing_task` reports each loop (`sensing.cc:149`) | Unaffected — skipping SoL shortens the loop |
| **AE settling** | Auto-gain corrects the *next* frame and its PI state dies each wake (`alif_image_processing_flow.md` §7.2) | At one frame per wake the AE loop never converges, so consecutive "frames" may differ in exposure by more than the `3.5σ` foreground threshold. This compounds §5.1 and is the mechanism behind the baseline's §9.4 |

That last row deserves emphasis: the baseline defers ISP feedback (§9.4) on the
assumption that it is a slow drift. At one frame per wake with a PI controller
that resets every wake, it is not a drift — it is a per-frame step.

---

## 7. Recommended order

1. **Settle the cadence question** (§5.1). Nothing else can be sized until then.
   Decide burst-per-wake vs. one-frame-per-wake; this is a product/power
   decision, not an algorithm one.
2. **Move the FSM scalars into `backup_4k_t`** (§5.2). Two lines, unblocks
   cross-wake arbitration state, testable immediately.
3. **Reorder `sensing.cc` and add the state machine** (§2.1, R4). ~70 lines, no
   library changes, and testable in `MODE_INSTALLATION` (§5.3) if the FSM is
   left out at first.
4. **Add the global-change guard** (§2.4). ~15 lines, caller-side.
5. **Port `StaticObjectFSM`** (§3.1). `D = 3` is 30 s at the deployed cadence,
   so this is worth doing on its own schedule.
6. **`Rebaseline` via option 3** (§3.2).
7. **Result accessor + telemetry schema** (§3.3). Needs gateway coordination —
   start the conversation early, land it last.

Steps 2–4 are independent of the blockers and worth doing regardless; steps 5–7
depend on step 1.

---

## 8. Corrections to feed back into the two documents

**`sol_people_detection_baseline.md`**

| § | Correction |
| --- | --- |
| §8 | `DwellConfig` and `StaticObjectFSM` do not exist on target — those rows describe the Python prototype (§3.1) |
| §8 | `sol::Pipeline` needs no split; the global-change guard works caller-side (§2.4) |
| §8 | `Rebaseline(region, gray)` cannot be called from `sensing.cc` — gray is not exposed (§3.2) |
| §8 | Add a row: `sensing.h` needs a SoL result accessor; none exists (§3.3) |
| §10.2 | Cadence is defined, not undefined: 120–3600 s. "Specify in seconds" does not rescue frame-count parameters (§5.1) |
| §12 Q4 | Retention exists (`backup_4k_t`, 4 KB); scalars fit, the model does not (§5.2) |
| §9.4 | At one frame per wake the ISP term is a per-frame step, not a slow drift (§6) |
| new | Installation mode never signals `sensing_task`, so the baseline cannot run there (§5.3) |

**`alif_image_processing_flow.md`** — no corrections. Everything cross-checked
in it held: the ISP stage order, the 320×320-vs-320×240 mismatch, the
`.bss.sram0` zeroing, the one-inference-per-wake cadence and the dead tracker
code were all confirmed against source.

---

## 9. Evidence index

| Claim | Source |
| --- | --- |
| Task entry, current SoL-before-inference order | `source/use_case/pg_object_detection/src/sensing.cc:89,119,125` |
| Frame buffer is read-only for SoL | `include/SignOfLifeDetector.hpp:64` |
| `Reset()` exists on the detector | `include/SignOfLifeDetector.hpp:73` |
| Learn-in counter lives inside `Apply` | `src/BackgroundModel.cc:62–71` |
| Model reset keeps arena capacity | `src/BackgroundModel.cc:29`, `include/BackgroundModel.hpp:41` |
| Scratch arena reset per `Process` | `src/SolPipeline.cc:40` |
| `FrameResult` contents; `Image` lifetime | `include/SolTypes.hpp:29,64` |
| No dwell parameters in the config | `include/SolTypes.hpp:80–110` |
| `signOfLife` is a per-frame area test | `src/SolPipeline.cc:114` |
| No SoL accessor | `include/sensing.h:29–33` |
| Unused `MotionNotification` | `pb/telemetry_payload.pb-c.h:219` |
| GW/Alif schema drift warning | `include/Production.h:66` |
| Sleep-policy detection flag at 0.5 | `src/UseCaseHandler.cc:57` |
| Wake-period bounds | `include/Production.h:69–72` |
| Retained region, layout and word-size rule | `source/hal/source/platform/ensemble/include/backup4k_sram.h:17–47`, `source/backup4k_sram.c:5` |
| Inference only in `MODE_OPERATIONAL` | `src/application.cc:634` |
| Detection struct fields | `include/Detection.hpp:21` |
| Dead tracker code | `include/BYTETracker.hpp`, `STrack.hpp`, `kalmanFilter.hpp`, `lapjv.hpp` — no instantiation in `source/use_case/` |
