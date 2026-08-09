# Sign-of-Life ↔ People-Detection — Baseline Flow

How the people detector and the Sign-of-Life (SoL) background-subtraction stage
hand off to each other, per frame.

Rationale, measurements, failure modes, deferred work and the ideal-world
variants are in **`sol_people_detection_rationale.md`**. Alignment with the
firmware as built is in **`sol_baseline_alignment_report.md`**. This document is
the flow only.

> ### The whole design is stateful, and the state must persist across frames
>
> Nothing here works on a single frame in isolation. Every rule below reads or
> writes state carried from earlier frames, and **in `MODE_OPERATIONAL` each
> frame is a separate cold boot** — the subsystem is power-cycled between wakes
> and `.bss.sram0` is re-zeroed, so today none of it survives.
>
> | What must persist | Size | Home |
> | --- | --- | --- |
> | `empty_run`, `frames_in_model`, `hits`, `state` | ~16 B | **fits** the existing 4 KB retained region (`backup_4k_t`, `uint32_t` fields only) |
> | **Background model** — per-pixel `mean` + `variance` | **614,656 B** @ QVGA | **no home** — 150× the retained region |
> | reported-region bookkeeping, if delayed commit is added later | tens of bytes | retained region |
>
> The counters can be made wake-persistent today; the model cannot. Until it is,
> the machine never leaves `WARMUP` and nothing below runs. This is the one
> blocker — see §7.

---

## 1. Rules

1. **People detection runs on every frame.** Its output is the only input to the
   state machine, which needs the detection *count*, not the boxes.
2. **The background model updates only while the scene is confirmed empty** —
   `n_persons == 0` for `N` consecutive frames. While anyone is in frame the
   model is frozen whole and SoL does not run at all.
3. **Whatever the scene contains when learning completes is the background.**
   The sensor is commissioned on a clean scene; that is the installer's
   responsibility. Anything already present is background and is never reported.
4. **`learn_frames` is a floor, not a window.** It is the minimum number of
   empty frames before decisions may start — not the end of learning. The model
   keeps updating on *every* empty frame for the life of the session and keeps
   converging as it goes; only the initial all-pixel phase ends at
   `learn_frames`.
5. **Freezing is immediate, resuming is not.** A single frame with a person
   freezes the model on that same frame; resuming takes `N` consecutive empty
   frames. Freezing is a pause — learn-in progress and the model are kept.
6. **A left object never decays.** The update is selective (foreground pixels
   are never updated), so an object stays foreground until it is deliberately
   folded back in by a re-baseline after it has been reported. Because it also
   never moves, persistence needs a counter, not a tracker — no IoU association,
   no track identity.
7. **`sign_of_life` means: the scene contains content the people detector does
   not explain.** A left object raises it; so does a person the detector has
   been missing for `N` frames.

---

## 2. Definitions

| Name | Meaning |
| --- | --- |
| `n_persons` | detections with `confidence ≥ conf_person` |
| **empty frame** | `n_persons == 0` |
| `empty_run` | consecutive empty frames, including now |
| `blobs[]` | SoL regions surviving `min_area` |
| `hits` | consecutive frames on which *some* blob reached `sol_area` |
| `sign_of_life` | edge — fires once per newly appeared object |

---

## 3. States

```mermaid
stateDiagram-v2
    [*] --> WARMUP

    WARMUP: WARMUP — model learning, on empty frames only. No decisions, flag forced false
    OCCUPIED: OCCUPIED — someone is, or recently was, in frame. Model frozen, SoL not run
    WATCH: WATCH — scene confirmed empty. Model updates, SoL decides every frame

    WARMUP --> OCCUPIED: person detected — learn-in pauses, progress kept
    OCCUPIED --> WARMUP: empty_run reaches N, learn-in not yet complete
    WARMUP --> WATCH: learn_frames empty frames accumulated

    WATCH --> OCCUPIED: person detected — freeze on the same frame
    OCCUPIED --> WATCH: empty_run reaches N, model valid

    WATCH --> WARMUP: global change — model reset, learn-in restarts
```

| State | Model | SoL layers | `sign_of_life` |
| --- | --- | --- | --- |
| `WARMUP` | updates **all** pixels | 0–1 only | forced `false` |
| `OCCUPIED` | **frozen — not called** | none | cleared on entry |
| `WATCH` | **keeps learning** — every empty frame, selective | all | set when `hits` reaches `D` |

**Learning does not stop at `WATCH`.** That transition marks the first frame on
which a decision is *allowed*, nothing more. Every empty frame after it still
updates the model — selectively, so a left object is preserved — and the
variance keeps tightening toward the noise floor. A model 100 frames into a
quiet room is materially better than one at frame 5, and no state change marks
the difference; `frames_in_model` is what reports it.

---

## 4. Per-frame flow

```mermaid
flowchart TD
    F(["frame — processed RGB888 320x240"]) --> DET["<b>people detection</b> — every frame<br/>n_persons = detections with conf >= conf_person"]

    DET --> Q1{"n_persons > 0 ?"}

    Q1 -->|yes| FRZ["<b>freeze, same frame</b><br/>state = OCCUPIED · empty_run = 0<br/>sign_of_life = false · hits = 0<br/>SoL not run at all"]
    FRZ --> OUT

    Q1 -->|no| INC["empty_run += 1"]
    INC --> Q2{"empty_run >= N ?"}

    Q2 -->|no| HOLD["still frozen — resume hysteresis<br/>state stays OCCUPIED"]
    HOLD --> OUT

    Q2 -->|yes| L01["<b>SoL layers 0-1</b> — the only place the model changes<br/>Gaussian blur → gray → running Gaussian average<br/>classify against the pre-update model, then update<br/>WARMUP → every pixel · WATCH → background pixels only"]

    L01 --> Q3{"frames_in_model >= learn_frames ?"}

    Q3 -->|no| WU["<b>state = WARMUP</b><br/>learn-in advanced by one empty frame<br/>no decisions, flag stays false"]
    WU --> OUT

    Q3 -->|yes| TOW["<b>state = WATCH</b>"]
    TOW --> GLOB{"foreground fraction<br/>> global_change_frac ?"}

    GLOB -->|yes| RST["<b>scene changed globally</b><br/>model reset · hits = 0 · state = WARMUP<br/>nothing reported"]
    RST --> OUT

    GLOB -->|no| L3["<b>SoL layers 3+</b><br/>open → close → connected components<br/>→ area gate at min_area"]

    L3 --> HIT{"any blob area >= sol_area ?"}
    HIT -->|no| Z["hits = 0"]
    Z --> OUT
    HIT -->|yes| INCH["hits += 1"]

    INCH --> Q4{"hits >= D ?"}
    Q4 -->|no| CLR["not yet persistent — flag unchanged"]
    Q4 -->|yes| SET["<b>sign_of_life = true</b><br/>report every qualifying blob<br/>bbox, area, centroid"]

    SET --> RB["<b>re-baseline those regions</b> · hits = 0<br/>mean = current gray, variance = init<br/>→ the objects become background"]

    RB --> OUT
    CLR --> OUT
    OUT(["telemetry: n_persons, state, empty_run, sign_of_life, blobs"])
```

**The state is derived, never stored.** It is a function of `n_persons`,
`empty_run` and `frames_in_model` — which is why the learn-in test is
`frames_in_model >= learn_frames` and not `state == WARMUP`. Testing the state
name would be a bug: an occupancy sets `OCCUPIED`, and nothing would ever set
`WARMUP` back, so an interrupted learn-in would silently start making decisions
on a half-built model.

---

## 5. Sequence of one full cycle

A person arrives, works, leaves and forgets a laptop. Phase lengths are given as
parameters, not frame counts — see §6.

```mermaid
sequenceDiagram
    participant CAM as Camera + ISP
    participant NN as People detector
    participant ARB as Arbitration — sensing.cc
    participant BG as Background model
    participant PER as Persistence counter
    participant TX as Telemetry

    Note over ARB: state = WARMUP

    loop arming · N empty frames
        CAM->>NN: frame, RGB888 320x240
        NN-->>ARB: n_persons = 0
        ARB->>ARB: empty_run += 1
    end
    Note right of ARB: empty_run below N — model not touched yet

    loop learn-in · learn_frames empty frames
        CAM->>NN: frame
        NN-->>ARB: n_persons = 0
        ARB->>BG: Apply(gray) — every pixel updates
        BG-->>ARB: mask, empty
    end
    Note over ARB,BG: learn_frames reached → state = WATCH.<br/>These frames are the golden background.

    loop watching · every empty frame
        CAM->>NN: frame
        NN-->>ARB: n_persons = 0
        ARB->>BG: Apply(gray) — selective, keeps learning
        BG-->>ARB: mask
        ARB->>PER: no blob at sol_area — hits = 0
    end

    Note over NN: person enters
    CAM->>NN: frame
    NN-->>ARB: n_persons > 0
    ARB->>ARB: state = OCCUPIED, empty_run = 0
    ARB->>PER: hits = 0
    Note over ARB,BG: Model frozen — Apply not called.<br/>SoL skipped entirely, including blur and gray.

    loop person present
        CAM->>NN: frame
        NN-->>ARB: n_persons > 0
    end

    Note over NN: person leaves, laptop stays

    loop resume hysteresis · until empty_run reaches N
        CAM->>NN: frame
        NN-->>ARB: n_persons = 0
        ARB->>ARB: empty_run += 1
    end

    ARB->>ARB: state = WATCH
    ARB->>BG: Apply(gray) — selective update
    BG-->>ARB: mask — laptop is foreground
    ARB->>PER: blob at sol_area — hits = 1

    loop accumulating persistence · until hits reaches D
        CAM->>NN: frame
        NN-->>ARB: n_persons = 0
        ARB->>BG: Apply(gray)
        BG-->>ARB: mask — laptop still foreground
        ARB->>PER: hits += 1
    end
    PER-->>ARB: hits = D → report

    ARB->>TX: sign_of_life = true, bbox / area / centroid
    ARB->>BG: Rebaseline(bbox) — mean := gray, variance := init
    Note over BG: Laptop is background now.<br/>The scene has a new normal.
```

| Phase | Length | What happens |
| --- | --- | --- |
| arming | `N` empty frames | `empty_run` climbs to `N`; the model is not touched |
| learn-in | `learn_frames` empty frames | all-pixel update — this frame set *is* the golden background |
| watching | until a person appears | SoL decides on every empty frame. The model is not finished: it keeps learning and gets more sensitive as it goes |
| occupied | while anyone is in frame | frozen on the first detected frame. The desk keeps its pre-arrival reference |
| resume hysteresis | `N` empty frames | still frozen; a short detector dropout does not resume it |
| persistence | `D` empty frames | the laptop is foreground from the first `WATCH` frame; `hits` climbs to `D` |
| report | 1 frame | `sign_of_life = true`, blobs reported, regions re-baselined |

Two latencies fall out:

```
time to a usable model   = ( N + learn_frames ) × wake_period
detection after departure = ( N + D )           × wake_period
```

The deployed `wake_period` is **10 s** (≈ 0.1 fps) — a firmware setting, not a
tuning parameter: `Production.h:75`, `sleep_policy.cc:26`.

---

## 6. Parameters

**Values are deliberately omitted — the research that fixes them is in
progress.** This table is the list of knobs, what each one controls, and what
will settle it. Current implemented defaults live in `sol::PipelineConfig`
(`SolTypes.hpp`); proposed starting points and the reasoning behind them are in
`sol_people_detection_rationale.md` §7, and should be treated as provisional.

| Symbol | Controls | Fixed by |
| --- | --- | --- |
| `conf_person` | which detections count as a person, and so when the model freezes | dropout **and hard-negative** rates from the 20-case eval suite — it is a two-sided trade (rationale §3.1) |
| `N` | resume hysteresis — empty frames required before unfreezing | detector flicker / dropout statistics |
| `learn_frames` | minimum empty frames before decisions start. A floor, not a window | M1 |
| `α` | background learning rate | M1, M5 |
| `min_area` | smallest blob kept by the area gate | M8 |
| `sol_area` | smallest blob that may raise the flag — the sensitivity floor | **M8** (TC-SOL-03: is a mug above it at QVGA?) |
| `D` | consecutive frames a blob must persist before reporting | M4 |
| `global_change_frac` | foreground fraction above which a frame is a changed scene, not objects | M6 (TC-SOL-06/07) |
| blur / open / close kernels | denoise and morphology, inherited from the SoL pipeline | existing BGS evaluation |

All of these are pixel- or frame-counted at the working resolution and cadence,
so any change to either silently retunes the detector.

---

## 7. Status

Not yet built, in the order they block each other:

1. **The background model does not survive a wake.** `.bss.sram0` is re-zeroed
   on every STOP cycle, so the model is rebuilt from frame 1 every 10 s and
   never leaves `WARMUP`. Everything below is moot until this is solved. The
   arbitration scalars fit the existing 4 KB retained region; the 614,656 B
   model does not.
2. **No persistence check on target** — `signOfLife` is a bare per-frame area
   test. The counter above is ~5 lines in `sensing.cc`; the full IoU-association
   FSM from `scr/fsm.py` is *not* needed unless per-object dwell is wanted
   later (rationale §9.5).
3. **No `Rebaseline`**, and no path for the SoL result to leave `sensing.cc`.

What already fits with no change: `BackgroundModel::Apply`, `Pipeline::Process`,
`SignOfLifeDetector::Reset()`. Details and costs in
`sol_baseline_alignment_report.md`.
