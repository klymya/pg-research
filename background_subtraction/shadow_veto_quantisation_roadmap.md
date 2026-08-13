# Shadow veto: three build modes and a `uint8` reference

Roadmap for quantising the HSV shadow veto and making its cost a **three-way
build choice** rather than the current on/off. Companion to the `uint8` model
state already shipped (`models_complexity_ranking.md` §*uint8 model state*).

> **Status (2026-08-13): implemented, with the mode split scoped down.**
> Python carries all three modes (`ShadowConfig.reference_dtype`); **C++ carries
> only `OFF` and `UINT8`** — a float reference is 12 B/px of persistent state on
> a target that has none to spare, so the embedded ports implement the quantised
> path outright, exactly as they do for the model state. That keeps
> `SOL_SHADOW_VETO` a plain `BOOL` and drops §4's tri-state CMake work entirely.
> Phases 1–4 are done; the measured results are in §3 and the doc pass in §6.

---

## 1. What is actually being quantised

The veto has three numerical parts and only one of them is floating point today:

| Part | Today | Storage? |
| --- | --- | --- |
| RGB → HSV conversion | **already int32 fixed point** (`bgr_to_hsv` reciprocal tables; `RgbToHsvPixel` uint8→uint8) | no |
| EW RGB background reference | **float32, 3 planes, 12 B/px** | **yes — 921,600 B at QVGA** |
| Ratio / saturation / hue comparisons | float32 scalars | no (registers) |

So "int32 vs fp32" is not an axis: the HSV conversion is integer in every mode
and always has been. The memory is entirely the **reference**; the comparison
arithmetic is a determinism question, not a memory one.

## 2. The three modes

| Mode | Reference | Veto arithmetic | Persistent arena @ QVGA |
| --- | --- | --- | ---: |
| `OFF` | not allocated | — | 153,856 B |
| `FLOAT` | float32, 12 B/px | float | 1,075,456 B |
| `UINT8` | uint8, 3 B/px | integer end-to-end | **384,256 B** |

`FLOAT` is today's behaviour, kept as the numerical reference. `UINT8` is the
new target. Against `FLOAT` it returns **691,200 B**; against the original
all-float pipeline (before the model went uint8) the arena drops 1,536,256 →
384,256 B, a **4× reduction**.

At QQVGA 160×120: `OFF` 38,656 B, `FLOAT` 268,864 B, `UINT8` 96,256 B.

**Why this unlocks the veto.** Free SRAM0 on the revB build is 1,016,048 B.
Turning the veto on costs 921,600 B in `FLOAT` — it fits, with 94 KB left, i.e.
it consumes essentially the whole headroom. In `UINT8` it costs 230,400 B and
leaves 785,648 B. That is the difference between "technically possible" and
"budgetable alongside anything else".

## 3. Evidence so far, and its limits

Final measurement, all three modes, `data/` at QVGA, on the two sequences that
actually carry shadow labels:

| Sequence | Mode | Precision | Recall | F1 | Ref B/px | SoL vs `float` |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| busStation | `off` | 0.8663 | 0.6322 | **0.7310** | 0 | 96.16 % |
| busStation | `float` | 0.8844 | 0.5305 | 0.6632 | 12 | — |
| busStation | `uint8` | 0.8840 | 0.5229 | 0.6571 | 3 | 99.84 % |
| traffic | `off` | 0.4877 | 0.8697 | 0.6250 | 0 | 94.52 % |
| traffic | `float` | 0.5238 | 0.8196 | **0.6391** | 12 | — |
| traffic | `uint8` | 0.5241 | 0.8179 | 0.6388 | 3 | 99.49 % |

**`uint8` vs `float` clears the gate**: ΔF1 −0.0061 (busStation) and −0.0002
(traffic), Sign-of-Life agreement 99.84 % and 99.49 %. Quantising the reference
costs less than quantising the model did.

**But note what the `off` rows say, and why the metric cannot settle it.**
Under the CDNET protocol, shadow-labelled pixels (groundtruth 50) are *excluded*
from scoring — they count as neither true nor false positive. So a veto that
correctly removes a shadow earns nothing, while every true-foreground pixel it
also removes is scored as a miss. The metric is structurally incapable of
rewarding this stage. On busStation it therefore reads as **+0.068 F1 for
turning the veto off**, and on traffic as −0.014 for the same. Neither figure
should be read as "the veto is worth it / not worth it": the veto's real job is
to stop a person's cast shadow inflating their blob area past
`area_threshold`, which pixel F1 never measures. That question needs a
different experiment, and this work does not answer it.

**Why these two sequences.** Counting CDNET shadow labels (groundtruth value
50) across the local `data/` tree:

| Sequence | Sampled gt frames carrying shadow label |
| --- | --- |
| busStation | 20 / 32 |
| traffic | 13 / 40 |
| boats | 1 / 200 |
| canoe, tramCrossroad, turnpike | **0** |

An earlier pass measured this change on busStation and `canoe`; `canoe` contains
**no labelled shadows at all**, so half that evidence said nothing about the
veto and it was replaced by `traffic`. There is no CDNET *shadow* category clip
in the tree, so the veto remains under-validated at any dtype. This work does
not fix that and should not be read as having done so.

## 4. Mechanism as built

The arena is a statically sized `.bss` array, so whether the reference exists
must be known at compile time — unchanged from before. What the scope-down
removed is the need for that switch to carry a *precision* as well: with only
`OFF` and `UINT8` on the target, `SOL_SHADOW_VETO` stays a plain CMake `BOOL`,
every existing invocation keeps working, and `SolPersistentBytes` collapses to a
single uniform expression because nothing persistent is float any more:

```cpp
constexpr size_t SolPersistentBytes(size_t width, size_t height,
                                    bool shadowEnabled = kSolShadowEnabled)
{
    return width * height * (2u + (shadowEnabled ? 3u : 0u)) * sizeof(uint8_t) +
           256u /* alignment slack */;
}
```

`ShadowSuppressor` holds `PersistentVector<uint8_t> m_reference` and quantises
`shadowAlpha`, the ratio bounds and the saturation/hue limits once in the
constructor. No `std::conditional`, no storage-policy overloads, no template on
`Pipeline` — all of which the three-mode design would have required.

**Python** carries all three modes, without the compile-time constraint:
`ShadowConfig.enabled` stays a bool and gains
`reference_dtype: "float32" | "uint8"`, plus `quantised` and
`reference_bytes_per_pixel` helpers. `"uint8"` switches the comparisons to the
integer path too, which is what keeps it bit-identical to C++. Turning `enabled`
into a tri-state enum would have touched every call site and test for no
expressive gain.

**Ratio bounds are Q16, not Q8.** In Q8, `0.40` rounds to `102/256 = 0.3984` — a
0.4 % threshold shift that disagreed with the float path on 62 of 65,536
`(V_frame, V_background)` pairs. Q16 costs nothing (`255 << 16` = 16.7 M keeps a
128× margin under `INT32_MAX`) and cuts that to **18 pairs, all at exactly ratio
0.40 or 0.95** — the irreducible ones, where the float path's `+1e-6` divide
guard tips an exact-boundary ratio the other way.

## 5. Corner cases the integer path introduces

1. **`V_background == 0` flips meaning.** The float test is
   `vf / (vb + 1e-6)`: at `vb == 0` the ratio is either enormous (`vf > 0`) or
   zero (`vf == 0`), and both fail the `[0.40, 0.95]` band — so a black
   background pixel is *never* shadow. The integer form
   `vf·256 >= ratioMinQ8·vb && vf·256 <= ratioMaxQ8·vb` gives `0 >= 0 && 0 <= 0`
   at `vf == vb == 0`, i.e. **shadow**. Needs an explicit `if (vb == 0) continue;`
   guard, and a test that pins it.
2. **Threshold quantisation.** `saturationDeltaMax = 40.0` and
   `hueDeltaMax = 12.0` are already integral, but a retune to e.g. 40.5 would
   round. Quantise once at construction and document the rounding.
3. **Ratio product width.** `vf·256` peaks at 65,280 and `ratioMaxQ8·vb` at
   61,965 — both exceed `int16`, so `int32` intermediates, same rule as the model.
4. **Reference dead zone.** `shadowAlpha = 0.1` gives the same 5-grey-level
   dead zone as the model mean. **Unlike the model there is no closed-form
   safety bound**: the model's rule (`1/(2α) < sqrt(threshold · var_floor)`)
   came from the algorithm, whereas the veto's tolerance depends on the ratio
   band, saturation and hue stability, all of which vary with scene brightness
   (±5 is 4 % of V on mid-grey, 25 % on a dark background; hue is undefined at
   zero saturation). Guard by analogy at `shadowAlpha > 0.071`, but treat §3's
   measurement as the actual evidence and re-measure if the veto thresholds are
   retuned.

## 6. Phases and estimate

Estimates are engineering-days for one person, excluding review.

| # | Phase | Deliverable | Est. | Status |
| --- | --- | --- | ---: | --- |
| 1 | **Python reference** | `ShadowConfig.reference_dtype` + `quantised` / `reference_bytes_per_pixel`; uint8 `_reference` with Q8 update in `bgs_model.py`; integer veto branch in `features.py` with the `V_bg == 0` guard; 6 new tests | 1.0 | **done** |
| 2 | **Evaluation** | Three modes measured on busStation + traffic (§3) | 0.5 | **done** |
| 3 | **Host C++** (`sign_of_life_cpp`) | uint8 reference, integer veto, `RgbToHsvPixel` extracted so the host matches the firmware's per-pixel conversion; bit-exactness vs Python over 60 frames | 1.0 | **done** |
| 4 | **Firmware** | Same source change; `SolPersistentBytes` all-uint8; `shadowAlpha` guard in `Init()`; native + revB builds veto-on and veto-off, arena sizes confirmed with `nm` | 1.0 | **done** |
| 5 | **Docs** | `CMakeLists.txt` / `build.sh` strings corrected; superseded-figures header added to the five analysis docs (their bodies left intact — the reasoning depends on the figures current when written) | 0.5 | **done** |

**Total ≈ 4 days estimated; delivered in one pass** because dropping the C++
float mode removed the tri-state CMake work, which was the riskiest part.

The gate between phase 2 and phase 3 — **ΔF1 ≥ −0.01 on busStation and traffic,
Sign-of-Life agreement ≥ 99 %** — was met on both sequences (§3).

## 7. Risks

| Risk | Severity | Mitigation |
| --- | --- | --- |
| Veto under-validated — no CDNET shadow-category clip locally | **med-high** | Phase 2 uses the only two clips with shadow labels; state the limitation rather than implying coverage. Fetching a shadow-category clip would be the real fix and is out of scope here |
| Three modes triple the validation surface | med | `FLOAT` is unchanged code paths; only `UINT8` is new. CI/manual checks need one build per mode, not per mode-pair |
| Retyping a `BOOL` CMake option breaks existing build trees | med | Legacy `ON`/`OFF` aliases + deprecation message; stale caches need a reconfigure, which the message should say |
| `vb == 0` divergence ships silently | med | Explicit guard plus a test that asserts float and integer paths agree on the full 0–255 × 0–255 `(vf, vb)` grid |
| Dead-zone bound is empirical, not derived | med | Documented as such; re-measure on any veto retune |
| Integer arithmetic buys no memory, only determinism | low | Stated up front so it is not sold as a saving; the real gain is bit-exact parity across the three ports |

## 8. What this does not do

- No change to the HSV conversion — already integer, nothing to gain.
- No change to the scratch arena; the veto's scratch cost was already removed
  when the conversion moved per-pixel.
- Does not address SRAM1, which sits at 98.29 % and is a scratch/decimation
  question (`sol_state_persistence_design.md` §6.4).
