# Background-Subtraction Models — Time-Complexity Ranking

Per-frame time complexity of every model in `scr/models/`, in Big-O notation,
with a ranking from cheapest to most expensive and a short analysis.

## Method

The complexity was derived **analytically from the source** (`scr/models/*.py`),
not by benchmarking. Each model's `apply(frame)` is a sequence of vectorised
NumPy operations; an operation over an array of `S` elements costs `O(S)`. The
per-frame cost is the sum of those operation costs, reduced to its dominant
term.

**Parameters** (all independent of each other):

| symbol | meaning | default | at QVGA 320×240 |
| --- | --- | --- | --- |
| `P` | pixels per frame, `H·W` | — | `76 800` |
| `C` | channels (gray → 1, colour → 3) | 1–3 | small constant |
| `K` | Gaussian modes per pixel (MOG) | 3 | small constant |
| `N` | samples per pixel (ViBe) | 20 | constant |
| `D` | retained PCA components (Eigen) | 10 | constant |
| `M` | history frames before PCA fit (Eigen) | 20 | constant |

**Conventions**

- Reported figures are the **steady-state per-frame** cost. Costs that occur
  once (Eigen's PCA fit) or only every few frames (adaptive median's update) are
  called out separately.
- Within one asymptotic class the ranking is broken by the **per-pixel constant
  factor** (how many element-wise passes over the frame), which Big-O hides.
- Ranking uses the default parameters (`C≤3, K=3, D=10, N=20`) to make the
  multipliers over `P` concrete.

## Summary — complexity, measured speed, and memory

One table, sorted by **time complexity** (per-frame). `N` is the measured
per-frame time relative to the fastest model (`X`); memory is the persistent
model state at QVGA 320×240 grayscale (`C = 1`), stored as **float32** — the C
`float` type used by the target (`alif_syntaxis_standart.md`: C99 / C++14, no
`double` needed) — with defaults `K = 3, N = 20, D = 10, M = 20`.

| # | Model | Time complexity | `N` (× fastest) | Memory (KB, float32) |
| --- | --- | --- | --- | --- |
| 1 | `static_frame_difference` | `O(P·C)` | 1.7 | 300 |
| 2 | `frame_difference` | `O(P·C)` | 1.7 | 300 |
| 3 | `running_average` | `O(P·C)` | 1.0 | 300 |
| 4 | `adaptive_median` | `O(P·C)` | **1.0 ← X** | 300 |
| 5 | `running_gaussian` | `O(P·C)` | 1.6 | 600 |
| 6 | `mixture_of_gaussians` | `O(P·K·(C+log K))` | 32 | 2 700 ¹ |
| 7 | `eigenbackground` | `O(P·D)` + `O(M²·P)` once | 1.9 ² | 3 300 ³ |
| 8 | `vibe` | `O(P·N)` | 11.7 | 6 000 ⁴ |

Fastest `X` ≈ 0.62 ms/frame (`adaptive_median` ≈ `running_average`). All figures
assume float32 storage (4 bytes/element); one full-frame plane = `P·4` = 300 KB.

Memory formulas (float32): tier-1 = `P·C·4`; running-Gaussian = `P·(C+1)·4`;
MoG = `P·K·(C+2)·4`; ViBe = `P·N·4`; Eigen (steady) = `P·(D+1)·4`.

¹ MoG's state is float32. (An earlier version silently upcast it to float64 —
5 400 KB — via a Python-float × bool multiply in `_update`; fixed to stay
float32.)
² inference only; excludes the one-time SVD and the empty first `M` frames.
³ steady state; peaks ~6 000 KB + SVD temporaries while training.
⁴ float32 samples; 1 500 KB if stored as uint8.

Note the inversion at rows 6–7: MoG has *lower* time complexity than ViBe yet
is measured slowest (`N = 32`) — see
[Why MoG measures slower than ViBe](#why-mog-measures-slower-than-vibe-smaller-big-o-larger-constant).

## Per-model analysis

| Model | Per-frame time | Dominant operation(s) | One-time / amortised | Space |
| --- | --- | --- | --- | --- |
| `StaticFrameDifference` | `O(P·C)` | abs-diff vs a fixed reference + threshold | — | `O(P·C)` |
| `FrameDifference` | `O(P·C)` | abs-diff vs previous frame + threshold; copy frame→prev | — | `O(P·C)` |
| `RunningAverageSubtractor` | `O(P·C)` | abs-diff + exponential background update | — | `O(P·C)` |
| `AdaptiveMedianSubtractor` | `O(P·C)` | abs-diff + threshold; ±1 median nudge every `r` frames | update amortised `O(P·C / r)` | `O(P·C)` |
| `RunningGaussianAverage` | `O(P·C)` | squared distance (sum over `C`) + mean/variance update | — | `O(P·C)` |
| `MixtureOfGaussians` | `O(P·K·(C + log K))` | `K`-mode distance/update `O(P·K·C)` + per-pixel mode `argsort` `O(P·K·log K)` | — | `O(P·K·C)` |
| `EigenBackground` | `O(P·D)` per frame | two matmuls `(D×P)` for projection + reconstruction | **PCA fit once at frame `M`: `O(M²·P)`**; emits empty masks for the first `M` frames | `O(M·P)` while training → `O(D·P)` after |
| `ViBe` | `O(P·N)` | compare pixel to its `N` samples + count matches | — | `O(P·N)` |

## Ranking (cheapest → most expensive, per frame)

| Rank | Model | Time complexity | Class |
| --- | --- | --- | --- |
| 1 | `StaticFrameDifference` | `O(P·C)` | linear in pixels |
| 2 | `FrameDifference` | `O(P·C)` | linear in pixels |
| 3 | `RunningAverageSubtractor` | `O(P·C)` | linear in pixels |
| 4 | `AdaptiveMedianSubtractor` | `O(P·C)` | linear in pixels |
| 5 | `RunningGaussianAverage` | `O(P·C)` | linear in pixels |
| 6 | `MixtureOfGaussians` | `O(P·K·(C + log K))` | linear × `K` modes |
| 7 | `EigenBackground` | `O(P·D)` + `O(M²·P)` once | linear × `D` components |
| 8 | `ViBe` | `O(P·N)` | linear × `N` samples |

Ranks 1–5 share the same asymptotic class `O(P·C)`; they are ordered by
increasing per-pixel arithmetic (a fixed-reference compare is the lightest, a
squared-distance-plus-variance update the heaviest of the group).

Concrete per-frame multipliers over `P` at the run's defaults (`C = 1`):
tier-1 ≈ 1, MoG ≈ `K·(C+log K)` ≈ 8, Eigen ≈ `2·D` ≈ 20 (naive matmul; BLAS
makes it far cheaper in practice), ViBe ≈ `N` ≈ 20.

## Conclusions & brief analysis

- **Five of the eight models are `O(P)`** — strictly linear in pixel count, with
  a tiny constant. Frame differencing, running average, adaptive median and the
  running Gaussian all cost one to a few element-wise passes per frame; they are
  the natural choice for real-time and for constrained targets (e.g. Cortex-M).
- **`RunningGaussianAverage` is the best cost/quality point on this data.** It
  stays in the `O(P·C)` tier yet gave the **highest F1** in the model sweep
  (`busStation` 0.726 fast, `canoe` 0.711) — linear cost, top accuracy.
- **`MixtureOfGaussians` adds a `K` factor** for multimodal backgrounds. The
  extra cost over tier-1 is the per-pixel `argsort` of `K` modes
  (`O(P·K·log K)`); with `K=3` it is a modest constant, not an order-of-magnitude
  jump.
- **`ViBe` has the highest *asymptotic* per-frame cost** (`O(P·N)`, `N=20`) and
  is the heaviest in memory (6 000 KB — 20 samples per pixel). *Measured*,
  though, MoG is the slowest (32× vs ViBe's 12×) — a constant-factor inversion
  explained in the empirical section. ViBe's F1 is competitive but not better
  than the far cheaper running Gaussian here, so the cost is hard to justify on
  this data.
- **`EigenBackground` is a special case.** Steady-state inference is `O(P·D)`,
  but it carries a one-time **PCA fit `O(M²·P)`** (dense SVD over an `M×P`
  matrix) and, critically, produces **empty output for the first `M` frames** —
  which is why it scored 0 at 0.1 fps (only 5–6 frames < `M=20`). It needs both
  many frames and real floating-point linear algebra (an SVD), so it is the
  poorest fit for streaming or embedded use.
- **For an embedded/Cortex-M port** the ranking maps cleanly onto feasibility:
  the tier-1 `O(P)` models are ideal; MOG is acceptable; ViBe's `O(P·N)` memory
  (20 buffers per pixel) is likely prohibitive; EigenBackground needs an SVD
  kernel (e.g. the Eigen library) and a warm-up buffer, making it the least
  embeddable.

> **Summary:** cost grows as *fixed factors* over the pixel count — nothing here
> is worse than linear-per-frame except Eigen's one-time SVD. Order:
> **frame-diff ≈ running-avg ≈ adaptive-median ≈ Wren-Gaussian (`O(P)`) <
> MOG (`O(P·K)`) < Eigen (`O(P·D)` + one-time `O(M²·P)`) < ViBe (`O(P·N)`)**.
> The running Gaussian is the standout: linear cost, best measured F1.

## Empirical per-frame multipliers (X · N)

The Big-O ranking above abstracts away constant factors; this section pins them
down by **measurement**. Each model was run on QVGA (320×240) **grayscale**
frames — the exact feature the pipeline fed the native models in
`results_model_pairs` (default `--feature gray`, so `C = 1`) — with the run's
default coefficients (`K = 3`, `N = 20`, `D = 10`, `M = 20`). Reported is the
**steady-state** per-frame `apply()` time (40-frame warm-up, 300-frame average,
NumPy on CPU), expressed as a multiple `N` of the fastest model `X`.

**Fastest `X` ≈ 0.62 ms/frame** (`adaptive_median` ≈ `running_average`).

| Model | ms/frame | `N` (× fastest) |
| --- | --- | --- |
| `adaptive_median` | 0.62 | **1.0**  ← `X` |
| `running_average` | 0.63 | 1.0 |
| `running_gaussian` | 1.00 | 1.6 |
| `static_frame_difference` | 1.04 | 1.7 |
| `frame_difference` | 1.06 | 1.7 |
| `eigenbackground` | 1.15 | 1.9 *(inference only)* |
| `vibe` | 7.22 | **11.7** |
| `mixture_of_gaussians` | 19.5 | **32** |

Grouped: **tier-1 ≈ 1–2×**, **ViBe ≈ 12×**, **MoG ≈ 32×**.

**How the multipliers relate to the coefficients**

- **ViBe ≈ 12×** — the `O(P·N)` sample loop (`N = 20`); the `(H,W,20)` compare +
  reduce lands near `N/2` in practice.
- **MoG ≈ 32×** — `O(P·K·(C+log K))` with `K = 3`, but driven by ~15 array
  passes over `(H,W,K)` plus a per-pixel `argsort`; the multiplier reflects op
  count and temporaries, not just `K`.
- **EigenBackground only 1.9×** — its `O(P·D)` inference is two `(D×P)` matmuls
  that run on **BLAS**, so `D = 10` is nearly free here. This excludes the
  one-time SVD `O(M²·P)` and the empty output for the first `M = 20` frames
  (why it scored 0 at 0.1 fps).

**Caveats**

1. **NumPy/CPU multipliers, not hardware-independent.** Eigen's low 1.9× is
   BLAS; MoG's 32× is array-temporary overhead. On scalar embedded hardware
   (Cortex-M, no BLAS/SIMD) the ordering shifts — Eigen's matmul rises toward
   `~D`, MoG's temporary overhead matters less relative to arithmetic.
2. **Steady-state only** — Eigen's one-time SVD and warm-up are not in `N`.
3. **Within tier-1 the ~2× spread is constant-factor.** Big-O treats
   frame-diff / running-avg / median / Wren as equal `O(P)`; the measured order
   (adaptive-median fastest, frame-difference 1.7×) is NumPy op/temporary noise,
   which *reorders* the analytical guess that frame differencing is cheapest.

### Why MoG measures slower than ViBe (smaller Big-O, larger constant)

MoG is `O(P·K)` with `K = 3`; ViBe is `O(P·N)` with `N = 20`. Asymptotically MoG
is *cheaper* — yet it measures ~2.7× **slower** (32× vs 12×). Big-O describes how
work scales with `P`; it deliberately drops the constant factor, and here the
constant inverts the ranking.

| | asymptotic | coefficient | constant factor (hidden) |
| --- | --- | --- | --- |
| MoG | `O(P·K)` | `K = 3` (small) | **huge** — ~30 array ops incl. a per-pixel `argsort` + 4 gathers |
| ViBe | `O(P·N)` | `N = 20` (large) | **tiny** — ~4 simple streaming ops |

- **MoG** runs ~30 distinct NumPy operations per frame, each allocating a new
  `(H,W,K)` / `(H,W,K,C)` temporary and making a full memory pass: `delta`,
  `squared_distance`, `matches`, `argmax`, `onehot`, `cumsum`, `take_along_axis`,
  weight/mean/variance updates, `np.where`, `clip`, weakest-mode replacement,
  renormalise — **then `_sort_modes`: a per-pixel `argsort` over `K` plus three
  more `take_along_axis` gathers.** `argsort` and gather are the costly ones:
  sorting per pixel and scattered indexing don't vectorise like streaming
  arithmetic.
- **ViBe** does ~4 operations: one fused broadcast compare
  `|gray − samples| ≤ radius` over `(H,W,N)`, one reduction, and a couple of
  random scatters on the few changed pixels — contiguous, SIMD-friendly,
  almost no temporaries.

Rough model: `time ≈ (ops × cost_per_op) × coefficient × P`. MoG pays
`~30 heavy ops × 3`, ViBe pays `~4 cheap ops × 20`; the op count and the *kind*
of ops (sort/gather/temporaries) outweigh MoG's smaller coefficient. On scalar
embedded hardware (no BLAS/SIMD, allocation cheaper relative to arithmetic) this
could flip back — there the `K = 3 < N = 20` advantage may reassert itself.

*The Big-O figures are source-based (analytical); the multipliers above are
measured on NumPy/CPU and would need re-measuring on the target hardware.*
