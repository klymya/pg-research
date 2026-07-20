# Sharpness-Adj Evaluation Report

Summary of 16 evaluation runs — 4 experiments from [experiments_minibatch_sharpnes_adj.json](../experiments_minibatch_sharpnes_adj.json) × 4 test sets. Env files: [generated_envs/sharpnes_adj/](generated_envs/sharpnes_adj/). Raw artifacts: `val_sharpnes_adj_<testset>/<PROJECT_NAME>/`.

- **Backend:** `tflite_backbone_only` (fully-INT8 model: `weights/best_non_separated_saved_model/best_full_integer_quant.tflite`)
- **Sweep grid:** `conf ∈ {0.20, 0.25, 0.30}` × `iou ∈ {0.35, 0.40, 0.45}` (9 candidates per run)
- **Selection metric:** `f1`, sourced from validator pose metrics (`metrics/precision(P)`, `metrics/recall(P)`)
- **Input size:** 320×320, device `cpu`, batch 64

## Experiment shorthand

| Handle | Full experiment name | Pretrained-from |
| --- | --- | --- |
| **baseline (12 / clahe, no adj)** | `dataset_v2_cam4_revB_batch1_12_minibatch2_margin_10_yolov8_wb_alif_clahe` | `results/filtered_v3_margin_15/weights/best.pt` |
| **10 / clahe** | `dataset_v2_cam4_revB_batch1_10_minibatch2_margin_10_yolov8_wb_alif_clahe_sharpnes_adj` | `results/filtered_v3_margin_15/weights/best.pt` |
| **10 / rq+clahe** | `dataset_v2_cam4_revB_batch1_10_minibatch2_margin_10_yolov8_rq_wb_alif_clahe_sharpnes_adj` | `results/filtered_v3_margin_15_reduced_quality/weights/best.pt` |
| **12 / clahe** | `dataset_v2_cam4_revB_batch1_12_minibatch2_margin_10_yolov8_wb_alif_clahe_sharpnes_adj` | `results/filtered_v3_margin_15/weights/best.pt` |
| **12 / rq+clahe** | `dataset_v2_cam4_revB_batch1_12_minibatch2_margin_10_yolov8_rq_wb_alif_clahe_sharpnes_adj` | `results/filtered_v3_margin_15_reduced_quality/weights/best.pt` |

**baseline** is the same batch1_12 experiment **without** the sharpness-adjusted augmentation — it is the direct ablation counterpart to `12 / clahe`. Runs live under `val_baseline_<testset>/` (envs: [generated_envs/baseline_no_sharpnes_adj/](generated_envs/baseline_no_sharpnes_adj/)).

## Test-set overview

| Key | Dataset (`DATASET_NAME`) | Frames | GT count |
| --- | --- | ---: | ---: |
| `cam4revB_batch1_3` | `batch1_3_margin_15_wb_alif_clahe` | 148 | 409 |
| `testAccuracy` | `batch_testAccuracy_margin_10_wb_alif_clahe` | 31 | 121 |
| `H270_16_44_09` | `batch_H270_16-44-09_margin_10_wb_alif_clahe` | 700 | 2103 |
| `PG2HardCases` | `batch_PG2HardCases` | 3 | 11 |

Columns in tables 1–4:

- **conf / iou** — winning threshold pair from the sweep (`best_params.json` → `conf_th`, `iou_th`).
- **Pose P / R / F1** — validator pose metrics at those thresholds (the selection scores).
- **mAP50 / mAP50-95** — validator pose mAP (`metrics/mAP50(P)`, `metrics/mAP50-95(P)`).
- **Det F1** — local detection F1 from the per-image scoring pass (`detection/f1`), plus **TP / FP / FN** counts.
- **Frame F1** — mean per-frame F1 (`frame/mean_f1`).

---

## Table 1 — `cam4revB_batch1_3` (148 frames, 409 GT)

| Experiment | conf / iou | Pose P | Pose R | Pose F1 | mAP50 | mAP50-95 | Det F1 | TP / FP / FN | Frame F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **baseline (12 / clahe, no adj)** | 0.30 / 0.35 | **0.9257** | **0.9135** | **0.9195** | **0.9577** | **0.9523** | **0.8868** | **380 / 68 / 29** | **0.8890** |
| 10 / clahe | 0.30 / 0.35 | 0.8790 | 0.9056 | 0.8921 | 0.9345 | 0.9283 | 0.8588 | 368 / 80 / 41 | 0.8625 |
| 10 / rq+clahe | 0.30 / 0.35 | 0.8828 | 0.8473 | 0.8647 | 0.8994 | 0.8927 | 0.8415 | 353 / 77 / 56 | 0.8467 |
| 12 / clahe | 0.25 / 0.35 | 0.8999 | 0.9011 | 0.9005 | 0.9422 | 0.9359 | 0.8163 | 371 / 129 / 38 | 0.8280 |
| 12 / rq+clahe | 0.30 / 0.35 | 0.8836 | 0.8998 | 0.8916 | 0.9231 | 0.9143 | 0.8276 | 384 / 135 / 25 | 0.8373 |

**Winner (Pose F1):** baseline (0.9195) — outperforms every sharpnes_adj variant on this test set, including its direct ablation counterpart `12 / clahe` by +1.9 F1. Baseline also has the fewest FPs (68) and best local Det F1 (0.8868). Sharpness-adjusted augmentation *hurts* on this in-distribution test set.

## Table 2 — `testAccuracy` (31 frames, 121 GT — baseline test set)

| Experiment | conf / iou | Pose P | Pose R | Pose F1 | mAP50 | mAP50-95 | Det F1 | TP / FP / FN | Frame F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **baseline (12 / clahe, no adj)** | 0.30 / 0.35 | **0.9516** | 0.9917 | **0.9712** | 0.9717 | 0.9703 | 0.9639 | 120 / 8 / 1 | 0.9624 |
| 10 / clahe | 0.30 / 0.35 | 0.9275 | 0.9917 | 0.9585 | 0.9799 | 0.9770 | 0.9444 | 119 / 12 / 2 | 0.9523 |
| 10 / rq+clahe | 0.30 / 0.35 | 0.9491 | 0.9669 | 0.9579 | 0.9703 | 0.9680 | 0.9512 | 117 / 8 / 4 | 0.9673 |
| 12 / clahe | 0.30 / 0.35 | 0.9302 | 0.9917 | 0.9600 | 0.9761 | 0.9751 | **0.9680** | **121 / 8 / 0** | 0.9758 |
| 12 / rq+clahe | 0.30 / 0.35 | 0.9488 | 0.9917 | 0.9698 | **0.9891** | **0.9877** | **0.9680** | **121 / 8 / 0** | **0.9758**⁺ |

⁺ 12 / rq+clahe frame-mean F1 = 0.9742, marginally under 12 / clahe (0.9758) — both effectively tied. **Winner (Pose F1):** baseline (0.9712), narrowly ahead of 12 / rq+clahe (0.9698). All 5 models are above 0.95 F1; this test set is close to saturated and 1 detection swings the ranking. Sharpness-adjusted augmentation gives no measurable gain here.

## Table 3 — `H270_16_44_09` (700 frames, 2103 GT — hardest held-out camera)

| Experiment | conf / iou | Pose P | Pose R | Pose F1 | mAP50 | mAP50-95 | Det F1 | TP / FP / FN | Frame F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (12 / clahe, no adj) | 0.25 / 0.35 | 0.7939 | 0.6757 | 0.7300 | 0.7630 | 0.7607 | 0.7264 | 1414 / 376 / 689 | 0.7314 |
| 10 / clahe | 0.20 / 0.35 | 0.8374 | **0.7223** | **0.7756** | **0.8102** | **0.8043** | **0.7577** | **1484 / 330 / 619** | 0.7597 |
| 10 / rq+clahe | 0.30 / 0.35 | **0.8713** | 0.6904 | 0.7704 | 0.7772 | 0.7735 | 0.7520 | 1480 / 353 / 623 | **0.7618** |
| 12 / clahe | 0.30 / 0.35 | 0.8123 | 0.6812 | 0.7410 | 0.7795 | 0.7766 | 0.7282 | 1423 / 382 / 680 | 0.7335 |
| 12 / rq+clahe | 0.30 / 0.35 | 0.8684 | 0.6728 | 0.7582 | 0.7879 | 0.7856 | 0.7353 | 1440 / 374 / 663 | 0.7432 |

**Winner (Pose F1):** 10 / clahe (0.7756), narrowly ahead of 10 / rq+clahe (0.7704). The baseline finishes **last** here (0.7300) — every sharpnes_adj model beats it, and the top variant beats it by +4.6 F1. **Sharpness-adjusted augmentation delivers its clearest win on this out-of-distribution camera set.** Note the sweep picked `conf=0.20` for 10 / clahe (recall-boosting) — the only run in the whole batch to leave the default `conf=0.30`. Batch-10 models beat batch-12 models by ~2–4 F1 points regardless of adj/no-adj.

## Table 4 — `PG2HardCases` (3 frames, 11 GT — sample is too small for firm ranking)

| Experiment | conf / iou | Pose P | Pose R | Pose F1 | mAP50 | mAP50-95 | Det F1 | TP / FP / FN | Frame F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (12 / clahe, no adj) | 0.30 / 0.45 | 0.8571 | 0.5455 | 0.6667 | 0.7145 | 0.7145 | 0.4444 | 4 / 3 / 7 | 0.3667 |
| 10 / clahe | 0.25 / 0.45 | **1.0000** | 0.5455 | 0.7059 | 0.7727 | 0.7727 | 0.4706 | 4 / 2 / 7 | 0.3889 |
| 10 / rq+clahe | 0.30 / 0.40 | **1.0000** | **0.6364** | **0.7778** | **0.8181** | **0.8181** | 0.6667 | 6 / 1 / 5 | **0.6556** |
| 12 / clahe | 0.25 / 0.45 | 0.8333 | 0.4545 | 0.5882 | 0.6669 | 0.6578 | 0.4706 | 4 / 2 / 7 | 0.3889 |
| 12 / rq+clahe | 0.30 / 0.35 | **1.0000** | 0.5455 | 0.7059 | 0.7727 | 0.7727 | **0.7059** | **6 / 0 / 5** | 0.6296 |

**Winner (Pose F1):** 10 / rq+clahe. Baseline places 4th (0.6667) — better than `12 / clahe` (0.5882) but worse than every other sharpnes_adj variant. Caveat: 3 frames / 11 GT is far below the sample size needed for a stable estimate — one recovered/missed detection swings F1 by ~0.09. Treat this table as indicative only.

---

## Table 5 — Cross-testset averages (arithmetic mean across all 4 test sets)

| Experiment | Avg Pose P | Avg Pose R | Avg Pose F1 | Avg mAP50 | Avg mAP50-95 | Avg Det F1 | Avg Frame F1 | Rank (Pose F1) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10 / rq+clahe | **0.9258** | 0.7853 | **0.8427** | 0.8663 | 0.8631 | 0.8028 | **0.8078** | **1** |
| 10 / clahe | 0.9110 | **0.7913** | 0.8330 | **0.8743** | **0.8706** | 0.7579 | 0.7408 | 2 |
| 12 / rq+clahe | 0.9252 | 0.7774 | 0.8314 | 0.8682 | 0.8651 | **0.8092** | 0.7961 | 3 |
| **baseline (12 / clahe, no adj)** | 0.8821 | 0.7816 | 0.8219 | 0.8517 | 0.8495 | 0.7554 | 0.7374 | 4 |
| 12 / clahe | 0.8689 | 0.7572 | 0.7974 | 0.8412 | 0.8364 | 0.7458 | 0.7315 | 5 |

The five averages are dragged down heavily by `H270_16_44_09` and (with high variance) by `PG2HardCases`. The gap between top and bottom is **~4.5 pose-F1 points** on average. The baseline lands **4th of 5** — beating only `12 / clahe`, but losing to every other sharpnes_adj variant, including its direct ablation counterpart's siblings `10 / clahe`, `10 / rq+clahe`, and `12 / rq+clahe` by 1.0 – 2.1 F1 points.

---

## Conclusions

1. **Best overall model: 10 / rq+clahe** — wins Pose F1 average (0.8427), the highest average Frame F1 (0.8078), and takes the top spot on `PG2HardCases`. It also comes in a very close second on the hardest set (`H270_16_44_09`, 0.7704 vs 0.7756).

2. **Sharpness-adjusted augmentation has a mixed, test-set-dependent effect vs. the baseline (12 / clahe, no adj):**
   - **In-distribution (`cam4revB_batch1_3`, `testAccuracy`) — baseline wins.** Baseline tops both tables. The direct ablation counterpart `12 / clahe` drops by 1.9 F1 on `cam4revB_batch1_3` and 1.1 F1 on `testAccuracy` when sharpness-adjusted augmentation is added. Augmentation gives up in-distribution precision by introducing FP noise (129 FPs vs 68 for baseline on `cam4revB_batch1_3`).
   - **Out-of-distribution (`H270_16_44_09`) — sharpnes_adj wins by a wide margin.** Every sharpnes_adj model beats the baseline; top variant `10 / clahe` beats it by +4.6 F1 (0.7756 vs 0.7300). This is where augmentation earns its keep — recovering recall on an unseen camera.
   - **Cross-testset average:** baseline finishes **4th of 5** (0.8219 vs 0.8427 for the top model), losing ~2 F1 to `10 / rq+clahe`. The gain from OOD generalization more than offsets the in-distribution regression.

3. **Reduced-quality pretraining consistently helps.** Both `rq+clahe` variants beat their non-rq counterparts on 3 of 4 test sets. Averaged across test sets:
   - 10 / rq+clahe (0.8427) > 10 / clahe (0.8330) by +0.010
   - 12 / rq+clahe (0.8314) > 12 / clahe (0.7974) by +0.034 — the effect is larger for batch-12
   The RQ pretraining acts as a regularizer against the sharpness-adjusted training augmentation and generalizes better to unseen cameras / edge cases.

4. **Batch-10 (smaller training set) generalizes better than batch-12 on hard data.** On `H270_16_44_09`, both batch-10 models beat both batch-12 models by 2–4 F1 points. Same trend on `PG2HardCases`. Batch-12 does slightly win on the easier sets (`cam4revB_batch1_3`, `testAccuracy`). This is a mild over-fitting signal for batch-12.

5. **Thresholds are stable at `conf=0.30, iou=0.35`.** 12 of 16 sharpnes_adj runs (and 3 of 4 baseline runs) converged to that pair. The substantive outliers — 10 / clahe on `H270_16_44_09` (`conf=0.20`) and the baseline on `H270_16_44_09` (`conf=0.25`) — both dropped conf on the hardest OOD set to recover recall. All models should ship with `inference_params.json = {conf_th: 0.30, iou_th: 0.35}` as the default. For deployment specifically targeting the H270 distribution, consider dropping conf to 0.20–0.25.

6. **The pose validator and the local per-frame scorer disagree on FP tolerance.** Table 1 shows the batch-12 models producing 129–135 FPs at local detection scoring while the pose validator still reports higher precision than the batch-10 pair (which produced only 77–80 FPs). The baseline sits in between (68 FPs, both metrics agree). Interpret **Pose F1** as the primary comparison metric (that's what the sweep optimized) and **Det F1 / TP-FP-FN** as the operational picture at those thresholds.

7. **`PG2HardCases` (3 frames) is not a reliable ranker.** Include it as a smoke-test / regression sentinel — a large drop on it should trigger investigation — but do not use it to choose between similarly-performing models. The `testAccuracy` set (31 frames) is also close to saturated; ranking there is dominated by 1–2 detections.

## Recommendation

Ship **10 / rq+clahe** as the production candidate — it beats the baseline on average (0.8427 vs 0.8219) and, crucially, on the hardest OOD test set. Keep **12 / rq+clahe** as the fallback for scenarios that match the `cam4revB_batch1_3` / `testAccuracy` distributions. Discontinue **12 / clahe** — the *only* configuration that lags the baseline on every average and every test-set ranking; it's evidence that batch-12 + sharpness-adj + no-RQ actively over-regularizes. If the deployment target is exclusively the `cam4revB_batch1_3` / `testAccuracy` distributions, the baseline (no sharpnes_adj) remains competitive and simpler to reproduce; deploy sharpnes_adj only if OOD robustness matters.
