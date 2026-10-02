#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_eval_int8_experiments.sh [EXP_NAME ...]

Evaluates the fully-INT8 model (tflite_backbone_only,
weights/best_non_separated_saved_model/best_full_integer_quant.tflite) of each experiment
on the four standard test sets, with the same protocol as the wiki INT8 tables and the
val_baseline_* prod runs: conf {0.20,0.25,0.30} x iou {0.35,0.40,0.45} sweep, selected by
pose F1.

Results are written into the experiment folder:
  people_detection_pg2/results/<EXP_NAME>/eval_int8/<testset>/
    eval.env               env file used for the run
    eval.log               full log
    best_params.json       selected conf/iou + selection/precision|recall|f1
    global_metrics.json    validator metrics + detection/tp|fp|fn
    sweep_results.csv      all 9 (conf, iou) candidates
    metrics_per_*.csv      per-frame / per-GT / per-prediction tables
    extended_report.html   slice-based report
    runs/, viz/            per-candidate validator outputs, per-frame overlays

With no arguments, evaluates the two bg_v2 experiments.
Run inside the yolo-export pyenv environment.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="$(cd -- "${SCRIPT_DIR}/../results" && pwd)"
DATA_DIR="/home/klymy/data/cam4_revB/yolo_dataset"

if [[ $# -gt 0 ]]; then
  EXPERIMENTS=("$@")
else
  EXPERIMENTS=(
    dataset_v2_cam4_revB_batch1_12_minibatch2_bg_v2_margin_10_yolov8_wb_alif_clahe
    dataset_v2_cam4_revB_batch1_12_minibatch2_bg_v2_margin_10_yolov8_rq_wb_alif_clahe
  )
fi

# key | dataset folder under DATA_DIR | holdout attribute bundle under SCRIPT_DIR
TEST_SETS=(
  "cam4revB_batch1_3|batch1_3_margin_15_wb_alif_clahe|holdout_data"
  "testAccuracy|batch_testAccuracy_margin_10_wb_alif_clahe|holdout_data_testAccuracy"
  "H270_16_44_09|batch_H270_16-44-09_margin_10_wb_alif_clahe|holdout_H270_16-44-09"
  "PG2HardCases|batch_PG2HardCases|holdout_PG2HardCases"
)

# The DeGirum fork is mandatory; upstream ultralytics silently gives wrong int8 results.
ULTRA_PATH="$(python -c 'import ultralytics; print(ultralytics.__file__)')"
if [[ "${ULTRA_PATH}" != *people_detection_pg2/ultralytics_yolov8* ]]; then
  echo "Error: ultralytics resolves to ${ULTRA_PATH}; run 'pyenv shell yolo-export' first." >&2
  exit 2
fi

# Check every input before starting a multi-hour run.
for EXP in "${EXPERIMENTS[@]}"; do
  for f in weights/best.pt weights/best_non_separated_saved_model/best_full_integer_quant.tflite; do
    if [[ ! -f "${RESULTS_DIR}/${EXP}/${f}" ]]; then
      echo "Error: missing ${RESULTS_DIR}/${EXP}/${f}" >&2
      exit 2
    fi
  done
done
for TS in "${TEST_SETS[@]}"; do
  IFS="|" read -r KEY DATASET_NAME ATTRS_DIR <<< "${TS}"
  for f in "${DATA_DIR}/${DATASET_NAME}/project_holdout.yaml" \
           "${SCRIPT_DIR}/${ATTRS_DIR}/frames_attributes.csv" \
           "${SCRIPT_DIR}/${ATTRS_DIR}/detections_attributes.csv"; do
    if [[ ! -f "${f}" ]]; then
      echo "Error: missing ${f}" >&2
      exit 2
    fi
  done
done

write_env() {
  local exp="$1" key="$2" dataset_name="$3" attrs_dir="$4" out_dir="$5"
  local exp_dir="${RESULTS_DIR}/${exp}"
  local dataset_pth="${DATA_DIR}/${dataset_name}"
  cat <<EOF
REPORT_BASE="${SCRIPT_DIR}"
FRAMES_ATTRS="${SCRIPT_DIR}/${attrs_dir}/frames_attributes.csv"
DETECTIONS_ATTRS="${SCRIPT_DIR}/${attrs_dir}/detections_attributes.csv"
INPUT_WIDTH=320
INPUT_HEIGHT=320
DATASET_NAME=${dataset_name}
EXP_NAME=${exp}
BASE_DATASET_PTH="${dataset_pth}"
SUBSET_NAME=holdout
DATASET_PTH="${dataset_pth}/project_holdout.yaml"
BASE_EXPERIMENT_DIR="${exp_dir}"
EVAL_BACKEND=tflite_backbone_only
WEIGHTS_PTH="${exp_dir}/weights/best.pt"
TFLITE_BACKBONE_PTH="${exp_dir}/weights/best_non_separated_saved_model/best_full_integer_quant.tflite"
TFLITE_OUTPUT_CORRECTION_MODEL_PTH=""
USE_INFERENCE_PARAMS=1
CONF_TH=0.35
IOU_TH=0.50
ENABLE_SWEEP=1
SWEEP_CONF_THS=0.20,0.25,0.30
SWEEP_IOU_THS=0.35,0.40,0.45
SELECTION_METRIC=f1
DEVICE=cpu
BATCH=64
PROJECT="${exp_dir}/eval_int8"
PROJECT_NAME=${key}
PROJECT_PTH="${out_dir}"
PREDICTIONS_PTH="${out_dir}/predictions.json"
GLOBAL_METRICS_PTH="${out_dir}/global_metrics.json"
SWEEP_RESULTS_PTH="${out_dir}/sweep_results.csv"
BEST_PARAMS_PTH="${out_dir}/best_params.json"
METRICS_PER_IMG="${out_dir}/metrics_per_img.csv"
METRICS_PER_GT="${out_dir}/metrics_per_gt.csv"
METRICS_PER_PRED="${out_dir}/metrics_per_pred.csv"
YOLO_HOLDOUT="${dataset_pth}"
VIZ_DIR="${out_dir}/viz"
VIZ_OUT_DIR="${out_dir}/viz"
TOP_N=6
F1_BINS=0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0
RUN_CONCLUSIONS_AGENT=0
EOF
}

FAILED=()
for EXP in "${EXPERIMENTS[@]}"; do
  for TS in "${TEST_SETS[@]}"; do
    IFS="|" read -r KEY DATASET_NAME ATTRS_DIR <<< "${TS}"
    OUT_DIR="${RESULTS_DIR}/${EXP}/eval_int8/${KEY}"
    mkdir -p "${OUT_DIR}"
    write_env "${EXP}" "${KEY}" "${DATASET_NAME}" "${ATTRS_DIR}" "${OUT_DIR}" > "${OUT_DIR}/eval.env"

    echo "[$(date '+%F %T')] ${EXP} / ${KEY} -> ${OUT_DIR}"
    if "${SCRIPT_DIR}/run_eval_and_viz.sh" "${OUT_DIR}/eval.env" > "${OUT_DIR}/eval.log" 2>&1; then
      echo "    done"
    else
      echo "    FAILED (see ${OUT_DIR}/eval.log)"
      FAILED+=("${EXP}/${KEY}")
    fi
  done
done

echo
echo "Summary (selected conf/iou, pose P/R/F1, TP/FP/FN):"
python - "${RESULTS_DIR}" "${EXPERIMENTS[@]}" <<'EOF'
import json, os, sys
results_dir, exps = sys.argv[1], sys.argv[2:]
for exp in exps:
    for key in ("cam4revB_batch1_3", "testAccuracy", "H270_16_44_09", "PG2HardCases"):
        d = os.path.join(results_dir, exp, "eval_int8", key)
        try:
            b = json.load(open(os.path.join(d, "best_params.json")))
            g = json.load(open(os.path.join(d, "global_metrics.json")))
        except FileNotFoundError:
            print(f"{exp:85} {key:18} (no results)")
            continue
        print(f"{exp:85} {key:18} {b['conf_th']:.2f}/{b['iou_th']:.2f}  "
              f"P={b['selection/precision']:.4f} R={b['selection/recall']:.4f} F1={b['selection/f1']:.4f}  "
              f"TP/FP/FN={g['detection/tp']}/{g['detection/fp']}/{g['detection/fn']}")
EOF

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo
  echo "Failed runs: ${FAILED[*]}" >&2
  exit 1
fi
