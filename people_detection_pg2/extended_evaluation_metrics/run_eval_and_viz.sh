#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_eval_and_viz.sh /path/to/.env

Runs sequentially:
  1) evaluation.py
  2) visualize_detections.py
  3) copy extended_report.ipynb into ${PROJECT_PTH}
  4) execute notebook + export HTML (nbconvert)

The provided env file path is passed via ENV_PATH, and both Python scripts load
their parameters from that file.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

ENV_FILE="${1:-}"
if [[ -z "${ENV_FILE}" ]]; then
  echo "Error: missing env file path." >&2
  usage >&2
  exit 2
fi
if [[ ! -f "${ENV_FILE}" ]]; then
  echo "Error: env file not found: ${ENV_FILE}" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$(cd -- "$(dirname -- "${ENV_FILE}")" && pwd)"

export ENV_PATH="${ENV_FILE}"

python "${SCRIPT_DIR}/evaluation.py"
python "${SCRIPT_DIR}/visualize_detections.py"

# Load env in bash (to read PROJECT_PTH / PROJECT / PROJECT_NAME).
# NOTE: this requires bash-compatible KEY=VALUE lines (no spaces around '=').
set -a
# shellcheck disable=SC1090
source "${ENV_FILE}"
set +a

OUT_DIR="${PROJECT_PTH-}"
if [[ -z "${OUT_DIR}" ]]; then
  if [[ -n "${PROJECT-}" && -n "${PROJECT_NAME-}" ]]; then
    OUT_DIR="${PROJECT}/${PROJECT_NAME}"
  else
    echo "Error: PROJECT_PTH not set in ${ENV_FILE} (and PROJECT/PROJECT_NAME fallback is missing)." >&2
    exit 2
  fi
fi

OUT_DIR="${OUT_DIR#./}"
if [[ "${OUT_DIR}" != /* ]]; then
  OUT_DIR="${ENV_DIR}/${OUT_DIR}"
fi

mkdir -p "${OUT_DIR}"

NOTEBOOK_SRC="${SCRIPT_DIR}/extended_report.ipynb"
NOTEBOOK_DST="${OUT_DIR}/extended_report.ipynb"
cp -f "${NOTEBOOK_SRC}" "${NOTEBOOK_DST}"
echo "Copied notebook to: ${NOTEBOOK_DST}"

execute_and_export_notebook() {
  if ! command -v jupyter >/dev/null 2>&1; then
    echo "Warning: 'jupyter' not found; skipping notebook execution/export." >&2
    return 1
  fi

  jupyter nbconvert --execute --to notebook --inplace "${NOTEBOOK_DST}" \
    --ExecutePreprocessor.timeout=600

  jupyter nbconvert --to html "${NOTEBOOK_DST}" \
    --output "extended_report.html" \
    --output-dir "${OUT_DIR}"
  echo "Exported HTML to: ${OUT_DIR}/extended_report.html"
}

# 1st pass: execute notebook and export HTML for Claude to analyze
execute_and_export_notebook

# Generate conclusions and inject them into the .ipynb
python "${SCRIPT_DIR}/generate_conclusions_claude.py"

# 2nd pass: re-execute notebook (now with conclusions) and export final HTML
execute_and_export_notebook