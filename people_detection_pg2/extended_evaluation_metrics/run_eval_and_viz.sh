#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_eval_and_viz.sh /path/to/.env

Runs sequentially:
  1) evaluation.py
  2) visualize_detections.py

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

export ENV_PATH="${ENV_FILE}"

python "${SCRIPT_DIR}/evaluation.py"
python "${SCRIPT_DIR}/visualize_detections.py"

