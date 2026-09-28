#!/usr/bin/env bash
# Merge yolo_dataset_v2/background_v2 (built by make_background_v2_dataset.sh) into
# batch1_12_minibatch2_bg_margin_10_wb_alif_clahe. The base dataset is copied, never modified.
#   train/val            : base + background_v2
#   holdout / holdout_v2 : unchanged (background_v2 has none)
#
# Usage: bash merge_background_v2_into_batch1_12.sh [--overwrite]
set -euo pipefail

YOLO=/home/klymy/data/cam4_revB/yolo_dataset_v2
BASE=batch1_12_minibatch2_bg_margin_10_wb_alif_clahe
UPDATE=background_v2
RESULT=batch1_12_minibatch2_bg_v2_margin_10_wb_alif_clahe

OVERWRITE=""
for arg in "$@"; do
    case $arg in
        --overwrite) OVERWRITE="--overwrite" ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")"
export PYENV_VERSION=yolo-export

for d in "$BASE" "$UPDATE"; do
    [[ -d $YOLO/$d ]] || { echo "Missing dataset: $YOLO/$d" >&2; exit 1; }
done
if [[ -e $YOLO/$RESULT && -z $OVERWRITE ]]; then
    echo "$YOLO/$RESULT already exists; pass --overwrite to rebuild" >&2
    exit 1
fi

count() { if [[ -d $1 ]]; then find "$1" -type f | wc -l; else echo 0; fi; }

declare -A expected
for split in train val holdout holdout_v2; do
    expected[$split]=$(( $(count "$YOLO/$BASE/images/$split") + $(count "$YOLO/$UPDATE/images/$split") ))
done

python merge_yolo_datasets.py --base-pth "$YOLO" --base-dataset "$BASE" \
    --update-dataset "$UPDATE" --result-dataset "$RESULT" $OVERWRITE

echo "== check"
D=$YOLO/$RESULT
ok=1
for split in train val holdout holdout_v2; do
    n_img=$(count "$D/images/$split")
    n_lbl=$(count "$D/labels/$split")
    echo "$split: images $n_img, labels $n_lbl (expected ${expected[$split]})"
    [[ $n_img -eq ${expected[$split]} && $n_lbl -eq $n_img ]] || ok=0
done
n_bg=$(find "$D/images" -name '4.2_background_v2*' | wc -l)
echo "background_v2 images in result: $n_bg (expected $(count "$YOLO/$UPDATE/images"))"
[[ $n_bg -eq $(count "$YOLO/$UPDATE/images") ]] || ok=0
[[ $ok -eq 1 ]] || { echo "CHECK FAILED" >&2; exit 1; }
ls "$D"/*.yaml
echo "Done: $D"
