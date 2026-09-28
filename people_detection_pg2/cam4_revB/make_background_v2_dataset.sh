#!/usr/bin/env bash
# Build yolo_dataset_v2/background_v2: one background-only YOLO dataset (empty labels, train only)
#   - 4.2_background_v2        -> jpg + ALIF WB + CLAHE (no sharpness aug); images already in
#                                 4.2_background are skipped (they are in the *_bg datasets already);
#                                 VAL_COUNT of them drawn at random (SEED) into val
#   - 4.2_background_v2_larry  -> jpg, resized to 640x480, NO image processing
#
# Usage: bash make_background_v2_dataset.sh [--overwrite] [--keep-intermediate]
set -euo pipefail

ROOT=/home/klymy/data/cam4_revB
YOLO_DIRNAME=yolo_dataset_v2
YOLO=$ROOT/$YOLO_DIRNAME
FINAL=background_v2
LARRY_SIZE=640x480
VAL_COUNT=10   # background_v2 images moved to val (larry stays all train)
SEED=42

# intermediates (deleted at the end unless --keep-intermediate)
V2_RAW=_background_v2_raw
V2_PROC=${V2_RAW}_wb_alif_clahe
LARRY=_background_v2_larry

OVERWRITE=""
KEEP=0
for arg in "$@"; do
    case $arg in
        --overwrite) OVERWRITE="--overwrite" ;;
        --keep-intermediate) KEEP=1 ;;
        *) echo "Unknown argument: $arg" >&2; exit 2 ;;
    esac
done

cd "$(dirname "$0")"
export PYENV_VERSION=yolo-export
python -c "import cv2, PIL" || { echo "yolo-export env missing cv2/PIL" >&2; exit 1; }
echo "python: $(command -v python) ($(python --version 2>&1))"

if [[ -e $YOLO/$FINAL && -z $OVERWRITE ]]; then
    echo "$YOLO/$FINAL already exists; pass --overwrite to rebuild" >&2
    exit 1
fi

echo "== 1/4 4.2_background_v2 -> YOLO (jpg + empty labels)"
python background2yolo_convert.py --root-pth "$ROOT" --yolo-output-dirname "$YOLO_DIRNAME" \
    --image-root "$ROOT/4.2_background_v2" --exclude-root "$ROOT/4.2_background" \
    --val-count "$VAL_COUNT" --seed "$SEED" \
    --dataset-name "$V2_RAW" --overwrite

echo "== 2/4 WB + CLAHE on 4.2_background_v2"
python preprocess_dataset.py --base-pth "$YOLO" --base-dataset "$V2_RAW" \
    --output-dataset "$V2_PROC" --mode wb_alif_clahe --no-sharpness-augmentation \
    --fail-on-issues --overwrite

echo "== 3/4 4.2_background_v2_larry -> YOLO (jpg, resize $LARRY_SIZE, no processing)"
python background2yolo_convert.py --root-pth "$ROOT" --yolo-output-dirname "$YOLO_DIRNAME" \
    --image-root "$ROOT/4.2_background_v2_larry" --resize "$LARRY_SIZE" \
    --dataset-name "$LARRY" --overwrite

echo "== 4/4 merge -> $FINAL"
python merge_yolo_datasets.py --base-pth "$YOLO" --base-dataset "$V2_PROC" \
    --update-dataset "$LARRY" --result-dataset "$FINAL" $OVERWRITE

# merge_yolo_datasets always creates holdout_v2 folders; backgrounds have no holdout, so drop them
rm -rf "$YOLO/$FINAL/images/holdout_v2" "$YOLO/$FINAL/labels/holdout_v2" "$YOLO/$FINAL/project_holdout_v2.yaml"
mv "$YOLO/$FINAL/background_conversion_summary.json" "$YOLO/$FINAL/background_conversion_summary_v2.json"
cp "$YOLO/$LARRY/background_conversion_summary.json" "$YOLO/$FINAL/background_conversion_summary_larry.json"

if [[ $KEEP -eq 0 ]]; then
    rm -rf "$YOLO/$V2_RAW" "$YOLO/$V2_PROC" "$YOLO/$LARRY"
fi

echo "== check"
D=$YOLO/$FINAL
n_img=$(find "$D/images" -type f | wc -l)
n_lbl=$(find "$D/labels" -type f -name '*.txt' | wc -l)
n_empty=$(find "$D/labels" -type f -name '*.txt' -empty | wc -l)
n_train=$(find "$D/images/train" -type f | wc -l)
n_val=$(find "$D/images/val" -type f | wc -l)
n_val_lbl=$(find "$D/labels/val" -type f -name '*.txt' | wc -l)
n_v2=$(find "$D/images" -name '4.2_background_v2____*' | wc -l)
n_larry=$(find "$D/images" -name '4.2_background_v2_larry____*' | wc -l)
echo "images: $n_img (train: $n_train, val: $n_val | v2: $n_v2, larry: $n_larry)  labels: $n_lbl (empty: $n_empty)"
[[ $n_img -eq $n_lbl && $n_lbl -eq $n_empty && $((n_v2 + n_larry)) -eq $n_img \
   && $n_val -eq $VAL_COUNT && $n_val_lbl -eq $VAL_COUNT ]] \
    || { echo "CHECK FAILED" >&2; exit 1; }
ls "$D"
echo "Done: $D"
