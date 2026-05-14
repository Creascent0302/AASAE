#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/home/liuzonghao/pretrained_models/Qwen2.5-VL-7B-Instruct}"
TARGET_LAYER_NAME="${TARGET_LAYER_NAME:-model.language_model.layers.20}"

SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints_sae}"
TRAIN_METHOD="${TRAIN_METHOD:-sym}"
SAE_TYPE="${SAE_TYPE:-VL_SAE}"
TOPK="${TOPK:-256}"
METHODS="${METHODS:-act}"
TEXT_TOPK="${TEXT_TOPK:-10}"

OVERLAY_ALPHA="${OVERLAY_ALPHA:-0.55}"
OVERLAY_GAMMA="${OVERLAY_GAMMA:-0.6}"
OVERLAY_CMAP="${OVERLAY_CMAP:-jet}"
OVERLAY_CLIP_LOW="${OVERLAY_CLIP_LOW:-0.05}"
OVERLAY_CLIP_HIGH="${OVERLAY_CLIP_HIGH:-0.95}"

AUX_PROJ_PATH="${AUX_PROJ_PATH:-${SAVE_DIR}/shared_best_aux_proj_${TRAIN_METHOD}.pth}"
SAE_CHECKPOINT="${SAE_CHECKPOINT:-${SAVE_DIR}/${SAE_TYPE}_${TRAIN_METHOD}_new_best_sae.pth}"

CSV_PATH="${CSV_PATH:-${ROOT_DIR}/checkpoints_sae/results/csv/features_${TRAIN_METHOD}.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/checkpoints_sae/results/text}"

mkdir -p "$OUTPUT_DIR"

python "${ROOT_DIR}/visualize.py" \
  --csv-path "$CSV_PATH" \
  --model-path "$MODEL_PATH" \
  --target-layer-name "$TARGET_LAYER_NAME" \
  --sae-checkpoint "$SAE_CHECKPOINT" \
  --sae-type "$SAE_TYPE" \
  --train-method "$TRAIN_METHOD" \
  --topk "$TOPK" \
  --aux-proj-path "$AUX_PROJ_PATH" \
  --save-dir "$SAVE_DIR" \
  --modality text \
  --method "$METHODS" \
  --text-topk "$TEXT_TOPK" \
  --output-dir "$OUTPUT_DIR" \
  --overlay-alpha "$OVERLAY_ALPHA" \
  --overlay-gamma "$OVERLAY_GAMMA" \
  --overlay-cmap "$OVERLAY_CMAP" \
  --overlay-clip-low "$OVERLAY_CLIP_LOW" \
  --overlay-clip-high "$OVERLAY_CLIP_HIGH"
