#!/usr/bin/env bash
set -e

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL_PATH="${MODEL_PATH:-/home/liuzonghao/pretrained_models/Qwen2.5-VL-7B-Instruct}"
DATASET_FILE="${DATASET_FILE:-/home/liuzonghao/AASAE/VL-SAE/CC3M/merged_cc3m_train.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/liuzonghao/AASAE/VL-SAE/CC3M/cc3m_jpg}"
TARGET_LAYER_NAME="${TARGET_LAYER_NAME:-model.language_model.layers.20}"

SAVE_DIR="${SAVE_DIR:-${ROOT_DIR}/checkpoints_sae}"
TRAIN_METHOD="${TRAIN_METHOD:-sym}"
SAE_TYPE="${SAE_TYPE:-VL_SAE}"
TOPK="${TOPK:-256}"

AUX_PROJ_PATH="${AUX_PROJ_PATH:-${SAVE_DIR}/shared_best_aux_proj_${TRAIN_METHOD}.pth}"
SAE_CHECKPOINT="${SAE_CHECKPOINT:-${SAVE_DIR}/${SAE_TYPE}_${TRAIN_METHOD}_new_best_sae.pth}"

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/checkpoints_sae/results/csv}"
OUTPUT_CSV="${OUTPUT_CSV:-${OUTPUT_DIR}/features_${TRAIN_METHOD}.csv}"

mkdir -p "$OUTPUT_DIR"

python "${ROOT_DIR}/feature_csv.py" \
  --model-path "$MODEL_PATH" \
  --dataset-file "$DATASET_FILE" \
  --image-folder "$IMAGE_FOLDER" \
  --target-layer-name "$TARGET_LAYER_NAME" \
  --sae-checkpoint "$SAE_CHECKPOINT" \
  --sae-type "$SAE_TYPE" \
  --train-method "$TRAIN_METHOD" \
  --topk "$TOPK" \
  --aux-proj-path "$AUX_PROJ_PATH" \
  --save-dir "$SAVE_DIR" \
  --output-csv "$OUTPUT_CSV"
