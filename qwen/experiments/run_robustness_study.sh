#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
QWEN_DIR="$(cd "${ROOT_DIR}/.." && pwd)"
EVAL_DIR="${QWEN_DIR}/eval"
BLOCK_DIR="${QWEN_DIR}/block_trainer"

MODEL_PATH="${MODEL_PATH:-/home/liuzonghao/pretrained_models/Qwen2.5-VL-7B-Instruct}"
DATASET_FILE="${DATASET_FILE:-/home/liuzonghao/AASAE/VL-SAE/CC3M/merged_cc3m_train_short.json}"
IMAGE_FOLDER="${IMAGE_FOLDER:-/home/liuzonghao/AASAE/VL-SAE/CC3M/cc3m_jpg}"
TEST_JSON="${TEST_JSON:-/home/liuzonghao/AASAE/VL-SAE/CC3M/merged_cc3m_test_short.json}"
TARGET_LAYER="${TARGET_LAYER:-model.language_model.layers.20}"
CHUNK_SIZE="${CHUNK_SIZE:-200}"
SEED="${SEED:-42}"
ASYM_USE_VIEWS="${ASYM_USE_VIEWS:-1}"

EXP_ROOT="${EXP_ROOT:-${QWEN_DIR}/experiments/robustness_$(date +%Y%m%d_%H%M%S)}"
SAVE_DIR="${SAVE_DIR:-${QWEN_DIR}/block_trainer/checkpoints_sae}"
OUT_CSV="${OUT_CSV:-${EXP_ROOT}/robustness_metrics.csv}"
OUT_PLOTS="${OUT_PLOTS:-${EXP_ROOT}/plots}"

mkdir -p "$EXP_ROOT"

python "${EVAL_DIR}/robustness_eval.py" \
  --test-json "$TEST_JSON" \
  --save-dir "$SAVE_DIR" \
  --methods "sym,asym" \
  --models "SAE_V,SAE_D,VL_SAE" \
  --noise-stds "0,0.02,0.05,0.1" \
  --dropout-rates "0,0.2,0.5" \
  --budget-fracs "1.0,0.5,0.25" \
  --chunk-size 100 \
  --eval-batch-size 8 \
  --seed "$SEED" \
  --out-csv "$OUT_CSV" \
  --asym_use_views "$ASYM_USE_VIEWS"

python "${EVAL_DIR}/plot_robustness.py" \
  --csv "$OUT_CSV" \
  --out-dir "$OUT_PLOTS"

echo "[run_robustness_study] Done. Results in: $EXP_ROOT"
