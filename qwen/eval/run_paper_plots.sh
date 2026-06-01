#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_FILE="${RESULTS_FILE:-${ROOT_DIR}/evaluation_results_all.txt}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/plots_paper_new}"
METHODS="${METHODS:-}"
MODELS="${MODELS:-}"

mkdir -p "$OUT_DIR"

cmd=(python "${ROOT_DIR}/plot_eval_results_paper.py" --results-file "$RESULTS_FILE" --out-dir "$OUT_DIR")
if [[ -n "$METHODS" ]]; then
  cmd+=(--methods "$METHODS")
fi
if [[ -n "$MODELS" ]]; then
  cmd+=(--models "$MODELS")
fi

"${cmd[@]}"

echo "[run_paper_plots] Done. Plots saved to: $OUT_DIR"
