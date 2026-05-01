#!/usr/bin/env bash
# E1 – Main results: full test split, all tools, Qwen3-8B (zero-shot).
#
# This produces the headline number in Table 1.
# Expected runtime: ~40-80 GPU-hours on A100 for the full test split.
#
# Usage:
#   ./E1_main_results.sh [max_samples]
#   ./E1_main_results.sh          # full test split (~2584 valid samples)
#   ./E1_main_results.sh 500      # quick 500-sample run for sanity check
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"   # experiment_scripts/ lives inside RadarAgent/
VENV="$PROJECT_DIR/.venv"
RESULTS_DIR="$PROJECT_DIR/outputs/results/experiments"

if [ ! -f "$VENV/bin/python" ]; then
    echo "ERROR: venv not found at $VENV"
    echo "Run: python3.10 -m venv $VENV && $VENV/bin/pip install -r $PROJECT_DIR/requirements.txt"
    exit 1
fi

MAX_SAMPLES="${1:-}"
OUT="$RESULTS_DIR/E1_main_results.json"
mkdir -p "$RESULTS_DIR"

CMD="HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH=$PROJECT_DIR \
  $VENV/bin/python $PROJECT_DIR/main.py evaluate \
  --split test \
  --temperature 0.3 \
  --output $OUT"

if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
    OUT="$RESULTS_DIR/E1_main_results_n${MAX_SAMPLES}.json"
    CMD="${CMD%$RESULTS_DIR/E1_main_results.json*} --output $OUT"
fi

echo "=== E1: Main Results (Qwen3-8B, all tools, test split) ==="
echo "Output: $OUT"
echo ""
cd "$PROJECT_DIR"
eval "$CMD"
