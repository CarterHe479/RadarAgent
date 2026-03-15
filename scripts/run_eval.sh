#!/usr/bin/env bash
# Evaluate RadarAgent on the test split.
# Usage: ./scripts/run_eval.sh [split] [max_samples]
#   split       – train | val | test  (default: test)
#   max_samples – integer limit       (default: all)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

SPLIT="${1:-test}"
MAX_SAMPLES="${2:-}"
OUTPUT="$PROJECT_DIR/outputs/results/eval_${SPLIT}.json"

cd "$PROJECT_DIR"

CMD="PYTHONPATH=$PROJECT_DIR python main.py --evaluate --split $SPLIT --output $OUTPUT"
if [ -n "$MAX_SAMPLES" ]; then
    CMD="$CMD --max-samples $MAX_SAMPLES"
fi

echo "Running: $CMD"
eval "$CMD"
