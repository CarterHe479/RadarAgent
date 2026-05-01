#!/usr/bin/env bash
# E3 – Backbone comparison: Qwen3-1.7B (smaller same-family model).
#
# Shows how the system scales with model size.
# Run on 500 samples to keep cost manageable; same 500 used for all backbone runs.
#
# Usage:
#   ./E3_backbone_qwen3_1b.sh [max_samples]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"   # experiment_scripts/ lives inside RadarAgent/
EXPERIMENT_DIR="$SCRIPT_DIR"
VENV="$PROJECT_DIR/.venv"
RESULTS_DIR="$PROJECT_DIR/outputs/results/experiments"

if [ ! -f "$VENV/bin/python" ]; then
    echo "ERROR: venv not found at $VENV"; exit 1
fi

MAX_SAMPLES="${1:-500}"
OUT="$RESULTS_DIR/E3_backbone_qwen3_1b_n${MAX_SAMPLES}.json"
mkdir -p "$RESULTS_DIR"

cd "$PROJECT_DIR"
echo "=== E3: Backbone – Qwen3-1.7B ($MAX_SAMPLES samples) ==="
echo "Output: $OUT"
echo ""

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONPATH="$PROJECT_DIR" \
  "$VENV/bin/python" - << PYEOF
import sys
sys.path.insert(0, "$EXPERIMENT_DIR")   # for shared.eval_utils
sys.path.insert(0, "$PROJECT_DIR")      # for agent, tools, config
import os; os.chdir("$PROJECT_DIR")

from shared.eval_utils import (
    setup_logging, load_model_and_tokenizer, make_restricted_agent,
    filter_split_ids, run_evaluation_loop, compute_and_save, print_result,
)
setup_logging()

MODEL = "Qwen/Qwen3-1.7B"
MAX   = int("$MAX_SAMPLES")

model, tokenizer = load_model_and_tokenizer(MODEL)

# All tools – same as E1 but smaller model
ALL_TOOLS = [
    "load_radar_sequence", "extract_radar_features",
    "analyze_joint_motion", "get_motion_text",
    "search_motions", "compare_motions", "visualize_motion",
]
agent = make_restricted_agent(model, tokenizer, ALL_TOOLS)

ids = filter_split_ids("test", max_samples=MAX)
hyps, refs, samples = run_evaluation_loop(agent, ids, temperature=0.3)
result = compute_and_save("E3_backbone_qwen3_1b_n${MAX_SAMPLES}", hyps, refs, samples,
                          extra_meta={"model": MODEL, "tools": "all", "max_samples": MAX})
print_result(result)
PYEOF
