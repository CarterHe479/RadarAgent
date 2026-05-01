#!/usr/bin/env bash
# E5 – Backbone comparison: Llama-3.1-8B-Instruct (cross-family generalization).
#
# Shows whether the tool-calling approach works beyond the Qwen family.
# Run on 500 samples (same IDs as E1/E3 for a fair comparison).
#
# Requirements:
#   - Download Llama-3.1-8B-Instruct first (requires HF token + Meta license):
#     huggingface-cli download meta-llama/Meta-Llama-3.1-8B-Instruct
#   - OR set LLAMA_MODEL_PATH to a local directory.
#
# Usage:
#   ./E5_backbone_llama.sh [max_samples]
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
LLAMA_MODEL="${LLAMA_MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
OUT="$RESULTS_DIR/E5_backbone_llama_n${MAX_SAMPLES}.json"
mkdir -p "$RESULTS_DIR"

cd "$PROJECT_DIR"
echo "=== E5: Backbone – Llama-3.1-8B-Instruct ($MAX_SAMPLES samples) ==="
echo "Model: $LLAMA_MODEL"
echo "Output: $OUT"
echo ""

# Note: Llama uses a different tool-call format than Qwen.
# We override generate() to use a plain-text prompt that includes tool results
# directly, since Llama's function-calling schema differs from Qwen's.
PYTHONPATH="$PROJECT_DIR" \
  "$VENV/bin/python" - << PYEOF
import sys
sys.path.insert(0, "$EXPERIMENT_DIR")   # for shared.eval_utils
sys.path.insert(0, "$PROJECT_DIR")      # for agent, tools, config
import os; os.chdir("$PROJECT_DIR")
import json, logging

from shared.eval_utils import setup_logging, filter_split_ids, compute_and_save, print_result
from tqdm import tqdm
from tools.data_retrieval import _read_descriptions
from tools.radar_processing import load_radar_sequence, extract_radar_features
from tools.joint_analysis import analyze_joint_motion
from agent.llm import strip_thinking, load_model
from transformers import pipeline
import torch, time

setup_logging()
logger = logging.getLogger(__name__)

MODEL = "$LLAMA_MODEL"
MAX   = int("$MAX_SAMPLES")

logger.info("Loading %s …", MODEL)
model, tokenizer = load_model(MODEL)

SYSTEM = (
    "You are an expert at understanding human motion from sensor data. "
    "You will be given structured feature data about a motion sequence. "
    "Respond with a single sentence (5-15 words) describing the action, "
    "e.g. 'a person walks forward' or 'the man waves his right hand'."
)

def describe_motion(mid: str) -> str:
    # Collect tool outputs as plain text (Llama doesn't support native tool calls here)
    try:
        seq = load_radar_sequence(mid)
        feats = extract_radar_features(mid)
        joints = analyze_joint_motion(mid)
    except Exception as e:
        return ""

    context = (
        f"Duration: {seq['duration_sec']}s, displacement: {seq['overall_displacement']:.2f}m.\n"
        f"Velocity: mean={feats['velocity']['mean_m_per_s']} m/s, max={feats['velocity']['max_m_per_s']} m/s.\n"
        f"Trajectory: {feats['trajectory_shape']}, dominant axis: {feats['dominant_motion_axis']}.\n"
        f"Periodicity: {feats['periodicity']['is_periodic']} "
        f"(period={feats['periodicity'].get('estimated_period_sec', 'N/A')}s).\n"
        f"Detected actions: {', '.join(joints['detected_actions'])}.\n"
        f"Most active body parts: {', '.join(joints['most_active_parts'])}.\n"
        f"Root trajectory: {joints['root_trajectory']}."
    )

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"Motion features:\n{context}\n\nDescribe this motion in one sentence."},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, temperature=0.3,
                             do_sample=True, pad_token_id=tokenizer.eos_token_id)
    new = out[0][inputs["input_ids"].shape[1]:]
    return strip_thinking(tokenizer.decode(new, skip_special_tokens=True))

ids = filter_split_ids("test", max_samples=MAX)
hypotheses, refs_list, samples = [], [], []

for mid in tqdm(ids, desc="Eval [E5]"):
    refs = _read_descriptions(mid)
    if not refs:
        continue
    t0 = time.perf_counter()
    gen = describe_motion(mid)
    elapsed = time.perf_counter() - t0
    hypotheses.append(gen)
    refs_list.append(refs)
    samples.append({"motion_id": mid, "generated": gen, "references": refs,
                    "elapsed_sec": round(elapsed, 3)})

result = compute_and_save("E5_backbone_llama_n${MAX_SAMPLES}", hypotheses, refs_list, samples,
                          extra_meta={"model": MODEL, "approach": "plain_text_features"})
print_result(result)
PYEOF
