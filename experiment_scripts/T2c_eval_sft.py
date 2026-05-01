#!/usr/bin/env python3
"""
T2c – Evaluate the LoRA SFT checkpoint on the test split.

Loads the merged (or PEFT-adapter) checkpoint from T2b and runs the full
evaluation on the test split using the same feature-context prompt as training.
Results are saved as a JSON for direct comparison with E1 (zero-shot).

Usage:
    PYTHONPATH=RadarAgent RadarAgent/.venv/bin/python experiment_scripts/T2c_eval_sft.py \
        --checkpoint outputs/sft/lora/final \
        [--max-samples 500] [--split test]
"""

from __future__ import annotations
import argparse, json, logging, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shared.eval_utils import (
    setup_logging, filter_split_ids, compute_and_save, print_result,
    PROJECT_DIR, EXPERIMENT_DIR,
)
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

from tqdm import tqdm
from tools.radar_processing import load_radar_sequence, extract_radar_features
from tools.joint_analysis import analyze_joint_motion
from tools.data_retrieval import _read_descriptions
from agent.llm import strip_thinking
from agent.prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

INSTRUCTION_TEMPLATE = (
    "Analyse motion {motion_id}.\n\n"
    "[Feature context]\n{feature_context}\n\n"
    "Describe this motion in one sentence."
)


def format_feature_context(seq, feats, joints) -> str:
    p = feats["periodicity"]
    period_str = (
        f"periodic ({p['estimated_period_sec']}s cycle)"
        if p["is_periodic"] else "aperiodic"
    )
    return (
        f"Duration: {seq['duration_sec']}s. "
        f"Net displacement: {seq['overall_displacement']:.2f}m. "
        f"Mean velocity: {feats['velocity']['mean_m_per_s']} m/s, "
        f"peak: {feats['velocity']['max_m_per_s']} m/s. "
        f"Motion: {period_str}, {feats['trajectory_shape']} trajectory. "
        f"Vertical range: {feats['vertical_dynamics']['vertical_range_m']:.2f}m. "
        f"Detected actions: {', '.join(joints['detected_actions'])}. "
        f"Most active parts: {', '.join(joints['most_active_parts'])}."
    )


def describe_with_sft(mid: str, model, tokenizer) -> str:
    import torch
    try:
        seq    = load_radar_sequence(mid)
        feats  = extract_radar_features(mid)
        joints = analyze_joint_motion(mid)
    except Exception:
        return ""

    ctx = format_feature_context(seq, feats, joints)
    messages = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": INSTRUCTION_TEMPLATE.format(
            motion_id=mid, feature_context=ctx
        )},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=64, temperature=0.0,
            do_sample=False, pad_token_id=tokenizer.eos_token_id,
        )
    new = out[0][inputs["input_ids"].shape[1]:]
    return strip_thinking(tokenizer.decode(new, skip_special_tokens=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="T2c: evaluate SFT checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to saved LoRA/merged model")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    setup_logging()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    ckpt = Path(args.checkpoint)
    logger.info("Loading checkpoint from %s", ckpt)

    # Try loading as a PEFT adapter first; fall back to full model
    base_name = "Qwen/Qwen3-8B"
    tokenizer  = AutoTokenizer.from_pretrained(str(ckpt), trust_remote_code=True)
    try:
        base  = AutoModelForCausalLM.from_pretrained(
            base_name, dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base, str(ckpt))
        logger.info("Loaded as PEFT adapter over %s", base_name)
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt), dtype=torch.bfloat16, device_map="auto", trust_remote_code=True,
        )
        logger.info("Loaded as full model")
    model.eval()

    ids = filter_split_ids(args.split, max_samples=args.max_samples)
    hypotheses, refs_list, samples = [], [], []

    for mid in tqdm(ids, desc="Eval [SFT]"):
        refs = _read_descriptions(mid)
        if not refs:
            continue
        t0 = time.perf_counter()
        gen = describe_with_sft(mid, model, tokenizer)
        elapsed = time.perf_counter() - t0
        hypotheses.append(gen)
        refs_list.append(refs)
        samples.append({"motion_id": mid, "generated": gen, "references": refs,
                         "elapsed_sec": round(elapsed, 3)})

    result = compute_and_save(
        f"T2_lora_sft_n{args.max_samples}",
        hypotheses, refs_list, samples,
        extra_meta={"checkpoint": str(ckpt), "approach": "LoRA_SFT"},
    )
    print_result(result)


if __name__ == "__main__":
    main()
