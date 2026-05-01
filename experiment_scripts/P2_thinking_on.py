#!/usr/bin/env python3
"""
P2 – Prompt ablation: Qwen 3 thinking mode ENABLED.

Re-enables <think>…</think> chain-of-thought to measure its effect on
metric scores.  The thinking tokens are still stripped before scoring,
but enabling thinking changes what the model attends to during generation.

Also measures latency impact of thinking vs non-thinking mode.

Usage:
    PYTHONPATH=RadarAgent RadarAgent/.venv/bin/python experiment_scripts/P2_thinking_on.py \
        [--max-samples 200]
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shared.eval_utils import (
    setup_logging, load_model_and_tokenizer, make_restricted_agent,
    filter_split_ids, run_evaluation_loop, compute_and_save, print_result,
)
import torch

ALL_TOOLS = [
    "load_radar_sequence", "extract_radar_features",
    "analyze_joint_motion", "get_motion_text",
    "search_motions", "compare_motions", "visualize_motion",
]


def patched_generate(original_generate):
    """Return a version of generate() with enable_thinking=True."""
    import functools

    @functools.wraps(original_generate)
    def wrapper(model, tokenizer, messages, tools, max_new_tokens=512, temperature=0.3):
        text = tokenizer.apply_chat_template(
            messages,
            tools=tools if tools else None,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,   # ← the only change vs the default
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(new_ids, skip_special_tokens=True)

    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser(description="P2: thinking mode ON")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    setup_logging()
    model, tokenizer = load_model_and_tokenizer()
    agent = make_restricted_agent(model, tokenizer, ALL_TOOLS)

    # Monkey-patch the generate function
    import agent.llm as llm_module
    original_generate = llm_module.generate
    llm_module.generate = patched_generate(original_generate)

    try:
        ids = filter_split_ids(args.split, max_samples=args.max_samples)
        hyps, refs, samples = run_evaluation_loop(agent, ids, temperature=0.3)
    finally:
        llm_module.generate = original_generate  # always restore

    # Compute average thinking length for analysis
    think_lengths = []
    import re
    THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    for s in samples:
        # Note: run_evaluation_loop strips thinking, so re-fetch from raw is unavailable here.
        # We log approximate token counts instead.
        think_lengths.append(s.get("n_tokens_approx", 0))

    result = compute_and_save(
        f"P2_thinking_on_n{args.max_samples}",
        hyps, refs, samples,
        extra_meta={
            "thinking": True,
            "avg_output_tokens": round(sum(think_lengths) / max(len(think_lengths), 1), 1),
        },
    )
    print_result(result)


if __name__ == "__main__":
    main()
