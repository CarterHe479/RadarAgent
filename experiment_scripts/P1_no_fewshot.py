#!/usr/bin/env python3
"""
P1 – Prompt ablation: no few-shot examples.

Removes the two few-shot (user, assistant) turns from the prompt to test
whether in-context examples matter for terse-caption generation.

Usage:
    PYTHONPATH=RadarAgent RadarAgent/.venv/bin/python experiment_scripts/P1_no_fewshot.py \
        [--max-samples 200]
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent))
from shared.eval_utils import (
    setup_logging, load_model_and_tokenizer, make_restricted_agent,
    filter_split_ids, run_evaluation_loop, compute_and_save, print_result,
)

ALL_TOOLS = [
    "load_radar_sequence", "extract_radar_features",
    "analyze_joint_motion", "get_motion_text",
    "search_motions", "compare_motions", "visualize_motion",
]


class NoFewShotAgent:
    """Wraps RadarAgent and monkey-patches _build_initial_messages to skip few-shot."""

    def __init__(self, base_agent):
        self._agent = base_agent

    def _build_initial_messages(self, user_query: str) -> List[dict]:
        from agent.prompts import SYSTEM_PROMPT
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_query},
        ]

    def run(self, user_query: str, temperature=0.3, max_new_tokens=512) -> str:
        # Temporarily replace the method
        original = self._agent._build_initial_messages
        self._agent._build_initial_messages = self._build_initial_messages
        try:
            return self._agent.run(user_query, temperature=temperature, max_new_tokens=max_new_tokens)
        finally:
            self._agent._build_initial_messages = original


def main() -> None:
    parser = argparse.ArgumentParser(description="P1: no few-shot examples")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    setup_logging()
    model, tokenizer = load_model_and_tokenizer()
    base_agent = make_restricted_agent(model, tokenizer, ALL_TOOLS)
    agent = NoFewShotAgent(base_agent)

    ids = filter_split_ids(args.split, max_samples=args.max_samples)
    hyps, refs, samples = run_evaluation_loop(agent, ids, temperature=0.3)

    result = compute_and_save(
        f"P1_no_fewshot_n{args.max_samples}",
        hyps, refs, samples,
        extra_meta={"few_shot": False},
    )
    print_result(result)


if __name__ == "__main__":
    main()
