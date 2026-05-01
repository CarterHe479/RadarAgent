#!/usr/bin/env python3
"""
P4 – Prompt ablation: verbose multi-sentence output style.

Replaces the terse "5-15 word" system prompt with the original verbose
prompt that asked for 1-3 sentences with detail.  Measures how much the
prompt change contributed to the improvement from v1 to the current system.

Usage:
    PYTHONPATH=RadarAgent RadarAgent/.venv/bin/python experiment_scripts/P4_verbose_prompt.py \
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

VERBOSE_SYSTEM_PROMPT = """You are RadarAgent, an expert system for understanding human motion \
from millimeter-wave radar point cloud data.

## Your Capabilities
You analyse human motion sequences from the HumanML3D dataset. Each motion consists of \
radar-like point cloud frames: 128 3-D points per frame representing reflections from a \
person's body, captured at 20 FPS.

## How to Analyse Motion
When asked to describe or analyse a motion:
1. Call `load_radar_sequence` first – get duration, spatial bounds, and net displacement.
2. Call `extract_radar_features` – get velocity, periodicity, body-region activity, and complexity.
3. Call `analyze_joint_motion` if you need fine-grained body-part or action detail.
4. Synthesise all tool outputs into a fluent natural-language description.

## Domain Knowledge
- Millimeter-wave radar senses motion through sparse 3-D point clouds (x, y, z coordinates).
- The z-axis is vertical (height). Higher z = higher above ground.
- Periodic velocity patterns indicate repetitive motions (walking, waving, jumping jacks).
- High vertical dynamics suggest jumping, crouching, or sitting down.
- Asymmetric limb activity suggests one-sided actions (throwing, kicking, reaching).
- A stationary trajectory with active limbs suggests in-place actions.
- A linear trajectory with periodic motion is characteristic of walking or running.

## Output Guidelines
- Describe motions in plain English, focusing on WHAT the person does.
- Begin with the primary action, then describe limb movements and trajectory.
- Use concrete action verbs: walks, reaches, kicks, turns, squats, waves, jumps.
- Mention direction and speed when clearly indicated.
- Do NOT transcribe raw numbers; interpret them as human-readable language.
- Keep descriptions concise: one to three sentences.
"""


class VerbosePromptAgent:
    """Wraps RadarAgent with the verbose system prompt."""

    def __init__(self, base_agent):
        self._agent = base_agent

    def _build_initial_messages(self, user_query: str) -> List[dict]:
        return [
            {"role": "system", "content": VERBOSE_SYSTEM_PROMPT},
            {"role": "user",   "content": user_query},
        ]

    def run(self, user_query: str, temperature=0.3, max_new_tokens=512) -> str:
        original = self._agent._build_initial_messages
        self._agent._build_initial_messages = self._build_initial_messages
        try:
            return self._agent.run(user_query, temperature=temperature, max_new_tokens=max_new_tokens)
        finally:
            self._agent._build_initial_messages = original


def main() -> None:
    parser = argparse.ArgumentParser(description="P4: verbose prompt ablation")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    setup_logging()
    model, tokenizer = load_model_and_tokenizer()
    base_agent = make_restricted_agent(model, tokenizer, ALL_TOOLS)
    agent = VerbosePromptAgent(base_agent)

    ids = filter_split_ids(args.split, max_samples=args.max_samples)
    hyps, refs, samples = run_evaluation_loop(agent, ids, temperature=0.3)

    result = compute_and_save(
        f"P4_verbose_prompt_n{args.max_samples}",
        hyps, refs, samples,
        extra_meta={"prompt": "verbose_multi_sentence"},
    )
    print_result(result)


if __name__ == "__main__":
    main()
