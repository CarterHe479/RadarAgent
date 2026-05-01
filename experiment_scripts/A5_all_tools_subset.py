#!/usr/bin/env python3
"""
A5 – Tool ablation: all tools on the same N-sample subset used for A1-A4.

This is the full system evaluated on the ablation subset, so results are
directly comparable to A1-A4 on equal footing.
(E1 runs all tools on the full test split; A5 is the ablation-scale version.)

Usage:
    PYTHONPATH=RadarAgent RadarAgent/.venv/bin/python experiment_scripts/A5_all_tools_subset.py \
        [--max-samples 500]
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shared.eval_utils import (
    setup_logging, load_model_and_tokenizer, make_restricted_agent,
    filter_split_ids, run_evaluation_loop, compute_and_save, print_result,
)

ALL_TOOLS = [
    "load_radar_sequence",
    "extract_radar_features",
    "analyze_joint_motion",
    "get_motion_text",
    "search_motions",
    "compare_motions",
    "visualize_motion",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="A5: all tools (ablation subset)")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    setup_logging()
    model, tokenizer = load_model_and_tokenizer()
    agent = make_restricted_agent(model, tokenizer, ALL_TOOLS)

    ids = filter_split_ids(args.split, max_samples=args.max_samples)
    hyps, refs, samples = run_evaluation_loop(agent, ids, temperature=0.3)

    result = compute_and_save(
        f"A5_all_tools_n{args.max_samples}",
        hyps, refs, samples,
        extra_meta={"tools": ALL_TOOLS, "description": "Full system on ablation subset"},
    )
    print_result(result)


if __name__ == "__main__":
    main()
