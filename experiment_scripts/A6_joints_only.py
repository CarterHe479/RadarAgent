#!/usr/bin/env python3
"""
A6 – Tool ablation: analyze_joint_motion only (no radar tools).

Isolates the radar signal contribution: if the agent can only use the
SMPL skeleton tool (which uses joint positions, not radar), how well
does it do compared to the full system?

This directly quantifies "how much does the radar data actually add?"

Usage:
    PYTHONPATH=RadarAgent RadarAgent/.venv/bin/python experiment_scripts/A6_joints_only.py \
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

TOOLS = ["analyze_joint_motion"]


def main() -> None:
    parser = argparse.ArgumentParser(description="A6: joints only (no radar)")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    setup_logging()
    model, tokenizer = load_model_and_tokenizer()
    agent = make_restricted_agent(model, tokenizer, TOOLS)

    ids = filter_split_ids(args.split, max_samples=args.max_samples)
    hyps, refs, samples = run_evaluation_loop(agent, ids, temperature=0.3)

    result = compute_and_save(
        f"A6_joints_only_n{args.max_samples}",
        hyps, refs, samples,
        extra_meta={"tools": TOOLS, "description": "Skeleton only, no radar data"},
    )
    print_result(result)


if __name__ == "__main__":
    main()
