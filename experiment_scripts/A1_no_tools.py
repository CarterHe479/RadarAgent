#!/usr/bin/env python3
"""
A1 – Tool ablation: No tools (raw LLM baseline).

The LLM receives only the motion_id in the query and no tool schemas.
This establishes the lower bound: what can the model produce with zero
data access?

Usage:
    PYTHONPATH=RadarAgent RadarAgent/.venv/bin/python experiment_scripts/A1_no_tools.py \
        [--max-samples 500] [--split test]
"""

from __future__ import annotations
import argparse, sys
from pathlib import Path

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from shared.eval_utils import (
    setup_logging, load_model_and_tokenizer, make_restricted_agent,
    filter_split_ids, run_evaluation_loop, compute_and_save, print_result,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="A1: No-tools ablation")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    setup_logging()

    model, tokenizer = load_model_and_tokenizer()

    # Pass NO tools – the LLM must answer purely from its weights + motion_id
    agent = make_restricted_agent(model, tokenizer, allowed_tools=[])

    ids = filter_split_ids(args.split, max_samples=args.max_samples)
    hyps, refs, samples = run_evaluation_loop(agent, ids, temperature=0.3)

    result = compute_and_save(
        f"A1_no_tools_n{args.max_samples}",
        hyps, refs, samples,
        extra_meta={"tools": [], "description": "Raw LLM, zero data access"},
    )
    print_result(result)


if __name__ == "__main__":
    main()
