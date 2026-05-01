#!/usr/bin/env python3
"""
A2 – Tool ablation: load_radar_sequence only.

The LLM can call only the basic sequence loader (duration, bounds,
displacement). Tests whether coarse spatial/temporal metadata is enough
to produce reasonable captions.

Usage:
    PYTHONPATH=RadarAgent RadarAgent/.venv/bin/python experiment_scripts/A2_radar_load_only.py \
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

TOOLS = ["load_radar_sequence"]


def main() -> None:
    parser = argparse.ArgumentParser(description="A2: load_radar_sequence only")
    parser.add_argument("--max-samples", type=int, default=500)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    setup_logging()
    model, tokenizer = load_model_and_tokenizer()
    agent = make_restricted_agent(model, tokenizer, TOOLS)

    ids = filter_split_ids(args.split, max_samples=args.max_samples)
    hyps, refs, samples = run_evaluation_loop(agent, ids, temperature=0.3)

    result = compute_and_save(
        f"A2_radar_load_only_n{args.max_samples}",
        hyps, refs, samples,
        extra_meta={"tools": TOOLS},
    )
    print_result(result)


if __name__ == "__main__":
    main()
