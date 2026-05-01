#!/usr/bin/env python3
"""
P3 – Prompt ablation: temperature sweep over {0.0, 0.3, 0.7}.

Evaluates the same 200-sample subset at three temperatures.
temperature=0.0 uses greedy decoding (most deterministic).
temperature=0.7 is the interactive default.

Results are saved as three separate JSON files for easy comparison.

Usage:
    PYTHONPATH=RadarAgent RadarAgent/.venv/bin/python experiment_scripts/P3_temperature_sweep.py \
        [--max-samples 200] [--temps 0.0 0.3 0.7]
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
    "load_radar_sequence", "extract_radar_features",
    "analyze_joint_motion", "get_motion_text",
    "search_motions", "compare_motions", "visualize_motion",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="P3: temperature sweep")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--split", default="test")
    parser.add_argument("--temps", type=float, nargs="+", default=[0.0, 0.3, 0.7])
    args = parser.parse_args()

    setup_logging()
    model, tokenizer = load_model_and_tokenizer()

    # Load IDs once; evaluate at each temperature using the same samples
    ids = filter_split_ids(args.split, max_samples=args.max_samples)

    all_results = {}
    for temp in args.temps:
        agent = make_restricted_agent(model, tokenizer, ALL_TOOLS)
        print(f"\n=== Temperature = {temp} ===")
        hyps, refs, samples = run_evaluation_loop(agent, ids, temperature=temp)
        name = f"P3_temp{str(temp).replace('.', 'p')}_n{args.max_samples}"
        result = compute_and_save(
            name, hyps, refs, samples,
            extra_meta={"temperature": temp},
        )
        print_result(result)
        all_results[temp] = result["metrics"]

    # Summary table
    print("\n=== TEMPERATURE SWEEP SUMMARY ===")
    header = f"{'Temp':<8}" + "  " + "  ".join(f"{'ROUGE-L':<10}{'BLEU-1':<10}{'BERTScore':<12}")
    print(header)
    for temp, m in sorted(all_results.items()):
        print(
            f"{temp:<8.1f}  "
            f"{m.get('ROUGE-L', 0):<10.2f}"
            f"{m.get('BLEU-1', 0):<10.2f}"
            f"{m.get('BERTScore', 0):<12.2f}"
        )


if __name__ == "__main__":
    main()
