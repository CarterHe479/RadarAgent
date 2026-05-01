#!/usr/bin/env python3
"""
Q5 – Analysis: efficiency metrics.

Reads one or more eval JSON files and reports per-sample timing and
token statistics.  Useful for the efficiency table in the paper:
  - Mean / median latency per sample
  - Throughput (samples per hour)
  - Approximate output token length

Usage:
    python experiment_scripts/Q5_efficiency.py \
        --results results1.json results2.json ... \
        [--labels "Zero-shot" "SFT"]
"""

from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import statistics

sys.path.insert(0, str(Path(__file__).parent))
from shared.eval_utils import RESULTS_DIR


def summarize(samples: list[dict]) -> dict:
    latencies   = [s.get("elapsed_sec", 0)       for s in samples]
    token_lens  = [s.get("n_tokens_approx", len(s["generated"].split()))
                   for s in samples]
    gen_lengths = [len(s["generated"].split()) for s in samples]

    def safe_mean(lst): return statistics.mean(lst) if lst else 0.0
    def safe_median(lst): return statistics.median(lst) if lst else 0.0
    def safe_stdev(lst): return statistics.stdev(lst) if len(lst) > 1 else 0.0

    mean_lat  = safe_mean(latencies)
    throughput = 3600 / mean_lat if mean_lat > 0 else float("inf")
    return {
        "n_samples":           len(samples),
        "latency_mean_sec":    round(safe_mean(latencies), 2),
        "latency_median_sec":  round(safe_median(latencies), 2),
        "latency_std_sec":     round(safe_stdev(latencies), 2),
        "latency_min_sec":     round(min(latencies, default=0), 2),
        "latency_max_sec":     round(max(latencies, default=0), 2),
        "throughput_per_hour": round(throughput, 1),
        "output_words_mean":   round(safe_mean(gen_lengths), 1),
        "output_words_median": round(safe_median(gen_lengths), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Q5: efficiency analysis")
    parser.add_argument("--results", nargs="+", required=True)
    parser.add_argument("--labels",  nargs="+", default=None)
    parser.add_argument("--output",  default=None)
    args = parser.parse_args()

    labels = args.labels or [Path(r).stem for r in args.results]
    if len(labels) < len(args.results):
        labels += [Path(r).stem for r in args.results[len(labels):]]

    all_summaries = {}
    for path, label in zip(args.results, labels):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        smry = summarize(data.get("samples", []))
        all_summaries[label] = smry

    # Print table
    cols = ["latency_mean_sec", "latency_median_sec", "throughput_per_hour",
            "output_words_mean"]
    col_heads = ["Mean latency(s)", "Median lat(s)", "Samples/hr", "Output words"]

    print(f"\n=== Efficiency Comparison ===\n")
    header = f"{'System':<28}" + "".join(f"  {h:>16}" for h in col_heads)
    print(header)
    print("─" * len(header))
    for label, smry in all_summaries.items():
        row = f"{label:<28}" + "".join(f"  {smry.get(c, 0):>16.2f}" for c in cols)
        print(row)

    out_path = Path(args.output) if args.output else RESULTS_DIR / "Q5_efficiency.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
