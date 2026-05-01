#!/usr/bin/env python3
"""
R3 – Robustness: sparse radar points (64 instead of 128 per frame).

Simulates a sparser sensor configuration by randomly subsampling each
frame of the (T, 128, 4) point cloud down to N_pts points.

Evaluated at N_pts = {128, 64, 32, 16} on the same 200-sample subset.

Usage:
    PYTHONPATH=RadarAgent RadarAgent/.venv/bin/python experiment_scripts/R3_sparse_points.py \
        [--max-samples 200] [--n-pts 128 64 32 16]
"""

from __future__ import annotations
import argparse, sys, os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shared.eval_utils import (
    setup_logging, load_model_and_tokenizer, make_restricted_agent,
    filter_split_ids, run_evaluation_loop, compute_and_save, print_result,
    PROJECT_DIR,
)
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np

ALL_TOOLS = [
    "load_radar_sequence", "extract_radar_features",
    "analyze_joint_motion",
]


def make_sparse_load(n_pts: int):
    """Return a patched _load_or_synthesise that subsamples to n_pts per frame."""
    import tools.radar_processing as rp
    original = rp._load_or_synthesise

    def sparse_load(motion_id: str) -> np.ndarray:
        pts = original(motion_id)           # (T, N_orig, 4)
        T, N_orig, D = pts.shape
        if n_pts >= N_orig:
            return pts
        # Random subsample (without replacement) per frame
        idx = np.random.choice(N_orig, size=n_pts, replace=False)
        return pts[:, idx, :]              # (T, n_pts, 4)

    return sparse_load, original


def main() -> None:
    parser = argparse.ArgumentParser(description="R3: sparse points robustness")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--split",  default="test")
    parser.add_argument("--n-pts",  type=int, nargs="+", default=[128, 64, 32, 16])
    args = parser.parse_args()

    setup_logging()
    model, tokenizer = load_model_and_tokenizer()

    ids = filter_split_ids(args.split, max_samples=args.max_samples)
    all_results = {}

    import tools.radar_processing as rp

    for n_pts in args.n_pts:
        print(f"\n=== N_pts = {n_pts} ===")
        sparse_fn, original_fn = make_sparse_load(n_pts)
        rp._load_or_synthesise = sparse_fn

        try:
            agent = make_restricted_agent(model, tokenizer, ALL_TOOLS)
            hyps, refs, samples = run_evaluation_loop(agent, ids, temperature=0.3)
        finally:
            rp._load_or_synthesise = original_fn

        name = f"R3_sparse_pts{n_pts}_n{args.max_samples}"
        result = compute_and_save(
            name, hyps, refs, samples,
            extra_meta={"points_per_frame": n_pts},
        )
        print_result(result)
        all_results[n_pts] = result["metrics"]

    print("\n=== SPARSITY SWEEP SUMMARY ===")
    print(f"{'N_pts':<10}" + "  " + "  ".join(f"{'ROUGE-L':>10}{'BERTScore':>12}"))
    for n, m in sorted(all_results.items(), reverse=True):
        print(f"{n:<10}  {m.get('ROUGE-L', 0):>10.2f}  {m.get('BERTScore', 0):>12.2f}")


if __name__ == "__main__":
    main()
