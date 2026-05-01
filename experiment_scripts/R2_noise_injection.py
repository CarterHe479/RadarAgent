#!/usr/bin/env python3
"""
R2 – Robustness: add Gaussian noise to radar point cloud coordinates.

Simulates real-world sensor noise at three levels:
  sigma=0.02m  (same as synthesis jitter, baseline noise)
  sigma=0.05m  (moderate noise)
  sigma=0.10m  (heavy noise)

Each sigma level is evaluated on the same 200-sample subset.

Usage:
    PYTHONPATH=RadarAgent RadarAgent/.venv/bin/python experiment_scripts/R2_noise_injection.py \
        [--max-samples 200] [--sigmas 0.02 0.05 0.10]
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


def make_noisy_load(sigma: float):
    """Return a patched _load_or_synthesise that adds noise to xyz coords."""
    import tools.radar_processing as rp
    original = rp._load_or_synthesise

    def noisy_load(motion_id: str) -> np.ndarray:
        pts = original(motion_id)           # (T, N, 4)
        noise = np.random.randn(*pts[:, :, :3].shape).astype(np.float32) * sigma
        pts_noisy = pts.copy()
        pts_noisy[:, :, :3] += noise
        return pts_noisy

    return noisy_load, original


def main() -> None:
    parser = argparse.ArgumentParser(description="R2: noise injection robustness")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--split",  default="test")
    parser.add_argument("--sigmas", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    args = parser.parse_args()

    setup_logging()
    model, tokenizer = load_model_and_tokenizer()

    ids = filter_split_ids(args.split, max_samples=args.max_samples)
    all_results = {}

    import tools.radar_processing as rp

    for sigma in args.sigmas:
        print(f"\n=== sigma = {sigma} m ===")
        noisy_fn, original_fn = make_noisy_load(sigma)
        rp._load_or_synthesise = noisy_fn

        try:
            agent = make_restricted_agent(model, tokenizer, ALL_TOOLS)
            hyps, refs, samples = run_evaluation_loop(agent, ids, temperature=0.3)
        finally:
            rp._load_or_synthesise = original_fn

        name = f"R2_noise_sigma{str(sigma).replace('.', 'p')}_n{args.max_samples}"
        result = compute_and_save(
            name, hyps, refs, samples,
            extra_meta={"noise_sigma_m": sigma},
        )
        print_result(result)
        all_results[sigma] = result["metrics"]

    print("\n=== NOISE SWEEP SUMMARY ===")
    print(f"{'Sigma':<10}" + "  " + "  ".join(f"{'ROUGE-L':>10}{'BERTScore':>12}"))
    for s, m in sorted(all_results.items()):
        print(f"{s:<10.2f}  {m.get('ROUGE-L', 0):>10.2f}  {m.get('BERTScore', 0):>12.2f}")


if __name__ == "__main__":
    main()
