#!/usr/bin/env python3
"""
R1 – Robustness: force on-the-fly radar synthesis (no pre-computed points).

Instead of loading pre-computed synthetic_points/rec_{id}.npy files,
forces the agent to synthesise radar point clouds on the fly from SMPL
joint positions for every sample.

This measures whether pre-computed vs synthesised point clouds produce
different captions, and validates that the synthesis pipeline produces
data of similar quality.

Usage:
    PYTHONPATH=RadarAgent RadarAgent/.venv/bin/python experiment_scripts/R1_no_precomputed.py \
        [--max-samples 200]
"""

from __future__ import annotations
import argparse, sys, os
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from shared.eval_utils import (
    setup_logging, load_model_and_tokenizer, make_restricted_agent,
    filter_split_ids, run_evaluation_loop, compute_and_save, print_result,
    PROJECT_DIR,
)
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

ALL_TOOLS = [
    "load_radar_sequence", "extract_radar_features",
    "analyze_joint_motion", "get_motion_text",
    "search_motions", "compare_motions", "visualize_motion",
]


def patch_load_to_force_synthesis():
    """Monkey-patch _resolve_points_path to always return None, forcing on-the-fly synthesis."""
    import tools.radar_processing as rp

    _original = rp._resolve_points_path

    def _no_precomputed(motion_id: str) -> Optional[Path]:
        return None   # always force synthesis from joint data

    rp._resolve_points_path = _no_precomputed
    return _original


def main() -> None:
    parser = argparse.ArgumentParser(description="R1: force on-the-fly synthesis")
    parser.add_argument("--max-samples", type=int, default=200)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    setup_logging()

    # Patch BEFORE loading model (the patch is Python-level, no GPU needed)
    original_fn = patch_load_to_force_synthesis()

    try:
        model, tokenizer = load_model_and_tokenizer()
        agent = make_restricted_agent(model, tokenizer, ALL_TOOLS)

        ids = filter_split_ids(args.split, max_samples=args.max_samples)
        hyps, refs, samples = run_evaluation_loop(agent, ids, temperature=0.3)
    finally:
        # Restore
        import tools.radar_processing as rp
        rp._resolve_points_path = original_fn

    result = compute_and_save(
        f"R1_no_precomputed_n{args.max_samples}",
        hyps, refs, samples,
        extra_meta={"synthesis": "on_the_fly", "no_precomputed": True},
    )
    print_result(result)


if __name__ == "__main__":
    main()
