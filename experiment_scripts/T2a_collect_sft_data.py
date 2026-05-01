#!/usr/bin/env python3
"""
T2a – Collect SFT training data: run tool pipeline on training samples.

For each training motion:
  1. Call the three core tools to get structured outputs.
  2. Format them as a compact feature text string (the LLM's "context").
  3. Pair with the first ground-truth reference description (the "target").
  4. Save as a JSONL file for fine-tuning.

This script does NOT run the LLM – only the Python tool functions.
Expected runtime: ~10-30 minutes for 1000 samples (CPU-only, no GPU needed).

Output format (one JSON per line):
{
  "motion_id": "000021",
  "feature_context": "<compact text summary of tool outputs>",
  "target": "a person walks forward casually",
  "all_targets": ["a person walks forward casually", "the man strolls ahead", ...]
}

Usage:
    PYTHONPATH=RadarAgent RadarAgent/.venv/bin/python experiment_scripts/T2a_collect_sft_data.py \
        [--max-samples 1000] [--split train] [--output path/to/sft_data.jsonl]
"""

from __future__ import annotations
import argparse, json, logging, sys, os
from pathlib import Path
from tqdm import tqdm

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from shared.eval_utils import setup_logging, filter_split_ids, EXPERIMENT_DIR, PROJECT_DIR
os.chdir(PROJECT_DIR)

sys.path.insert(0, str(PROJECT_DIR))
from tools.radar_processing import load_radar_sequence, extract_radar_features
from tools.joint_analysis import analyze_joint_motion
from tools.data_retrieval import _read_descriptions

logger = logging.getLogger(__name__)


def format_feature_context(seq: dict, feats: dict, joints: dict) -> str:
    """Convert tool outputs into a compact human-readable string for the LLM."""
    p = feats["periodicity"]
    period_str = (
        f"periodic ({p['estimated_period_sec']}s cycle)"
        if p["is_periodic"]
        else "aperiodic"
    )

    return (
        f"Duration: {seq['duration_sec']}s. "
        f"Net displacement: {seq['overall_displacement']:.2f}m. "
        f"Mean velocity: {feats['velocity']['mean_m_per_s']} m/s, "
        f"peak: {feats['velocity']['max_m_per_s']} m/s. "
        f"Motion: {period_str}, {feats['trajectory_shape']} trajectory, "
        f"dominant axis: {feats['dominant_motion_axis']}. "
        f"Vertical range: {feats['vertical_dynamics']['vertical_range_m']:.2f}m. "
        f"Upper body: {feats['upper_body']['activity_level']}, "
        f"lower body: {feats['lower_body']['activity_level']}. "
        f"Detected actions: {', '.join(joints['detected_actions'])}. "
        f"Most active parts: {', '.join(joints['most_active_parts'])}. "
        f"Root: {joints['root_trajectory']}. "
        f"Symmetry: {joints['symmetry']}."
    )


def collect_sample(motion_id: str) -> dict | None:
    """Return a training sample dict for one motion_id, or None on error."""
    try:
        seq    = load_radar_sequence(motion_id)
        feats  = extract_radar_features(motion_id)
        joints = analyze_joint_motion(motion_id)
    except Exception as e:
        logger.debug("Skipping %s: %s", motion_id, e)
        return None

    targets = _read_descriptions(motion_id)
    if not targets:
        return None

    return {
        "motion_id":       motion_id,
        "feature_context": format_feature_context(seq, feats, joints),
        "target":          targets[0],          # primary reference
        "all_targets":     targets,             # all references (for metric cross-ref)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="T2a: collect SFT data from tools")
    parser.add_argument("--max-samples", type=int, default=1000)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    setup_logging()

    out_path = Path(args.output) if args.output else (
        PROJECT_DIR / "outputs" / "sft" / f"sft_data_{args.split}_n{args.max_samples}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ids = filter_split_ids(args.split, max_samples=args.max_samples)
    logger.info("Collecting SFT data for %d motions from '%s' split …", len(ids), args.split)

    n_ok, n_skip = 0, 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for mid in tqdm(ids, desc="Collecting"):
            sample = collect_sample(mid)
            if sample is None:
                n_skip += 1
                continue
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
            n_ok += 1

    logger.info("Done. %d samples written, %d skipped. Output: %s", n_ok, n_skip, out_path)
    print(f"\nSFT data saved to: {out_path}")
    print(f"Total samples: {n_ok}  (skipped: {n_skip})")


if __name__ == "__main__":
    main()
