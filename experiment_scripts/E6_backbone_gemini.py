#!/usr/bin/env python3
"""
E6 – Closed-source backbone comparison: Gemini 2.0 Flash.

Uses the same plain-text feature context as E5 (no native tool-calling schema
required), but calls the Gemini API instead of a local model.  Measures
whether a state-of-the-art closed-source LLM can match or beat the open
Qwen 3 8B RadarAgent when given identical feature summaries.

Requirements:
    pip install google-genai
    GOOGLE_API_KEY env var (or pass --api-key).

Usage (from RadarAgent/):
    PYTHONPATH=. python experiment_scripts/E6_backbone_gemini.py [max_samples]
    PYTHONPATH=. python experiment_scripts/E6_backbone_gemini.py 500
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
EXPERIMENT_DIR = Path(__file__).parent
PROJECT_DIR    = EXPERIMENT_DIR.parent
sys.path.insert(0, str(EXPERIMENT_DIR))   # shared.eval_utils
sys.path.insert(0, str(PROJECT_DIR))      # agent, tools, config
os.chdir(PROJECT_DIR)

from shared.eval_utils import (
    setup_logging,
    filter_split_ids,
    compute_and_save,
    print_result,
)
from tools.data_retrieval import _read_descriptions
from tools.radar_processing import load_radar_sequence, extract_radar_features
from tools.joint_analysis import analyze_joint_motion

setup_logging()
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
GEMINI_MODEL   = "gemini-2.0-flash"
DEFAULT_SAMPLES = 500

SYSTEM_PROMPT = (
    "You are an expert at understanding human motion from sensor data. "
    "You will be given structured feature data about a motion sequence extracted "
    "from millimeter-wave radar and skeleton joint analysis. "
    "Respond with a single sentence of 5–15 words describing the primary action, "
    "matching HumanML3D annotation style: "
    "'a person walks forward', 'the man waves his right hand', "
    "'a person jogs in a circle', 'the person squats down and stands back up'. "
    "Start with 'a person', 'the person', 'a man', or 'the man'. "
    "Name only the primary action – do NOT include speeds, distances, or sensor readings. "
    "Do NOT write more than one sentence."
)


def _build_feature_context(mid: str) -> str | None:
    """Run the three core tools and format results as a text context string."""
    try:
        seq    = load_radar_sequence(mid)
        feats  = extract_radar_features(mid)
        joints = analyze_joint_motion(mid)
    except Exception as exc:
        logger.warning("Tool error for %s: %s", mid, exc)
        return None

    ctx = (
        f"Duration: {seq['duration_sec']}s, "
        f"displacement: {seq['overall_displacement']:.2f}m.\n"
        f"Velocity: mean={feats['velocity']['mean_m_per_s']} m/s, "
        f"max={feats['velocity']['max_m_per_s']} m/s.\n"
        f"Trajectory shape: {feats['trajectory_shape']}, "
        f"dominant axis: {feats['dominant_motion_axis']}.\n"
        f"Periodic motion: {feats['periodicity']['is_periodic']} "
        f"(period={feats['periodicity'].get('estimated_period_sec', 'N/A')}s).\n"
        f"Detected actions: {', '.join(joints['detected_actions']) or 'none'}.\n"
        f"Most active body parts: {', '.join(joints['most_active_parts'])}.\n"
        f"Root trajectory: {joints['root_trajectory']}."
    )
    return ctx


def describe_motion_gemini(client, mid: str) -> str:
    """Call Gemini API to produce a one-sentence motion description."""
    ctx = _build_feature_context(mid)
    if ctx is None:
        return ""

    prompt = f"Motion sensor features:\n{ctx}\n\nDescribe this motion in one sentence."

    from google.genai import types as genai_types

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.3,
                max_output_tokens=64,
            ),
        )
        text = response.text.strip()
        # Strip any markdown formatting the model might add
        text = text.strip("`").strip("*").strip()
        return text
    except Exception as exc:
        logger.error("Gemini API error for %s: %s", mid, exc)
        return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="E6 – Gemini backbone comparison")
    parser.add_argument("max_samples", nargs="?", type=int, default=DEFAULT_SAMPLES,
                        help=f"Max test samples to evaluate (default: {DEFAULT_SAMPLES})")
    parser.add_argument("--api-key", default=None,
                        help="Gemini API key (overrides GOOGLE_API_KEY env var)")
    parser.add_argument("--model", default=GEMINI_MODEL,
                        help=f"Gemini model to use (default: {GEMINI_MODEL})")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("GOOGLE_API_KEY", "")
    if not api_key:
        logger.error("No Gemini API key provided. Set GOOGLE_API_KEY or use --api-key.")
        sys.exit(1)

    model_name   = args.model
    max_samples  = args.max_samples
    experiment_id = f"E6_backbone_gemini_n{max_samples}"

    logger.info("=== E6: Backbone – %s (%d samples) ===", model_name, max_samples)

    # Initialise Gemini client
    from google import genai
    client = genai.Client(api_key=api_key)

    # Filter to samples that have actual data files
    ids = filter_split_ids("test", max_samples=max_samples)
    logger.info("Evaluating %d samples …", len(ids))

    from tqdm import tqdm
    hypotheses, refs_list, samples = [], [], []

    for mid in tqdm(ids, desc=f"Eval [E6/{model_name}]"):
        refs = _read_descriptions(mid)
        if not refs:
            continue

        t0  = time.perf_counter()
        gen = describe_motion_gemini(client, mid)
        elapsed = time.perf_counter() - t0

        hypotheses.append(gen)
        refs_list.append(refs)
        samples.append({
            "motion_id":       mid,
            "generated":       gen,
            "references":      refs,
            "elapsed_sec":     round(elapsed, 3),
            "n_tokens_approx": len(gen.split()),
        })

    result = compute_and_save(
        experiment_id,
        hypotheses,
        refs_list,
        samples,
        extra_meta={
            "model":    model_name,
            "approach": "plain_text_features_via_gemini_api",
        },
    )
    print_result(result)


if __name__ == "__main__":
    main()
