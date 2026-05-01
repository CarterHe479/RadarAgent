#!/usr/bin/env python3
"""
Q4 – Analysis: error taxonomy.

Classifies each sample's error into one of four categories:
  1. CORRECT       – generated captures the right action
  2. WRONG_ACTION  – wrong primary action verb (e.g. "kick" vs reference "walk")
  3. RIGHT_ACT_WRONG_DETAIL – correct action but wrong detail (e.g. "waves right" vs "waves left")
  4. MULTI_ACTION_MISS – reference has 2+ actions, agent captured ≤1
  5. HALLUCINATION – agent mentions an action absent from all references

Reports counts and examples for each category.

Usage:
    python experiment_scripts/Q4_error_taxonomy.py \
        --results RadarAgent/outputs/results/experiments/E1_main_results.json
"""

from __future__ import annotations
import argparse, json, sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shared.eval_utils import RESULTS_DIR

ACTION_VOCAB = {
    "walk": ["walk", "stroll"],
    "run":  ["run", "jog", "sprint"],
    "jump": ["jump", "hop", "leap"],
    "squat": ["squat", "crouch", "bend down", "crouch down"],
    "wave": ["wave"],
    "kick": ["kick"],
    "raise_arm": ["raise", "lift arm", "raise arm"],
    "turn": ["turn", "rotate", "spin"],
    "sit":  ["sit"],
    "stand": ["stand up"],
    "throw": ["throw", "toss"],
    "punch": ["punch"],
    "pick_up": ["pick up", "pickup"],
    "climb": ["climb", "stairs"],
    "stretch": ["stretch", "reach"],
}

SIDE_WORDS = ["left", "right", "forward", "backward", "backwards", "back"]


def get_actions(text: str) -> set[str]:
    t = text.lower()
    return {cat for cat, kws in ACTION_VOCAB.items() if any(kw in t for kw in kws)}


def classify(gen: str, refs: list[str]) -> tuple[str, str]:
    """Return (error_type, reason_string)."""
    gen_acts  = get_actions(gen)
    ref_acts  = set()
    all_ref_acts_per = []
    for r in refs:
        acts = get_actions(r)
        ref_acts |= acts
        all_ref_acts_per.append(acts)

    # How many distinct action categories in the primary reference?
    primary_acts = all_ref_acts_per[0] if all_ref_acts_per else set()

    # 1. Correct: generated actions overlap with reference actions
    if gen_acts & ref_acts:
        # Check if detail (side/direction) differs
        gen_sides = {w for w in SIDE_WORDS if w in gen.lower()}
        ref0_sides = {w for w in SIDE_WORDS if w in refs[0].lower()}
        if gen_sides and ref0_sides and not (gen_sides & ref0_sides):
            return "RIGHT_ACT_WRONG_DETAIL", f"gen_sides={gen_sides} ref_sides={ref0_sides}"
        return "CORRECT", ""

    # 2. Multi-action miss: reference has ≥2 distinct actions and we caught 0
    if len(primary_acts) >= 2 and not (gen_acts & primary_acts):
        return "MULTI_ACTION_MISS", f"ref_acts={primary_acts} gen_acts={gen_acts}"

    # 3. Hallucination: gen mentions actions not in ANY reference
    hallucinated = gen_acts - ref_acts
    if hallucinated:
        return "HALLUCINATION", f"hallucinated={hallucinated}"

    # 4. Wrong action
    if ref_acts and not (gen_acts & ref_acts):
        return "WRONG_ACTION", f"ref={ref_acts} gen={gen_acts}"

    return "OTHER", ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Q4: error taxonomy")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output",  default=None)
    parser.add_argument("--examples", type=int, default=3,
                        help="Number of examples to show per category")
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    taxonomy: dict[str, list] = {
        "CORRECT": [], "WRONG_ACTION": [], "RIGHT_ACT_WRONG_DETAIL": [],
        "MULTI_ACTION_MISS": [], "HALLUCINATION": [], "OTHER": [],
    }

    for s in samples:
        err_type, reason = classify(s["generated"], s["references"])
        taxonomy[err_type].append({
            "motion_id": s["motion_id"],
            "generated": s["generated"],
            "ref":       s["references"][0],
            "reason":    reason,
        })

    n = len(samples)
    print(f"\n=== Error Taxonomy ({n} samples) ===\n")
    print(f"{'Category':<28}  {'Count':>6}  {'%':>6}")
    print("─" * 44)
    for cat in ["CORRECT", "WRONG_ACTION", "RIGHT_ACT_WRONG_DETAIL",
                "MULTI_ACTION_MISS", "HALLUCINATION", "OTHER"]:
        c = len(taxonomy[cat])
        print(f"{cat:<28}  {c:>6}  {c/n:>6.1%}")

    # Show examples for each error type
    for cat in ["WRONG_ACTION", "HALLUCINATION", "MULTI_ACTION_MISS", "RIGHT_ACT_WRONG_DETAIL"]:
        examples = taxonomy[cat][:args.examples]
        if not examples:
            continue
        print(f"\n--- {cat} examples ---")
        for ex in examples:
            print(f"  [{ex['motion_id']}]")
            print(f"    Ref: {ex['ref']}")
            print(f"    Gen: {ex['generated']}")
            if ex["reason"]:
                print(f"    Why: {ex['reason']}")

    out_path = Path(args.output) if args.output else RESULTS_DIR / "Q4_error_taxonomy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": args.results,
        "n_samples": n,
        "counts": {cat: len(v) for cat, v in taxonomy.items()},
        "pct":    {cat: round(len(v)/n, 3) for cat, v in taxonomy.items()},
        "examples": {cat: v[:5] for cat, v in taxonomy.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
