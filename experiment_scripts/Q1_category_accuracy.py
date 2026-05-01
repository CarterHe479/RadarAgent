#!/usr/bin/env python3
"""
Q1 – Analysis: per-action-category accuracy.

Reads an existing eval JSON (e.g. from E1), groups samples by the dominant
action verb in the reference annotation, and reports precision/recall per
action category.

Requires no model inference.

Usage:
    python experiment_scripts/Q1_category_accuracy.py \
        --results RadarAgent/outputs/results/experiments/E1_main_results.json \
        [--output outputs/results/experiments/Q1_category_accuracy.json]
"""

from __future__ import annotations
import argparse, json, sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shared.eval_utils import RESULTS_DIR

# Action verb groups: each group maps a canonical category to synonyms/partial matches
ACTION_VOCAB = {
    "walk":      ["walk", "stroll", "step"],
    "run":       ["run", "jog", "sprint"],
    "jump":      ["jump", "hop", "leap", "spring"],
    "squat":     ["squat", "crouch", "kneel", "bend down"],
    "wave":      ["wave", "wave hand"],
    "kick":      ["kick"],
    "raise_arm": ["raise", "lift arm", "raise arm", "lift hand"],
    "turn":      ["turn", "rotate", "spin", "pivot"],
    "sit":       ["sit", "sit down"],
    "stand":     ["stand", "stand up"],
    "throw":     ["throw", "toss"],
    "punch":     ["punch", "hit"],
    "stretch":   ["stretch", "reach"],
    "climb":     ["climb", "stairs", "stair", "step up"],
    "dance":     ["dance"],
    "pick_up":   ["pick up", "pickup", "pick", "grab", "lift object"],
}


def categorize(text: str) -> list[str]:
    text_l = text.lower()
    cats = []
    for cat, keywords in ACTION_VOCAB.items():
        if any(kw in text_l for kw in keywords):
            cats.append(cat)
    return cats if cats else ["other"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Q1: per-category accuracy")
    parser.add_argument("--results", required=True, help="Path to eval JSON (e.g. E1)")
    parser.add_argument("--output",  default=None)
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]

    # Per-category stats
    stats = defaultdict(lambda: {"tp": 0, "fn": 0, "fp": 0, "total": 0, "examples": []})

    for s in samples:
        ref_cats = categorize(s["references"][0])
        gen_cats = categorize(s["generated"])

        for cat in ref_cats:
            stats[cat]["total"] += 1
            if cat in gen_cats:
                stats[cat]["tp"] += 1
            else:
                stats[cat]["fn"] += 1
                if len(stats[cat]["examples"]) < 3:
                    stats[cat]["examples"].append({
                        "motion_id": s["motion_id"],
                        "ref": s["references"][0],
                        "gen": s["generated"],
                    })

        for cat in gen_cats:
            if cat not in ref_cats:
                stats[cat]["fp"] += 1

    # Compute precision, recall, F1
    rows = []
    for cat in sorted(stats, key=lambda c: -stats[c]["total"]):
        tp = stats[cat]["tp"]
        fn = stats[cat]["fn"]
        fp = stats[cat]["fp"]
        total = stats[cat]["total"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        rows.append({
            "category":  cat,
            "total_ref": total,
            "tp": tp, "fn": fn, "fp": fp,
            "precision": round(precision, 3),
            "recall":    round(recall, 3),
            "f1":        round(f1, 3),
            "examples":  stats[cat]["examples"],
        })

    # Print table
    print(f"\n{'Category':<16} {'Count':>6} {'Precision':>10} {'Recall':>8} {'F1':>6}")
    print("─" * 52)
    for r in rows:
        print(
            f"{r['category']:<16} {r['total_ref']:>6} "
            f"{r['precision']:>10.1%} {r['recall']:>8.1%} {r['f1']:>6.3f}"
        )

    macro_f1 = sum(r["f1"] for r in rows) / len(rows)
    print(f"\nMacro F1: {macro_f1:.3f}  (over {len(rows)} categories, {len(samples)} samples)")

    # Save
    out_path = Path(args.output) if args.output else RESULTS_DIR / "Q1_category_accuracy.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out = {"source": args.results, "macro_f1": round(macro_f1, 3), "per_category": rows}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
