#!/usr/bin/env python3
"""
Q2 – Analysis: tool call frequency and usage patterns.

Reads an existing eval JSON and analyses which tools the agent invoked,
per action category and overall.  Useful to show the agent is behaving
intelligently (e.g. using analyze_joint_motion more for complex actions).

Note: the eval JSON must contain the raw generated text BEFORE thinking was
stripped (i.e. the intermediate tool_call blocks).  If your eval JSON has
already-stripped output, this script falls back to reporting from 'samples'.
For full tool-call logging, set agent logging to DEBUG before running eval.

Usage:
    python experiment_scripts/Q2_tool_call_freq.py \
        --results RadarAgent/outputs/results/experiments/E1_main_results.json
"""

from __future__ import annotations
import argparse, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from shared.eval_utils import RESULTS_DIR

# Tool names to look for
TOOLS = [
    "load_radar_sequence",
    "extract_radar_features",
    "analyze_joint_motion",
    "get_motion_text",
    "search_motions",
    "compare_motions",
    "visualize_motion",
]

TOOL_CALL_RE = re.compile(r'"name"\s*:\s*"([^"]+)"', re.DOTALL)


def count_tool_calls_in_text(text: str) -> Counter:
    """Count how many times each tool was called in the raw generated text."""
    hits = TOOL_CALL_RE.findall(text)
    return Counter(h for h in hits if h in TOOLS)


# Simple action categorizer (same logic as Q1)
ACTION_VOCAB = {
    "walk": ["walk"], "run": ["run", "jog"], "jump": ["jump", "hop"],
    "squat": ["squat", "crouch"], "wave": ["wave"], "kick": ["kick"],
    "raise_arm": ["raise", "lift arm"], "turn": ["turn", "rotate"],
}


def categorize(text: str) -> str:
    text_l = text.lower()
    for cat, kws in ACTION_VOCAB.items():
        if any(kw in text_l for kw in kws):
            return cat
    return "other"


def main() -> None:
    parser = argparse.ArgumentParser(description="Q2: tool call frequency analysis")
    parser.add_argument("--results", required=True)
    parser.add_argument("--output",  default=None)
    args = parser.parse_args()

    with open(args.results, encoding="utf-8") as f:
        data = json.load(f)

    samples = data["samples"]
    n = len(samples)

    # Overall tool call frequency
    total_calls = Counter()
    per_cat_calls: dict[str, Counter] = defaultdict(Counter)
    n_calls_per_sample = []

    for s in samples:
        # The 'generated' field may not contain tool calls (they're stripped).
        # We do our best with what's in the file.
        text = s.get("generated", "")
        calls = count_tool_calls_in_text(text)
        total_calls.update(calls)
        n_calls_per_sample.append(sum(calls.values()))
        cat = categorize(s["references"][0])
        per_cat_calls[cat].update(calls)

    print(f"\n=== Tool Call Frequency ({n} samples) ===\n")
    print(f"{'Tool':<30}  {'Count':>8}  {'Per sample':>12}")
    print("─" * 56)
    for tool in TOOLS:
        c = total_calls.get(tool, 0)
        print(f"{tool:<30}  {c:>8}  {c/n:>12.2f}")
    print(f"\nAvg total calls per sample: {sum(n_calls_per_sample)/max(n,1):.2f}")

    if any(per_cat_calls.values()):
        print("\n=== Tool Usage by Action Category ===")
        cats = sorted(per_cat_calls)
        print(f"{'Category':<14}", end="")
        for t in TOOLS:
            print(f"  {t[:12]:>12}", end="")
        print()
        print("─" * (14 + len(TOOLS) * 14))
        for cat in cats:
            cc = per_cat_calls[cat]
            n_cat = sum(1 for s in samples if categorize(s["references"][0]) == cat) or 1
            print(f"{cat:<14}", end="")
            for t in TOOLS:
                print(f"  {cc.get(t, 0)/n_cat:>12.2f}", end="")
            print()

    out_path = Path(args.output) if args.output else RESULTS_DIR / "Q2_tool_call_freq.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "source": args.results,
        "n_samples": n,
        "total_calls": dict(total_calls),
        "avg_calls_per_sample": round(sum(n_calls_per_sample) / max(n, 1), 2),
        "per_category": {cat: dict(c) for cat, c in per_cat_calls.items()},
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to: {out_path}")


if __name__ == "__main__":
    main()
