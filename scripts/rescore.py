"""
Rescore the existing eval_test.json with Fix 1 (strip think) + Fix 2 (valid data only).
Run: PYTHONPATH=. .venv/bin/python scripts/rescore.py
"""
import json
import re
import sys

sys.path.insert(0, "/home/carter/radar_llm/RadarAgent")
from config import JOINTS_DIR, SYNTHETIC_POINTS_DIR
from eval.metrics import compute_all_metrics

THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

results_path = "/home/carter/radar_llm/RadarAgent/outputs/results/eval_test.json"
with open(results_path) as f:
    data = json.load(f)

print(f"Loaded {len(data['samples'])} samples from {results_path}", flush=True)

clean = []
for s in data["samples"]:
    # Fix 1: strip <think> tags
    gen = THINK_RE.sub("", s["generated"]).strip()

    # Fix 2: skip samples with no data files
    mid = s["motion_id"]
    has_data = (
        (SYNTHETIC_POINTS_DIR / f"rec_{mid}.npy").exists()
        or (JOINTS_DIR / f"{mid}.npy").exists()
    )
    if not has_data:
        print(f"  Skipping {mid} (no data)", flush=True)
        continue

    clean.append({"motion_id": mid, "generated": gen, "references": s["references"]})

print(f"\nSamples after filtering: {len(clean)}", flush=True)
print("Computing metrics ...", flush=True)

hyps = [s["generated"] for s in clean]
refs = [s["references"] for s in clean]
all_refs = [r for rr in refs for r in rr]

m = compute_all_metrics(hyps, refs, corpus_refs=all_refs)

print("\n=== RESCORED METRICS (Fix1: no <think>, Fix2: valid data only) ===", flush=True)
baseline = {"ROUGE-1": 4.74, "ROUGE-L": 3.84, "BLEU-1": 3.44, "BLEU-4": 0.28,
            "METEOR": 9.2, "CIDEr": 0.32, "BERTScore": 79.42, "SimCSE": 23.19}
for k, v in m.items():
    delta = v - baseline.get(k, 0)
    print(f"  {k:12s}: {v:6.2f}  (was {baseline.get(k, '-'):6.2f}, delta {delta:+.2f})", flush=True)

print("\nSample outputs (clean):", flush=True)
for s in clean[:5]:
    print(f"  [{s['motion_id']}]", flush=True)
    print(f"    Gen: {s['generated'][:120]!r}", flush=True)
    print(f"    Ref: {s['references'][0]!r}", flush=True)
