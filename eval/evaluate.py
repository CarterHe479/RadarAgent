"""
Evaluation loop: run RadarAgent on a dataset split and compute metrics.

Usage:
    python main.py --evaluate --split test --output results/test_results.json

For each motion in the split:
  1. Query: "Analyse the radar point cloud for motion {id} and describe what
             the person is doing."
  2. Run the agent → collect generated description.
  3. Load ground-truth descriptions from texts/{id}.txt.
  4. Compute all metrics.
  5. Save per-sample results + aggregate scores.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from tqdm import tqdm

from config import SPLITS_DIR, RESULTS_DIR, JOINTS_DIR, SYNTHETIC_POINTS_DIR
from tools.data_retrieval import _read_descriptions
from eval.metrics import compute_all_metrics
from agent.llm import strip_thinking

if TYPE_CHECKING:
    from agent.agent import RadarAgent

logger = logging.getLogger(__name__)


QUERY_TEMPLATE = (
    "Analyse the radar point cloud for motion {motion_id} and describe "
    "what the person is doing in one to three sentences."
)


def load_split_ids(split: str) -> List[str]:
    path = SPLITS_DIR / f"{split}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _has_motion_data(motion_id: str) -> bool:
    """Return True if joint or synthetic point-cloud data exists for motion_id."""
    if (SYNTHETIC_POINTS_DIR / f"rec_{motion_id}.npy").exists():
        return True
    if (JOINTS_DIR / f"{motion_id}.npy").exists():
        return True
    return False


def run_evaluation(
    agent: "RadarAgent",
    split: str = "test",
    max_samples: Optional[int] = None,
    output_path: Optional[Path] = None,
    temperature: float = 0.3,
) -> dict:
    """Run the agent over a split and return aggregated metric scores.

    Args:
        agent:        Initialised RadarAgent.
        split:        One of "train", "val", "test".
        max_samples:  Limit evaluation to this many samples (None = all).
        output_path:  Path to save the JSON results file.
        temperature:  Generation temperature to use during evaluation (default 0.3
                      for more deterministic output; lower than the interactive default).

    Returns:
        Dict with "metrics" and "samples" keys.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = RESULTS_DIR / f"eval_{split}.json"

    ids = load_split_ids(split)

    # Drop motion IDs that have no data; the agent can only produce error messages
    # for those, which would unfairly drag down all metrics.
    available_ids = [mid for mid in ids if _has_motion_data(mid)]
    n_skipped = len(ids) - len(available_ids)
    if n_skipped:
        logger.info(
            "Skipping %d/%d motion IDs that have no data files.", n_skipped, len(ids)
        )
    ids = available_ids

    if max_samples:
        ids = ids[:max_samples]

    hypotheses:      List[str]       = []
    references_list: List[List[str]] = []
    samples:         List[dict]      = []
    all_refs:        List[str]       = []  # for CIDEr IDF

    logger.info("Evaluating %d motions from split '%s' …", len(ids), split)

    for motion_id in tqdm(ids, desc=f"Eval [{split}]"):
        refs = _read_descriptions(motion_id)
        if not refs:
            logger.warning("No text annotations for %s – skipping.", motion_id)
            continue

        query = QUERY_TEMPLATE.format(motion_id=motion_id)
        try:
            generated = agent.run(query, temperature=temperature)
        except Exception as exc:
            logger.error("Agent failed on %s: %s", motion_id, exc)
            generated = ""

        # Safety: strip any residual <think>...</think> blocks before scoring.
        generated = strip_thinking(generated)

        hypotheses.append(generated)
        references_list.append(refs)
        all_refs.extend(refs)

        samples.append({
            "motion_id":   motion_id,
            "generated":   generated,
            "references":  refs,
        })

    if not hypotheses:
        logger.error("No samples were successfully evaluated.")
        return {}

    logger.info("Computing metrics over %d samples …", len(hypotheses))
    metrics = compute_all_metrics(hypotheses, references_list, corpus_refs=all_refs)

    result = {
        "split":   split,
        "n_samples": len(samples),
        "metrics": metrics,
        "samples": samples,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", output_path)

    return result


def print_metrics_table(metrics: dict, title: str = "Evaluation Results") -> None:
    """Pretty-print metrics in a format comparable to Table 1 of the paper."""
    paper_baselines = {
        "RadarLLM":  {"ROUGE-L": 36.0, "BLEU-1": 48.0, "BLEU-4": 11.4,
                      "METEOR": 33.7, "BERTScore": 83.3},
        "AvatarGPT": {"ROUGE-L": 30.0, "BLEU-1": 36.3, "BLEU-4":  5.0,
                      "METEOR": 28.3, "BERTScore": 82.4},
        "MotionGPT": {"ROUGE-L": 29.4, "BLEU-1": 37.6, "BLEU-4":  5.0,
                      "METEOR": 26.1, "BERTScore": 82.6},
    }

    cols = ["ROUGE-1", "ROUGE-L", "BLEU-1", "BLEU-4", "METEOR",
            "CIDEr", "BERTScore", "SimCSE"]
    col_w = 10

    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")
    header = f"{'Model':<18}" + "".join(f"{c:>{col_w}}" for c in cols)
    print(header)
    print("─" * len(header))

    # Paper baselines (partial columns)
    for name, base_scores in paper_baselines.items():
        row = f"{name:<18}"
        for c in cols:
            v = base_scores.get(c, "–")
            row += f"{v if isinstance(v, str) else f'{v:.1f}':>{col_w}}"
        print(row)

    print("─" * len(header))
    row = f"{'RadarAgent (ours)':<18}"
    for c in cols:
        v = metrics.get(c, "–")
        row += f"{v if isinstance(v, str) else f'{v:.1f}':>{col_w}}"
    print(row)
    print(f"{'─'*60}\n")
