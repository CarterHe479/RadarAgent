"""
Shared evaluation utilities for all RadarAgent experiments.

Every experiment script imports from here to get a consistent evaluation loop,
output path conventions, and results table printer.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

# ── Path setup (works no matter where the script is invoked from) ─────────────
# experiment_scripts/ is now a sub-directory of RadarAgent/, so:
#   __file__  →  RadarAgent/experiment_scripts/shared/eval_utils.py
#   .parent   →  RadarAgent/experiment_scripts/shared/
#   .parent   →  RadarAgent/experiment_scripts/   (EXPERIMENT_DIR)
#   .parent   →  RadarAgent/                       (PROJECT_DIR)
EXPERIMENT_DIR = Path(__file__).parent.parent   # RadarAgent/experiment_scripts/
PROJECT_DIR    = EXPERIMENT_DIR.parent          # RadarAgent/
RESULTS_DIR    = PROJECT_DIR / "outputs" / "results" / "experiments"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# Must chdir so that config.py resolves DATA_DIR relative to RadarAgent/
os.chdir(PROJECT_DIR)


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )


def load_agent(model_name: Optional[str] = None):
    """Load model + tokenizer and return a RadarAgent instance."""
    from agent.agent import RadarAgent
    from config import MODEL_NAME
    name = model_name or MODEL_NAME
    logging.getLogger(__name__).info("Loading model: %s", name)
    return RadarAgent(model_name=name)


def load_model_and_tokenizer(model_name: Optional[str] = None):
    """Return (model, tokenizer) without wrapping in RadarAgent."""
    from agent.llm import load_model
    from config import MODEL_NAME
    name = model_name or MODEL_NAME
    return load_model(name)


def filter_split_ids(split: str, max_samples: Optional[int] = None) -> List[str]:
    """Return motion IDs from the split file that actually have data files."""
    from config import SPLITS_DIR, JOINTS_DIR, SYNTHETIC_POINTS_DIR

    path = SPLITS_DIR / f"{split}.txt"
    all_ids = [l.strip() for l in path.read_text().splitlines() if l.strip()]
    available = [
        mid for mid in all_ids
        if (SYNTHETIC_POINTS_DIR / f"rec_{mid}.npy").exists()
        or (JOINTS_DIR / f"{mid}.npy").exists()
    ]
    if max_samples:
        available = available[:max_samples]
    return available


def run_evaluation_loop(
    agent,
    motion_ids: List[str],
    query_fn: Callable[[str], str] = None,
    temperature: float = 0.3,
    max_new_tokens: int = 512,
) -> tuple[List[str], List[List[str]], List[Dict]]:
    """
    Core evaluation loop shared by all experiment scripts.

    Args:
        agent:          Initialised RadarAgent (or compatible object with .run()).
        motion_ids:     List of motion IDs to evaluate.
        query_fn:       Maps motion_id -> query string. Default uses standard template.
        temperature:    Decoding temperature.
        max_new_tokens: Max tokens per generation step.

    Returns:
        (hypotheses, references_list, samples)
        samples is a list of dicts: {motion_id, generated, references, elapsed_sec, n_tokens_approx}
    """
    from tqdm import tqdm
    from tools.data_retrieval import _read_descriptions
    from agent.llm import strip_thinking

    if query_fn is None:
        def query_fn(mid: str) -> str:
            return (
                f"Analyse the radar point cloud for motion {mid} and describe "
                "what the person is doing in one to three sentences."
            )

    hypotheses: List[str] = []
    references_list: List[List[str]] = []
    samples: List[Dict] = []
    logger = logging.getLogger(__name__)

    for mid in tqdm(motion_ids, desc="Evaluating"):
        refs = _read_descriptions(mid)
        if not refs:
            logger.warning("No text annotations for %s – skipping.", mid)
            continue

        t0 = time.perf_counter()
        try:
            generated = agent.run(
                query_fn(mid),
                temperature=temperature,
                max_new_tokens=max_new_tokens,
            )
        except Exception as exc:
            logger.error("Agent failed on %s: %s", mid, exc)
            generated = ""
        elapsed = time.perf_counter() - t0

        generated = strip_thinking(generated)

        hypotheses.append(generated)
        references_list.append(refs)
        samples.append({
            "motion_id":        mid,
            "generated":        generated,
            "references":       refs,
            "elapsed_sec":      round(elapsed, 3),
            "n_tokens_approx":  len(generated.split()),
        })

    return hypotheses, references_list, samples


def compute_and_save(
    experiment_name: str,
    hypotheses: List[str],
    references_list: List[List[str]],
    samples: List[Dict],
    extra_meta: Optional[Dict] = None,
) -> Dict:
    """Compute all metrics and save results JSON. Returns the full result dict."""
    from eval.metrics import compute_all_metrics

    logger = logging.getLogger(__name__)
    logger.info("Computing metrics over %d samples …", len(hypotheses))

    all_refs = [r for refs in references_list for r in refs]
    metrics = compute_all_metrics(hypotheses, references_list, corpus_refs=all_refs)

    result = {
        "experiment":   experiment_name,
        "n_samples":    len(samples),
        "metrics":      metrics,
        "meta":         extra_meta or {},
        "samples":      samples,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{experiment_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Results saved to %s", out_path)

    return result


def print_result(result: Dict) -> None:
    """Pretty-print metrics for one experiment."""
    m = result.get("metrics", {})
    cols = ["ROUGE-1", "ROUGE-L", "BLEU-1", "BLEU-4", "METEOR", "CIDEr", "BERTScore", "SimCSE"]
    print(f"\n{'─'*70}")
    print(f"  {result.get('experiment', 'Experiment')}  (n={result.get('n_samples', '?')})")
    print(f"{'─'*70}")
    header = f"{'Metric':<14}" + "  " + "  ".join(f"{c:<10}" for c in cols)
    print(header)
    row    = f"{'Score':<14}" + "  " + "  ".join(
        f"{m.get(c, '-'):<10}" if isinstance(m.get(c), str)
        else f"{m.get(c, 0):<10.2f}"
        for c in cols
    )
    print(row)
    print(f"{'─'*70}\n")


def make_restricted_agent(model, tokenizer, allowed_tools: List[str]):
    """
    Build a RadarAgent that exposes only the specified tools.

    Useful for tool ablation experiments. Pass allowed_tools=[] for the
    no-tools baseline (raw LLM).
    """
    from agent.agent import RadarAgent

    agent = RadarAgent(model=model, tokenizer=tokenizer)

    # Filter schemas exposed to the LLM
    agent.tool_schemas = [
        s for s in agent.tool_schemas
        if s["function"]["name"] in allowed_tools
    ]

    # Filter callable map (for actual dispatch)
    agent.tool_map = {
        k: v for k, v in agent.tool_map.items()
        if k in allowed_tools
    }

    return agent
