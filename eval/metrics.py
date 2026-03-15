"""
Evaluation metrics matching those used in the RadarLLM paper (Table 1).

Metrics:
  ROUGE-1, ROUGE-L  – via rouge-score
  BLEU-1, BLEU-4    – via nltk
  METEOR            – via nltk
  CIDEr             – custom TF-IDF weighted n-gram implementation
  BERTScore         – via bert-score
  SimCSE            – cosine similarity of sentence embeddings
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Sequence, List, Optional

import numpy as np


# ── ROUGE ─────────────────────────────────────────────────────────────────────

def compute_rouge(hypothesis: str, references: List[str]) -> dict:
    """Compute ROUGE-1 and ROUGE-L, taking the best score over references."""
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    best = {"rouge1": 0.0, "rougeL": 0.0}
    for ref in references:
        scores = scorer.score(ref, hypothesis)
        best["rouge1"] = max(best["rouge1"], scores["rouge1"].fmeasure)
        best["rougeL"] = max(best["rougeL"], scores["rougeL"].fmeasure)
    return best


# ── BLEU ──────────────────────────────────────────────────────────────────────

def compute_bleu(hypothesis: str, references: List[str]) -> dict:
    """Compute BLEU-1 and BLEU-4 (corpus-level, best over references)."""
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    sf = SmoothingFunction().method1
    hyp_tokens = hypothesis.lower().split()
    ref_tokens = [r.lower().split() for r in references]

    bleu1 = sentence_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=sf)
    bleu4 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25,) * 4, smoothing_function=sf)
    return {"bleu1": bleu1, "bleu4": bleu4}


# ── METEOR ────────────────────────────────────────────────────────────────────

def compute_meteor(hypothesis: str, references: List[str]) -> float:
    """Compute METEOR, best over references."""
    import nltk
    try:
        nltk.data.find("wordnet")
    except LookupError:
        nltk.download("wordnet", quiet=True)
    try:
        nltk.data.find("omw-1.4")
    except LookupError:
        nltk.download("omw-1.4", quiet=True)

    from nltk.translate.meteor_score import meteor_score

    hyp_tokens = hypothesis.lower().split()
    best = 0.0
    for ref in references:
        ref_tokens = ref.lower().split()
        score = meteor_score([ref_tokens], hyp_tokens)
        best = max(best, score)
    return best


# ── CIDEr (custom) ────────────────────────────────────────────────────────────

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1))


def compute_cider(
    hypothesis: str,
    references: List[str],
    corpus_refs: Optional[List[List[str]]] = None,
    n_max: int = 4,
) -> float:
    """Compute CIDEr-D score.

    Args:
        hypothesis:  Generated sentence.
        references:  Reference sentences for this sample.
        corpus_refs: All reference sentences in the corpus (for IDF).
                     If None, IDF is computed from `references` only.
        n_max:       Maximum n-gram order (paper uses 4).
    """
    if corpus_refs is None:
        corpus_refs = references

    hyp_tokens = hypothesis.lower().split()
    ref_tokens_list = [r.lower().split() for r in references]
    all_tokens_list = [r.lower().split() for r in corpus_refs]

    scores_n = []
    for n in range(1, n_max + 1):
        # IDF from corpus
        doc_freq: Counter = Counter()
        for ref_toks in all_tokens_list:
            doc_freq.update(set(_ngrams(ref_toks, n)))
        N_docs = len(all_tokens_list)
        idf = {ng: math.log((N_docs + 1.0) / (df + 1.0)) for ng, df in doc_freq.items()}

        def _tfidf_vec(tokens: list[str]) -> dict:
            grams = _ngrams(tokens, n)
            total = sum(grams.values()) or 1
            return {g: (cnt / total) * idf.get(g, 0.0) for g, cnt in grams.items()}

        hyp_vec = _tfidf_vec(hyp_tokens)

        ref_scores = []
        for ref_toks in ref_tokens_list:
            ref_vec = _tfidf_vec(ref_toks)
            keys = set(hyp_vec) | set(ref_vec)
            h = np.array([hyp_vec.get(k, 0.0) for k in keys])
            r = np.array([ref_vec.get(k, 0.0) for k in keys])
            norm_h = np.linalg.norm(h)
            norm_r = np.linalg.norm(r)
            if norm_h == 0 or norm_r == 0:
                ref_scores.append(0.0)
            else:
                ref_scores.append(float(np.dot(h, r) / (norm_h * norm_r)))
        scores_n.append(np.mean(ref_scores) if ref_scores else 0.0)

    return float(np.mean(scores_n)) * 10.0  # CIDEr-D scale


# ── BERTScore ─────────────────────────────────────────────────────────────────

def compute_bertscore(
    hypotheses: List[str],
    references_list: List[List[str]],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    device: Optional[str] = None,
) -> dict:
    """Compute BERTScore F1 (mean over samples).

    Args:
        hypotheses:      List of generated sentences.
        references_list: For each hypothesis, the list of references.
        model_type:      BERTScore model.
        device:          'cpu' or 'cuda'; None = auto-detect.

    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    from bert_score import score as bert_score_fn
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # bert_score expects a flat refs list; use the first reference per sample
    flat_refs = [refs[0] for refs in references_list]

    P, R, F = bert_score_fn(
        hypotheses,
        flat_refs,
        model_type=model_type,
        device=device,
        verbose=False,
    )
    return {
        "precision": float(P.mean()),
        "recall":    float(R.mean()),
        "f1":        float(F.mean()),
    }


# ── SimCSE ────────────────────────────────────────────────────────────────────

def compute_simcse(
    hypotheses: List[str],
    references_list: List[List[str]],
    model_name: str = "all-MiniLM-L6-v2",
) -> float:
    """Compute SimCSE: cosine similarity between sentence embeddings.

    Uses sentence-transformers as a proxy for SimCSE semantic similarity.
    Best score over references, averaged over samples.
    """
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)

    all_sents = hypotheses + [r for refs in references_list for r in refs]
    all_embs = model.encode(
        all_sents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=128,
        show_progress_bar=False,
    )

    n = len(hypotheses)
    hyp_embs = all_embs[:n]

    idx = n
    scores = []
    for i, refs in enumerate(references_list):
        ref_embs = all_embs[idx: idx + len(refs)]
        idx += len(refs)
        sims = (hyp_embs[i] @ ref_embs.T)   # cosine (normalised)
        scores.append(float(sims.max()))

    return float(np.mean(scores)) if scores else 0.0


# ── Aggregate scorer ──────────────────────────────────────────────────────────

def compute_all_metrics(
    hypotheses: List[str],
    references_list: List[List[str]],
    corpus_refs: Optional[List[str]] = None,
) -> dict:
    """Compute all metrics used in the paper.

    Args:
        hypotheses:      Generated motion descriptions (one per sample).
        references_list: Ground-truth descriptions for each sample.
        corpus_refs:     All reference strings in the eval set (for CIDEr IDF).

    Returns:
        Dict of metric_name → score (0–100 scale for ROUGE/BLEU/METEOR/CIDEr,
        0–1 for BERTScore F1 and SimCSE).
    """
    assert len(hypotheses) == len(references_list)

    rouge1_scores, rougeL_scores = [], []
    bleu1_scores, bleu4_scores = [], []
    meteor_scores = []
    cider_scores  = []

    all_corpus = corpus_refs if corpus_refs else [r for refs in references_list for r in refs]

    for hyp, refs in zip(hypotheses, references_list):
        r = compute_rouge(hyp, refs)
        rouge1_scores.append(r["rouge1"])
        rougeL_scores.append(r["rougeL"])

        b = compute_bleu(hyp, refs)
        bleu1_scores.append(b["bleu1"])
        bleu4_scores.append(b["bleu4"])

        meteor_scores.append(compute_meteor(hyp, refs))
        cider_scores.append(compute_cider(hyp, refs, corpus_refs=all_corpus))

    bert_result = compute_bertscore(hypotheses, references_list)
    simcse_score = compute_simcse(hypotheses, references_list)

    def _mean100(lst: list[float]) -> float:
        return round(float(np.mean(lst)) * 100, 2)

    return {
        "ROUGE-1":    _mean100(rouge1_scores),
        "ROUGE-L":    _mean100(rougeL_scores),
        "BLEU-1":     _mean100(bleu1_scores),
        "BLEU-4":     _mean100(bleu4_scores),
        "METEOR":     _mean100(meteor_scores),
        "CIDEr":      round(float(np.mean(cider_scores)), 2),
        "BERTScore":  round(bert_result["f1"] * 100, 2),
        "SimCSE":     round(simcse_score * 100, 2),
    }
