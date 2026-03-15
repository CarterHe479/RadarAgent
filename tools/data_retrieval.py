"""
Data retrieval tools: get_motion_text and search_motions.

Text annotation format (texts/XXXXXX.txt), one line per description:
    original_description#POS_tagged_sentence#start_time#end_time

We always take index 0 (the plain English description).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
from typing import Optional, List

from config import TEXTS_DIR, SPLITS_DIR, RADAR_DATA_DIR


# ── helpers ───────────────────────────────────────────────────────────────────

def _read_descriptions(motion_id: str) -> list[str]:
    """Return plain-English descriptions for a motion ID.

    Tries both XXXXXX.txt and MXXXXXX.txt (mirrored) automatically.
    """
    for cand in [
        TEXTS_DIR / f"{motion_id}.txt",
        TEXTS_DIR / f"M{motion_id}.txt",
    ]:
        if cand.exists():
            lines = cand.read_text(encoding="utf-8").strip().splitlines()
            descs = []
            for line in lines:
                parts = line.split("#")
                if parts:
                    desc = parts[0].strip()
                    if desc:
                        descs.append(desc)
            return descs
    return []


# ── Tool 3: get_motion_text ───────────────────────────────────────────────────

def get_motion_text(motion_id: str) -> dict:
    """Return all text annotations for a motion from the HumanML3D dataset."""
    descs = _read_descriptions(motion_id)
    if not descs:
        return {
            "motion_id": motion_id,
            "descriptions": [],
            "num_descriptions": 0,
            "error": f"No text file found for motion_id={motion_id!r}",
        }
    return {
        "motion_id": motion_id,
        "descriptions": descs,
        "num_descriptions": len(descs),
    }


# ── Search index (lazy, cached in module-level variable) ─────────────────────

class _SearchIndex:
    """Sentence-embedding index over HumanML3D text descriptions."""

    def __init__(self) -> None:
        self._embeddings: Optional[np.ndarray] = None
        self._motion_ids: List[str] = []
        self._descriptions: List[List[str]] = []

    def _build(self) -> None:
        """Load all text files and build embeddings (first-call only)."""
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")

        # collect all motion IDs from the all.txt split file
        all_ids_path = SPLITS_DIR / "all.txt"
        if all_ids_path.exists():
            ids = [l.strip() for l in all_ids_path.read_text().splitlines() if l.strip()]
        else:
            # fall back: glob the texts directory
            ids = [p.stem for p in TEXTS_DIR.glob("*.txt") if not p.stem.startswith("M")]

        self._motion_ids = []
        self._descriptions = []
        first_descs: List[str] = []   # one representative sentence per motion

        for mid in ids:
            descs = _read_descriptions(mid)
            if not descs:
                continue
            self._motion_ids.append(mid)
            self._descriptions.append(descs)
            first_descs.append(descs[0])

        self._embeddings = model.encode(
            first_descs,
            batch_size=256,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)

    def search(self, query: str, top_k: int = 5) -> List[dict]:
        if self._embeddings is None:
            self._build()

        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        q_emb = model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype(np.float32)                     # (1, D)

        scores = (self._embeddings @ q_emb.T).flatten()   # cosine similarity
        top_idx = np.argsort(-scores)[:top_k]

        return [
            {
                "motion_id": self._motion_ids[i],
                "descriptions": self._descriptions[i],
                "similarity_score": round(float(scores[i]), 4),
            }
            for i in top_idx
        ]


_INDEX = _SearchIndex()


# ── Tool 4: search_motions ────────────────────────────────────────────────────

def search_motions(query: str, top_k: int = 5) -> dict:
    """Search for motions matching a natural language query.

    Uses sentence-transformer embeddings (all-MiniLM-L6-v2) and cosine
    similarity.  The index is built lazily on first call and cached.
    """
    results = _INDEX.search(query, top_k=top_k)
    return {
        "query": query,
        "results": results,
    }
