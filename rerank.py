from __future__ import annotations

from typing import Any, Sequence

from sentence_transformers import CrossEncoder

_rerankers: dict[str, CrossEncoder] = {}


def get_reranker(model_name: str) -> CrossEncoder:
    if model_name not in _rerankers:
        _rerankers[model_name] = CrossEncoder(model_name)
    return _rerankers[model_name]


def rerank_results(
    question: str,
    candidates: Sequence[dict[str, Any]],
    model_name: str,
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Task 6 — cross-encoder reranking on (question, chunk) pairs.
    `candidates` must include key 'content'.
    """
    if not candidates:
        return []

    reranker = get_reranker(model_name)
    pairs = [[question, c["content"]] for c in candidates]
    scores = reranker.predict(pairs, show_progress_bar=False)

    scored = []
    for c, s in zip(candidates, scores):
        row = dict(c)
        row["rerank_score"] = float(s)
        scored.append(row)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored[:top_k]
