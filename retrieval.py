from __future__ import annotations

from typing import Any

import psycopg
from pgvector.psycopg import register_vector

from config import HYBRID_KEYWORD_WEIGHT, HYBRID_VECTOR_WEIGHT
from sql_queries import HYBRID_BASE_SQL, SEARCH_CHUNKS_SQL


def _safe_websearch_query(q: str) -> str:
    q = " ".join(q.split())
    if len(q) < 2:
        return "deep learning"
    return q[:800]


def hybrid_search(
    conn_str: str,
    doc_name: str,
    query_vec,
    query_text: str,
    top_n: int,
) -> list[tuple[Any, ...]]:
    """
    Task 5 — combine vector similarity with full-text rank (weights from config).
    Returns rows: id, chapter, section, page_number, content, vec_sim, kw_rank, hybrid_score
    """
    qtext = _safe_websearch_query(query_text)

    with psycopg.connect(conn_str) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(
                HYBRID_BASE_SQL,
                (query_vec, qtext, doc_name),
            )
            rows = cur.fetchall()

    if not rows:
        return []

    kw_vals = [float(r[6] or 0) for r in rows]
    max_kw = max(kw_vals) if kw_vals else 0.0
    if max_kw <= 0:
        max_kw = 1e-9

    scored: list[tuple[Any, ...]] = []
    for r in rows:
        rid = r[0]
        chapter = r[1]
        section = r[2]
        page = r[3]
        content = r[4]
        vec_sim = float(r[5] or 0)
        kw_rank = float(r[6] or 0)
        norm_kw = kw_rank / max_kw
        hybrid = HYBRID_VECTOR_WEIGHT * vec_sim + HYBRID_KEYWORD_WEIGHT * norm_kw
        scored.append(
            (rid, chapter, section, page, content, vec_sim, kw_rank, hybrid)
        )

    scored.sort(key=lambda x: x[7], reverse=True)
    return scored[:top_n]


def retrieve_topk(conn_str, doc_name, query_vec, k):
    """Vector-only top-k (chapter, section, page_number, content, cosine_sim)."""
    with psycopg.connect(conn_str) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            cur.execute(SEARCH_CHUNKS_SQL, (query_vec, doc_name, query_vec, k))
            rows = cur.fetchall()
    return rows


def retrieve_for_answer(
    conn_str: str,
    doc_name: str,
    embedder,
    question: str,
    expanded_question: str,
    retrieval_candidates: int,
    rerank_top_k: int,
    reranker_model: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Full pipeline: embed expanded query → hybrid top-N → cross-encoder rerank → top-K dicts.
    """
    import numpy as np
    from rerank import rerank_results

    qvec = embedder.encode(
        [expanded_question], normalize_embeddings=True
    )[0].astype(np.float32)

    hybrid_rows = hybrid_search(
        conn_str, doc_name, qvec, expanded_question, retrieval_candidates
    )

    candidates: list[dict[str, Any]] = []
    for r in hybrid_rows:
        candidates.append(
            {
                "id": r[0],
                "chapter": r[1] or "",
                "section": r[2] or "",
                "page_number": r[3],
                "content": r[4] or "",
                "vector_similarity": r[5],
                "keyword_rank": r[6],
                "hybrid_score": r[7],
            }
        )

    reranked = rerank_results(
        question, candidates, reranker_model, rerank_top_k
    )
    return candidates, reranked
