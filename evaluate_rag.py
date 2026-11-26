"""Bonus: run a fixed question list through the pipeline (no LLM — retrieval only)."""
from __future__ import annotations

import json

import numpy as np

from config import (
    CROSS_ENCODER_MODEL,
    DOC_NAME,
    EMBED_MODEL_NAME,
    PG_CONN_STR,
    RETRIEVAL_CANDIDATES,
    RERANK_TOP_K,
)
from embeddings import load_embedder
from query_expansion import expand_query
from retrieval import retrieve_for_answer

QUESTIONS = [
    "What is gradient descent?",
    "Explain stochastic gradient descent.",
    "What is representation learning?",
    "What are convolutional networks?",
    "What is backpropagation?",
    "What chapter discusses autoencoders?",
    "What is the vanishing gradient problem?",
    "Explain dropout regularization.",
    "What is batch normalization?",
    "What is a restricted Boltzmann machine?",
]


def main() -> None:
    embedder = load_embedder(EMBED_MODEL_NAME)
    report = []
    for q in QUESTIONS:
        exp = expand_query(q)
        pool, reranked = retrieve_for_answer(
            PG_CONN_STR,
            DOC_NAME,
            embedder,
            q,
            exp,
            RETRIEVAL_CANDIDATES,
            RERANK_TOP_K,
            CROSS_ENCODER_MODEL,
        )
        top = reranked[0] if reranked else {}
        report.append(
            {
                "question": q,
                "expanded": exp,
                "top_section": top.get("section"),
                "top_page": top.get("page_number"),
                "rerank_score": top.get("rerank_score"),
            }
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
