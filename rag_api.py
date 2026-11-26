import os
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

from config import (
    CROSS_ENCODER_MODEL,
    DOC_NAME,
    EMBED_MODEL_NAME,
    HF_API_KEY,
    PG_CONN_STR,
    RERANK_TOP_K,
    RETRIEVAL_CANDIDATES,
)
from embeddings import load_embedder
from query_expansion import expand_query
from retrieval import retrieve_for_answer

# Generation LLM: use LLM_* for any OpenAI-compatible API (OpenAI, Groq, Ollama, etc.).
# If LLM_BASE_URL is unset, falls back to Hugging Face Inference *Router* (needs token with Inference Providers permission).
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "").strip()

HF_BASE_URL = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")
HF_MODEL_NAME = os.getenv("HF_MODEL_NAME", "Qwen/Qwen3-Coder-Next:novita")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")


def generate_text(
    client: OpenAI, model: str, context: str, question: str
) -> str:
    prompt = f"""
You are an AI assistant that answers questions strictly using the provided CONTEXT.

Your task is to answer the QUESTION using ONLY the information that appears in the CONTEXT.

STRICT RULES:
- Use ONLY the information inside the CONTEXT.
- Do NOT use prior knowledge or external information.
- Do NOT guess or infer information that is not explicitly stated.
- If the answer is not clearly present in the CONTEXT, respond exactly with:
"The answer is not found in the provided context."
- Do NOT add explanations that are not supported by the CONTEXT.
- Prefer quoting or closely paraphrasing the CONTEXT.

Answer style:
- Keep the answer clear and concise.
- Use bullet points when listing multiple facts.

Output format:

Answer:
<answer based strictly on the context>

Source:
<chapter / section / page if available>

--------------------------------

CONTEXT:
{context}

QUESTION:
{question}
"""

    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1024,
    )
    return completion.choices[0].message.content.strip()


def _serialize_chunk(c: dict[str, Any]) -> dict[str, Any]:
    out = {
        "chapter": c.get("chapter") or "",
        "section": c.get("section") or "",
        "page_number": c.get("page_number"),
        "vector_similarity": c.get("vector_similarity"),
        "keyword_rank": c.get("keyword_rank"),
        "hybrid_score": c.get("hybrid_score"),
        "rerank_score": c.get("rerank_score"),
        "content_preview": (c.get("content") or "")[:1200],
    }
    return out


app = FastAPI(title="RAG FastAPI", version="2.0.0")

embedder = load_embedder(EMBED_MODEL_NAME)


def _build_llm_client_and_model() -> tuple[OpenAI, str] | tuple[None, None]:
    """Return (OpenAI client, model id). Prefer LLM_* env over HF Router."""
    if LLM_BASE_URL:
        key = LLM_API_KEY if LLM_API_KEY else "ollama"
        model = LLM_MODEL or "llama3.2"
        return OpenAI(base_url=LLM_BASE_URL.rstrip("/"), api_key=key), model

    api_key = os.getenv("HF_API_KEY") or os.getenv("HF_TOKEN") or HF_API_KEY
    if not api_key or api_key.strip() == "" or api_key == "hf_...":
        return None, None
    return (
        OpenAI(base_url=HF_BASE_URL.rstrip("/"), api_key=api_key),
        HF_MODEL_NAME,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask")
def ask_question(payload: AskRequest) -> dict[str, Any]:
    client, llm_model = _build_llm_client_and_model()
    if client is None or not llm_model:
        raise HTTPException(
            status_code=500,
            detail=(
                "No LLM configured. Set LLM_BASE_URL + LLM_MODEL (and LLM_API_KEY if needed), "
                "or set HF_API_KEY / HF_TOKEN for Hugging Face Router."
            ),
        )

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        expanded = expand_query(question)

        _pool, reranked = retrieve_for_answer(
            PG_CONN_STR,
            DOC_NAME,
            embedder,
            question,
            expanded,
            RETRIEVAL_CANDIDATES,
            RERANK_TOP_K,
            CROSS_ENCODER_MODEL,
        )

        top_for_llm = [_serialize_chunk(c) for c in reranked]

        if reranked:
            context = "\n\n".join(
                [
                    (
                        f"CHAPTER: {c.get('chapter') or 'n/a'}\n"
                        f"SECTION: {c.get('section') or 'n/a'}\n"
                        f"PAGE: {c.get('page_number') if c.get('page_number') is not None else 'n/a'}\n"
                        f"{c.get('content', '')}"
                    )
                    for c in reranked
                ]
            )
            generated_answer = generate_text(
                client=client, model=llm_model, context=context, question=question
            )
        else:
            generated_answer = "The answer is not found in the provided context."

        return {
            "question": question,
            "expanded_query": expanded,
            "Generated Answer": generated_answer,
            "Top chunks after hybrid search (pool)": [_serialize_chunk(c) for c in _pool[:8]],
            "Final context chunks (reranked for LLM)": top_for_llm,
        }
    except HTTPException:
        raise
    except Exception as exc:
        msg = str(exc)
        if "401" in msg or "Invalid username or password" in msg:
            raise HTTPException(
                status_code=401,
                detail=(
                    "Hugging Face authentication failed. Verify token value/scope and "
                    "use only raw token in env var (without 'Bearer ')."
                ),
            ) from exc
        if "403" in msg and (
            "Inference Providers" in msg or "inference" in msg.lower()
        ):
            raise HTTPException(
                status_code=502,
                detail=(
                    "Hugging Face returned 403: this token cannot use Inference Providers "
                    "(Router). Fix: at https://huggingface.co/settings/tokens create a token "
                    "with permission to use Inference / Inference Providers, or use a different "
                    "account token. Alternative: set LLM_BASE_URL + LLM_MODEL to an "
                    "OpenAI-compatible server (e.g. Ollama at http://localhost:11434/v1 with "
                    "LLM_MODEL=llama3.2)."
                ),
            ) from exc
        raise HTTPException(status_code=500, detail=f"RAG pipeline failed: {exc}") from exc
