# Advanced RAG System

A retrieval-augmented generation (RAG) system that answers questions about a long technical book (the *Deep Learning* textbook by Goodfellow, Bengio & Courville), grounded strictly in the retrieved text.

It goes beyond basic vector search with **section-aware chunking**, **hybrid search**, **query expansion**, and **cross-encoder reranking**, served through a **FastAPI** endpoint.

![System diagram](system.png)

## Pipeline

1. **Load** the book text from PDF (`pdf_loader.py`, PyMuPDF / pypdf).
2. **Chunk by structure** (`chunking.py`): split into sections, infer the chapter of each section, then cut sections into overlapping token chunks (450 tokens, 90 overlap) that keep section and page metadata.
3. **Embed and store** (`embeddings.py`, `ingest.py`): `all-MiniLM-L6-v2` embeddings stored in **PostgreSQL + pgvector**.
4. **Expand the query** (`query_expansion.py`): common deep learning acronyms are expanded (e.g. "SGD" → "stochastic gradient descent", "CNN" → "convolutional neural network") while keeping the original wording.
5. **Hybrid search** (`retrieval.py`): combines vector similarity (weight 0.7) with PostgreSQL full-text keyword search (weight 0.3) to build a pool of 20 candidates.
6. **Rerank** (`rerank.py`): a cross-encoder (`ms-marco-MiniLM-L-6-v2`) rescores the candidates and keeps the top 5.
7. **Generate** (`rag_api.py`): the LLM answers only from the top chunks. Any OpenAI-compatible API works (OpenAI, Groq, Ollama…), with Hugging Face Inference as a fallback.

## API

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/ask` | `{"question": "..."}` → grounded answer |

## Evaluation

`evaluate_rag.py` runs a fixed set of 10 deep learning questions (gradient descent, backpropagation, dropout, batch normalization, RBMs…) through retrieval and reranking only, and reports the expanded query, top section, page, and rerank score for each.

## Run locally

Requirements: Python 3.10+ and PostgreSQL with the `pgvector` extension.

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
PG_CONN_STR=postgresql://postgres:postgres@localhost:5432/rag
DOC_NAME=deep_learning_book
LLM_BASE_URL=https://api.groq.com/openai/v1   # any OpenAI-compatible endpoint
LLM_API_KEY=your_key
LLM_MODEL=your_model
```

Place the book's PDF or text in the project folder (not included in this repo), then:

```bash
python ingest.py                 # chunk, embed and store
uvicorn rag_api:app --reload     # start the API
python evaluate_rag.py           # optional: retrieval report
```

## Tech

Python · FastAPI · PostgreSQL · pgvector · sentence-transformers · cross-encoder reranking · OpenAI-compatible LLM APIs · PyMuPDF
