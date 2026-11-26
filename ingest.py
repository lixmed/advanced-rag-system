"""
Re-ingest the book with current CHUNK_SIZE / CHUNK_OVERLAP and metadata (Task 3–4).
Prefers Deep_Learning_Book_2.txt ([PAGE n] markers) when present.
Run: python ingest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from config import CHUNK_OVERLAP, CHUNK_SIZE, DOC_NAME, EMBED_MODEL_NAME, PG_CONN_STR
from chunking import build_chunks, split_sections
from db import init_db, upsert_chunks
from embeddings import embed_chunks, load_embedder


def _pick_book_text(project_dir: Path) -> str:
    candidates = [
        project_dir / "Deep_Learning_Book_2.txt",
        project_dir / "Deep_Learning_Book.txt",
    ]
    for p in candidates:
        if p.is_file():
            print(f"Using book file: {p.name}")
            return p.read_text(encoding="utf-8", errors="replace")
    raise FileNotFoundError(
        "No Deep_Learning_Book_2.txt or Deep_Learning_Book.txt in project folder."
    )


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    text = _pick_book_text(project_dir)

    print("Loading embedding model...")
    embedder = load_embedder(EMBED_MODEL_NAME)
    dim = embedder.get_embedding_dimension()

    print("Initializing database (drops existing rag_cv_chunks)...")
    init_db(PG_CONN_STR, dim)

    print("Splitting sections...")
    sections = split_sections(text)
    print(f"   Sections: {len(sections)}")

    print(f"Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = build_chunks(
        sections,
        EMBED_MODEL_NAME,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP,
    )
    print(f"   Chunks: {len(chunks)}")

    texts = [c["content"] for c in chunks]
    print("Embedding...")
    vectors = embed_chunks(texts, embedder)

    print("Storing in PostgreSQL...")
    upsert_chunks(PG_CONN_STR, DOC_NAME, chunks, vectors)
    print("Ingest complete for doc_name =", DOC_NAME)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Error:", e, file=sys.stderr)
        sys.exit(1)
