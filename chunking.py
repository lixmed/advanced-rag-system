from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from transformers import AutoTokenizer

# ============================================================
# 1️⃣ SPLIT SECTIONS (optional [PAGE n] markers from PDF extract)
# ============================================================

SECTION_HEADER = re.compile(r"^[A-Z][A-Z\s]{3,}$")
PAGE_MARKER = re.compile(r"^\[PAGE\s+(\d+)\]\s*$", re.IGNORECASE)


def infer_chapter(section: str) -> str:
    """Rough chapter label from section title (lab Task 4 metadata)."""
    s = (section or "").strip()
    m = re.search(r"chapter\s*(\d+)", s, re.IGNORECASE)
    if m:
        return f"Chapter {m.group(1)}"
    m = re.match(r"^(\d{1,2})\s", s)
    if m:
        return f"Chapter {m.group(1)}"
    return ""


def split_sections(text: str) -> List[Dict[str, Any]]:
    sections: List[Dict[str, Any]] = []
    current = "general"
    buffer: List[str] = []
    current_page = 1
    buffer_start_page: Optional[int] = None

    for line in text.splitlines():
        line_strip = line.strip()

        pm = PAGE_MARKER.match(line_strip)
        if pm:
            current_page = int(pm.group(1))
            continue

        if SECTION_HEADER.match(line_strip) and len(line_strip) >= 10:
            if buffer:
                sections.append(
                    {
                        "chapter": "",
                        "section": current,
                        "content": "\n".join(buffer).strip(),
                        "page_number": buffer_start_page,
                    }
                )

            current = line_strip.lower()
            buffer = []
            buffer_start_page = None
            continue

        if not buffer:
            buffer_start_page = current_page
        buffer.append(line)

    if buffer:
        sections.append(
            {
                "chapter": "",
                "section": current,
                "content": "\n".join(buffer).strip(),
                "page_number": buffer_start_page,
            }
        )

    return sections


# ============================================================
# 2️⃣ SPLIT ENTRIES (paragraphs)
# ============================================================


def split_section_entries(section_name: str, content: str) -> List[str]:
    paragraphs = [
        p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()
    ]
    return paragraphs if paragraphs else [content]


# ============================================================
# 3️⃣ CHUNK TEXT
# ============================================================


def chunk_text(text, tokenizer, chunk_size=300, overlap=50):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(tokens), step):
        window = tokens[i : i + chunk_size]
        if not window:
            break
        chunk = tokenizer.decode(window)
        chunks.append(chunk.strip())
        if i + chunk_size >= len(tokens):
            break
    return chunks


# ============================================================
# 4️⃣ BUILD CHUNKS
# ============================================================


def build_chunks(
    sections,
    embed_model_name,
    chunk_size=300,
    overlap=50,
    tokenizer=None,
):
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(embed_model_name)

    chunks = []

    for sec in sections:
        section_name = sec["section"]
        text = sec["content"]
        base_page: Optional[int] = sec.get("page_number")
        chapter = sec.get("chapter") or infer_chapter(section_name)
        if not chapter:
            chapter = section_name.title()

        entries = split_section_entries(section_name, text)

        for entry_id, entry in enumerate(entries):
            sub_chunks = chunk_text(entry, tokenizer, chunk_size, overlap)

            for idx, chunk in enumerate(sub_chunks):
                token_len = len(
                    tokenizer.encode(chunk, add_special_tokens=False)
                )

                chunks.append(
                    {
                        "chapter": chapter,
                        "section": section_name,
                        "page_number": base_page,
                        "entry_id": entry_id,
                        "chunk_in_entry": idx,
                        "token_len": token_len,
                        "content": chunk,
                    }
                )

    return chunks


# ============================================================
# 5️⃣ UPSERT TO DATABASE
# ============================================================


def upsert_chunks(conn_str, doc_name, chunks, vectors):
    from db import upsert_chunks as db_upsert_chunks

    return db_upsert_chunks(conn_str, doc_name, chunks, vectors)
