def get_create_chunks_table_sql(dim: int) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS rag_cv_chunks (
    id SERIAL PRIMARY KEY,
    doc_name TEXT NOT NULL,
    chapter TEXT,
    section TEXT,
    page_number INT,
    chunk_index INT,
    content TEXT NOT NULL,
    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', coalesce(content, ''))) STORED,
    embedding VECTOR({dim})
);
CREATE INDEX IF NOT EXISTS rag_cv_chunks_doc_name_idx ON rag_cv_chunks (doc_name);
CREATE INDEX IF NOT EXISTS rag_cv_chunks_content_tsv_idx ON rag_cv_chunks USING GIN (content_tsv);
"""


INSERT_CHUNK_SQL = """
INSERT INTO rag_cv_chunks
(doc_name, chapter, section, page_number, chunk_index, content, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s)
"""

# Vector-only: section, content, sim first (notebook-friendly), then chapter, page
SEARCH_CHUNKS_SQL = """
SELECT section, content, 1 - (embedding <=> %s::vector) AS cosine_sim,
       chapter, page_number
FROM rag_cv_chunks
WHERE doc_name = %s
ORDER BY embedding <=> %s::vector
LIMIT %s
"""

# Hybrid: full doc scan, combine scores in Python (Task 5)
HYBRID_BASE_SQL = """
SELECT id, chapter, section, page_number, content,
       1 - (embedding <=> %s::vector) AS vec_sim,
       ts_rank_cd(content_tsv, websearch_to_tsquery('english', %s)) AS kw_rank
FROM rag_cv_chunks
WHERE doc_name = %s
"""

limit = 5
DELETE_CHUNKS_SQL = "DELETE FROM rag_cv_chunks;"
COUNT_CHUNKS_SQL = "SELECT COUNT(*) FROM rag_cv_chunks"

PREIVEW_CHUNKS_SQL = f"""SELECT doc_name, chapter, section, page_number, content
            FROM rag_cv_chunks
            LIMIT {limit}
            """
