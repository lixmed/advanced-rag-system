import psycopg
from pgvector.psycopg import register_vector

from sql_queries import (
    INSERT_CHUNK_SQL,
    PREIVEW_CHUNKS_SQL,
    COUNT_CHUNKS_SQL,
    get_create_chunks_table_sql,
    DELETE_CHUNKS_SQL,
)


def init_db(conn_str: str, dim: int) -> None:
    with psycopg.connect(conn_str) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS rag_cv_chunks CASCADE")
            # get_create_chunks_table_sql may contain multiple statements
            for stmt in get_create_chunks_table_sql(dim).strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    cur.execute(stmt)

        conn.commit()


def upsert_chunks(conn_str, doc_name, chunks, vectors):
    with psycopg.connect(conn_str) as conn:
        register_vector(conn)

        with conn.cursor() as cur:
            for i, (chunk, vec) in enumerate(zip(chunks, vectors)):
                cur.execute(
                    INSERT_CHUNK_SQL,
                    (
                        doc_name,
                        chunk.get("chapter") or "",
                        chunk.get("section") or "",
                        chunk.get("page_number"),
                        i,
                        chunk["content"],
                        vec,
                    ),
                )

        conn.commit()


def delete_all_chunks(conn_str):
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(DELETE_CHUNKS_SQL)
        conn.commit()

    print("All chunks deleted from rag_cv_chunks")


def count_chunks(conn_str):
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(COUNT_CHUNKS_SQL)
            count = cur.fetchone()[0]

    print("Total stored chunks:", count)

    return count


def preview_chunks(conn_str):
    with psycopg.connect(conn_str) as conn:
        with conn.cursor() as cur:
            cur.execute(PREIVEW_CHUNKS_SQL)
            rows = cur.fetchall()

    print("\nStored Chunks Preview\n")

    for row in rows:
        doc, chapter, sec, page, content = row
        print("DOC:", doc)
        print("CHAPTER:", chapter)
        print("SECTION:", sec)
        print("PAGE:", page)
        print(content[:200])
        print("-" * 40)
