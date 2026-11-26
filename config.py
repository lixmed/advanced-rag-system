import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
PG_CONN_STR = os.getenv(
    "PG_CONN_STR",
    "postgresql://postgres:postgres@localhost:5432/rag",
)

# Task 3 — chunking (lab: 450 / 90)
CHUNK_SIZE = 450
CHUNK_OVERLAP = 90

# Legacy alias used by some notebooks
TOP_K = 5

# Retrieval pipeline
RETRIEVAL_CANDIDATES = 20  # vector + hybrid pool size before rerank
RERANK_TOP_K = 5  # chunks passed to the LLM
HYBRID_VECTOR_WEIGHT = float(os.getenv("HYBRID_VECTOR_WEIGHT", "0.7"))
HYBRID_KEYWORD_WEIGHT = float(os.getenv("HYBRID_KEYWORD_WEIGHT", "0.3"))

CROSS_ENCODER_MODEL = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

DOC_NAME = os.getenv("DOC_NAME", "deep_learning_book")

HF_API_KEY = os.getenv("HF_API_KEY", "")
