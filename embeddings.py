import numpy as np
from sentence_transformers import SentenceTransformer

def load_embedder(model_name):

    model = SentenceTransformer(model_name)

    return model


def embed_chunks(chunks, embedder):

    vecs = embedder.encode(
        chunks,
        batch_size=32,
        normalize_embeddings=True
    )

    return np.asarray(vecs, dtype=np.float32)