from sentence_transformers import SentenceTransformer#embedding mode == converts text to vectors
import numpy as np

_embedder = None
_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"


def get_embedder(model_name: str = _DEFAULT_MODEL_NAME) -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(model_name)
    return _embedder


def embed_chunks(chunks):
    embedder = get_embedder()
    embeddings = embedder.encode(chunks)
    return np.array(embeddings).astype("float32")
