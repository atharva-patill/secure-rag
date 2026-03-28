
from sentence_transformers import SentenceTransformer#embedding mode == converts text to vectorsha
import numpy as np#vectore store

_embedder = None
_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"#pre-trained embedding model


def get_embedder(model_name: str = _DEFAULT_MODEL_NAME) -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(model_name)
    return _embedder


def embed_chunks(chunks):#embedd chunks into float32 format
    embedder = get_embedder()
    embeddings = embedder.encode(chunks)#text > vector
    return np.array(embeddings).astype("float32")
