import numpy as np
from secure_rag.vector_store import VectorStore


def test_vector_search():
    embeddings = np.random.rand(5, 384).astype("float32")
    store = VectorStore(embeddings)

    query = np.random.rand(1, 384).astype("float32")
    distances, indices = store.search(query, k=2)

    assert len(indices) == 2