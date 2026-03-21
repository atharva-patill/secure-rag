import faiss#Facebook AI similarity search
import numpy as np


class VectorStore:
    def __init__(self, embeddings):
        embeddings = np.array(embeddings).astype("float32")
        if embeddings.size == 0:
            raise ValueError("Embeddings array cannot be empty.")
        if embeddings.ndim != 2:
            raise ValueError("Embeddings must be a 2D array.")

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

    def search(self, query_vector, k=2):
        query_vector = np.array(query_vector).astype("float32")
        if query_vector.ndim != 2:
            raise ValueError("Query vector must be a 2D array.")

        limit = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, limit)
        return distances, indices[0]
