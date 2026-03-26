import faiss#Facebook AI similarity search
import numpy as np#handles vector formatting and validation


class VectorStore:
    def __init__(self, embeddings):
        embeddings = np.array(embeddings).astype("float32")#validates correct format 
        if embeddings.size == 0:#empty check
            raise ValueError("Embeddings array cannot be empty.")
        if embeddings.ndim != 2:#dimension check 
            raise ValueError("Embeddings must be a 2D array.")

        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

    def search(self, query_vector, k=2):#search fn 
        query_vector = np.array(query_vector).astype("float32")
        if query_vector.ndim != 2:
            raise ValueError("Query vector must be a 2D array.")

        limit = min(k, self.index.ntotal)
        distances, indices = self.index.search(query_vector, limit)
        return distances, indices[0].tolist()
