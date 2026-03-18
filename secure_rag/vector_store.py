import faiss
import numpy as np

class VectorStore:
    def __init__(self, embeddings):
        embeddings=np.array(embeddings).astype("float32")
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

    def search(self, query_vector, k=2):
        query_vector=np.array(query_vector).astype("float32")
        distances, indices = self.index.search(query_vector, k)
        return distances, indices[0] 