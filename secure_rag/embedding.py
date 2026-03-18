from sentence_transformers import SentenceTransformer
import numpy as np

# Load embedding model

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def embed_chunks(chunks):
    embeddings=embedder.encode(chunks)

    return np.array(embeddings).astype("float32")