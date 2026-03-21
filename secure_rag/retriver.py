import numpy as np
from .embedding import embed_chunks#from embedding module

def retrieve(query, vector_store, chunks, k=2):#params==> user input,faiss index,original text chunks,number of results
    #query->embedding
    query_vector = embed_chunks([query])
    query_vector = np.array(query_vector).astype("float32")
    #searches vectore store
    distances,indices=vector_store.search(query_vector,k)
    #reuturns matching chunks
    return [chunks[i] for i in indices]