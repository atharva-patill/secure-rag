import numpy as np
from .embedding import embed_chunks#from embedding module

def retrieve(query, vector_store, chunks, k=2):#params==> user input,faiss index,original text chunks,number of results
    #query->vector
    query_vector = embed_chunks([query])
    query_vector = np.array(query_vector).astype("float32")#ensures correct format(beacuse faiss require float 32 vectors)
    #searches vectore store
    distances,indices=vector_store.search(query_vector,k)
    #reuturns matching indices to  raw text chunks only
    return [str(chunks[int(i)]).strip() for i in indices if int(i) >= 0]
