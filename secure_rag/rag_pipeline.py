import logging
import numpy as np

from embedding import embed_chunks
from vector_store import VectorStore
from retriver import retrieve
from masker import mask_text
from generator import generate_answer
from pdf_loader import load_pdf,chunk_text


# Logging 

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


#made data loading configurbale instead of hardcoding
def load_data(file_path):
    with open("data.txt", "r") as f:
        return f.read()




#building rag system using provided path
def build_rag(file_path):
    text=load_data(file_path)
    chunks=[line for line in text.split("\n") if line.strip()]
    embeddings =embed_chunks(chunks)
    embeddings=np.array(embeddings.astype("float32"))

    vectore_store=VectorStore(embeddings)
    return vectore_store,chunks




# RAG Pipeline

def rag_answer(query: str,vector_store,chunks):

    masked_query = mask_text(query)

    # Log original + masked query
    logging.info(f"Original Query: {query}")
    logging.info(f"Masked Query: {masked_query}")

    # Retrieval
    context_chunks = retrieve(masked_query, vector_store, chunks)
    context = "\n".join(context_chunks)
    context = mask_text(context)

    logging.info(f"Retrieved Context: {context}")

    # Generation
    return generate_answer(context, masked_query)