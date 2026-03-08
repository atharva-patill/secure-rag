from pdf_loader import load_pdf,chunk_text
from embedding import embed_chunks
from  vector_store import VectorStore

def ingest_pdf(pdf_path):
    #loads the pdf
    text=load_pdf(pdf_path)
    #splits into chunks
    chunks=chunk_text(text)
    #embed splitted chunks
    embeddings=embed_chunks(chunks)
    #creating vector store
    vector_store=VectorStore(embeddings)

    print("PDF sucessfully ingested")
    print("Total chunks:",len(chunks))

    return vector_store,chunks