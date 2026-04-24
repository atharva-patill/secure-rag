import logging#for debugging and tracking
from pathlib import Path #connects all modules

from .embedding import embed_chunks
from .generator import generate_answer
from .masker import mask_text
from .pdf_loader import load_pdf ,chunk_text
from .retriever import retrieve
from .vector_store import VectorStore

logger = logging.getLogger(__name__)#logging specific to file


def load_data(file_path):#fn load data
    path = Path(file_path)#convert to path for validation checks
    if not path.exists():#validation
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():#validate if it's a file
        raise ValueError(f"Expected a file path, got: {path}")

    suffix = path.suffix.lower()#detects file type
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")#handles text
    if suffix == ".pdf":
        return load_pdf(path)#handles pdf

    raise ValueError(#error
        f"Unsupported file format: {suffix or 'unknown'}. Supported formats are .txt and .pdf."
    )


def build_rag(file_path, use_masking=True):
    text = load_data(file_path)
    if use_masking:
        text = mask_text(text)#masks before chunking
        #data flow : load_data() → mask_text() → chunk_text() → embed_chunks() → VectorStore
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No usable text chunks were found in the provided file.")

    embeddings = embed_chunks(chunks)#embeddings
    vector_store = VectorStore(embeddings)#vectore store
    return vector_store, chunks


def rag_answer(query: str, vector_store, chunks, mask_mode: str = "raw"):
    """
    RAG answer generation with configurable masking mode.
    
    mask_mode options:
    - "raw":  No masking anywhere (query stays raw, context stays raw)
    - "post": Mask context only before LLM (query stays raw)
    - "pre":  No masking here (pre-handled in build_rag with use_masking=True)
    
    Key rule: NEVER mask query in any mode.
    """
    context_chunks = retrieve(query, vector_store, chunks)
    context = "\n".join(context_chunks)

    if mask_mode == "post":
        context = mask_text(context)

    return generate_answer(context, query)
