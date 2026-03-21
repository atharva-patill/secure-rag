import logging#for debugging and tracking
from pathlib import Path #connects all modules

from .embedding import embed_chunks
from .generator import generate_answer
from .masker import mask_text
from .pdf_loader import load_pdf
from .retriver import retrieve
from .vector_store import VectorStore

logger = logging.getLogger(__name__)#logging specific to file


def load_data(file_path):
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise ValueError(f"Expected a file path, got: {path}")

    suffix = path.suffix.lower()
    if suffix == ".txt":
        return path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return load_pdf(path)

    raise ValueError(
        f"Unsupported file format: {suffix or 'unknown'}. Supported formats are .txt and .pdf."
    )


def build_rag(file_path):
    text = load_data(file_path)
    chunks = [line.strip() for line in text.splitlines() if line.strip()]
    if not chunks:
        raise ValueError("No usable text chunks were found in the provided file.")

    embeddings = embed_chunks(chunks)
    vector_store = VectorStore(embeddings)
    return vector_store, chunks


def rag_answer(query: str, vector_store, chunks):
    masked_query = mask_text(query)

    logger.info("Original Query: %s", query)
    logger.info("Masked Query: %s", masked_query)

    context_chunks = retrieve(masked_query, vector_store, chunks)
    context = "\n".join(context_chunks)
    context = mask_text(context)

    logger.info("Retrieved Context: %s", context)
    return generate_answer(context, masked_query)
