import logging#for debugging and tracking
import re
from pathlib import Path #connects all modules
#imports
from .detection import load_detector_stack
from .embedding import embed_chunks
from .generator import generate_answer
from .masker import mask_text
from .pdf_loader import chunk_record, load_pdf, split_into_records
from .policies import load_policy
from .query_router import AGGREGATE_K, DEFAULT_SINGLE_K, classify_query, routed_k
from .retriever import retrieve
from .vector_store import VectorStore

logger = logging.getLogger(__name__)#logging specific to file


def clean_input_text(text: str) -> str:#helper fn that takes RAW text and returns cleaned text
    cleaned_lines = []#empty arrary 
    skip_block = False

    for line in text.splitlines():#processe input text line by line 
        stripped = line.strip()#removes whitespace

        if skip_block:
            if not stripped:
                skip_block = False
            continue

        if stripped.startswith(("Context:", "Question:")):#skip condition
            skip_block = True
            continue

        if (#often appearing cli transcripts
            not stripped
            or stripped in {"[/INST]", "Thinking...", "Exiting", "Secure RAG Chat", "Type 'exit' to quit"}
            or stripped.startswith(("You:", "LLM:"))
            or re.fullmatch(r"[╭╰─│]+", stripped)
        ):
            if not stripped:
                cleaned_lines.append(line)
            continue

        cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def load_data(file_path):#fn load data
    path = Path(file_path)#convert to path for validation checks
    if not path.exists():#validation
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():#validate if it's a file
        raise ValueError(f"Expected a file path, got: {path}")

    suffix = path.suffix.lower()#detects file type
    if suffix == ".txt":
        return clean_input_text(path.read_text(encoding="utf-8"))#handles text
    if suffix == ".pdf":
        return clean_input_text(load_pdf(path))#handles pdf

    raise ValueError(#error
        f"Unsupported file format: {suffix or 'unknown'}. Supported formats are .txt and .pdf."
    )


def build_rag(file_path, policy=None, detector_profile="medical"):
    text = load_data(file_path)
    records = split_into_records(text)#Instead of treating the full document as one flat string, it splits it on blank lines, so each patient record becomes its own unit.
    chunks = []#building chunks record by record
    if policy is None:
        policy = load_policy("medical")
    detectors = load_detector_stack(detector_profile)

    for record in records:
        record = mask_text(record, policy=policy, detectors=detectors)#masks each record before chunking
        #data flow : load_data() → split_into_records() → mask_text() → chunk_record() → embed_chunks() → VectorStore
        chunks.extend(chunk_record(record))

    if not chunks:
        raise ValueError("No usable text chunks were found in the provided file.")

    embeddings = embed_chunks(chunks)#embeddings
    vector_store = VectorStore(embeddings)#vectore store
    return vector_store, chunks


def _truncate_at_stop_marker(text: str) -> str:
    stop_markers = ("\nContext:", "\nQuestion:", "[/INST]")
    positions = [text.find(m) for m in stop_markers if m in text]
    if positions:
        text = text[: min(positions)]
    return text.strip()


def rag_answer(query: str, vector_store, chunks, routed: bool = False, k: int | None = None):
    """
    Answer a query using retrieved context.

    Default production: routed=False -> K=DEFAULT_SINGLE_K (2).
    Routed mode: routed=True -> K = classify_query(query) ? AGGREGATE_K : DEFAULT_SINGLE_K
    (single->2, multi->20). Research-only k override: if k is not None, use that
    value directly (not exposed via CLI).
    """
    if k is not None:
        selected_k = int(k)
    elif routed:
        selected_k = routed_k(query, enabled=True)
    else:
        selected_k = DEFAULT_SINGLE_K
    context_chunks = retrieve(query, vector_store, chunks, k=selected_k)
    context = "\n\n".join(chunk for chunk in context_chunks if chunk)

    response = "".join(generate_answer(context, f"{query}\n\nAnswer:"))
    yield _truncate_at_stop_marker(response)
