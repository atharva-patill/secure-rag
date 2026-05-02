#Pytest automatically discovers test files and test functions. 
from secure_rag.pdf_loader import chunk_record, chunk_text, split_into_records


def test_chunking_basic():#chunk test fn 
    text = "word " * 1000  # 1000 words
    chunks = chunk_text(text, chunk_size=200)

    assert len(chunks) > 1
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)


def test_split_into_records_preserves_blank_line_boundaries():
    text = "Arjun Sharma has fever.\n\nPriya Patel has chest pain.\n\nSneha Reddy uses inhaler."

    records = split_into_records(text)

    assert records == [
        "Arjun Sharma has fever.",
        "Priya Patel has chest pain.",
        "Sneha Reddy uses inhaler.",
    ]


def test_chunk_record_keeps_short_record_intact():
    record = "Arjun Sharma has persistent fever."

    assert chunk_record(record, chunk_size=10) == [record]


def test_chunk_record_splits_long_record_only_within_record():
    record = "word " * 520

    chunks = chunk_record(record, chunk_size=200, overlap=50)

    assert len(chunks) > 1
    assert all(len(chunk.split()) <= 200 for chunk in chunks)
