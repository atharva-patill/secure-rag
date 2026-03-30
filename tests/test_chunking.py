from secure_rag.pdf_loader import chunk_text


def test_chunking_basic():#chunk test fn 
    text = "word " * 1000  # 1000 words
    chunks = chunk_text(text, chunk_size=200)

    assert len(chunks) > 1
    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)