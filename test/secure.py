from secure_rag import build_rag, rag_answer

vector_store, chunks = build_rag("data.txt")

print(rag_answer("what is rag?", vector_store, chunks))

