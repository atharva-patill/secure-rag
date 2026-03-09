from rag_pipeline import rag_answer

query = "8484848488 what is RAG and RAG decoding?"

response = rag_answer(query)

print("\nFinal Answer:\n")

for token in response:
    print(token, end="", flush=True)

print()