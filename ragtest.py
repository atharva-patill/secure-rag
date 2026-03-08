from rag_pipeline import rag_answer

query = "what does rag combine?"

response = rag_answer(query)

print("\nFinal Answer:\n")

for token in response:
    print(token, end="", flush=True)

print()