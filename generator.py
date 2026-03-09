import os
from openai import OpenAI
from dotenv import load_dotenv


# Load environment

load_dotenv()
client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.environ["HF_API_KEY"]
)
#llm build prompt + steaming reaponse
def generate_answer(context, query):
    messages = [
    {
        "role": "system",
        "content": (
            "You are a strict RAG assistant.\n"
            "Answer ONLY using the provided context.\n"
            "If the answer is not explicitly present, reply exactly with: I don't know.\n"
            "Do not explain anything extra.\n"
        )
    },
    {
        "role": "user",
        "content": f"Context:\n{context}\n\nQuestion:\n{query}"
    }
]
#completion model used
    completion = client.chat.completions.create(
        model="HuggingFaceH4/zephyr-7b-beta:featherless-ai",
        messages=messages,
        max_tokens=200,
        temperature=0.3,
        stream=True         #enable streaming
    )


    # Stream tokens back (using chunks instead of big JSON object)
    for chunk in completion:
        if len(chunk.choices) == 0:
            continue

        content = chunk.choices[0].delta.content if chunk.choices[0].delta else None

        if content:
            yield content