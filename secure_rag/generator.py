import os#reads env variables

from dotenv import load_dotenv#loads .env
from openai import OpenAI#openAI sdk -> hugging face router

_client = None
_DEFAULT_BASE_URL = "https://router.huggingface.co/v1"#HF router end point 
_DEFAULT_MODEL = "HuggingFaceH4/zephyr-7b-beta:featherless-ai"#llm model


def get_client() -> OpenAI:#creates LLM client when called
    global _client#allowing reuse
    if _client is None:#lazyloading
        load_dotenv()#reads dotenv
        api_key = os.getenv("HF_API_KEY")#gets api key
        if not api_key:#validates the API key
            raise RuntimeError(
                "HF_API_KEY is not set. Set the environment variable before generating answers."
            )
        _client = OpenAI(#Creating Client
            base_url=os.getenv("HF_BASE_URL", _DEFAULT_BASE_URL),
            api_key=api_key,
        )
    return _client


def generate_answer(context, query):
    client = get_client()
    messages = [    #system prompt 
        {
            "role": "system",
            "content": (
                "You are a RAG assistant.\n"
                "You MUST answer ONLY using the provided context.\n"
                "If the answer is present in the context, use it.\n"
                "Do not use outside knowledge.\n"
                "If the answer is not present, say 'I don't know'."
            ),
        },
        {
            "role": "user",     #passing retrived text and user query
            "content": f"Context:\n{context}\n\nQuestion:\n{query}",
        },
    ]

    completion = client.chat.completions.create(    #model call
        model=os.getenv("HF_MODEL", _DEFAULT_MODEL),
        messages=messages,#prompt
        max_tokens=200,#response limit
        temperature=0.3,#minimize randomness
        stream=True,#response == token->token(streaming)
    )
    buffer = ""

    for chunk in completion:
        if len(chunk.choices) == 0:
            continue

        content = chunk.choices[0].delta.content if chunk.choices[0].delta else None
        if content:
            buffer += content

    cleaned = (
        buffer.replace("Answer:", "")
        .replace("Final Answer:", "")
        .replace("?", "", 1)
        .strip()
    )

    for token in cleaned.split(" "):
        yield token + " "