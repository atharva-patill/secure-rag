import json
import os#reads env variables
from urllib import request

from dotenv import load_dotenv#loads .env
from openai import OpenAI#openAI sdk -> hugging face router

_client = None
_DEFAULT_BASE_URL = "https://router.huggingface.co/v1"#HF router end point 
_DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"#llm model
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()


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


def _build_ollama_prompt(context, query) -> str:
    return (
        "You are a RAG assistant. Answer only from the provided context. "
        "If the answer is not present, say 'I don't know'.\n\n"
        f"Context:\n{context}\n\nQuestion:\n{query}"
    )


def _build_hf_messages(context, query):
    system_content = (
        "You are Secure RAG, a privacy-aware clinical retrieval assistant.\n\n"
        "Your responses MUST be based ONLY on the information provided in the retrieved context.\n\n"
        "Core Rules:\n"
        "1. Never use outside medical knowledge.\n"
        "2. Never infer missing information.\n"
        "3. Never fabricate diagnoses, treatments, medications, or patient details.\n"
        "4. If the answer is not explicitly present in the retrieved context, reply exactly:\n"
        "\"I don't know.\"\n\n"
        "Clinical Reasoning:\n"
        "Before answering, determine whether the user's question requests:\n"
        "- A SINGLE patient-specific answer, or\n"
        "- An AGGREGATE answer involving multiple patients or records.\n\n"
        "Aggregate Queries:\n"
        "If the user asks questions such as:\n"
        "- Which patients...\n"
        "- Who all...\n"
        "- List all...\n"
        "- How many...\n"
        "- Which records...\n"
        "- Summarize...\n\n"
        "then return ALL matching records found in the retrieved context. "
        "Do NOT ask for clarification simply because multiple patients match.\n\n"
        "Patient-Specific Queries:\n"
        "If the user's question expects ONE patient-specific answer (for example a treatment, diagnosis, medication, follow-up plan, or contact information) "
        "and the retrieved context contains multiple patient records with different valid answers, DO NOT choose one arbitrarily.\n\n"
        "Instead:\n"
        "- Explain that multiple patient records match the query.\n"
        "- Ask the user to identify the intended patient.\n"
        "- Request only identifiers present in the retrieved records, such as:\n"
        "  • Patient name\n"
        "  • Patient ID\n"
        "  • Age\n"
        "  • Admission date\n\n"
        "Never:\n"
        "- Merge treatments from different patients.\n"
        "- Combine diagnoses from multiple patients.\n"
        "- Guess which patient the user intended.\n"
        "- Recommend a treatment that is not explicitly associated with the identified patient.\n\n"
        "If exactly one patient record answers the question, answer normally using only that record."
    )
    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"},
    ]


_LAST_OLLAMA_METADATA: dict = {}
_LAST_OLLAMA_METADATA_FIELDS = [
    "done",
    "done_reason",
    "eval_count",
    "prompt_eval_count",
    "total_duration",
    "load_duration",
    "prompt_eval_duration",
    "eval_duration",
    "model",
    "created_at",
]


def get_last_ollama_metadata() -> dict:
    """Return a copy of the last Ollama streaming final metadata (research instrumentation)."""
    return dict(_LAST_OLLAMA_METADATA)


def _clear_last_ollama_metadata():
    global _LAST_OLLAMA_METADATA
    _LAST_OLLAMA_METADATA = {}


def _generate_ollama(context, query, num_predict=None, temperature=None):
    global _LAST_OLLAMA_METADATA
    _LAST_OLLAMA_METADATA = {}
    prompt = _build_ollama_prompt(context, query)
    data = {
        "model": os.getenv("OLLAMA_MODEL", "llama3.2"),#system prompt for local ollama model
        "prompt": prompt,
        "stream": True,
    }
    # Research-only budget control: expose num_predict and temperature without altering production defaults.
    options = {}
    if num_predict is not None:
        options["num_predict"] = int(num_predict)
        # also set top-level for older Ollama compatibility
        data["num_predict"] = int(num_predict)
    if temperature is not None:
        options["temperature"] = float(temperature)
        data["temperature"] = float(temperature)
    if options:
        data["options"] = options
    payload = json.dumps(data).encode("utf-8")
    req = request.Request(
        os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") + "/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    # Capture final streaming metadata (done, done_reason, eval_count, etc.) separately from text.
    # The text is still yielded exactly as before to preserve production behavior.
    final_metadata: dict = {}
    with request.urlopen(req) as response:
        for line in response:
            if not line:
                continue
            chunk = json.loads(line.decode("utf-8"))
            # Capture provider metadata when present (final chunk has done=True).
            # We record whatever fields are present without assuming completeness.
            if chunk.get("done") is not None or chunk.get("done_reason") is not None or chunk.get("eval_count") is not None:
                # Keep the most recent chunk that looks like a terminal/metadata chunk
                for k in ["done", "done_reason", "eval_count", "prompt_eval_count", "total_duration", "load_duration", "prompt_eval_duration", "eval_duration", "model", "created_at"]:
                    if k in chunk:
                        final_metadata[k] = chunk[k]
                # Also capture any other done-related keys that may appear
                if "done" in chunk:
                    final_metadata["done"] = chunk["done"]
                if "done_reason" in chunk:
                    final_metadata["done_reason"] = chunk["done_reason"]
            # Also capture metadata from final done=True chunk even if it has no "response"
            if chunk.get("done") is True:
                for k in ["done", "done_reason", "eval_count", "prompt_eval_count", "total_duration", "load_duration", "prompt_eval_duration", "eval_duration"]:
                    if k in chunk and k not in final_metadata:
                        final_metadata[k] = chunk[k]
                # If no done_reason but done==True, we still store done
            if chunk.get("response"):
                yield chunk["response"]
    # Store separately for research instrumentation (outside the text stream)
    _LAST_OLLAMA_METADATA = dict(final_metadata)

def generate_answer(context, query, max_tokens=None, num_predict=None, temperature=None):
    if LLM_PROVIDER == "ollama":
        # Map generic max_tokens -> num_predict for Ollama; num_predict takes precedence if both provided.
        budget = num_predict if num_predict is not None else max_tokens
        yield from _generate_ollama(context, query, num_predict=budget, temperature=temperature)
        return
    # For non-ollama, clear stale Ollama metadata so research code does not reuse old values
    _clear_last_ollama_metadata()
    client = get_client()
    # Reuse canonical HF message construction; do not duplicate prompt strings.
    messages = _build_hf_messages(context, query)

    # Resolve generation budget: explicit param overrides production default (200).
    eff_max_tokens = int(max_tokens) if max_tokens is not None else 200
    if num_predict is not None and max_tokens is None:
        eff_max_tokens = int(num_predict)
    eff_temperature = float(temperature) if temperature is not None else 0.3

    completion = client.chat.completions.create(    #model call
        model=os.getenv("HF_MODEL", _DEFAULT_MODEL),
        messages=messages,#prompt
        max_tokens=eff_max_tokens,#response limit
        temperature=eff_temperature,#minimize randomness
        stream=True,#response == token->token(streaming)
    )
    prefix_buffer = ""      #fn to check llm response starting with "answer" 
    prefix_removed = False

    for chunk in completion:
        if len(chunk.choices) == 0:
            continue

        content = chunk.choices[0].delta.content if chunk.choices[0].delta else None #delaying the response untill prefix decision is made
        if not content:
            continue

        if not prefix_removed:#safely strip the response 
            prefix_buffer += content

            # remove Answer: safely even if split across chunks
            if "Answer:" in prefix_buffer or "Final Answer:" in prefix_buffer:
                prefix_buffer = (
                    prefix_buffer.replace("Answer:", "")
                    .replace("Final Answer:", "")
                    .replace("?", "", 1)
                )
                prefix_removed = True

                if prefix_buffer:
                    yield prefix_buffer
                continue

            # if real answer starts directly, release it
            if len(prefix_buffer) > 20:
                prefix_removed = True
                yield prefix_buffer
                prefix_buffer = ""
                continue

        else:
            yield content

    if not prefix_removed and prefix_buffer:
        yield prefix_buffer
