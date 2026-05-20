from __future__ import annotations

import os
import requests
from typing import Optional


GEMINI_ENDPOINT = os.getenv("GEMINI_API_ENDPOINT")
GEMINI_KEY = os.getenv("GEMINI_API_KEY")


import os
from dotenv import load_dotenv
#mport google.generativeai as genai
from google import genai

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

MODEL_NAME = "gemini-2.5-flash"

def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    print("Building prompt for Gemini with retrieved chunks:")
    print(retrieved_chunks)
    context = "\n\n".join(
        f"""
Source:
Document: {item["document_name"]}
Page: {item["page_number"]}

Text:
{item["text_excerpt"]}
"""
        for item in retrieved_chunks
    )

    return f"""
You are a helpful PDF chatbot.

Answer the question ONLY based on the provided context.

If the answer is not contained in the context, say:
"I could not find relevant information in the PDF."

Context:
{context}

Question:
{question}

Answer:
"""


def generate_with_gemini(question: str, retrieved_chunks: list[dict]):
    """
    Generate answer from retrieved RAG chunks.
    """

    if not retrieved_chunks:
        return "I could not find relevant information in the PDF."

    prompt = build_prompt(question, retrieved_chunks)

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    return response.text.strip()




''''
def _build_prompt(question: str, chunks: list[dict]) -> str:
    prompt = [
        "You are a helpful assistant. Use the provided document excerpts to answer the question concisely and provide source citations (document name and page).",
        "Question:",
        question,
        "\nRelevant excerpts:\n",
    ]
    for i, c in enumerate(chunks, start=1):
        prompt.append(f"[{i}] {c.get('document_name','unknown')} (page {c.get('page_number','?')}): {c.get('text_excerpt','')}")

    prompt.append("\nGive a final answer and include a short list of citations referencing the excerpts above.")
    return "\n".join(prompt)


def generate_with_gemini(question: str, retrieved_chunks: list[dict], timeout: int = 20) -> Optional[str]:
    """Call an external Gemini-compatible endpoint to generate an answer.

    Expects the endpoint at `GEMINI_API_ENDPOINT` and the key in `GEMINI_API_KEY`.
    The exact API shape may vary; this wrapper assumes a JSON POST returning a `text` field.
    If environment variables are not set or the call fails, returns None.
    """
    if not GEMINI_ENDPOINT or not GEMINI_KEY:
        return None

    prompt = _build_prompt(question, retrieved_chunks)
    headers = {"Authorization": f"Bearer {GEMINI_KEY}", "Content-Type": "application/json"}
    payload = {"prompt": prompt, "max_tokens": 512}

    try:
        resp = requests.post(GEMINI_ENDPOINT, json=payload, headers=headers, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # Try common response keys
        if isinstance(data, dict):
            for key in ("text", "answer", "content", "generated_text"):
                if key in data and isinstance(data[key], str):
                    return data[key]
            # Some APIs return choices
            if "choices" in data and isinstance(data["choices"], list) and data["choices"]:
                choice = data["choices"][0]
                if isinstance(choice, dict):
                    return choice.get("text") or choice.get("message") or choice.get("content")
        return None
    except Exception:
        return None

'''
def generate_answer_if_configured(question: str, retrieved_chunks: list[dict]) -> Optional[str]:
    return generate_with_gemini(question, retrieved_chunks)



