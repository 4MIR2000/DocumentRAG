from __future__ import annotations

import heapq
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Iterable
from typing import Callable

import fitz
import pymupdf4llm
import os
import numpy as np
from google import genai
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from pathlib import Path
from dotenv import load_dotenv
#try:
from langchain_text_splitters import MarkdownTextSplitter
#except ImportError:  # pragma: no cover - fallback for environments that have not installed the package yet
#    MarkdownTextSplitter = None

load_dotenv()

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
PARAGRAPH_SPLIT = re.compile(r"\n\s*\n+")
WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9]{3,}")

CHUNK_SIZE_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 120
MARKDOWN_CHUNK_SIZE_CHARS = 2200
MARKDOWN_CHUNK_OVERLAP_CHARS = 300

EMBEDDING_MODEL = "gemini-embedding-001"
_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def _embed_text(text: str, task_type: str) -> list[float]:
    print(f"Embedding text for task '{task_type}' with Gemini...")

    try:
        result = _client.models.embed_content(
            model=EMBEDDING_MODEL,
            contents=text,
            config={
                "task_type": task_type,
            },
        )
    except Exception as exc:
        print(f"Error during embedding for task '{task_type}': {exc}")
        raise 
    
    result = _client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config={
            "task_type": task_type,
        },
    )
    print(f"Embedding finished for task '{task_type}'")
    return result.embeddings[0].values


def _normalize_vector(vector: list[float]) -> list[float]:
    arr = np.array(vector, dtype=np.float32)
    norm = np.linalg.norm(arr)
    if norm == 0:
        return arr.tolist()
    return (arr / norm).tolist()


def embed_document(text: str) -> list[float]:
    return _normalize_vector(_embed_text(text, "RETRIEVAL_DOCUMENT"))


def embed_query(text: str) -> list[float]:
    return _normalize_vector(_embed_text(text, "RETRIEVAL_QUERY"))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    return float(np.dot(np.array(a), np.array(b)))



@dataclass(frozen=True)
class UploadedDocument:
    document_id: str
    file_name: str
    content: bytes


@dataclass(frozen=True)
class PageRecord:
    document_id: str
    document_name: str
    page_number: int
    text: str


@dataclass(frozen=True)
class ChunkRecord:
    document_id: str
    document_name: str
    page_number: int
    chunk_index: int
    text: str


@dataclass
class RAGIndex:
    documents: list[UploadedDocument] = field(default_factory=list)
    chunks: list[ChunkRecord] = field(default_factory=list)
    #idf: dict[str, float] = field(default_factory=dict)
    #chunk_vectors: list[dict[str, float]] = field(default_factory=list)
    chunk_embeddings: list[list[float]] = field(default_factory=list)


@dataclass
class RetrievedChunk:
    chunk: ChunkRecord
    score: float


def _clean_text(text: str) -> str:
    text = text or ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _token_count(text: str) -> int:
    return len(WORD_RE.findall(text))


def _terms(text: str) -> list[str]:
    tokens = [token.lower() for token in WORD_RE.findall(text)]
    if len(tokens) < 2:
        return tokens

    ngrams = [f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1)]
    return tokens + ngrams


def _normalized_tfidf_vector(terms: list[str], idf: dict[str, float]) -> dict[str, float]:
    if not terms:
        return {}

    term_frequencies = Counter(terms)
    total_terms = len(terms)
    weights: dict[str, float] = {}

    for term, frequency in term_frequencies.items():
        idf_score = idf.get(term)
        if idf_score is None:
            continue
        weights[term] = (frequency / total_terms) * idf_score

    norm = math.sqrt(sum(weight * weight for weight in weights.values()))
    if norm == 0:
        return {}

    return {term: weight / norm for term, weight in weights.items()}


def _split_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in SENTENCE_SPLIT.split(text) if s.strip()]
    return sentences or ([text.strip()] if text.strip() else [])


def _page_to_markdown(page: fitz.Page, page_number: int) -> str:
    try:
        markdown = page.get_text("markdown")
    except Exception:
        markdown = page.get_text("text")

    markdown = _clean_text(markdown)
    if not markdown:
        return ""

    return f"## Seite {page_number}\n\n{markdown}"


def _split_markdown_text(markdown_text: str) -> list[str]:
    markdown_text = markdown_text.strip()
    if not markdown_text:
        return []

    if MarkdownTextSplitter is not None:
        splitter = MarkdownTextSplitter(
            chunk_size=MARKDOWN_CHUNK_SIZE_CHARS,
            chunk_overlap=MARKDOWN_CHUNK_OVERLAP_CHARS,
        )
        chunks = [chunk.strip() for chunk in splitter.split_text(markdown_text) if chunk.strip()]

        return chunks
    # Small fallback so the app still runs if the optional package is missing.
'''
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0

    for paragraph in re.split(r"\n\s*\n+", markdown_text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        if current and current_length + len(paragraph) + 2 > MARKDOWN_CHUNK_SIZE_CHARS:
            chunks.append("\n\n".join(current).strip())
            overlap = current[-1:] if current else []
            current = overlap.copy()
            current_length = len("\n\n".join(current))

        current.append(paragraph)
        current_length += len(paragraph) + 2

    if current:
        chunks.append("\n\n".join(current).strip())

    return [chunk for chunk in chunks if chunk]
'''


def _split_paragraphs(text: str) -> list[str]:
    paragraphs = [p.strip() for p in PARAGRAPH_SPLIT.split(text) if p.strip()]
    return paragraphs or _split_sentences(text)


def _make_overlap(sentences: list[str], overlap_tokens: int) -> list[str]:
    overlap: list[str] = []
    total = 0

    for sentence in reversed(sentences):
        sentence_tokens = _token_count(sentence)
        if total + sentence_tokens > overlap_tokens and overlap:
            break
        overlap.insert(0, sentence)
        total += sentence_tokens

    return overlap


def _split_long_text(text: str, max_tokens: int) -> list[str]:
    sentences = _split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = _token_count(sentence)

        if current and current_tokens + sentence_tokens > max_tokens:
            chunks.append(" ".join(current).strip())
            current = []
            current_tokens = 0

        current.append(sentence)
        current_tokens += sentence_tokens

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


def _split_page_into_chunks(
    page_text: str,
    max_tokens: int = CHUNK_SIZE_TOKENS,
    overlap_tokens: int = CHUNK_OVERLAP_TOKENS,
) -> list[str]:
    """
    Page-aware recursive chunking:
    1. Keep page boundaries.
    2. Prefer paragraph-level chunks.
    3. Split oversized paragraphs by sentences.
    4. Add sentence-based overlap between chunks.
    """

    paragraphs = _split_paragraphs(page_text)
    raw_chunks: list[str] = []

    current_parts: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        paragraph_tokens = _token_count(paragraph)

        if paragraph_tokens > max_tokens:
            if current_parts:
                raw_chunks.append("\n\n".join(current_parts).strip())
                current_parts = []
                current_tokens = 0

            raw_chunks.extend(_split_long_text(paragraph, max_tokens))
            continue

        if current_parts and current_tokens + paragraph_tokens > max_tokens:
            raw_chunks.append("\n\n".join(current_parts).strip())
            current_parts = []
            current_tokens = 0

        current_parts.append(paragraph)
        current_tokens += paragraph_tokens

    if current_parts:
        raw_chunks.append("\n\n".join(current_parts).strip())

    if not raw_chunks:
        return []

    final_chunks: list[str] = []

    previous_sentences: list[str] = []
    for raw_chunk in raw_chunks:
        chunk_sentences = _split_sentences(raw_chunk)

        if previous_sentences and overlap_tokens > 0:
            overlap = _make_overlap(previous_sentences, overlap_tokens)
            chunk_text = " ".join(overlap + chunk_sentences).strip()
        else:
            chunk_text = " ".join(chunk_sentences).strip()

        if chunk_text:
            final_chunks.append(chunk_text)

        previous_sentences = chunk_sentences

    return final_chunks


def extract_pages_from_pdf(document: UploadedDocument) -> list[PageRecord]:
    pdf = fitz.open(stream=document.content, filetype="pdf")
    pages: list[PageRecord] = []
    #md=pymupdf4llm.to_markdown(pdf)
    #print("Extracted markdown from PDF: ", md[:500] if md else "no markdown")
    for page_number, _ in enumerate(pdf, start=1):
        #text = _page_to_markdown(page, page_number)
        text = pymupdf4llm.to_markdown(pdf, pages=[page_number-1])

        if text:
            pages.append(
                PageRecord(
                    document_id=document.document_id,
                    document_name=document.file_name,
                    page_number=page_number,
                    text=text,
                )
            )


    if not pages:
        raise ValueError(f"{document.file_name} enthält keinen extrahierbaren Text.")

    return pages


def build_chunks(pages: Iterable[PageRecord]) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    chunk_index = 0
    for page in pages:
        for page_chunk in _split_markdown_text(page.text):
            chunks.append(
                ChunkRecord(
                    document_id=page.document_id,
                    document_name=page.document_name,
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    text=page_chunk,
                )
            )
            chunk_index += 1
    if not chunks:
        raise ValueError("Es konnten keine Chunks erzeugt werden.")
    return chunks


def build_index(documents: list[UploadedDocument], pages: list[PageRecord]) -> RAGIndex:
    chunks = build_chunks(pages)

    chunk_embeddings = []
    for chunk in chunks:
        print("chunks built")
        embedding = embed_document(chunk.text)
        print("embedding done")
        chunk_embeddings.append(embedding)

    return RAGIndex(
        documents=documents,
        chunks=chunks,
        chunk_embeddings=chunk_embeddings,
    )

''''
def build_index(documents: list[UploadedDocument], pages: list[PageRecord]) -> RAGIndex:
    chunks = build_chunks(pages)

    document_frequency: Counter[str] = Counter()
    chunk_terms: list[list[str]] = []
    for chunk in chunks:
        terms = _terms(chunk.text)
        chunk_terms.append(terms)
        document_frequency.update(set(terms))

    total_chunks = len(chunks)
    idf = {
        term: math.log((1 + total_chunks) / (1 + frequency)) + 1.0
        for term, frequency in document_frequency.items()
    }
    chunk_vectors = [_normalized_tfidf_vector(terms, idf) for terms in chunk_terms]

    return RAGIndex(
        documents=documents,
        chunks=chunks,
        idf=idf,
        chunk_vectors=chunk_vectors,
    )
'''

def encode_query(index: RAGIndex, query: str) -> dict[str, float]:
    if not index.idf:
        raise ValueError("Index ist noch nicht aufgebaut.")

    return _normalized_tfidf_vector(_terms(query), index.idf)


def retrieve(
    index: RAGIndex,
    query: str,
    top_k: int = 4,
    score_threshold: float = 0.55,
) -> list[RetrievedChunk]:
    if not index.chunk_embeddings:
        raise ValueError("Index ist noch nicht aufgebaut.")

    if top_k <= 0:
        return []

    query_embedding = embed_query(query)

    scored_chunks: list[tuple[float, int]] = []

    for chunk_index, chunk_embedding in enumerate(index.chunk_embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding)

        # Score in der Konsole ausgeben
        print(
            f"Chunk {chunk_index} | "
            f"Score: {score:.4f} | "
            f"Dokument: {index.chunks[chunk_index].document_name}"
        )

        # Nur Chunks über Threshold speichern
        if score >= score_threshold:
            scored_chunks.append((score, chunk_index))

    best_matches = heapq.nlargest(
        top_k,
        scored_chunks,
        key=lambda item: item[0],
    )

    return [
        RetrievedChunk(
            chunk=index.chunks[chunk_index],
            score=float(score),
        )
        for score, chunk_index in best_matches
    ]

'''
def retrieve(index: RAGIndex, query: str, top_k: int = 4) -> list[RetrievedChunk]:
    query_vector = encode_query(index, query)
    if not query_vector or top_k <= 0:
        return []

    scored_chunks: list[tuple[float, int]] = []
    for chunk_index, chunk_vector in enumerate(index.chunk_vectors):
        score = sum(query_vector.get(term, 0.0) * weight for term, weight in chunk_vector.items())
        if score > 0:
            scored_chunks.append((score, chunk_index))

    best_matches = heapq.nlargest(top_k, scored_chunks, key=lambda item: item[0])
    return [
        RetrievedChunk(chunk=index.chunks[chunk_index], score=float(score))
        for score, chunk_index in best_matches
    ]
'''

def _question_keywords(question: str) -> set[str]:
    return {word.lower() for word in WORD_RE.findall(question) if len(word) > 3}


def _rank_sentence(question: str, chunk_text: str) -> tuple[str, float]:
    keywords = _question_keywords(question)
    sentences = _split_sentences(chunk_text)
    if not sentences:
        return chunk_text[:400], 0.0

    best_sentence = sentences[0]
    best_score = -1.0
    for sentence in sentences:
        sentence_words = {word.lower() for word in WORD_RE.findall(sentence)}
        overlap = len(keywords & sentence_words)
        density = overlap / max(len(sentence_words), 1)
        score = overlap + density
        if score > best_score:
            best_sentence = sentence
            best_score = score
    return best_sentence, best_score

'''
def summarize_answer(question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
    if not retrieved_chunks:
        return "Ich habe dazu in den hochgeladenen Dokumenten keine passende Stelle gefunden."

    best_sentences: list[str] = []
    seen = set()
    for item in retrieved_chunks[:2]:
        sentence, _ = _rank_sentence(question, item.chunk.text)
        normalized = sentence.lower()
        if normalized not in seen:
            seen.add(normalized)
            best_sentences.append(sentence)

    if not best_sentences:
        best_sentences = [retrieved_chunks[0].chunk.text[:400]]

    documents = []
    for item in retrieved_chunks:
        if item.chunk.document_name not in documents:
            documents.append(item.chunk.document_name)

    prefix = f"Basierend auf {', '.join(documents)}: " if documents else ""
    return prefix + " ".join(best_sentences)
'''

def citation_payload(retrieved_chunks: list[RetrievedChunk], limit: int = 3) -> list[dict]:
    payload: list[dict] = []
    for item in retrieved_chunks[:limit]:
        payload.append(
            {
                "document_name": item.chunk.document_name,
                "page_number": item.chunk.page_number,
                "chunk_index": item.chunk.chunk_index,
                "text_excerpt": item.chunk.text,
            }
        )
    return payload



BENCHMARK_ITEMS = [
    {
        "question": "Welche Anschlussmöglichkeiten bietet das Gerät?",
        "expected_answer": "USB, Ethernet und WLAN",
    },
    {
        "question": "Wie kann man von einem mobilen Gerät drucken, ohne mit demselben WLAN wie der Drucker verbunden zu sein?",
        "expected_answer": (
            "Man kann Wi-Fi Direct verwenden, um das mobile Gerät direkt "
            "mit dem Drucker zu verbinden und kabellos zu drucken, ohne "
            "mit einem bestehenden WLAN verbunden zu sein."
        ),
    },
    {
        "question": "Was sollte man tun, bevor man das Netzkabel des Druckers abzieht?",
        "expected_answer": (
            "Man sollte den Drucker über die Netztaste ausschalten und warten, "
            "bis keine Betriebsgeräusche mehr zu hören sind, bevor man das "
            "Netzkabel abzieht."
        ),
    },
    {
        "question": "Wie wird ein Papierstau behoben?",
        "expected_answer": "Papierfach öffnen, vorsichtig festsitzendes Papier entfernen und Abdeckung schließen",
    },
    {
        "question": "Wie lange gilt die Herstellergarantie?",
        "expected_answer": "1 Jahr Herstellergarantie",
    },
]


def retrieval_score(expected_answer: str, generated_answer: str, retrieved_excerpt: str) -> float:
    expected_tokens = {token.lower() for token in WORD_RE.findall(expected_answer)}
    generated_tokens = {token.lower() for token in WORD_RE.findall(generated_answer)}
    excerpt_tokens = {token.lower() for token in WORD_RE.findall(retrieved_excerpt)}

    keyword_recall = len(expected_tokens & excerpt_tokens) / max(len(expected_tokens), 1)
    answer_overlap = len(expected_tokens & generated_tokens) / max(len(expected_tokens), 1)
    lexical_similarity = SequenceMatcher(None, expected_answer.lower(), retrieved_excerpt.lower()).ratio()

    score = 100.0 * (0.5 * keyword_recall + 0.3 * answer_overlap + 0.2 * lexical_similarity)
    return round(score, 1)


def build_demo_index() -> RAGIndex:
    demo_path = Path(__file__).parent.parent / "demo_handbuch.pdf"

    if not demo_path.exists():
        raise FileNotFoundError(f"Demo-PDF nicht gefunden: {demo_path}")

    document = UploadedDocument(
        document_id="demo-handbook",
        file_name="demo_handbuch.pdf",
        content=demo_path.read_bytes(),
    )

    pages = extract_pages_from_pdf(document)

    return build_index([document], pages)

def ragas_benchmark(
    index: RAGIndex,
    answer_function: Callable[[RAGIndex, str], tuple[str, list[RetrievedChunk]]],
) -> tuple[list[dict], dict]:
    rows = []

    for benchmark_item in BENCHMARK_ITEMS:
        question = benchmark_item["question"]
        expected_answer = benchmark_item["expected_answer"]

        generated_answer, retrieved = answer_function(index, question)

        contexts = [item.chunk.text for item in retrieved]

        rows.append(
            {
                "question": question,
                "answer": generated_answer,
                "contexts": contexts,
                "ground_truth": expected_answer,
            }
        )

    dataset = Dataset.from_list(rows)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=os.getenv("GEMINI_API_KEY"),
    )

    print("Starting RAGas benchmark evaluation with Gemini...")
    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=llm,
        embeddings=embeddings,
    )

    df = result.to_pandas()

    average_scores = {
        "faithfulness": round(float(df["faithfulness"].mean()), 3),
        "answer_relevancy": round(float(df["answer_relevancy"].mean()), 3),
        "context_precision": round(float(df["context_precision"].mean()), 3),
        "context_recall": round(float(df["context_recall"].mean()), 3),
    }

    return df.to_dict("records"), average_scores