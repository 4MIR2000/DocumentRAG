from __future__ import annotations

import os
import threading
import uuid
from dataclasses import dataclass, field
from typing import Annotated
from dotenv import load_dotenv
import traceback
from fastapi import HTTPException

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from backend.rag import (
    BENCHMARK_ITEMS,
    UploadedDocument,
    build_demo_index,
    build_index,
    citation_payload,
    extract_pages_from_pdf,
    retrieval_score,
    retrieve,
    ragas_benchmark,
)

load_dotenv()  # Lädt die .env Datei

from backend.rag import (
    BENCHMARK_ITEMS,
    UploadedDocument,
    build_demo_index,
    build_index,
    citation_payload,
    extract_pages_from_pdf,
    retrieval_score,
    retrieve,
)
from backend.llm import generate_answer_if_configured
from backend.schemas import BenchmarkItem, BenchmarkResponse, ChatRequest, ChatResponse, UploadResponse


@dataclass
class ConversationStore:
    documents: list[UploadedDocument] = field(default_factory=list)
    index: object | None = None


app = FastAPI(title="RAG Document Chat API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_stores: dict[str, ConversationStore] = {}
_lock = threading.Lock()


def _get_store(conversation_id: str) -> ConversationStore:
    with _lock:
        if conversation_id not in _stores:
            _stores[conversation_id] = ConversationStore()
        return _stores[conversation_id]


def _rebuild_store(store: ConversationStore):
    pages = []
    for document in store.documents:
        pages.extend(extract_pages_from_pdf(document))
        print("Extracted pages from document:", document.file_name)
    store.index = build_index(store.documents, pages)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(
    conversation_id: Annotated[str, Form(...)],
    files: Annotated[list[UploadFile], File(...)],
):
    if not files:
        raise HTTPException(status_code=400, detail="Mindestens ein PDF muss hochgeladen werden.")

    pdf_documents: list[UploadedDocument] = []
    for uploaded_file in files:
        if not uploaded_file.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"{uploaded_file.filename} ist kein PDF.")
        pdf_documents.append(
            UploadedDocument(
                document_id=str(uuid.uuid4()),
                file_name=uploaded_file.filename,
                content=await uploaded_file.read(),
            )
        )

    store = _get_store(conversation_id)
    store.documents.extend(pdf_documents)

    _rebuild_store(store)


    return UploadResponse(
        conversation_id=conversation_id,
        document_count=len(store.documents),
        chunk_count=len(store.index.chunks) if store.index else 0,
        documents=[document.file_name for document in store.documents],
    )

def answer_question_with_index(index, question: str):
    retrieved_chunks = retrieve(index, question, top_k=6)

    chunk_payload = [
        {
            "document_name": c.document_name,
            "page_number": c.page_number,
            "chunk_index": c.chunk_index,
            "text_excerpt": c.text,
        }
        for c in (rc.chunk for rc in retrieved_chunks)
    ]

    answer = generate_answer_if_configured(question, chunk_payload)

    return answer, retrieved_chunks

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    store = _get_store(request.conversation_id)

    if store.index is None:
        raise HTTPException(status_code=400, detail="Bitte zuerst PDFs hochladen und indexieren.")

    answer, retrieved_chunks = answer_question_with_index(
        store.index,
        request.question,
    )

    return ChatResponse(
        answer=answer,
        citations=citation_payload(retrieved_chunks),
        chunk_count=len(retrieved_chunks),
    )



@app.get("/benchmark")
def benchmark():
    index = build_demo_index()

    items, average_scores = ragas_benchmark(
        index=index,
        answer_function=answer_question_with_index,
    )

    return {
        "average_scores": average_scores,
        "items": items,
    }


'''
@app.get("/benchmark", response_model=BenchmarkResponse)
def benchmark():
    index = build_demo_index()
    items: list[BenchmarkItem] = []
    for benchmark_item in BENCHMARK_ITEMS:
        retrieved = retrieve(index, benchmark_item["question"], top_k=3)
        generated_answer = summarize_answer(benchmark_item["question"], retrieved)
        source = retrieved[0].chunk if retrieved else None
        source_excerpt = source.text[:500] if source else ""
        score = retrieval_score(benchmark_item["expected_answer"], generated_answer, source_excerpt)
        items.append(
            BenchmarkItem(
                question=benchmark_item["question"],
                expected_answer=benchmark_item["expected_answer"],
                generated_answer=generated_answer,
                retrieval_score=score,
                source_document=source.document_name if source else "",
                source_page=source.page_number if source else 0,
                source_excerpt=source_excerpt,
            )
        )

    average_score = round(sum(item.retrieval_score for item in items) / max(len(items), 1), 1)
    return BenchmarkResponse(average_retrieval_score=average_score, items=items)

'''