from pydantic import BaseModel, Field


class Citation(BaseModel):
    document_name: str
    page_number: int
    chunk_index: int
    text_excerpt: str


class ChatRequest(BaseModel):
    conversation_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    chunk_count: int


class UploadResponse(BaseModel):
    conversation_id: str
    document_count: int
    chunk_count: int
    documents: list[str]


class BenchmarkItem(BaseModel):
    question: str
    expected_answer: str
    generated_answer: str
    retrieval_score: float
    source_document: str
    source_page: int
    source_excerpt: str


class BenchmarkResponse(BaseModel):
    average_retrieval_score: float
    items: list[BenchmarkItem]