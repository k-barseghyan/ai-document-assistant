from typing import Literal

from pydantic import BaseModel, Field, field_validator


class QuestionRequest(BaseModel):
    question: str = Field(
        min_length=3,
        max_length=500,
        description="User question about uploaded documents"
    )
    answer_mode: Literal["rag_only", "hybrid"] = Field(
        default="rag_only",
        description=(
            "Answer mode: rag_only uses only uploaded documents; hybrid uses "
            "uploaded documents first and general model knowledge when needed"
        ),
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must be non-empty")
        return value


class AnswerSource(BaseModel):
    filename: str
    document_id: str
    chunk_index: int
    score: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[AnswerSource] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str
    content: str = Field(min_length=1)

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        allowed_roles = {"system", "user", "assistant"}
        if value not in allowed_roles:
            raise ValueError("role must be one of: system, user, assistant")
        return value

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("content must be non-empty")
        return value


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)


class ChatResponse(BaseModel):
    answer: str


class DocumentUploadResponse(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    stored_chunk_count: int


class DevEmbeddingRequest(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must be non-empty")
        return value


class DevEmbeddingResponse(BaseModel):
    model: str
    dimension: int
    preview: list[float]


class DevChunksRequest(BaseModel):
    text: str = Field(min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must be non-empty")
        return value


class DevChunkPreview(BaseModel):
    index: int
    char_count: int
    preview: str


class DevChunksResponse(BaseModel):
    chunk_count: int
    chunks: list[DevChunkPreview]


class DevRetrieveRequest(BaseModel):
    question: str = Field(min_length=1)
    limit: int = Field(default=3, ge=1, le=10)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must be non-empty")
        return value


class RetrievedChunkResponse(BaseModel):
    score: float
    document_id: str
    document_hash: str
    filename: str
    chunk_index: int
    chunk_hash: str
    text: str
    char_count: int


class DevRetrieveResponse(BaseModel):
    matches: list[RetrievedChunkResponse]


class DevRagEvaluateRequest(BaseModel):
    question: str = Field(min_length=1)
    answer_mode: Literal["rag_only", "hybrid"] = Field(default="rag_only")
    expected_keywords: list[str] = Field(default_factory=list)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must be non-empty")
        return value

    @field_validator("expected_keywords")
    @classmethod
    def validate_expected_keywords(cls, value: list[str]) -> list[str]:
        keywords = [keyword.strip() for keyword in value]
        if any(not keyword for keyword in keywords):
            raise ValueError("expected_keywords must contain only non-empty strings")
        return keywords


class DevRagEvaluateMatch(BaseModel):
    score: float
    filename: str
    document_id: str
    chunk_index: int
    char_count: int
    used_in_context: bool
    preview: str


class DevRagEvaluateResponse(BaseModel):
    question: str
    answer_mode: Literal["rag_only", "hybrid"]
    retrieval_limit: int
    context_limit: int
    min_relevance_score: float
    retrieved_count: int
    used_context_count: int
    matches: list[DevRagEvaluateMatch]
    answer: str
    missing_expected_keywords_in_context: list[str]
    missing_expected_keywords_in_answer: list[str]


class DevVectorStoreClearResponse(BaseModel):
    collection: str
    deleted: bool
