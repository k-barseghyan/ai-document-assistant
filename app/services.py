import os
from dataclasses import dataclass

from fastapi import HTTPException

from app.embeddings.ollama_embedding_client import OllamaEmbeddingClient
from app.llm.ollama_client import OllamaClient
from app.rag.chunker import PREVIEW_CHARS
from app.schemas import (
    AnswerResponse,
    AnswerSource,
    ChatRequest,
    ChatResponse,
    DevRagEvaluateMatch,
    DevRagEvaluateRequest,
    DevRagEvaluateResponse,
    QuestionRequest,
)
from app.vector_store.qdrant_client import QdrantVectorStoreClient, RetrievedChunk


def _get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _get_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default

    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


DEFAULT_CHAT_SYSTEM_MESSAGE = (
    "You are a helpful AI assistant. Answer clearly and naturally."
)
NO_CONTEXT_ANSWER = "I do not know based on the uploaded documents."
RAG_ONLY_SYSTEM_MESSAGE = (
    "You are a deterministic API assistant.\n"
    "Answer only from the provided document context.\n"
    "Do not use outside knowledge.\n"
    f'If the context is insufficient, answer exactly: "{NO_CONTEXT_ANSWER}"'
)
HYBRID_SYSTEM_MESSAGE = (
    "You are a deterministic API assistant.\n"
    "Use uploaded document context when it is available.\n"
    "When document context is missing or incomplete, you may use general knowledge.\n"
    "Clearly distinguish document-supported information from general knowledge.\n"
    "Do not invent document sources or imply that general knowledge came from "
    "uploaded documents."
)
RAG_RETRIEVAL_LIMIT = _get_int_env("RAG_RETRIEVAL_LIMIT", 5)
RAG_CONTEXT_LIMIT = _get_int_env("RAG_CONTEXT_LIMIT", 3)
RAG_MIN_RELEVANCE_SCORE = _get_float_env("RAG_MIN_RELEVANCE_SCORE", 0.55)
RAG_ONLY_ANSWER_INSTRUCTIONS = (
    "You must answer using only the provided document context.\n"
    f'If the context does not contain the answer, say: "{NO_CONTEXT_ANSWER}"\n'
    "Keep the answer brief and direct.\n"
    "Do not mention filenames, chunk numbers, source references, or citations inside the answer text.\n"
    'The application returns sources separately in the "sources" field.'
)
HYBRID_ANSWER_INSTRUCTIONS = (
    "Use the uploaded document context first when it contains relevant information.\n"
    "If the document context fully answers the question, answer normally from the document context.\n"
    "If document context is missing or incomplete, say that clearly and then add general model knowledge.\n"
    "When both document context and general knowledge are used, clearly separate them.\n"
    "Do not invent document sources.\n"
    "Do not imply that general knowledge came from uploaded documents.\n"
    "Keep the answer brief and direct.\n"
    "Do not mention filenames, chunk numbers, source references, or citations inside the answer text.\n"
    'The application returns document sources separately in the "sources" field.'
)


def _validate_rag_config() -> None:
    if RAG_RETRIEVAL_LIMIT <= 0:
        raise ValueError("RAG_RETRIEVAL_LIMIT must be positive")

    if RAG_CONTEXT_LIMIT <= 0:
        raise ValueError("RAG_CONTEXT_LIMIT must be positive")

    if not 0 <= RAG_MIN_RELEVANCE_SCORE <= 1:
        raise ValueError("RAG_MIN_RELEVANCE_SCORE must be between 0 and 1 inclusive")

    if RAG_CONTEXT_LIMIT > RAG_RETRIEVAL_LIMIT:
        raise ValueError("RAG_CONTEXT_LIMIT must not exceed RAG_RETRIEVAL_LIMIT")


_validate_rag_config()


@dataclass(frozen=True)
class _RagRetrievalResult:
    collection_exists: bool
    retrieved_chunks: list[RetrievedChunk]
    context_chunks: list[RetrievedChunk]


class QuestionService:
    def __init__(
        self,
        llm_client: OllamaClient,
        embedding_client: OllamaEmbeddingClient,
        vector_store: QdrantVectorStoreClient,
    ):
        self.llm_client = llm_client
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    def answer_question(self, request: QuestionRequest) -> AnswerResponse:
        question = request.question.strip()
        if not question:
            raise ValueError("question must be non-empty")

        retrieval = self._retrieve_relevant_chunks(question)
        answer = self._generate_answer(
            question=question,
            answer_mode=request.answer_mode,
            collection_exists=retrieval.collection_exists,
            context_chunks=retrieval.context_chunks,
        )

        return AnswerResponse(
            answer=answer,
            sources=[
                AnswerSource(
                    filename=chunk.filename,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    score=chunk.score,
                )
                for chunk in retrieval.context_chunks
            ],
        )

    def evaluate_rag(self, request: DevRagEvaluateRequest) -> DevRagEvaluateResponse:
        question = request.question.strip()
        if not question:
            raise ValueError("question must be non-empty")

        retrieval = self._retrieve_relevant_chunks(question)
        answer = self._generate_answer(
            question=question,
            answer_mode=request.answer_mode,
            collection_exists=retrieval.collection_exists,
            context_chunks=retrieval.context_chunks,
        )

        context_chunk_refs = {
            _chunk_ref(chunk)
            for chunk in retrieval.context_chunks
        }

        return DevRagEvaluateResponse(
            question=question,
            answer_mode=request.answer_mode,
            retrieval_limit=RAG_RETRIEVAL_LIMIT,
            context_limit=RAG_CONTEXT_LIMIT,
            min_relevance_score=RAG_MIN_RELEVANCE_SCORE,
            retrieved_count=len(retrieval.retrieved_chunks),
            used_context_count=len(retrieval.context_chunks),
            matches=[
                DevRagEvaluateMatch(
                    score=chunk.score,
                    filename=chunk.filename,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    char_count=chunk.char_count,
                    used_in_context=_chunk_ref(chunk) in context_chunk_refs,
                    preview=_preview_text(chunk.text),
                )
                for chunk in retrieval.retrieved_chunks
            ],
            answer=answer,
            missing_expected_keywords=_missing_expected_keywords(
                answer=answer,
                expected_keywords=request.expected_keywords,
            ),
        )

    def _retrieve_relevant_chunks(self, question: str) -> _RagRetrievalResult:
        try:
            collection_exists = self.vector_store.collection_exists()
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="Could not connect to Qdrant",
            ) from exc

        if not collection_exists:
            return _RagRetrievalResult(
                collection_exists=False,
                retrieved_chunks=[],
                context_chunks=[],
            )

        try:
            question_vector = self.embedding_client.embed_text(question)
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="Could not create question embedding",
            ) from exc

        try:
            chunks = self.vector_store.search_similar_chunks(
                query_vector=question_vector,
                limit=RAG_RETRIEVAL_LIMIT,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="Could not search Qdrant",
            ) from exc

        return _RagRetrievalResult(
            collection_exists=True,
            retrieved_chunks=chunks,
            context_chunks=_select_relevant_chunks(chunks),
        )

    def _generate_answer(
        self,
        question: str,
        answer_mode: str,
        collection_exists: bool,
        context_chunks: list[RetrievedChunk],
    ) -> str:
        if answer_mode == "rag_only":
            if not collection_exists or not context_chunks:
                return NO_CONTEXT_ANSWER

            final_prompt = _build_rag_only_prompt(
                question=question,
                chunks=context_chunks,
            )
            return self.llm_client.generate(
                final_prompt,
                system=RAG_ONLY_SYSTEM_MESSAGE,
            )

        if answer_mode == "hybrid":
            final_prompt = _build_hybrid_prompt(
                question=question,
                chunks=context_chunks,
            )
            return self.llm_client.generate(
                final_prompt,
                system=HYBRID_SYSTEM_MESSAGE,
            )

        raise ValueError("answer_mode must be one of: rag_only, hybrid")


class ChatService:
    def __init__(self, llm_client: OllamaClient):
        self.llm_client = llm_client

    def answer_message(self, request: ChatRequest) -> ChatResponse:
        messages = [message.model_dump() for message in request.messages]

        if not any(message["role"] == "system" for message in messages):
            messages.insert(
                0,
                {
                    "role": "system",
                    "content": DEFAULT_CHAT_SYSTEM_MESSAGE,
                },
            )

        answer = self.llm_client.chat(messages)
        return ChatResponse(answer=answer)


def get_question_service() -> QuestionService:
    return QuestionService(
        llm_client=OllamaClient(),
        embedding_client=OllamaEmbeddingClient(),
        vector_store=QdrantVectorStoreClient(),
    )


def get_chat_service() -> ChatService:
    return ChatService(llm_client=OllamaClient())


def _build_rag_only_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        _format_context_chunk(index=index, chunk=chunk)
        for index, chunk in enumerate(chunks, start=1)
    )

    return (
        f"{RAG_ONLY_ANSWER_INSTRUCTIONS}\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def _build_hybrid_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    if chunks:
        context = "\n\n".join(
            _format_context_chunk(index=index, chunk=chunk)
            for index, chunk in enumerate(chunks, start=1)
        )
    else:
        context = "No relevant uploaded document context is available."

    return (
        f"{HYBRID_ANSWER_INSTRUCTIONS}\n\n"
        f"Document context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer:"
    )


def _format_context_chunk(index: int, chunk: RetrievedChunk) -> str:
    filename = _single_line(chunk.filename)
    text = chunk.text.strip()
    return (
        f"[{index}] filename={filename}, chunk_index={chunk.chunk_index}\n"
        f"{text}"
    )


def _single_line(value: str) -> str:
    return " ".join(value.split())


def _select_relevant_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return [
        chunk
        for chunk in chunks
        if chunk.score >= RAG_MIN_RELEVANCE_SCORE
    ][:RAG_CONTEXT_LIMIT]


def _chunk_ref(chunk: RetrievedChunk) -> tuple[str, int, str]:
    return (chunk.document_id, chunk.chunk_index, chunk.chunk_hash)


def _preview_text(text: str) -> str:
    return _single_line(text.strip())[:PREVIEW_CHARS]


def _missing_expected_keywords(
    answer: str,
    expected_keywords: list[str],
) -> list[str]:
    normalized_answer = answer.casefold()
    return [
        keyword
        for keyword in expected_keywords
        if keyword.casefold() not in normalized_answer
    ]
