from fastapi import HTTPException

from app.embeddings.ollama_embedding_client import OllamaEmbeddingClient
from app.llm.ollama_client import OllamaClient
from app.schemas import (
    AnswerResponse,
    AnswerSource,
    ChatRequest,
    ChatResponse,
    QuestionRequest,
)
from app.vector_store.qdrant_client import QdrantVectorStoreClient, RetrievedChunk


DEFAULT_CHAT_SYSTEM_MESSAGE = (
    "You are a helpful AI assistant. Answer clearly and naturally."
)
NO_CONTEXT_ANSWER = "I do not know based on the uploaded documents."
QUESTION_RETRIEVAL_LIMIT = 3
RAG_ANSWER_INSTRUCTIONS = (
    "You must answer using only the provided document context.\n"
    'If the context does not contain the answer, say: "I do not know based on the uploaded documents."\n'
    "Keep the answer brief and direct.\n"
    "Do not mention filenames, chunk numbers, source references, or citations inside the answer text.\n"
    'The application returns sources separately in the "sources" field.'
)


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

        try:
            collection_exists = self.vector_store.collection_exists()
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="Could not connect to Qdrant",
            ) from exc

        if not collection_exists:
            return AnswerResponse(answer=NO_CONTEXT_ANSWER)

        question_vector = self.embedding_client.embed_text(question)

        try:
            chunks = self.vector_store.search_similar_chunks(
                query_vector=question_vector,
                limit=QUESTION_RETRIEVAL_LIMIT,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail="Could not search Qdrant",
            ) from exc

        if not chunks:
            return AnswerResponse(answer=NO_CONTEXT_ANSWER)

        final_prompt = _build_rag_prompt(question=question, chunks=chunks)
        answer = self.llm_client.generate(final_prompt)
        return AnswerResponse(
            answer=answer,
            sources=[
                AnswerSource(
                    filename=chunk.filename,
                    document_id=chunk.document_id,
                    chunk_index=chunk.chunk_index,
                    score=chunk.score,
                )
                for chunk in chunks
            ],
        )


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


def _build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        _format_context_chunk(index=index, chunk=chunk)
        for index, chunk in enumerate(chunks, start=1)
    )

    return (
        f"{RAG_ANSWER_INSTRUCTIONS}\n\n"
        f"Context:\n{context}\n\n"
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
