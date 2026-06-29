from fastapi import APIRouter, Depends, HTTPException

from app.embeddings.ollama_embedding_client import OllamaEmbeddingClient
from app.rag.chunker import PREVIEW_CHARS, chunk_text
from app.schemas import (
    DevChunkPreview,
    DevChunksRequest,
    DevChunksResponse,
    DevEmbeddingRequest,
    DevEmbeddingResponse,
)

router = APIRouter(prefix="/dev", tags=["dev"])


def get_embedding_client() -> OllamaEmbeddingClient:
    return OllamaEmbeddingClient()


@router.post("/embeddings", response_model=DevEmbeddingResponse)
def create_embedding(
    request: DevEmbeddingRequest,
    embedding_client: OllamaEmbeddingClient = Depends(get_embedding_client),
):
    try:
        vector = embedding_client.embed_text(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return DevEmbeddingResponse(
        model=embedding_client.model,
        dimension=len(vector),
        preview=vector[:5],
    )


@router.post("/chunks", response_model=DevChunksResponse)
def create_chunks(request: DevChunksRequest):
    try:
        chunks = chunk_text(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return DevChunksResponse(
        chunk_count=len(chunks),
        chunks=[
            DevChunkPreview(
                index=chunk.index,
                char_count=chunk.char_count,
                preview=chunk.text[:PREVIEW_CHARS],
            )
            for chunk in chunks
        ],
    )
