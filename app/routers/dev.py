from fastapi import APIRouter, Depends, HTTPException

from app.embeddings.ollama_embedding_client import OllamaEmbeddingClient
from app.rag.chunker import PREVIEW_CHARS, chunk_text
from app.schemas import (
    DevChunkPreview,
    DevChunksRequest,
    DevChunksResponse,
    DevEmbeddingRequest,
    DevEmbeddingResponse,
    DevRetrieveRequest,
    DevRetrieveResponse,
    RetrievedChunkResponse,
)
from app.vector_store.qdrant_client import QdrantVectorStoreClient

router = APIRouter(prefix="/dev", tags=["dev"])


def get_embedding_client() -> OllamaEmbeddingClient:
    return OllamaEmbeddingClient()


def get_vector_store_client() -> QdrantVectorStoreClient:
    return QdrantVectorStoreClient()


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


@router.post("/retrieve", response_model=DevRetrieveResponse)
def retrieve_chunks(
    request: DevRetrieveRequest,
    embedding_client: OllamaEmbeddingClient = Depends(get_embedding_client),
    vector_store: QdrantVectorStoreClient = Depends(get_vector_store_client),
):
    try:
        if not vector_store.collection_exists():
            raise HTTPException(
                status_code=404,
                detail="No documents have been ingested yet",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not connect to Qdrant",
        ) from exc

    try:
        question_vector = embedding_client.embed_text(request.question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        matches = vector_store.search_similar_chunks(
            query_vector=question_vector,
            limit=request.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not search Qdrant",
        ) from exc

    return DevRetrieveResponse(
        matches=[
            RetrievedChunkResponse(
                score=match.score,
                document_id=match.document_id,
                filename=match.filename,
                chunk_index=match.chunk_index,
                text=match.text,
                char_count=match.char_count,
            )
            for match in matches
        ],
    )
