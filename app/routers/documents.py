from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.embeddings.ollama_embedding_client import OllamaEmbeddingClient
from app.rag.chunker import chunk_text
from app.schemas import DocumentUploadResponse
from app.vector_store.qdrant_client import QdrantVectorStoreClient


router = APIRouter(prefix="/documents", tags=["documents"])

SUPPORTED_CONTENT_TYPE = "text/plain"
SUPPORTED_EXTENSION = ".txt"


def get_embedding_client() -> OllamaEmbeddingClient:
    return OllamaEmbeddingClient()


def get_vector_store_client() -> QdrantVectorStoreClient:
    return QdrantVectorStoreClient()


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    embedding_client: OllamaEmbeddingClient = Depends(get_embedding_client),
    vector_store: QdrantVectorStoreClient = Depends(get_vector_store_client),
):
    filename = _validate_text_file(file)
    file_bytes = await file.read()

    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file must be valid UTF-8 text",
        ) from exc

    if not text.strip():
        raise HTTPException(status_code=400, detail="Uploaded text content is blank")

    try:
        chunks = chunk_text(text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if not chunks:
        raise HTTPException(status_code=400, detail="Uploaded text produced no chunks")

    document_id = str(uuid4())
    first_embedding = embedding_client.embed_text(chunks[0].text)

    try:
        if not vector_store.collection_exists():
            vector_store.create_collection(vector_size=len(first_embedding))
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not prepare Qdrant collection",
        ) from exc

    embeddings = [first_embedding]
    for chunk in chunks[1:]:
        embeddings.append(embedding_client.embed_text(chunk.text))

    try:
        stored_chunk_count = vector_store.upsert_chunks(
            document_id=document_id,
            filename=filename,
            chunks=chunks,
            embeddings=embeddings,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="Could not store document chunks in Qdrant",
        ) from exc

    return DocumentUploadResponse(
        document_id=document_id,
        filename=filename,
        chunk_count=len(chunks),
        stored_chunk_count=stored_chunk_count,
    )


def _validate_text_file(file: UploadFile) -> str:
    filename = file.filename or ""
    if Path(filename).suffix.lower() != SUPPORTED_EXTENSION:
        raise HTTPException(
            status_code=415,
            detail="Only .txt files are supported",
        )

    content_type = (file.content_type or "").split(";", maxsplit=1)[0].strip().lower()
    if content_type and content_type != SUPPORTED_CONTENT_TYPE:
        raise HTTPException(
            status_code=415,
            detail="Only text/plain uploads are supported",
        )

    return filename
