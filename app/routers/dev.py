from fastapi import APIRouter, Depends, HTTPException

from app.embeddings.ollama_embedding_client import OllamaEmbeddingClient
from app.schemas import DevEmbeddingRequest, DevEmbeddingResponse

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
