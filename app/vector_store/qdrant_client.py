from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from uuid import uuid4

from qdrant_client import QdrantClient, models

from app.rag.chunker import TextChunk


DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "document_chunks"


@dataclass(frozen=True)
class RetrievedChunk:
    score: float
    document_id: str
    filename: str
    chunk_index: int
    text: str
    char_count: int


class QdrantVectorStoreClient:
    def __init__(
        self,
        url: str | None = None,
        collection_name: str | None = None,
    ):
        self.url = url or os.getenv("QDRANT_URL") or DEFAULT_QDRANT_URL
        self.collection_name = (
            collection_name
            or os.getenv("QDRANT_COLLECTION")
            or DEFAULT_QDRANT_COLLECTION
        )
        self.client = QdrantClient(url=self.url)

    def list_collections(self) -> list[str]:
        collections = self.client.get_collections()
        return [collection.name for collection in collections.collections]

    def collection_exists(self) -> bool:
        return self.client.collection_exists(collection_name=self.collection_name)

    def create_collection(self, vector_size: int) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size must be positive")

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        points = [
            models.PointStruct(
                id=str(uuid4()),
                vector=[float(value) for value in embedding],
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": chunk.index,
                    "text": chunk.text,
                    "char_count": chunk.char_count,
                },
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]

        if not points:
            return 0

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True,
        )
        return len(points)

    def search_similar_chunks(
        self,
        query_vector: list[float],
        limit: int,
    ) -> list[RetrievedChunk]:
        if not query_vector:
            raise ValueError("query_vector must be non-empty")
        if limit <= 0:
            raise ValueError("limit must be positive")
        if not all(
            isinstance(value, Real) and not isinstance(value, bool)
            for value in query_vector
        ):
            raise ValueError("query_vector must contain only numeric values")

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=[float(value) for value in query_vector],
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        return [
            RetrievedChunk(
                score=float(point.score),
                document_id=str(point.payload["document_id"]),
                filename=str(point.payload["filename"]),
                chunk_index=int(point.payload["chunk_index"]),
                text=str(point.payload["text"]),
                char_count=int(point.payload["char_count"]),
            )
            for point in response.points
            if point.payload is not None
        ]
