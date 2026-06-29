from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import sha256
from numbers import Real
from uuid import UUID

from qdrant_client import QdrantClient, models

from app.rag.chunker import TextChunk


DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "document_chunks"


@dataclass(frozen=True)
class RetrievedChunk:
    score: float
    document_id: str
    document_hash: str
    filename: str
    chunk_index: int
    chunk_hash: str
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

    def delete_collection(self) -> bool:
        if not self.collection_exists():
            return False

        self.client.delete_collection(collection_name=self.collection_name)
        return True

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
        document_hash: str,
        filename: str,
        chunks: Sequence[TextChunk],
        embeddings: Sequence[Sequence[float]],
    ) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")

        points: list[models.PointStruct] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            chunk_hash = _chunk_hash(chunk.text)
            points.append(
                models.PointStruct(
                    id=_point_id(
                        document_id=document_id,
                        chunk_index=chunk.index,
                        chunk_hash=chunk_hash,
                    ),
                    vector=[float(value) for value in embedding],
                    payload={
                        "document_id": document_id,
                        "document_hash": document_hash,
                        "filename": filename,
                        "chunk_index": chunk.index,
                        "chunk_hash": chunk_hash,
                        "text": chunk.text,
                        "char_count": chunk.char_count,
                    },
                )
            )

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

        chunks: list[RetrievedChunk] = []
        for point in response.points:
            if point.payload is None:
                continue

            payload = point.payload
            text = str(payload["text"])
            document_id = str(payload["document_id"])
            chunks.append(
                RetrievedChunk(
                    score=float(point.score),
                    document_id=document_id,
                    document_hash=str(payload.get("document_hash", document_id)),
                    filename=str(payload["filename"]),
                    chunk_index=int(payload["chunk_index"]),
                    chunk_hash=str(payload.get("chunk_hash", _chunk_hash(text))),
                    text=text,
                    char_count=int(payload["char_count"]),
                )
            )

        return chunks


def _chunk_hash(text: str) -> str:
    return _hash_text(_normalize_chunk_text_for_hash(text))


def _point_id(document_id: str, chunk_index: int, chunk_hash: str) -> str:
    point_id_hash = _hash_text(f"{document_id}:{chunk_index}:{chunk_hash}")
    # Qdrant string point IDs are UUID-shaped, so derive one from this SHA-256.
    return str(UUID(hex=point_id_hash[:32]))


def _normalize_chunk_text_for_hash(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _hash_text(text: str) -> str:
    return sha256(text.encode("utf-8")).hexdigest()
