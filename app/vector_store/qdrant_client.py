from __future__ import annotations

import os

from qdrant_client import QdrantClient


DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_QDRANT_COLLECTION = "document_chunks"


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
