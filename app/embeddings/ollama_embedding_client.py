from __future__ import annotations

import os
from numbers import Real

import httpx

from app.exceptions import LLMClientError


DEFAULT_OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"


class OllamaEmbeddingClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        ).rstrip("/")
        self.model = (
            model
            or os.getenv("OLLAMA_EMBEDDING_MODEL")
            or DEFAULT_OLLAMA_EMBEDDING_MODEL
        )
        self.timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
        )

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("text must be non-empty")

        try:
            response = httpx.post(
                f"{self.base_url}/api/embed",
                json={
                    "model": self.model,
                    "input": text,
                    "truncate": False,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code
            raise LLMClientError(f"Ollama request failed with HTTP {status_code}") from exc
        except httpx.TimeoutException as exc:
            raise LLMClientError("Ollama request timed out") from exc
        except httpx.ConnectError as exc:
            raise LLMClientError("Could not connect to Ollama") from exc
        except httpx.RequestError as exc:
            raise LLMClientError("Could not connect to Ollama") from exc
        except ValueError as exc:
            raise LLMClientError("Ollama returned an invalid JSON response") from exc

        embeddings = data.get("embeddings")
        if not embeddings or not isinstance(embeddings, list):
            raise LLMClientError("Ollama returned no embeddings")

        vector = embeddings[0]
        if not vector or not isinstance(vector, list):
            raise LLMClientError("Ollama returned an invalid embedding")

        if not all(
            isinstance(value, Real) and not isinstance(value, bool)
            for value in vector
        ):
            raise LLMClientError("Ollama returned a non-numeric embedding")

        return [float(value) for value in vector]
