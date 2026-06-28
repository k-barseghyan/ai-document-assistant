from __future__ import annotations

import os

import httpx

from app.exceptions import EmptyAnswerError, LLMClientError


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
    ):
        self.base_url = (
            base_url or os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
        ).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL") or "qwen3:14b"
        self.timeout = (
            timeout
            if timeout is not None
            else float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "300"))
        )

    def generate(self, prompt: str) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
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

        answer = data.get("response", "")
        if not answer.strip():
            raise EmptyAnswerError("Ollama returned an empty answer")

        return answer

    def chat(self, messages: list[dict[str, str]]) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {
                        "num_predict": 300,
                        "temperature": 0.2,
                    },
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

        answer = data.get("message", {}).get("content", "")
        if not answer.strip():
            raise EmptyAnswerError("Ollama returned an empty answer")

        return answer
