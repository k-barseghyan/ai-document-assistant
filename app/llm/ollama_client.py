from __future__ import annotations

import os

import httpx

from app.exceptions import EmptyAnswerError, LLMClientError


DEFAULT_API_SYSTEM_MESSAGE = (
    "Answer briefly and directly. Do not guess. "
    "If you are not sure, say that you do not know."
)


class OllamaClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float | None = None,
        api_temperature: float | None = None,
        api_num_predict: int | None = None,
        chat_temperature: float | None = None,
        chat_num_predict: int | None = None,
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
        self.api_options = {
            "temperature": (
                api_temperature
                if api_temperature is not None
                else float(os.getenv("OLLAMA_API_TEMPERATURE", "0"))
            ),
        }
        api_num_predict_env = os.getenv("OLLAMA_API_NUM_PREDICT")
        if api_num_predict is not None:
            self.api_options["num_predict"] = api_num_predict
        elif api_num_predict_env is not None and api_num_predict_env.strip():
            self.api_options["num_predict"] = int(api_num_predict_env)

        self.chat_options = {
            "num_predict": (
                chat_num_predict
                if chat_num_predict is not None
                else int(os.getenv("OLLAMA_CHAT_NUM_PREDICT", "400"))
            ),
            "temperature": (
                chat_temperature
                if chat_temperature is not None
                else float(os.getenv("OLLAMA_CHAT_TEMPERATURE", "0.7"))
            ),
        }

    def generate(self, prompt: str, system: str = DEFAULT_API_SYSTEM_MESSAGE) -> str:
        try:
            response = httpx.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "system": system,
                    "stream": False,
                    "options": self.api_options,
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
                    "options": self.chat_options,
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
