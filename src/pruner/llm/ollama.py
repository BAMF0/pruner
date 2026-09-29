"""Local Ollama provider.

The default, because it costs nothing and needs no key. Uses Ollama's structured
output support (``format`` accepting a JSON schema), which makes small models
dramatically more reliable at returning parseable JSON.
"""

from __future__ import annotations

from typing import Any

import httpx

from pruner.config import LlmConfig
from pruner.llm.base import Provider, ProviderError

DEFAULT_BASE_URL = "http://localhost:11434"


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self, config: LlmConfig) -> None:
        super().__init__(config)
        self.base_url = (config.base_url or DEFAULT_BASE_URL).rstrip("/")
        self._client = httpx.Client(timeout=config.timeout)

    def complete(self, system: str, user: str, *, schema: dict[str, Any]) -> str:
        payload = {
            "model": self.config.model,
            "stream": False,
            "format": schema,
            "options": {"temperature": self.config.temperature},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        try:
            response = self._client.post(f"{self.base_url}/api/chat", json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"cannot reach Ollama at {self.base_url}: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"Ollama returned HTTP {response.status_code}: {response.text[:300]}"
            )

        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderError("Ollama returned non-JSON") from exc

        content = (body.get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProviderError("Ollama returned an empty message")
        return content

    def close(self) -> None:
        self._client.close()
