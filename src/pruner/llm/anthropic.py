"""Anthropic Messages API provider."""

from __future__ import annotations

import json
import os
from typing import Any

import httpx

from pruner.config import LlmConfig
from pruner.llm.base import Provider, ProviderError

DEFAULT_BASE_URL = "https://api.anthropic.com"
API_VERSION = "2023-06-01"
DEFAULT_KEY_ENV = "ANTHROPIC_API_KEY"


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, config: LlmConfig) -> None:
        super().__init__(config)
        env = config.api_key_env or DEFAULT_KEY_ENV
        key = os.environ.get(env, "").strip()
        if not key:
            raise ProviderError(
                f"no Anthropic API key in ${env}; set it or use --llm none / provider = 'ollama'"
            )
        self.base_url = (config.base_url or DEFAULT_BASE_URL).rstrip("/")
        self._client = httpx.Client(
            timeout=config.timeout,
            headers={
                "x-api-key": key,
                "anthropic-version": API_VERSION,
                "content-type": "application/json",
            },
        )

    def complete(self, system: str, user: str, *, schema: dict[str, Any]) -> str:
        # The Messages API has no native JSON-schema mode, so the schema is
        # supplied in-prompt and a leading "{" is pre-filled to suppress preamble.
        instruction = (
            f"{user}\n\nReturn a single JSON object conforming to this schema:\n"
            f"{json.dumps(schema, indent=2)}"
        )
        payload = {
            "model": self.config.model,
            "max_tokens": 1024,
            "temperature": self.config.temperature,
            "system": system,
            "messages": [
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": "{"},
            ],
        }
        try:
            response = self._client.post(f"{self.base_url}/v1/messages", json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"cannot reach Anthropic: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"Anthropic returned HTTP {response.status_code}: {response.text[:300]}"
            )

        body = response.json()
        blocks = body.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        if not text.strip():
            raise ProviderError("Anthropic returned no text content")
        # Re-attach the pre-filled brace.
        return "{" + text

    def close(self) -> None:
        self._client.close()
