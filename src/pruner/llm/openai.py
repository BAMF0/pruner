"""OpenAI-compatible chat-completions provider.

Serves both ``openai`` and ``openrouter``, since OpenRouter implements the same
wire format. Uses ``response_format: json_schema`` where the endpoint supports it
and degrades to ``json_object`` on the (common) rejection of strict schemas by
proxies and third-party models.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

from pruner.config import LlmConfig
from pruner.llm.base import Provider, ProviderError

DEFAULTS: dict[str, tuple[str, str]] = {
    # provider name -> (base url, api key env var)
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
}


class OpenAICompatibleProvider(Provider):
    def __init__(self, config: LlmConfig, *, name: str) -> None:
        super().__init__(config)
        self.name = name
        default_url, default_env = DEFAULTS.get(name, DEFAULTS["openai"])
        env = config.api_key_env or default_env
        key = os.environ.get(env, "").strip()
        if not key:
            raise ProviderError(
                f"no API key in ${env}; set it or use --llm none / provider = 'ollama'"
            )
        self.base_url = (config.base_url or default_url).rstrip("/")
        self._client = httpx.Client(
            timeout=config.timeout,
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        self._strict_schema_supported = True

    def complete(self, system: str, user: str, *, schema: dict[str, Any]) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        if self._strict_schema_supported:
            text = self._post(messages, self._schema_format(schema), tolerate_400=True)
            if text is not None:
                return text
            # Endpoint rejected the strict schema; stop trying it for this run.
            self._strict_schema_supported = False

        result = self._post(messages, {"type": "json_object"}, tolerate_400=False)
        assert result is not None
        return result

    @staticmethod
    def _schema_format(schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "json_schema",
            "json_schema": {"name": "bug_assessment", "strict": True, "schema": schema},
        }

    def _post(
        self,
        messages: list[dict[str, str]],
        response_format: dict[str, Any],
        *,
        tolerate_400: bool,
    ) -> str | None:
        payload = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "messages": messages,
            "response_format": response_format,
        }
        try:
            response = self._client.post(f"{self.base_url}/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise ProviderError(f"cannot reach {self.name}: {exc}") from exc

        if response.status_code == 400 and tolerate_400:
            return None
        if response.status_code >= 400:
            raise ProviderError(
                f"{self.name} returned HTTP {response.status_code}: {response.text[:300]}"
            )

        body = response.json()
        choices = body.get("choices") or []
        if not choices:
            raise ProviderError(f"{self.name} returned no choices")
        content = (choices[0].get("message") or {}).get("content")
        if not isinstance(content, str) or not content.strip():
            raise ProviderError(f"{self.name} returned empty content")
        return content

    def close(self) -> None:
        self._client.close()
