"""LLM provider selection.

``provider = "none"`` yields no provider at all, and :class:`~pruner.llm.base.Analyzer`
then returns "no opinion" for every bug. That makes the entire pipeline runnable
and testable with zero inference -- useful for tuning rule thresholds against a
real backlog before spending anything on tokens.
"""

from __future__ import annotations

from pruner.config import LlmConfig
from pruner.llm.base import Analyzer, Provider, ProviderError
from pruner.store import Store


def build_provider(config: LlmConfig) -> Provider | None:
    """Instantiate the configured provider, or ``None`` when disabled."""
    match config.provider:
        case "none":
            return None
        case "ollama":
            from pruner.llm.ollama import OllamaProvider

            return OllamaProvider(config)
        case "anthropic":
            from pruner.llm.anthropic import AnthropicProvider

            return AnthropicProvider(config)
        case "openai" | "openrouter" as name:
            from pruner.llm.openai import OpenAICompatibleProvider

            return OpenAICompatibleProvider(config, name=name)
        case other:  # pragma: no cover - guarded by the config Literal
            raise ProviderError(f"unknown LLM provider: {other}")


def build_analyzer(config: LlmConfig, store: Store | None = None) -> Analyzer:
    return Analyzer(build_provider(config), config, store)


__all__ = ["Analyzer", "Provider", "ProviderError", "build_analyzer", "build_provider"]
