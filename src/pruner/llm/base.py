"""LLM provider interface and the advisory analyzer.

Providers do one thing: turn a system+user prompt into raw text. All parsing,
validation, retrying, caching and failure handling is shared, so adding a provider
is a dozen lines and cannot change decision semantics.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any

from pruner.config import LlmConfig
from pruner.llm.prompt import SYSTEM_PROMPT, build_user_prompt
from pruner.llm.schema import RESPONSE_SCHEMA, parse_response
from pruner.models import BugSnapshot, LlmVerdict, RuleHit
from pruner.store import Store, verdict_fingerprint

log = logging.getLogger(__name__)


class ProviderError(RuntimeError):
    """The provider could not be reached or returned an error."""


class Provider(ABC):
    """A text-in, text-out completion backend."""

    name: str = "provider"

    def __init__(self, config: LlmConfig) -> None:
        self.config = config

    @property
    def model_id(self) -> str:
        """Identifier recorded on verdicts and used as part of the cache key."""
        return f"{self.name}:{self.config.model}"

    @abstractmethod
    def complete(self, system: str, user: str, *, schema: dict[str, Any]) -> str:
        """Return the model's raw response text."""

    def close(self) -> None:  # pragma: no cover - most providers need nothing
        return None


class Analyzer:
    """Produces advisory verdicts, with caching and fail-safe behaviour.

    Any failure -- unreachable provider, unparseable output, exhausted retries --
    yields :meth:`LlmVerdict.no_opinion`, which the policy treats as silence. The
    LLM can therefore only ever reduce the set of actions taken, never expand it.
    """

    def __init__(
        self,
        provider: Provider | None,
        config: LlmConfig,
        store: Store | None = None,
    ) -> None:
        self.provider = provider
        self.config = config
        self.store = store

    @property
    def enabled(self) -> bool:
        return self.provider is not None

    @property
    def model_id(self) -> str:
        return self.provider.model_id if self.provider else "none"

    def assess(
        self,
        bug: BugSnapshot,
        hits: tuple[RuleHit, ...],
        *,
        package: str,
        use_cache: bool = True,
    ) -> LlmVerdict:
        if self.provider is None:
            return LlmVerdict.no_opinion(model="none")

        fingerprint = verdict_fingerprint(bug)
        model_id = self.provider.model_id

        if use_cache and self.store is not None:
            cached = self.store.get_verdict(bug.id, model_id, fingerprint)
            if cached is not None:
                return cached

        user = build_user_prompt(bug, hits, self.config, package=package)
        verdict = self._call(bug.id, user, model_id)

        if self.store is not None:
            self.store.put_verdict(bug.id, model_id, fingerprint, verdict)
        return verdict

    def _call(self, bug_id: int, user: str, model_id: str) -> LlmVerdict:
        assert self.provider is not None
        last_problem = ""
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                raw = self.provider.complete(SYSTEM_PROMPT, user, schema=RESPONSE_SCHEMA)
            except ProviderError as exc:
                last_problem = str(exc)
                log.warning("bug #%s: LLM call failed (attempt %d): %s", bug_id, attempt, exc)
                continue

            if verdict := parse_response(raw, model=model_id):
                return verdict

            last_problem = "unparseable response"
            log.warning(
                "bug #%s: could not parse LLM response (attempt %d): %r",
                bug_id,
                attempt,
                raw[:200],
            )

        log.warning(
            "bug #%s: no usable LLM verdict (%s); recording no opinion", bug_id, last_problem
        )
        return LlmVerdict.no_opinion(model=model_id, failed=True)

    def close(self) -> None:
        if self.provider is not None:
            self.provider.close()
