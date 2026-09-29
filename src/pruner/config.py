"""Configuration: policy thresholds, rule toggles, LLM provider settings.

All tunables live in a TOML file (``pruner.toml`` by default) so that a run is
reproducible and reviewable. Nothing that affects which bugs get touched is
hardcoded in the rules themselves.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pruner.models import BugKind, Importance, SupportPolicy

DEFAULT_CONFIG_FILENAMES: tuple[str, ...] = ("pruner.toml", ".pruner.toml")

SERVICE_ROOTS: dict[str, str] = {
    "production": "https://api.launchpad.net/devel",
    "staging": "https://api.staging.launchpad.net/devel",
}

class LaunchpadConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    service: Literal["production", "staging"] = "production"
    distribution: str = "ubuntu"

    support_policy: SupportPolicy = SupportPolicy.STANDARD
    """How to decide which releases are still worth fixing bugs on.

    ``standard`` (default) treats a release as end-of-life once *standard* support
    ends, even if Launchpad still calls it Supported because of ESM. This matters
    a great deal: Launchpad reports trusty, xenial, bionic and focal as Supported,
    so trusting it alone would make the EOL rules fire on almost nothing.

    ``launchpad`` trusts Launchpad's status. ``explicit`` uses ``live_series``.
    See :mod:`pruner.lp.series`.
    """

    live_series: tuple[str, ...] = ()
    """Series to treat as live under ``support_policy = "explicit"``."""

    timeout: float = 30.0
    max_retries: int = 5
    max_concurrency: int = 4
    """Kept deliberately low. Launchpad throttles, and a backlog sweep is not
    something that needs to finish in ten seconds."""

    user_agent: str = "pruner/0.1 (Launchpad backlog triage; +https://launchpad.net)"

    @property
    def api_root(self) -> str:
        return SERVICE_ROOTS[self.service]


class SafetyConfig(BaseModel):
    """Hard exclusions. Every one of these is a veto on touching a bug."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_quiet_days: int = 180
    """A bug touched more recently than this is never actioned, no matter what."""

    protect_importances: tuple[Importance, ...] = (Importance.CRITICAL, Importance.HIGH)
    protect_users_affected: int = 5
    protect_duplicates: int = 3

    protect_tags: tuple[str, ...] = (
        "regression-*",
        "rls-*",
        "block-proposed*",
        "sru-*",
        "verification-*",
        "champagne",
        "patch",
    )
    """fnmatch patterns. A bug carrying any matching tag is left alone."""

    skip_private: bool = True
    skip_security: bool = True
    skip_with_patch: bool = True
    skip_assigned: bool = True
    skip_milestoned: bool = True
    skip_with_dev_activity: bool = True

    max_actions_per_run: int = 50
    """Circuit breaker. A policy bug should not be able to rewrite a whole backlog."""

    action_delay_seconds: float = 1.0


class RulesConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: tuple[str, ...] = (
        "eol_series_tasks",
        "eol_series_tag",
        "eol_apport_release",
        "likely_fixed",
        "removed_from_archive",
        "empty_report",
    )

    min_desc_chars: int = 120
    """Below this many characters of actual prose, a report is considered empty."""

    fixed_upstream_statuses: tuple[str, ...] = (
        "RESOLVED",
        "CLOSED",
        "FIXED",
        "VERIFIED",
        "Fix Released",
        "RESOLVED FIXED",
    )
    """Upstream ``remote_status`` values that suggest the bug is fixed elsewhere."""


class LlmConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["ollama", "anthropic", "openai", "openrouter", "none"] = "ollama"
    model: str = "qwen2.5:7b"
    base_url: str = ""
    """Override the provider's default endpoint. Mostly for a remote Ollama."""

    api_key_env: str = ""
    """Environment variable holding the API key. Defaults per provider."""

    timeout: float = 120.0
    max_attempts: int = 2
    """One retry on unparseable JSON, then we record "no opinion" and move on."""

    temperature: float = 0.0

    veto_threshold: float = Field(default=0.6, ge=0.0, le=1.0)
    """Minimum confidence for the LLM to block a rule-proposed action."""

    reclassify_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    """Minimum confidence for the LLM to turn needs-info into invalid."""

    allow_llm_reclassify_to_invalid: bool = True

    reclassify_kinds: tuple[BugKind, ...] = (BugKind.SUPPORT_QUESTION, BugKind.SPAM)
    """Bug kinds the model may use to turn ``needs-info`` into ``invalid``.

    Feature requests are excluded on purpose: Ubuntu convention keeps them open at
    Wishlist importance rather than closing them as Invalid. Add
    ``"feature-request"`` here if your project disagrees.
    """

    max_description_chars: int = 6000
    max_comments: int = 6
    max_comment_chars: int = 800

    @property
    def enabled(self) -> bool:
        return self.provider != "none"


class CommentConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    marker: str = "[pruner-automated-triage]"
    """Stable token included in every comment we post. Used to recognise our own
    prior comments so re-runs are idempotent."""

    signature: str = (
        "This comment was posted by an automated backlog triage tool. "
        "If this bug is still reproducible on a supported Ubuntu release, please "
        "reply with the details above and set the status back to New or Confirmed - "
        "that is all it takes to keep it open."
    )

    include_rule_names: bool = True
    """Include which rule fired. Transparency beats looking magical."""


class Config(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    launchpad: LaunchpadConfig = LaunchpadConfig()
    safety: SafetyConfig = SafetyConfig()
    rules: RulesConfig = RulesConfig()
    llm: LlmConfig = LlmConfig()
    comment: CommentConfig = CommentConfig()

    source_path: Path | None = None

    @model_validator(mode="after")
    def _check_rules_known(self) -> Self:
        from pruner.rules import unknown_rule_names

        if unknown := unknown_rule_names(self.rules.enabled):
            raise ValueError(
                f"unknown rule(s) in [rules].enabled: {', '.join(sorted(unknown))}"
            )
        return self


def find_config(start: Path | None = None) -> Path | None:
    """Locate a config file by walking up from ``start`` (default: cwd)."""
    current = (start or Path.cwd()).resolve()
    for directory in (current, *current.parents):
        for name in DEFAULT_CONFIG_FILENAMES:
            candidate = directory / name
            if candidate.is_file():
                return candidate
    return None


def load_config(path: Path | None = None, *, search: bool = True) -> Config:
    """Load and validate configuration.

    With no explicit path, searches upwards for ``pruner.toml``; if none exists the
    built-in defaults are used, which are deliberately the conservative ones.
    """
    resolved = path or (find_config() if search else None)
    if resolved is None:
        return Config()

    with resolved.open("rb") as handle:
        raw: dict[str, Any] = tomllib.load(handle)

    raw.pop("source_path", None)
    return Config(**raw, source_path=resolved)
