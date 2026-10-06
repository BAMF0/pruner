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

from pruner.models import BugKind, Importance, RuleClaim, SupportPolicy

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
    """Parallel HTTP requests.

    Fetching is latency-bound, not bandwidth-bound: a request costs ~230ms
    against production Launchpad while often returning a couple of hundred bytes,
    and a fully enriched bug needs nine of them. Launchpad is HTTP/1.1 only, so
    there is no multiplexing to exploit -- this is a pool of parallel connections.

    Measured end-to-end on previously-unfetched packages (first touch, so no
    Launchpad-side cache warming): 230ms/request sequential, 95ms at 4 workers
    (2.4x), 45ms at 8 (5.1x). The pipeline reaches roughly 85% of the
    depth-limited optimum at a given worker count, so the worker count is the
    knob that matters.

    4 is the default because Launchpad publishes no rate limits, launchpadlib
    itself is sequential, and a backlog sweep is not urgent. If you are working
    through a large package, ``--concurrency 8`` roughly halves the time again.
    """

    chunk_size: int = 100
    """Bugs processed per pipeline pass. Bounds peak memory on large backlogs."""

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


class AgeConfig(BaseModel):
    """Hardening of ``needs-info`` into ``wont-fix`` for very old bugs.

    This only ever *hardens* an action the rules already authorised. It never
    creates eligibility: a bug no prune rule flagged is untouched no matter how
    old it is. See :mod:`pruner.policy`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    wont_fix_after_days: int = 2555
    """Bugs older than this many days get Won't Fix instead of Incomplete.

    ~7 years: comfortably older than every release Launchpad still lists as
    supported, so an over-age bug concerns nothing we ship. ``0`` disables age
    escalation entirely.
    """

    claims: tuple[RuleClaim, ...] = (RuleClaim.LIFECYCLE,)
    """Which rule claims age may escalate, mirroring how the LLM's veto is
    scoped by :class:`~pruner.models.RuleClaim`.

    Only ``lifecycle`` by default: "old AND on a dead release" is the airtight
    case. A ``quality`` hit (``empty_report``) still just asks for information --
    a thin report being old is not itself a reason to close it. ``existence``
    is pointless to list since ``removed_from_archive`` already proposes
    ``invalid`` directly.
    """


class AuthConfig(BaseModel):
    """How the write path authenticates to Launchpad.

    The credential itself never appears in this file -- config names an
    environment variable or a path, exactly as ``llm.api_key_env`` does. See
    :mod:`pruner.secrets`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    token_env: str = "PRUNER_LP_CREDENTIALS"
    """Environment variable holding a serialised launchpadlib credential.

    The value is the OAuth 1.0a credential for the account you want to act as
    (a bot, typically), produced once by authorising in a browser. It never
    touches this file, the keyring, or any other disk location.
    """

    credentials_file: Path | None = None
    """Path to a launchpadlib credentials file. Must be ``chmod 600``."""

    allow_interactive: bool = True
    """Fall back to launchpadlib's keyring/browser flow when nothing above
    supplied a credential. On by default so a laptop ``pruner apply`` keeps
    working for a human; set ``false`` in automation so a revoked token fails
    fast instead of blocking on a browser prompt."""

    @model_validator(mode="before")
    @classmethod
    def _reject_literal_secrets(cls, data: Any) -> Any:
        """Catch the one mistake that matters here: a credential pasted into
        ``pruner.toml``. ``extra="forbid"`` would already reject these keys,
        but with an unhelpful message, and this is the error worth naming."""
        if isinstance(data, dict):
            forbidden = {"token", "access_token", "secret", "consumer_secret"} & set(data)
            if forbidden:
                keys = ", ".join(sorted(forbidden))
                raise ValueError(
                    f"[auth] must not contain credential material ({keys}). Put the "
                    "serialised credential in the environment variable named by "
                    "token_env, or in a chmod 600 file named by credentials_file."
                )
        return data


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
    age: AgeConfig = AgeConfig()
    auth: AuthConfig = AuthConfig()
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
