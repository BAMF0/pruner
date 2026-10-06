"""Core data model for pruner.

Everything downstream of :mod:`pruner.lp.read` operates on these types rather than
raw Launchpad JSON. Rules are pure functions over :class:`BugSnapshot`, which makes
them trivially unit-testable against recorded fixtures.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SeriesStatus(StrEnum):
    """Launchpad `distro_series.status` values.

    Verified against ``/devel/ubuntu/series``. We never hardcode EOL dates; the
    distribution itself is the source of truth for what is still alive.
    """

    FUTURE = "Future"
    EXPERIMENTAL = "Experimental"
    DEVELOPMENT = "Active Development"
    FROZEN = "Pre-release Freeze"
    CURRENT = "Current Stable Release"
    SUPPORTED = "Supported"
    OBSOLETE = "Obsolete"

    @property
    def is_obsolete(self) -> bool:
        return self is SeriesStatus.OBSOLETE


class SupportPolicy(StrEnum):
    """How to decide which releases are still worth fixing bugs on.

    Lives here rather than in :mod:`pruner.lp.series` so that :mod:`pruner.config`
    can reference it without importing the Launchpad client. See
    :mod:`pruner.lp.series` for the full rationale and for why Launchpad's own
    ``status`` field is insufficient.
    """

    STANDARD = "standard"
    """End-of-life once *standard* support ends, even if Launchpad still reports
    the series as Supported because of ESM."""

    LAUNCHPAD = "launchpad"
    """Trust Launchpad's status. ESM releases count as supported."""

    EXPLICIT = "explicit"
    """The operator names the live series."""


class BugTaskStatus(StrEnum):
    NEW = "New"
    INCOMPLETE = "Incomplete"
    OPINION = "Opinion"
    INVALID = "Invalid"
    WONT_FIX = "Won't Fix"
    DEFERRED = "Deferred"
    EXPIRED = "Expired"
    CONFIRMED = "Confirmed"
    TRIAGED = "Triaged"
    IN_PROGRESS = "In Progress"
    FIX_COMMITTED = "Fix Committed"
    FIX_RELEASED = "Fix Released"
    DOES_NOT_EXIST = "Does not exist"
    UNKNOWN = "Unknown"


#: Statuses that mean somebody is already acting on the bug, or it is already closed.
#: A bug with any task in one of these states is never a prune candidate.
PROGRESSING_STATUSES: frozenset[BugTaskStatus] = frozenset(
    {
        BugTaskStatus.IN_PROGRESS,
        BugTaskStatus.FIX_COMMITTED,
        BugTaskStatus.FIX_RELEASED,
    }
)

#: Statuses that mean the task is already resolved//closed, so there is nothing to prune.
CLOSED_STATUSES: frozenset[BugTaskStatus] = frozenset(
    {
        BugTaskStatus.INVALID,
        BugTaskStatus.WONT_FIX,
        BugTaskStatus.EXPIRED,
        BugTaskStatus.OPINION,
        BugTaskStatus.FIX_RELEASED,
        BugTaskStatus.DOES_NOT_EXIST,
    }
)

#: The statuses we consider "open backlog" and therefore fetch by default.
DEFAULT_FETCH_STATUSES: tuple[BugTaskStatus, ...] = (
    BugTaskStatus.NEW,
    BugTaskStatus.CONFIRMED,
    BugTaskStatus.TRIAGED,
    BugTaskStatus.INCOMPLETE,
)


class Importance(StrEnum):
    UNKNOWN = "Unknown"
    CRITICAL = "Critical"
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    WISHLIST = "Wishlist"
    UNDECIDED = "Undecided"


class Action(StrEnum):
    """What the pipeline proposes to do to a bug."""

    KEEP = "keep"
    """Leave the bug completely alone."""

    NEEDS_INFO = "needs-info"
    """Set the task to Incomplete and ask for confirmation. Reversible: any reply
    reopens it, and Launchpad's janitor expires it if nobody answers."""

    INVALID = "invalid"
    """Set the task to Invalid. Used for not-a-bug and gone-from-the-archive cases."""

    WONT_FIX = "wont-fix"
    """Set the task to Won't Fix because the report is too old to verify against
    anything we still ship.

    Deliberately distinct from ``INVALID``: Won't Fix does not claim the report
    was never a real bug, only that we will not act on it. Only :mod:`pruner.policy`
    may emit it, as an age escalation of an already-eligible ``needs-info`` -- no
    rule proposes it and the LLM is never offered it, because "how old is this
    bug" is a deterministic fact the model has nothing to add to.
    """

    ESCALATE = "escalate"
    """Flag for human attention without changing anything.

    Reserved: the model may recommend it, and it is accepted and reported, but no
    rule proposes it and :mod:`pruner.policy` never emits it, so nothing acts on
    it today. It exists because "this needs a person" is a genuinely distinct
    outcome from "leave it alone", and a future `escalate` action would only ever
    add a tag -- but until something actually consumes the tag, emitting it would
    just be noise on other people's bugs.
    """

    @property
    def mutates_status(self) -> bool:
        return self in (Action.NEEDS_INFO, Action.INVALID, Action.WONT_FIX)


class IsABug(StrEnum):
    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"


class BugKind(StrEnum):
    DEFECT = "defect"
    FEATURE_REQUEST = "feature-request"
    SUPPORT_QUESTION = "support-question"
    PACKAGING = "packaging"
    DOCUMENTATION = "documentation"
    UPSTREAM = "upstream"
    SPAM = "spam"
    UNCLEAR = "unclear"


class RuleClaim(StrEnum):
    """What kind of assertion a rule makes.

    This determines how the LLM is allowed to veto it, and the distinction is
    essential rather than cosmetic:

    A ``LIFECYCLE`` rule claims "this concerns a release we no longer ship". An LLM
    reporting "this is a well-written, genuine, reproducible defect" does **not**
    contradict that claim -- a real defect on a dead release is still unverifiable.
    If a generic "it's a real bug" veto were applied to lifecycle rules, the
    best-written EOL reports would be exactly the ones never pruned, which is
    backwards and would make the EOL rules useless.

    A ``QUALITY`` rule claims "there is not enough here to act on". That claim
    *is* directly contradicted by "a triager could reproduce this", so the LLM
    gets a full veto.

    An ``EXISTENCE`` rule claims "this package is not in the archive any more",
    which is a matter of archive state that the LLM has no basis to overrule.
    """

    LIFECYCLE = "lifecycle"
    QUALITY = "quality"
    EXISTENCE = "existence"


# ---------------------------------------------------------------------------
# Launchpad-derived snapshots
# ---------------------------------------------------------------------------

#: Matches Launchpad's ``bug_target_name`` for distribution (source package) tasks,
#: e.g. ``"vim (Ubuntu)"`` or ``"vim (Ubuntu Jammy)"``. Both formats verified live.
_TARGET_RE = re.compile(
    r"^(?P<package>[^\s(]+)\s+\((?P<distro>[^\s)]+)(?:\s+(?P<series>[^\s)]+))?\)$"
)


def _lp_name(link: str | None) -> str | None:
    """Extract the trailing name from a Launchpad API link.

    ``https://api.launchpad.net/devel/~pitti`` -> ``pitti``
    """
    if not link:
        return None
    return link.rstrip("/").rsplit("/", 1)[-1].lstrip("~") or None


def _dt(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


class TaskSnapshot(BaseModel):
    """A single Launchpad bug task (one bug can have many)."""

    model_config = ConfigDict(frozen=True)

    target_name: str
    """Raw ``bug_target_name``, e.g. ``"vim (Ubuntu Jammy)"``."""

    package: str | None = None
    distribution: str | None = None
    """Lowercased distribution name, e.g. ``"ubuntu"``. ``None`` for upstream tasks."""

    series: str | None = None
    """Lowercased series name, e.g. ``"jammy"``. ``None`` for the generic distro task."""

    status: BugTaskStatus = BugTaskStatus.UNKNOWN
    importance: Importance = Importance.UNDECIDED
    assignee: str | None = None
    milestone: str | None = None
    is_complete: bool = False
    date_created: datetime | None = None
    date_incomplete: datetime | None = None
    self_link: str | None = None

    @classmethod
    def from_api(cls, raw: dict[str, Any]) -> Self:
        target = str(raw.get("bug_target_name") or "")
        package: str | None = None
        distribution: str | None = None
        series: str | None = None

        if match := _TARGET_RE.match(target):
            package = match["package"]
            distribution = match["distro"].lower()
            series = match["series"].lower() if match["series"] else None
        elif target:
            # Upstream project task: bare name with no parenthesised target.
            package = target

        return cls(
            target_name=target,
            package=package,
            distribution=distribution,
            series=series,
            status=_coerce(BugTaskStatus, raw.get("status"), BugTaskStatus.UNKNOWN),
            importance=_coerce(Importance, raw.get("importance"), Importance.UNDECIDED),
            assignee=_lp_name(raw.get("assignee_link")),
            milestone=_lp_name(raw.get("milestone_link")),
            is_complete=bool(raw.get("is_complete", False)),
            date_created=_dt(raw.get("date_created")),
            date_incomplete=_dt(raw.get("date_incomplete")),
            self_link=raw.get("self_link"),
        )

    def is_distro_task(self, distribution: str) -> bool:
        return self.distribution == distribution.lower()


class ApportInfo(BaseModel):
    """Structured apport metadata recovered from a bug description."""

    model_config = ConfigDict(frozen=True)

    distro_release: str | None = None
    """Numeric release as reported, e.g. ``"14.04"``."""

    package: str | None = None
    """Binary package name from the ``Package:`` line, e.g. ``"vim-gtk"``."""

    version: str | None = None
    """Package version from the ``Package:`` line, e.g. ``"2:7.4.052-1ubuntu3"``."""

    problem_type: str | None = None

    @property
    def is_empty(self) -> bool:
        return not any((self.distro_release, self.package, self.version, self.problem_type))


class BugSnapshot(BaseModel):
    """Everything pruner needs to know about one bug.

    Deliberately flat and serialisable: snapshots are cached in sqlite so re-runs
    and LLM calls are idempotent, and tests replay recorded snapshots with no network.
    """

    model_config = ConfigDict(frozen=True)

    id: int
    title: str
    description: str = ""
    web_link: str = ""
    tags: tuple[str, ...] = ()

    private: bool = False
    information_type: str = "Public"
    security_related: bool = False
    cve_count: int = 0
    vulnerability_count: int = 0

    latest_patch_uploaded: datetime | None = None
    linked_mp_count: int = 0
    linked_branch_count: int = 0
    attachment_count: int = 0
    patch_attachment_count: int = 0
    """Attachments whose ``type`` is ``Patch``.

    Tracked in addition to ``latest_patch_uploaded`` as cheap redundancy. The
    attachments collection has to be fetched anyway for ``attachment_count``, and
    the exact semantics of ``latest_patch_uploaded`` across duplicates and
    upstream tasks are not documented. Discarding a contributor's patch because a
    single field was unset is not a failure worth risking for zero saving.
    """

    duplicate_of: int | None = None
    number_of_duplicates: int = 0
    users_affected_count: int = 0
    heat: int = 0
    message_count: int = 0

    date_created: datetime | None = None
    date_last_updated: datetime | None = None
    date_last_message: datetime | None = None

    tasks: tuple[TaskSnapshot, ...] = ()
    apport: ApportInfo = ApportInfo()

    comment_texts: tuple[str, ...] = ()
    """Comment bodies (excluding the description), oldest first. May be truncated."""

    remote_bug_statuses: tuple[str, ...] = ()
    """``remote_status`` of each linked upstream bug watch, used by the
    ``likely_fixed`` rule."""

    enriched: bool = False
    """True once the expensive sub-collections (CVEs, merge proposals, branches,
    attachments, comments, watches) have been fetched.

    ``fetch`` only enriches bugs that survive a cheap prefilter, because doing so
    costs ~6 extra API calls per bug. A snapshot that was never enriched is
    treated as un-actionable by :mod:`pruner.rules.exclusions`, so the optimisation
    can only ever cause us to *skip* a bug, never to wrongly act on one."""

    prefilter_reason: str | None = None
    """Why enrichment was skipped, for reporting."""

    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # -- derived helpers ---------------------------------------------------

    @property
    def has_patch(self) -> bool:
        return self.latest_patch_uploaded is not None or self.patch_attachment_count > 0

    @property
    def has_dev_activity(self) -> bool:
        return bool(self.linked_mp_count or self.linked_branch_count)

    def distro_tasks(self, distribution: str) -> tuple[TaskSnapshot, ...]:
        return tuple(t for t in self.tasks if t.is_distro_task(distribution))

    def open_distro_tasks(self, distribution: str) -> tuple[TaskSnapshot, ...]:
        return tuple(
            t
            for t in self.distro_tasks(distribution)
            if t.status not in CLOSED_STATUSES and not t.is_complete
        )

    def quiet_days(self, *, now: datetime | None = None) -> float:
        """Days since anything happened on this bug. ``inf`` if unknown."""
        reference = max(
            (d for d in (self.date_last_updated, self.date_last_message) if d),
            default=None,
        )
        if reference is None:
            return float("inf")
        return ((now or datetime.now(UTC)) - reference).total_seconds() / 86400.0

    def age_days(self, *, now: datetime | None = None) -> float | None:
        """Days since this bug was reported. ``None`` if ``date_created`` is unknown.

        Deliberately asymmetric with :meth:`quiet_days`, which returns ``inf`` when
        the dates are missing. ``quiet_days`` feeds a *veto* (the ``recently_active``
        exclusion), so failing open towards "quiet" only ever costs an opportunity
        to prune. ``age_days`` feeds an *authorisation* (closing a bug as Won't
        Fix), so it must fail closed: if we do not know how old a bug is, it is
        not old enough to close.
        """
        if self.date_created is None:
            return None
        return ((now or datetime.now(UTC)) - self.date_created).total_seconds() / 86400.0

    @classmethod
    def from_api(
        cls,
        bug: dict[str, Any],
        *,
        tasks: list[dict[str, Any]],
        cve_count: int = 0,
        vulnerability_count: int = 0,
        linked_mp_count: int = 0,
        linked_branch_count: int = 0,
        attachment_count: int = 0,
        patch_attachment_count: int = 0,
        comment_texts: tuple[str, ...] = (),
        remote_bug_statuses: tuple[str, ...] = (),
        apport: ApportInfo | None = None,
        enriched: bool = False,
        prefilter_reason: str | None = None,
    ) -> Self:
        from pruner.apport import parse_apport  # local import avoids a cycle

        description = str(bug.get("description") or "")
        return cls(
            id=int(bug["id"]),
            title=str(bug.get("title") or ""),
            description=description,
            web_link=str(bug.get("web_link") or ""),
            tags=tuple(bug.get("tags") or ()),
            private=bool(bug.get("private", False)),
            information_type=str(bug.get("information_type") or "Public"),
            security_related=bool(bug.get("security_related", False)),
            cve_count=cve_count,
            vulnerability_count=vulnerability_count,
            latest_patch_uploaded=_dt(bug.get("latest_patch_uploaded")),
            linked_mp_count=linked_mp_count,
            linked_branch_count=linked_branch_count,
            attachment_count=attachment_count,
            patch_attachment_count=patch_attachment_count,
            duplicate_of=_bug_id(bug.get("duplicate_of_link")),
            number_of_duplicates=int(bug.get("number_of_duplicates") or 0),
            users_affected_count=int(bug.get("users_affected_count") or 0),
            heat=int(bug.get("heat") or 0),
            message_count=int(bug.get("message_count") or 0),
            date_created=_dt(bug.get("date_created")),
            date_last_updated=_dt(bug.get("date_last_updated")),
            date_last_message=_dt(bug.get("date_last_message")),
            tasks=tuple(TaskSnapshot.from_api(t) for t in tasks),
            apport=apport if apport is not None else parse_apport(description),
            comment_texts=comment_texts,
            remote_bug_statuses=remote_bug_statuses,
            enriched=enriched,
            prefilter_reason=prefilter_reason,
        )


def _bug_id(link: str | None) -> int | None:
    name = _lp_name(link)
    return int(name) if name and name.isdigit() else None


def _coerce[E: StrEnum](enum: type[E], value: Any, default: E) -> E:
    """Tolerantly map an API string onto an enum, falling back rather than raising.

    Launchpad can grow new status values; an unknown one must not crash a run.
    """
    if isinstance(value, enum):
        return value
    if isinstance(value, str):
        try:
            return enum(value)
        except ValueError:
            return default
    return default


# ---------------------------------------------------------------------------
# Analysis results
# ---------------------------------------------------------------------------


class RuleHit(BaseModel):
    """A deterministic rule firing on a bug."""

    model_config = ConfigDict(frozen=True)

    rule: str
    action: Action
    """The action this rule would propose if nothing blocks it."""

    claim: RuleClaim
    """What the rule asserts, which governs how the LLM may veto it.

    Deliberately has no default: a rule author must state what kind of claim they
    are making, because getting it wrong silently changes the veto semantics.
    """

    reason: str
    """Human-readable justification, shown in the report and the posted comment."""

    evidence: dict[str, str] = Field(default_factory=dict)
    """Concrete values behind the decision, e.g. ``{"series": "trusty"}``."""


class Exclusion(BaseModel):
    """A hard-exclusion rule firing. Any exclusion forces ``Action.KEEP``."""

    model_config = ConfigDict(frozen=True)

    rule: str
    reason: str


class LlmVerdict(BaseModel):
    """Advisory LLM assessment. Never sufficient on its own to act."""

    model_config = ConfigDict(frozen=True)

    is_actually_a_bug: IsABug = IsABug.UNCLEAR
    bug_kind: BugKind = BugKind.UNCLEAR
    needs_more_info: bool = False
    missing_info: tuple[str, ...] = ()
    reproducible_from_report: bool = False
    releases_mentioned: tuple[str, ...] = ()
    recommendation: Action = Action.KEEP
    confidence: float = 0.0
    rationale: str = ""
    suggested_comment_points: tuple[str, ...] = ()

    model: str = ""
    """Provider:model that produced this verdict, recorded for auditability."""

    failed: bool = False
    """True if the provider errored or returned unusable output. Treated as
    "no opinion" -- never as consent to act."""

    @classmethod
    def no_opinion(cls, *, model: str = "", failed: bool = False) -> Self:
        return cls(model=model, failed=failed)


class Decision(BaseModel):
    """Final fused outcome for one bug: what we will do and exactly why."""

    bug_id: int
    action: Action
    reason: str

    exclusions: tuple[Exclusion, ...] = ()
    rule_hits: tuple[RuleHit, ...] = ()
    verdict: LlmVerdict | None = None

    rule_action: Action = Action.KEEP
    """What the rules alone proposed, before the LLM was consulted."""

    llm_vetoed: bool = False
    llm_reclassified: bool = False
    age_escalated: bool = False
    """True when a rule-eligible ``needs-info`` was hardened to ``wont-fix``
    because the bug is older than ``[age].wont_fix_after_days``. Deterministic,
    like the rules; the LLM has no say in it."""
    policy_branch: str = ""
    """Which branch of :mod:`pruner.policy` produced ``action``."""

    @property
    def actionable(self) -> bool:
        return self.action is not Action.KEEP

    @property
    def rule_names(self) -> tuple[str, ...]:
        return tuple(h.rule for h in self.rule_hits)
