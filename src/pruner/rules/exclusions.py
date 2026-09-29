"""Hard exclusions: vetoes on touching a bug at all.

Every check here answers "is there any reason a human would be annoyed that a bot
touched this bug?" If any fires, the final action is forced to ``keep`` regardless
of what prune rules or the LLM think.

These run *after* prune rules in the report (so you can see what was nearly
actioned and why it was spared) but they take absolute precedence in
:mod:`pruner.policy`.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime

from pruner.config import Config
from pruner.lp.series import SeriesTable
from pruner.models import (
    PROGRESSING_STATUSES,
    BugSnapshot,
    Exclusion,
)
from pruner.rules.signals import (
    live_release_evidence,
    matched_tags,
    open_target_tasks,
    target_tasks,
)

ExclusionFunc = Callable[[BugSnapshot, Config, SeriesTable, str, datetime], Exclusion | None]


def _incomplete_snapshot(
    bug: BugSnapshot, config: Config, table: SeriesTable, package: str, now: datetime
) -> Exclusion | None:
    """Refuse to act on a snapshot that was never fully enriched.

    ``fetch`` skips the expensive sub-collection lookups for bugs that fail a cheap
    prefilter. Those snapshots are missing CVE/patch/merge-proposal counts, so
    treating them as clean would be unsound. Failing closed here is what makes the
    fetch-time optimisation safe.
    """
    if not bug.enriched:
        reason = bug.prefilter_reason or "snapshot not fully fetched"
        return Exclusion(rule="incomplete_snapshot", reason=reason)
    return None


def _private(
    bug: BugSnapshot, config: Config, table: SeriesTable, package: str, now: datetime
) -> Exclusion | None:
    if not config.safety.skip_private:
        return None
    if bug.private or bug.information_type not in ("Public", "Public Security"):
        return Exclusion(
            rule="private",
            reason=f"bug is not public (information_type={bug.information_type})",
        )
    return None


def _security(
    bug: BugSnapshot, config: Config, table: SeriesTable, package: str, now: datetime
) -> Exclusion | None:
    """Security bugs are never auto-triaged. Wrongly closing one is unacceptable."""
    if not config.safety.skip_security:
        return None
    if bug.security_related:
        return Exclusion(rule="security", reason="flagged security_related")
    if bug.cve_count:
        return Exclusion(rule="security", reason=f"has {bug.cve_count} linked CVE(s)")
    if bug.vulnerability_count:
        return Exclusion(
            rule="security", reason=f"has {bug.vulnerability_count} linked vulnerability record(s)"
        )
    if bug.information_type == "Public Security":
        return Exclusion(rule="security", reason="information_type is Public Security")
    return None


def _has_patch(
    bug: BugSnapshot, config: Config, table: SeriesTable, package: str, now: datetime
) -> Exclusion | None:
    """Somebody did work here. Closing it would discard a contribution."""
    if not config.safety.skip_with_patch:
        return None
    if bug.patch_attachment_count:
        return Exclusion(
            rule="has_patch",
            reason=f"{bug.patch_attachment_count} attachment(s) of type Patch",
        )
    if bug.latest_patch_uploaded is not None:
        return Exclusion(
            rule="has_patch",
            reason=f"a patch was attached ({bug.latest_patch_uploaded.date()})",
        )
    return None


def _dev_activity(
    bug: BugSnapshot, config: Config, table: SeriesTable, package: str, now: datetime
) -> Exclusion | None:
    if not config.safety.skip_with_dev_activity:
        return None
    if bug.linked_mp_count:
        return Exclusion(
            rule="dev_activity", reason=f"{bug.linked_mp_count} linked merge proposal(s)"
        )
    if bug.linked_branch_count:
        return Exclusion(
            rule="dev_activity", reason=f"{bug.linked_branch_count} linked branch(es)"
        )
    return None


def _assigned(
    bug: BugSnapshot, config: Config, table: SeriesTable, package: str, now: datetime
) -> Exclusion | None:
    if not config.safety.skip_assigned:
        return None
    for task in target_tasks(bug, table.distribution, package):
        if task.assignee:
            return Exclusion(
                rule="assigned", reason=f"assigned to ~{task.assignee} on {task.target_name}"
            )
    return None


def _milestoned(
    bug: BugSnapshot, config: Config, table: SeriesTable, package: str, now: datetime
) -> Exclusion | None:
    if not config.safety.skip_milestoned:
        return None
    for task in target_tasks(bug, table.distribution, package):
        if task.milestone:
            return Exclusion(
                rule="milestoned", reason=f"targeted to milestone {task.milestone}"
            )
    return None


def _protected_importance(
    bug: BugSnapshot, config: Config, table: SeriesTable, package: str, now: datetime
) -> Exclusion | None:
    protected = set(config.safety.protect_importances)
    for task in target_tasks(bug, table.distribution, package):
        if task.importance in protected:
            return Exclusion(
                rule="protected_importance",
                reason=f"importance is {task.importance} on {task.target_name}",
            )
    return None


def _progressing(
    bug: BugSnapshot, config: Config, table: SeriesTable, package: str, now: datetime
) -> Exclusion | None:
    for task in bug.tasks:
        if task.status in PROGRESSING_STATUSES:
            return Exclusion(
                rule="progressing",
                reason=f"{task.target_name} is {task.status}",
            )
    return None


def _duplicate(
    bug: BugSnapshot, config: Config, table: SeriesTable, package: str, now: datetime
) -> Exclusion | None:
    """Already a duplicate: its status is governed by the master bug."""
    if bug.duplicate_of is not None:
        return Exclusion(
            rule="duplicate", reason=f"already marked a duplicate of #{bug.duplicate_of}"
        )
    return None


def _no_open_task(
    bug: BugSnapshot, config: Config, table: SeriesTable, package: str, now: datetime
) -> Exclusion | None:
    if not open_target_tasks(bug, table.distribution, package):
        return Exclusion(
            rule="no_open_task",
            reason=f"no open {table.distribution}/{package} task to act on",
        )
    return None


def _affects_live_release(
    bug: BugSnapshot, config: Config, table: SeriesTable, package: str, now: datetime
) -> Exclusion | None:
    """The single most important exclusion.

    If there is *any* indication the bug concerns a release that is still alive --
    an open task on a live series, a live series tag, apport metadata, or a mention
    anywhere in the text -- then it is not an EOL bug and we leave it alone.
    """
    if evidence := live_release_evidence(bug, table, package):
        return Exclusion(
            rule="affects_live_release",
            reason="may affect a supported release: " + "; ".join(evidence[:4]),
        )
    return None


def _recently_active(
    bug: BugSnapshot, config: Config, table: SeriesTable, package: str, now: datetime
) -> Exclusion | None:
    quiet = bug.quiet_days(now=now)
    threshold = config.safety.min_quiet_days
    if quiet < threshold:
        return Exclusion(
            rule="recently_active",
            reason=f"activity {quiet:.0f} days ago, under the {threshold}-day quiet period",
        )
    return None


def _popular(
    bug: BugSnapshot, config: Config, table: SeriesTable, package: str, now: datetime
) -> Exclusion | None:
    """Lots of people care. Even if stale, this deserves a human."""
    if bug.users_affected_count >= config.safety.protect_users_affected:
        return Exclusion(
            rule="popular",
            reason=f"{bug.users_affected_count} users marked themselves affected",
        )
    if bug.number_of_duplicates >= config.safety.protect_duplicates:
        return Exclusion(
            rule="popular", reason=f"{bug.number_of_duplicates} duplicates point here"
        )
    return None


def _protected_tag(
    bug: BugSnapshot, config: Config, table: SeriesTable, package: str, now: datetime
) -> Exclusion | None:
    if hits := matched_tags(bug.tags, config.safety.protect_tags):
        return Exclusion(
            rule="protected_tag", reason=f"carries protected tag(s): {', '.join(hits)}"
        )
    return None


def _already_triaged(
    bug: BugSnapshot, config: Config, table: SeriesTable, package: str, now: datetime
) -> Exclusion | None:
    """Idempotency: never comment on the same bug twice.

    A bug we already pinged has had its chance to be confirmed; nagging it again
    would be rude and would reset Launchpad's expiry clock.
    """
    marker = config.comment.marker
    if marker and any(marker in text for text in bug.comment_texts):
        return Exclusion(
            rule="already_triaged", reason="this tool has already commented on the bug"
        )
    return None


#: Evaluation order. Cheap and categorical checks first so the recorded reason is
#: the most informative one, and so the prefilter can reuse the early entries.
EXCLUSIONS: tuple[tuple[str, ExclusionFunc], ...] = (
    ("incomplete_snapshot", _incomplete_snapshot),
    ("private", _private),
    ("security", _security),
    ("duplicate", _duplicate),
    ("has_patch", _has_patch),
    ("dev_activity", _dev_activity),
    ("assigned", _assigned),
    ("milestoned", _milestoned),
    ("protected_importance", _protected_importance),
    ("progressing", _progressing),
    ("protected_tag", _protected_tag),
    ("no_open_task", _no_open_task),
    ("recently_active", _recently_active),
    ("popular", _popular),
    ("affects_live_release", _affects_live_release),
    ("already_triaged", _already_triaged),
)

#: Exclusions computable from a phase-1 snapshot (bug entry + tasks only), i.e.
#: without the extra per-bug sub-collection requests. Used by ``fetch`` to avoid
#: enriching bugs that are obviously untouchable.
PREFILTER_EXCLUSIONS: frozenset[str] = frozenset(
    {
        "private",
        "security",
        "duplicate",
        "has_patch",
        "assigned",
        "milestoned",
        "protected_importance",
        "progressing",
        "protected_tag",
        "no_open_task",
        "recently_active",
        "popular",
    }
)


def evaluate_exclusions(
    bug: BugSnapshot,
    config: Config,
    table: SeriesTable,
    package: str,
    *,
    now: datetime,
    only: frozenset[str] | None = None,
) -> tuple[Exclusion, ...]:
    """Run exclusions and return every one that fired.

    All of them run, not just the first: the report is far more useful when it
    shows every reason a bug is protected.
    """
    results: list[Exclusion] = []
    for name, func in EXCLUSIONS:
        if only is not None and name not in only:
            continue
        if found := func(bug, config, table, package, now):
            results.append(found)
    return tuple(results)


def prefilter(
    bug: BugSnapshot,
    config: Config,
    table: SeriesTable,
    package: str,
    *,
    now: datetime,
) -> Exclusion | None:
    """First exclusion computable without enrichment, if any.

    ``fetch`` uses this to decide whether a bug is worth ~6 more API calls.
    """
    found = evaluate_exclusions(
        bug, config, table, package, now=now, only=PREFILTER_EXCLUSIONS
    )
    return found[0] if found else None
