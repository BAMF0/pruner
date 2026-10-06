"""Applying approved decisions, and undoing them.

Order of operations per bug is deliberate: **comment first, then change status.**
If the status change succeeds but the comment fails, the reporter sees an
unexplained status flip from a bot, which is the worst outcome. Commenting first
means the failure mode is a comment explaining a change that did not happen --
confusing, but self-evidently harmless and easy to spot.

Every bug is also re-checked against live Launchpad state immediately before
mutation. Between ``analyze`` and ``apply`` a human may have triaged the bug, and
clobbering that would be unforgivable.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from pruner.audit import AuditLog, AuditRecord, TaskChange
from pruner.comment import compose_comment
from pruner.config import Config
from pruner.lp.write import BugWriter, WriteError
from pruner.models import (
    CLOSED_STATUSES,
    Action,
    BugSnapshot,
    BugTaskStatus,
    Decision,
)
from pruner.rules.signals import open_target_tasks

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]

#: Statuses a task may be in for us to still consider changing it. Anything else
#: means somebody has moved on since the analysis.
_ACTIONABLE_STATUSES: frozenset[BugTaskStatus] = frozenset(
    {
        BugTaskStatus.NEW,
        BugTaskStatus.CONFIRMED,
        BugTaskStatus.TRIAGED,
        BugTaskStatus.INCOMPLETE,
    }
)

_ACTION_STATUS: dict[Action, BugTaskStatus] = {
    Action.NEEDS_INFO: BugTaskStatus.INCOMPLETE,
    Action.INVALID: BugTaskStatus.INVALID,
    Action.WONT_FIX: BugTaskStatus.WONT_FIX,
}


class ApplyStats(BaseModel):
    applied: int = 0
    skipped: int = 0
    failed: int = 0
    reasons: dict[str, int] = Field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped += 1
        self.reasons[reason] = self.reasons.get(reason, 0) + 1


def apply_decisions(
    decisions: list[Decision],
    bugs: dict[int, BugSnapshot],
    *,
    config: Config,
    package: str,
    writer: BugWriter,
    audit: AuditLog,
    run_id: str,
    dry_run: bool,
    limit: int | None = None,
    progress: ProgressCallback | None = None,
) -> ApplyStats:
    stats = ApplyStats()
    actionable = [d for d in decisions if d.actionable]

    cap = config.safety.max_actions_per_run
    if limit is not None:
        cap = min(cap, limit)
    if len(actionable) > cap:
        log.warning(
            "%d bugs approved but max_actions_per_run is %d; applying the first %d",
            len(actionable),
            cap,
            cap,
        )
        actionable = actionable[:cap]

    for index, decision in enumerate(actionable, start=1):
        if progress:
            progress(index, len(actionable), f"bug #{decision.bug_id}")

        bug = bugs.get(decision.bug_id)
        if bug is None:
            stats.skip("no snapshot")
            continue

        record = AuditRecord(
            run_id=run_id,
            bug_id=bug.id,
            action=str(decision.action),
            distribution=config.launchpad.distribution,
            package=package,
            service=config.launchpad.service,
            actor=writer.actor,
            dry_run=dry_run,
            rules=list(decision.rule_names),
            reason=decision.reason,
            policy_branch=decision.policy_branch,
            llm_model=decision.verdict.model if decision.verdict else "",
        )

        try:
            _apply_one(
                bug, decision, config, package, writer, record, dry_run=dry_run
            )
        except WriteError as exc:
            record.outcome = "failed"
            record.error = str(exc)
            audit.append(record)
            stats.failed += 1
            log.error("bug #%s: %s", bug.id, exc)
            continue

        audit.append(record)
        if record.outcome == "applied":
            stats.applied += 1
        else:
            stats.skip(record.error or "skipped")

        if not dry_run and config.safety.action_delay_seconds > 0:
            time.sleep(config.safety.action_delay_seconds)

    return stats


def _apply_one(
    bug: BugSnapshot,
    decision: Decision,
    config: Config,
    package: str,
    writer: BugWriter,
    record: AuditRecord,
    *,
    dry_run: bool,
) -> None:
    target_status = _ACTION_STATUS.get(decision.action)
    if target_status is None:
        record.outcome = "skipped"
        record.error = f"no status mapping for {decision.action}"
        return

    tasks = [
        t
        for t in open_target_tasks(bug, config.launchpad.distribution, package)
        if t.self_link
    ]
    if not tasks:
        record.outcome = "skipped"
        record.error = "no open task with a usable link"
        return

    # Re-read live state so a human's triage since `analyze` is never clobbered.
    changes: list[TaskChange] = []
    for task in tasks:
        assert task.self_link is not None
        current = writer.task_status(task.self_link)
        try:
            current_status = BugTaskStatus(current)
        except ValueError:
            record.outcome = "skipped"
            record.error = f"unrecognised live status {current!r} on {task.target_name}"
            return

        if current_status in CLOSED_STATUSES:
            record.outcome = "skipped"
            record.error = f"{task.target_name} is already {current_status}"
            return
        if current_status not in _ACTIONABLE_STATUSES:
            record.outcome = "skipped"
            record.error = (
                f"{task.target_name} is now {current_status}, which changed since analysis"
            )
            return
        if current_status is target_status:
            continue

        changes.append(
            TaskChange(
                task_link=task.self_link,
                target_name=task.target_name,
                previous_status=str(current_status),
                new_status=str(target_status),
            )
        )

    if not changes:
        record.outcome = "skipped"
        record.error = f"already {target_status}"
        return

    body = compose_comment(bug, decision, config, package=package)
    record.comment_body = body
    record.tags_added = ["pruner-triaged"]

    if dry_run:
        record.task_changes = changes
        record.outcome = "skipped"
        record.error = "dry run"
        return

    # Comment first: an unexplained bot status change is worse than an
    # explanation of a change that then failed to happen.
    writer.add_comment(bug.id, body)
    record.comment_posted = True

    for change in changes:
        writer.set_task_status(change.task_link, BugTaskStatus(change.new_status))
        record.task_changes.append(change)

    writer.add_tags(bug.id, record.tags_added)
    record.outcome = "applied"


def rollback_run(
    run_id: str,
    *,
    config: Config,
    writer: BugWriter,
    audit: AuditLog,
    new_run_id: str,
    dry_run: bool,
    progress: ProgressCallback | None = None,
) -> ApplyStats:
    """Restore the statuses changed by a previous run.

    Reverts only records that genuinely applied and have not already been
    reverted, and only when the task is still in the state this tool left it in.
    If somebody has since triaged the bug themselves, their state wins and we
    leave it alone.
    """
    stats = ApplyStats()
    records = audit.applied_for_run(run_id)
    if not records:
        log.warning("no applied records found for run %s", run_id)
        return stats

    for index, original in enumerate(records, start=1):
        if progress:
            progress(index, len(records), f"bug #{original.bug_id}")

        revert = AuditRecord(
            run_id=new_run_id,
            bug_id=original.bug_id,
            action="rollback",
            distribution=original.distribution,
            package=original.package,
            service=config.launchpad.service,
            actor=writer.actor,
            dry_run=dry_run,
            reverts_run_id=run_id,
            reason=f"rollback of run {run_id}",
        )

        try:
            restored = _revert_one(original, writer, revert, dry_run=dry_run)
        except WriteError as exc:
            revert.outcome = "failed"
            revert.error = str(exc)
            audit.append(revert)
            stats.failed += 1
            continue

        audit.append(revert)
        if restored:
            stats.applied += 1
        else:
            stats.skip(revert.error or "skipped")

        if not dry_run and config.safety.action_delay_seconds > 0:
            time.sleep(config.safety.action_delay_seconds)

    return stats


def _revert_one(
    original: AuditRecord,
    writer: BugWriter,
    revert: AuditRecord,
    *,
    dry_run: bool,
) -> bool:
    planned: list[TaskChange] = []
    for change in original.task_changes:
        live = writer.task_status(change.task_link)
        if live != change.new_status:
            revert.outcome = "skipped"
            revert.error = (
                f"{change.target_name} is now {live!r}, not the {change.new_status!r} "
                "this tool set; leaving it alone"
            )
            return False
        planned.append(
            TaskChange(
                task_link=change.task_link,
                target_name=change.target_name,
                previous_status=change.new_status,
                new_status=change.previous_status,
            )
        )

    if not planned:
        revert.outcome = "skipped"
        revert.error = "nothing to revert"
        return False

    if dry_run:
        revert.task_changes = planned
        revert.outcome = "skipped"
        revert.error = "dry run"
        return False

    body = (
        "Reverting an automated triage change: this bug's status is being restored "
        f"to {planned[0].new_status} because the triage pass that changed it "
        f"({original.run_id}) was rolled back. Apologies for the noise.\n\n"
        f"--\n{revert.reason}"
    )
    writer.add_comment(original.bug_id, body)
    revert.comment_posted = True
    revert.comment_body = body

    for change in planned:
        writer.set_task_status(change.task_link, BugTaskStatus(change.new_status))
        revert.task_changes.append(change)

    revert.outcome = "applied"
    revert.timestamp = datetime.now(UTC).isoformat()
    return True
