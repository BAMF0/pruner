"""Append-only audit log.

One JSONL record per attempted mutation, written *before* success is known and
then updated, so a crash mid-run still leaves evidence of what was in flight.
This file is what :func:`pruner.cli.rollback` replays, so it records the prior
status of every task it changed.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, Field

DEFAULT_AUDIT_FILENAME = "audit.jsonl"


class TaskChange(BaseModel):
    """A single task status transition, with enough detail to undo it."""

    task_link: str
    target_name: str
    previous_status: str
    new_status: str


class AuditRecord(BaseModel):
    run_id: str
    bug_id: int
    action: str
    distribution: str
    package: str
    service: str
    actor: str = ""
    """Launchpad username of the account that performed the write.

    Recorded because writes may come from a bot account rather than the person
    who ran the command -- "who did this" is no longer implicit then, and the
    audit log is the whole accountability story."""
    dry_run: bool
    outcome: Literal["pending", "applied", "failed", "skipped", "reverted"] = "pending"
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())

    task_changes: list[TaskChange] = Field(default_factory=list)
    comment_posted: bool = False
    comment_body: str = ""
    tags_added: list[str] = Field(default_factory=list)

    rules: list[str] = Field(default_factory=list)
    reason: str = ""
    policy_branch: str = ""
    llm_model: str = ""
    error: str = ""

    reverts_run_id: str = ""
    """Set on records written by ``rollback``."""


class AuditLog:
    """Append-only JSONL writer/reader.

    Each append is flushed and ``fsync``-ed. A backlog sweep does far more damage
    if the record of it is lost than the few milliseconds per write cost.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def open(cls, state_dir: Path) -> Self:
        return cls(state_dir / DEFAULT_AUDIT_FILENAME)

    def append(self, record: AuditRecord) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(record.model_dump_json() + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def records(self) -> list[AuditRecord]:
        if not self.path.exists():
            return []
        out: list[AuditRecord] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload: dict[str, Any] = json.loads(stripped)
            except ValueError:
                continue
            try:
                out.append(AuditRecord.model_validate(payload))
            except ValueError:
                continue
        return out

    def applied_for_run(self, run_id: str) -> list[AuditRecord]:
        """Records that actually changed something, in the order applied.

        Only genuine applications are returned: pending, failed, dry-run and
        already-reverted records are excluded so rollback cannot double-revert or
        "undo" something that never happened.
        """
        reverted = {
            r.reverts_run_id + ":" + str(r.bug_id)
            for r in self.records()
            if r.reverts_run_id and r.outcome == "applied"
        }
        return [
            r
            for r in self.records()
            if r.run_id == run_id
            and r.outcome == "applied"
            and not r.dry_run
            and f"{run_id}:{r.bug_id}" not in reverted
        ]

    def run_ids(self) -> list[str]:
        seen: dict[str, None] = {}
        for record in self.records():
            seen.setdefault(record.run_id, None)
        return list(seen)
