"""Local state: bug snapshot cache, LLM verdict cache, decisions and runs.

Why a store at all:

* **Cost.** LLM verdicts are cached against a fingerprint of exactly the text the
  model was shown, so re-running ``analyze`` after tweaking a threshold costs no
  inference.
* **Politeness.** Bug snapshots are cached against ``date_last_updated``, so a
  second ``fetch`` re-downloads only what actually changed.
* **Auditability.** Decisions are persisted per run, so the report you reviewed
  and the actions ``apply`` performs are provably the same set.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pruner.lp.archive import ArchiveIndex
from pruner.lp.series import SeriesTable
from pruner.models import BugSnapshot, Decision, LlmVerdict

DEFAULT_STATE_DIR = Path(".pruner")
DB_FILENAME = "cache.db"

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bugs (
    bug_id            INTEGER NOT NULL,
    distribution      TEXT    NOT NULL,
    package           TEXT    NOT NULL,
    date_last_updated TEXT,
    fetched_at        TEXT    NOT NULL,
    payload           TEXT    NOT NULL,
    PRIMARY KEY (distribution, package, bug_id)
);

CREATE TABLE IF NOT EXISTS verdicts (
    bug_id      INTEGER NOT NULL,
    model       TEXT    NOT NULL,
    fingerprint TEXT    NOT NULL,
    created_at  TEXT    NOT NULL,
    payload     TEXT    NOT NULL,
    PRIMARY KEY (bug_id, model, fingerprint)
);

CREATE TABLE IF NOT EXISTS decisions (
    run_id     TEXT    NOT NULL,
    bug_id     INTEGER NOT NULL,
    action     TEXT    NOT NULL,
    created_at TEXT    NOT NULL,
    payload    TEXT    NOT NULL,
    PRIMARY KEY (run_id, bug_id)
);

CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    kind         TEXT NOT NULL,
    distribution TEXT NOT NULL,
    package      TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    details      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS series_cache (
    distribution TEXT PRIMARY KEY,
    fetched_at   TEXT NOT NULL,
    payload      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS archive_cache (
    distribution TEXT NOT NULL,
    package      TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    payload      TEXT NOT NULL,
    PRIMARY KEY (distribution, package)
);

CREATE INDEX IF NOT EXISTS idx_decisions_run ON decisions (run_id, action);
"""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def verdict_fingerprint(bug: BugSnapshot) -> str:
    """Hash of exactly the bug text an LLM is shown.

    Keyed on the *inputs to the model*, not on ``date_last_updated``: a bug whose
    only change was a status flip does not need re-scoring, while an edited
    description or a new comment does.
    """
    digest = hashlib.sha256()
    digest.update(bug.title.encode())
    digest.update(b"\0")
    digest.update(bug.description.encode())
    for comment in bug.comment_texts:
        digest.update(b"\0")
        digest.update(comment.encode())
    return digest.hexdigest()[:32]


class Store:
    """SQLite-backed state. Safe to delete at any time; it is only a cache."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        self._conn.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        self._conn.commit()

    @classmethod
    def open(cls, state_dir: Path | None = None) -> Store:
        directory = state_dir or DEFAULT_STATE_DIR
        return cls(directory / DB_FILENAME)

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- bug snapshots -----------------------------------------------------

    def put_bug(self, distribution: str, package: str, bug: BugSnapshot) -> None:
        self._conn.execute(
            """
            INSERT INTO bugs
                (bug_id, distribution, package, date_last_updated, fetched_at, payload)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (distribution, package, bug_id) DO UPDATE SET
                date_last_updated = excluded.date_last_updated,
                fetched_at        = excluded.fetched_at,
                payload           = excluded.payload
            """,
            (
                bug.id,
                distribution,
                package,
                bug.date_last_updated.isoformat() if bug.date_last_updated else None,
                _now(),
                bug.model_dump_json(),
            ),
        )
        self._conn.commit()

    def put_bugs(self, distribution: str, package: str, bugs: list[BugSnapshot]) -> None:
        for bug in bugs:
            self.put_bug(distribution, package, bug)

    def get_bug(self, distribution: str, package: str, bug_id: int) -> BugSnapshot | None:
        row = self._conn.execute(
            "SELECT payload FROM bugs WHERE distribution = ? AND package = ? AND bug_id = ?",
            (distribution, package, bug_id),
        ).fetchone()
        return BugSnapshot.model_validate_json(row["payload"]) if row else None

    def cached_last_updated(self, distribution: str, package: str) -> dict[int, str | None]:
        """``{bug_id: date_last_updated}`` for cache-freshness checks in ``fetch``."""
        rows = self._conn.execute(
            "SELECT bug_id, date_last_updated FROM bugs WHERE distribution = ? AND package = ?",
            (distribution, package),
        ).fetchall()
        return {int(r["bug_id"]): r["date_last_updated"] for r in rows}

    def iter_bugs(self, distribution: str, package: str) -> Iterator[BugSnapshot]:
        cursor = self._conn.execute(
            "SELECT payload FROM bugs WHERE distribution = ? AND package = ? ORDER BY bug_id",
            (distribution, package),
        )
        with closing(cursor):
            for row in cursor:
                yield BugSnapshot.model_validate_json(row["payload"])

    def bug_count(self, distribution: str, package: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) AS n FROM bugs WHERE distribution = ? AND package = ?",
            (distribution, package),
        ).fetchone()
        return int(row["n"])

    # -- LLM verdicts ------------------------------------------------------

    def get_verdict(self, bug_id: int, model: str, fingerprint: str) -> LlmVerdict | None:
        row = self._conn.execute(
            "SELECT payload FROM verdicts WHERE bug_id = ? AND model = ? AND fingerprint = ?",
            (bug_id, model, fingerprint),
        ).fetchone()
        return LlmVerdict.model_validate_json(row["payload"]) if row else None

    def put_verdict(
        self, bug_id: int, model: str, fingerprint: str, verdict: LlmVerdict
    ) -> None:
        # Failed verdicts are never cached: a transient provider outage must not
        # poison later runs with a permanent "no opinion".
        if verdict.failed:
            return
        self._conn.execute(
            """
            INSERT INTO verdicts (bug_id, model, fingerprint, created_at, payload)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (bug_id, model, fingerprint) DO UPDATE SET
                payload    = excluded.payload,
                created_at = excluded.created_at
            """,
            (bug_id, model, fingerprint, _now(), verdict.model_dump_json()),
        )
        self._conn.commit()

    # -- decisions and runs ------------------------------------------------

    def start_run(
        self, run_id: str, kind: str, distribution: str, package: str, details: dict[str, Any]
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO runs (run_id, kind, distribution, package, created_at, details)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id) DO UPDATE SET details = excluded.details
            """,
            (run_id, kind, distribution, package, _now(), json.dumps(details, default=str)),
        )
        self._conn.commit()

    def put_decision(self, run_id: str, decision: Decision) -> None:
        self._conn.execute(
            """
            INSERT INTO decisions (run_id, bug_id, action, created_at, payload)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (run_id, bug_id) DO UPDATE SET
                action     = excluded.action,
                payload    = excluded.payload,
                created_at = excluded.created_at
            """,
            (run_id, decision.bug_id, str(decision.action), _now(), decision.model_dump_json()),
        )
        self._conn.commit()

    def put_decisions(self, run_id: str, decisions: list[Decision]) -> None:
        for decision in decisions:
            self.put_decision(run_id, decision)

    def get_decisions(self, run_id: str) -> list[Decision]:
        rows = self._conn.execute(
            "SELECT payload FROM decisions WHERE run_id = ? ORDER BY bug_id",
            (run_id,),
        ).fetchall()
        return [Decision.model_validate_json(r["payload"]) for r in rows]

    def latest_run(self, kind: str, distribution: str, package: str) -> str | None:
        row = self._conn.execute(
            """
            SELECT run_id FROM runs
            WHERE kind = ? AND distribution = ? AND package = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (kind, distribution, package),
        ).fetchone()
        return str(row["run_id"]) if row else None

    def run_details(self, run_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT kind, distribution, package, created_at, details FROM runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        if not row:
            return None
        return {
            "kind": row["kind"],
            "distribution": row["distribution"],
            "package": row["package"],
            "created_at": row["created_at"],
            "details": json.loads(row["details"]),
        }

    # -- reference data ----------------------------------------------------

    def put_series(self, table: SeriesTable) -> None:
        self._conn.execute(
            """
            INSERT INTO series_cache (distribution, fetched_at, payload) VALUES (?, ?, ?)
            ON CONFLICT (distribution) DO UPDATE SET
                fetched_at = excluded.fetched_at, payload = excluded.payload
            """,
            (table.distribution, _now(), table.model_dump_json()),
        )
        self._conn.commit()

    def get_series(self, distribution: str) -> SeriesTable | None:
        row = self._conn.execute(
            "SELECT payload FROM series_cache WHERE distribution = ?", (distribution,)
        ).fetchone()
        return SeriesTable.model_validate_json(row["payload"]) if row else None

    def put_archive(self, index: ArchiveIndex) -> None:
        self._conn.execute(
            """
            INSERT INTO archive_cache (distribution, package, fetched_at, payload)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (distribution, package) DO UPDATE SET
                fetched_at = excluded.fetched_at, payload = excluded.payload
            """,
            (index.distribution, index.package, _now(), index.model_dump_json()),
        )
        self._conn.commit()

    def get_archive(self, distribution: str, package: str) -> ArchiveIndex | None:
        row = self._conn.execute(
            "SELECT payload FROM archive_cache WHERE distribution = ? AND package = ?",
            (distribution, package),
        ).fetchone()
        return ArchiveIndex.model_validate_json(row["payload"]) if row else None
