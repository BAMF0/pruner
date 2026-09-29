"""Fetching a package's bug backlog into local snapshots.

Uses two phases per bug, because full enrichment costs roughly seven extra API
calls and most of a stale backlog is excluded on cheap grounds anyway:

* **Phase 1** -- the bug entry plus its tasks (2 calls). Enough to evaluate the
  cheap exclusions in :data:`~pruner.rules.exclusions.PREFILTER_EXCLUSIONS`.
* **Phase 2** -- only for bugs that survive: CVEs, merge proposals, branches,
  attachments, comments and upstream watches.

A snapshot that never reached phase 2 is stored with ``enriched=False``, and
:func:`~pruner.rules.exclusions._incomplete_snapshot` refuses to act on such a
snapshot. So the optimisation can only cause a bug to be *skipped*, never acted on
with missing information.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel

from pruner.config import Config
from pruner.lp.read import LaunchpadError, NotFound, ReadClient
from pruner.lp.series import SeriesTable, load_series_table
from pruner.models import DEFAULT_FETCH_STATUSES, BugSnapshot, BugTaskStatus
from pruner.rules.exclusions import prefilter
from pruner.store import Store

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class FetchStats(BaseModel):
    package: str
    tasks_found: int = 0
    bugs_seen: int = 0
    from_cache: int = 0
    fetched: int = 0
    enriched: int = 0
    prefiltered: int = 0
    errors: int = 0
    prefilter_reasons: dict[str, int] = {}

    def note_prefilter(self, rule: str) -> None:
        self.prefilter_reasons[rule] = self.prefilter_reasons.get(rule, 0) + 1


class Fetcher:
    def __init__(
        self,
        client: ReadClient,
        config: Config,
        store: Store,
        series: SeriesTable,
    ) -> None:
        self.client = client
        self.config = config
        self.store = store
        self.series = series

    @property
    def distribution(self) -> str:
        return self.config.launchpad.distribution

    def ensure_package_exists(self, package: str) -> None:
        """Fail fast and loudly on a typo'd package name.

        Important because ``removed_from_archive`` proposes ``invalid``; a typo
        must surface here as an error rather than downstream as "this package
        doesn't exist, close everything".
        """
        try:
            self.client.source_package(self.distribution, package)
        except NotFound as exc:
            raise NotFound(
                f"{package!r} is not a known source package in {self.distribution}"
            ) from exc

    def fetch(
        self,
        package: str,
        *,
        statuses: Iterable[BugTaskStatus] | None = None,
        limit: int | None = None,
        refresh: bool = False,
        progress: ProgressCallback | None = None,
    ) -> FetchStats:
        now = datetime.now(UTC)
        stats = FetchStats(package=package)
        status_values = tuple(
            str(s) for s in (statuses if statuses is not None else DEFAULT_FETCH_STATUSES)
        )

        tasks = self.client.search_tasks(
            self.distribution, package, statuses=status_values, limit=limit
        )
        stats.tasks_found = len(tasks)

        bug_ids = _unique_bug_ids(tasks)
        stats.bugs_seen = len(bug_ids)
        cached = self.store.cached_last_updated(self.distribution, package)

        for index, bug_id in enumerate(bug_ids, start=1):
            if progress:
                progress(index, len(bug_ids), f"bug #{bug_id}")
            try:
                snapshot, reused = self._fetch_one(
                    bug_id, package, cached, now=now, refresh=refresh
                )
            except NotFound:
                log.warning("bug #%s vanished while fetching; skipping", bug_id)
                stats.errors += 1
                continue
            except LaunchpadError:
                log.warning("failed to fetch bug #%s; skipping", bug_id, exc_info=True)
                stats.errors += 1
                continue

            if reused:
                stats.from_cache += 1
            else:
                stats.fetched += 1
                self.store.put_bug(self.distribution, package, snapshot)

            if snapshot.enriched:
                stats.enriched += 1
            else:
                stats.prefiltered += 1
                stats.note_prefilter(_reason_rule(snapshot.prefilter_reason))

        return stats

    # -- internals ---------------------------------------------------------

    def _fetch_one(
        self,
        bug_id: int,
        package: str,
        cached: dict[int, str | None],
        *,
        now: datetime,
        refresh: bool,
    ) -> tuple[BugSnapshot, bool]:
        """Return ``(snapshot, reused_from_cache)``.

        We always GET the bug entry, because ``searchTasks`` does not report the
        bug's ``date_last_updated`` and so cannot tell us whether the cache is
        stale. That single call then lets us skip up to seven more.
        """
        raw = self.client.bug(bug_id)
        last_updated = raw.get("date_last_updated")

        cache_is_fresh = not refresh and bug_id in cached and cached[bug_id] == last_updated
        if cache_is_fresh and (
            existing := self.store.get_bug(self.distribution, package, bug_id)
        ):
            return existing, True

        task_entries = self.client.bug_tasks(bug_id)
        phase1 = BugSnapshot.from_api(raw, tasks=task_entries)

        skip = prefilter(phase1, self.config, self.series, package, now=now)
        if skip is not None:
            return (
                phase1.model_copy(
                    update={"enriched": False, "prefilter_reason": f"{skip.rule}: {skip.reason}"}
                ),
                False,
            )

        return self._enrich(raw, task_entries, bug_id), False

    def _enrich(
        self, raw: dict[str, Any], task_entries: list[dict[str, Any]], bug_id: int
    ) -> BugSnapshot:
        base = f"/bugs/{bug_id}"
        attachments = self.client.collection(f"{base}/attachments")
        messages = self.client.bug_messages(bug_id)
        watches = self.client.bug_watches(bug_id)

        return BugSnapshot.from_api(
            raw,
            tasks=task_entries,
            cve_count=self.client.count(f"{base}/cves"),
            vulnerability_count=self.client.count(f"{base}/vulnerabilities"),
            linked_mp_count=self.client.count(f"{base}/linked_merge_proposals"),
            linked_branch_count=self.client.count(f"{base}/linked_branches"),
            attachment_count=len(attachments),
            patch_attachment_count=sum(1 for a in attachments if a.get("type") == "Patch"),
            comment_texts=_comment_texts(messages),
            remote_bug_statuses=tuple(
                str(w.get("remote_status") or "") for w in watches if w.get("remote_status")
            ),
            enriched=True,
        )


def _comment_texts(messages: list[dict[str, Any]]) -> tuple[str, ...]:
    """Comment bodies, excluding the description.

    Launchpad returns the description as message 0 of ``/messages``; it is already
    on the snapshot, so including it again would double-count it in every
    text scan and in the LLM prompt.
    """
    bodies = [str(m.get("content") or "").strip() for m in messages[1:]]
    return tuple(b for b in bodies if b)


def _unique_bug_ids(tasks: list[dict[str, Any]]) -> list[int]:
    """Bug IDs from search results, de-duplicated and stably ordered.

    One bug can appear several times when it has series nominations.
    """
    seen: dict[int, None] = {}
    for task in tasks:
        link = str(task.get("bug_link") or "")
        tail = link.rstrip("/").rsplit("/", 1)[-1]
        if tail.isdigit():
            seen.setdefault(int(tail), None)
    return list(seen)


def _reason_rule(reason: str | None) -> str:
    if not reason:
        return "unknown"
    return reason.split(":", 1)[0]


def load_series(
    client: ReadClient, config: Config, store: Store, *, refresh: bool = False
) -> SeriesTable:
    """Series table, from cache unless refreshed.

    Cached because it changes a couple of times a year, but refreshable because
    getting it wrong is how you accidentally treat a supported release as dead.
    The cache is also invalidated when the configured support policy changes,
    since the policy is baked into the resolved table.
    """
    distribution = config.launchpad.distribution
    cached = store.get_series(distribution) if not refresh else None
    if cached is not None and cached.policy == config.launchpad.support_policy:
        return cached

    table = load_series_table(
        client,
        distribution,
        policy=config.launchpad.support_policy,
        live_series=config.launchpad.live_series,
    )
    store.put_series(table)
    return table
