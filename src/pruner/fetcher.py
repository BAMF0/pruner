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
import threading
from collections.abc import Callable, Iterable, Iterator, Sequence
from datetime import UTC, datetime
from functools import partial
from typing import Any, Final

from pydantic import BaseModel

from pruner.config import Config
from pruner.lp.read import NotFound, ReadClient
from pruner.lp.series import SeriesTable, load_series_table
from pruner.models import DEFAULT_FETCH_STATUSES, BugSnapshot, BugTaskStatus
from pruner.rules.exclusions import prefilter
from pruner.store import Store

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]

#: Sentinel distinguishing "no cached date recorded" from "cached date is NULL".
_MISSING: Final = object()


class FetchStats(BaseModel):
    package: str
    tasks_found: int = 0
    bugs_seen: int = 0
    from_cache: int = 0
    fetched: int = 0
    enriched: int = 0
    prefiltered: int = 0
    errors: int = 0
    requests: int = 0
    """HTTP requests issued. Reported so a slow run is diagnosable, and asserted
    on in tests to catch an accidental extra round-trip per bug."""

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
        *,
        chunk_size: int | None = None,
    ) -> None:
        self.client = client
        self.config = config
        self.store = store
        self.series = series
        self.chunk_size = max(1, chunk_size or config.launchpad.chunk_size)
        """Bugs per pipeline pass. Bounds peak memory on a large backlog (all raw
        payloads for a chunk are held at once) and means an interrupted run keeps
        the chunks it already committed."""

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
        stage: ProgressCallback | None = None,
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
        if progress:
            # Publish the total before the first chunk finishes; otherwise the
            # bar sits at "0/?" for the whole of chunk one.
            progress(0, len(bug_ids), "")
        cached = self.store.cached_last_updated(self.distribution, package)
        etags = self.store.cached_etags(self.distribution, package)

        done = 0
        for chunk in _chunks(bug_ids, self.chunk_size):
            self._fetch_chunk(
                chunk, package, cached, etags, stats, now=now, refresh=refresh, stage=stage
            )
            done += len(chunk)
            if progress:
                progress(done, len(bug_ids), f"{stats.requests} requests")

        return stats

    # -- internals ---------------------------------------------------------

    def _stage_counter(
        self, stage: ProgressCallback | None, name: str, total: int
    ) -> Callable[[], None] | None:
        """Build a thread-safe ``on_done`` for one :meth:`ReadClient.gather` batch.

        Each stage knows its exact job count up front, so the reported progress
        is measured rather than estimated. The counter is locked because the
        callback fires on pool workers; ``stage`` is invoked outside the lock so
        that the renderer's own lock is never taken while holding this one.

        Returns ``None`` when nobody is listening, which keeps the
        no-progress path allocation-free.
        """
        if stage is None or total == 0:
            return None

        lock = threading.Lock()
        done = 0

        def on_done() -> None:
            nonlocal done
            with lock:
                done += 1
                current = done
            stage(current, total, name)

        return on_done

    def _fetch_chunk(
        self,
        bug_ids: list[int],
        package: str,
        cached: dict[int, str | None],
        etags: dict[int, str | None],
        stats: FetchStats,
        *,
        now: datetime,
        refresh: bool,
        stage: ProgressCallback | None = None,
    ) -> None:
        """Fetch one chunk of bugs as a series of wide, parallel stages.

        Structured this way rather than bug-by-bug for two reasons. First, it
        maximises parallelism: a bug needs up to nine requests, and doing them
        one bug at a time would leave most of the connection pool idle. Second,
        every sqlite call happens here on the calling thread -- the connection is
        not thread-safe, and the worker threads only ever do HTTP and parsing.
        """
        before = self.client.request_count

        # -- Stage 1: bug entries, revalidating against stored ETags ---------
        entries = self.client.gather(
            [
                partial(self.client.bug_conditional, bug_id, None if refresh else etags.get(bug_id))
                for bug_id in bug_ids
            ],
            on_done=self._stage_counter(stage, "entries", len(bug_ids)),
        )

        fresh_etags: dict[int, str | None] = {}
        raws: dict[int, dict[str, Any]] = {}
        reused: list[BugSnapshot] = []

        for bug_id, result in zip(bug_ids, entries, strict=True):
            if isinstance(result, BaseException):
                self._note_error(bug_id, result, stats)
                continue

            fresh_etags[bug_id] = result.etag

            if result.not_modified:
                # Unchanged since we last looked; the cached snapshot stands and
                # costs no further requests.
                if existing := self.store.get_bug(self.distribution, package, bug_id):
                    reused.append(existing)
                    continue
                # The ETag matched but the snapshot is gone (cache cleared, or a
                # partial earlier run). Re-fetch unconditionally rather than
                # silently dropping the bug.
                retry = self.client.gather(
                    [partial(self.client.bug_conditional, bug_id, None)]
                )[0]
                if isinstance(retry, BaseException) or retry.payload is None:
                    self._note_error(bug_id, retry, stats)
                    continue
                fresh_etags[bug_id] = retry.etag
                raws[bug_id] = retry.payload
                continue

            assert result.payload is not None
            raw = result.payload
            stale = cached.get(bug_id, _MISSING)
            if (
                not refresh
                and stale != _MISSING
                and stale == raw.get("date_last_updated")
                and (existing := self.store.get_bug(self.distribution, package, bug_id))
            ):
                reused.append(existing)
                continue
            raws[bug_id] = raw

        # -- Stage 2: bug tasks for everything we actually need to rebuild ----
        pending = sorted(raws)
        task_results = self.client.gather(
            [partial(self.client.bug_tasks, bug_id) for bug_id in pending],
            on_done=self._stage_counter(stage, "tasks", len(pending)),
        )

        phase1: dict[int, BugSnapshot] = {}
        for bug_id, task_result in zip(pending, task_results, strict=True):
            if isinstance(task_result, BaseException):
                self._note_error(bug_id, task_result, stats)
                raws.pop(bug_id, None)
                continue
            phase1[bug_id] = BugSnapshot.from_api(raws[bug_id], tasks=task_result)

        # -- Stage 3: cheap prefilter (pure CPU, no I/O) ----------------------
        to_enrich: list[int] = []
        finished: list[BugSnapshot] = []
        for bug_id in sorted(phase1):
            snapshot = phase1[bug_id]
            skip = prefilter(snapshot, self.config, self.series, package, now=now)
            if skip is None:
                to_enrich.append(bug_id)
            else:
                finished.append(
                    snapshot.model_copy(
                        update={
                            "enriched": False,
                            "prefilter_reason": f"{skip.rule}: {skip.reason}",
                        }
                    )
                )

        # -- Stage 4: enrichment, all requests for all survivors at once ------
        finished.extend(self._enrich_many(to_enrich, phase1, stats, stage=stage))

        # -- Stage 5: persist, deterministically ------------------------------
        for snapshot in reused:
            stats.from_cache += 1
            self._note_shape(snapshot, stats)

        finished.sort(key=lambda b: b.id)
        self.store.put_bugs(self.distribution, package, finished, etags=fresh_etags)
        for snapshot in finished:
            stats.fetched += 1
            self._note_shape(snapshot, stats)

        stats.requests += self.client.request_count - before

    def _enrich_many(
        self,
        bug_ids: list[int],
        phase1: dict[int, BugSnapshot],
        stats: FetchStats,
        *,
        stage: ProgressCallback | None = None,
    ) -> list[BugSnapshot]:
        """Run every enrichment request for every survivor as one flat batch.

        Submitting all ``7 x N`` requests together is the point: the pool stays
        saturated regardless of how many bugs are in flight.
        """
        if not bug_ids:
            return []

        jobs: list[Callable[[], Any]] = []
        keys: list[tuple[int, str]] = []
        for bug_id in bug_ids:
            base = f"/bugs/{bug_id}"
            for kind, job in (
                ("attachments", partial(self.client.collection, f"{base}/attachments")),
                ("messages", partial(self.client.bug_messages, bug_id)),
                ("watches", partial(self.client.bug_watches, bug_id)),
                ("cves", partial(self.client.count, f"{base}/cves")),
                ("vulnerabilities", partial(self.client.count, f"{base}/vulnerabilities")),
                ("mps", partial(self.client.count, f"{base}/linked_merge_proposals")),
                ("branches", partial(self.client.count, f"{base}/linked_branches")),
            ):
                keys.append((bug_id, kind))
                jobs.append(job)

        results = self.client.gather(
            jobs, on_done=self._stage_counter(stage, "enrich", len(jobs))
        )

        collected: dict[int, dict[str, Any]] = {bug_id: {} for bug_id in bug_ids}
        failed: set[int] = set()
        for (bug_id, kind), result in zip(keys, results, strict=True):
            if isinstance(result, BaseException):
                # One missing sub-collection makes the whole snapshot untrustworthy,
                # so the bug is dropped rather than stored half-enriched. Report the
                # first failure per bug only, to keep the log readable.
                if bug_id not in failed:
                    self._note_error(bug_id, result, stats)
                    failed.add(bug_id)
                continue
            collected[bug_id][kind] = result

        out: list[BugSnapshot] = []
        for bug_id in bug_ids:
            if bug_id in failed:
                continue
            parts = collected[bug_id]
            attachments: list[dict[str, Any]] = parts["attachments"]
            watches: list[dict[str, Any]] = parts["watches"]
            out.append(
                phase1[bug_id].model_copy(
                    update={
                        "cve_count": parts["cves"],
                        "vulnerability_count": parts["vulnerabilities"],
                        "linked_mp_count": parts["mps"],
                        "linked_branch_count": parts["branches"],
                        "attachment_count": len(attachments),
                        "patch_attachment_count": sum(
                            1 for a in attachments if a.get("type") == "Patch"
                        ),
                        "comment_texts": _comment_texts(parts["messages"]),
                        "remote_bug_statuses": tuple(
                            str(w.get("remote_status") or "")
                            for w in watches
                            if w.get("remote_status")
                        ),
                        "enriched": True,
                    }
                )
            )
        return out

    def _note_error(self, bug_id: int, error: object, stats: FetchStats) -> None:
        stats.errors += 1
        if isinstance(error, NotFound):
            log.warning("bug #%s vanished while fetching; skipping", bug_id)
        else:
            log.warning("failed to fetch bug #%s; skipping (%s)", bug_id, error)

    @staticmethod
    def _note_shape(snapshot: BugSnapshot, stats: FetchStats) -> None:
        if snapshot.enriched:
            stats.enriched += 1
        else:
            stats.prefiltered += 1
            stats.note_prefilter(_reason_rule(snapshot.prefilter_reason))


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


def _chunks(items: Sequence[int], size: int) -> Iterator[list[int]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


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
