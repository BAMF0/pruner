"""Concurrency, conditional GETs, and the fetch pipeline's request budget.

The request-count assertions are the important ones. Fetching is latency-bound,
so an accidentally reintroduced round-trip per bug is a real regression that no
behavioural test would notice -- it would just make the tool slower.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from pruner.config import Config, LaunchpadConfig
from pruner.fetcher import Fetcher
from pruner.lp.archive import fetch_archive_index
from pruner.lp.read import (
    Conditional,
    LaunchpadError,
    NotFound,
    ReadClient,
    normalise_etag,
)
from pruner.lp.series import SeriesTable
from pruner.store import Store
from tests.conftest import SERIES_ENTRIES, load_fixture

# ---------------------------------------------------------------------------
# ETag normalisation
# ---------------------------------------------------------------------------


class TestEtagNormalisation:
    """Apache's ``mod_deflate`` appends ``-gzip`` to the ETag of a compressed
    response but compares ``If-None-Match`` against the uncompressed tag. Since
    httpx requests gzip by default, echoing the tag back verbatim always misses.

    Verified against production Launchpad: suffixed tag -> 200 with 3743 bytes,
    stripped tag -> 304 with 0 bytes. Getting this wrong fails silently, as a
    permanent full download rather than an error.
    """

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ('"abc-gzip"', '"abc"'),
            ('"abc"', '"abc"'),
            ("abc-gzip", "abc"),
            ('"a-b-c-gzip"', '"a-b-c"'),
            (None, None),
            ("", None),
        ],
    )
    def test_strips_only_the_trailing_suffix(self, given: str | None, expected: str | None) -> None:
        assert normalise_etag(given) == expected

    def test_does_not_strip_gzip_mid_token(self) -> None:
        assert normalise_etag('"gzip-abc"') == '"gzip-abc"'


# ---------------------------------------------------------------------------
# gather()
# ---------------------------------------------------------------------------


def _client(concurrency: int, transport: httpx.MockTransport | None = None) -> ReadClient:
    kwargs: dict[str, Any] = {}
    if transport is not None:
        kwargs["client"] = httpx.Client(transport=transport, base_url="https://lp.test")
    return ReadClient(LaunchpadConfig(), concurrency=concurrency, **kwargs)


class TestGather:
    def test_preserves_order(self) -> None:
        with _client(4) as client:
            jobs = [(lambda n=n: n) for n in range(20)]
            assert client.gather(jobs) == list(range(20))

    def test_empty(self) -> None:
        with _client(4) as client:
            assert client.gather([]) == []

    def test_captures_exceptions_per_job(self) -> None:
        """One unreachable bug must not abort a whole batch."""
        boom = NotFound("gone")

        def fail() -> int:
            raise boom

        with _client(4) as client:
            results = client.gather([lambda: 1, fail, lambda: 3])
        assert results[0] == 1
        assert results[1] is boom
        assert results[2] == 3

    def test_keyboard_interrupt_propagates(self) -> None:
        """Ctrl-C during a long sweep must actually stop it, so BaseException is
        deliberately not caught."""

        def interrupt() -> int:
            raise KeyboardInterrupt

        with _client(1) as client, pytest.raises(KeyboardInterrupt):
            client.gather([interrupt])

    def test_concurrency_one_runs_inline(self) -> None:
        """Keeps the sequential path genuinely sequential and single-threaded, so
        it is a clean reference to compare the parallel path against."""
        seen: set[int] = set()

        def record() -> None:
            seen.add(threading.get_ident())

        with _client(1) as client:
            client.gather([record for _ in range(5)])
        assert seen == {threading.get_ident()}

    def test_actually_runs_in_parallel(self) -> None:
        """Guards against the pool silently degrading to serial execution."""

        def slow() -> None:
            time.sleep(0.05)

        with _client(8) as client:
            start = time.perf_counter()
            client.gather([slow for _ in range(8)])
            elapsed = time.perf_counter() - start
        assert elapsed < 0.20, f"8 x 50ms took {elapsed:.2f}s; not parallel"

    def test_results_match_sequential(self) -> None:
        jobs = [(lambda n=n: n * 2) for n in range(30)]
        with _client(1) as a, _client(6) as b:
            assert a.gather(jobs) == b.gather(jobs)


class TestConnectionPoolSizing:
    def test_pool_matches_worker_count(self) -> None:
        """If the HTTP pool is smaller than the worker count the requests queue
        behind each other and the parallelism is silently undone. Verified by
        inspecting the real connection pool, since that is the thing that bites.
        """
        client = ReadClient(LaunchpadConfig(), concurrency=6)
        try:
            pool = client._client._transport._pool  # type: ignore[attr-defined]
            assert pool._max_connections == 6
            assert pool._max_keepalive_connections == 6
        finally:
            client.close()

    def test_defaults_to_configured_concurrency(self) -> None:
        with ReadClient(LaunchpadConfig(max_concurrency=3)) as client:
            assert client.concurrency == 3

    def test_explicit_override_wins(self) -> None:
        with ReadClient(LaunchpadConfig(max_concurrency=3), concurrency=7) as client:
            assert client.concurrency == 7

    def test_falsy_override_falls_back_to_config(self) -> None:
        """0 and None both mean "unspecified", not "no workers"."""
        with ReadClient(LaunchpadConfig(max_concurrency=5), concurrency=0) as client:
            assert client.concurrency == 5

    def test_never_below_one(self) -> None:
        with ReadClient(LaunchpadConfig(max_concurrency=0)) as client:
            assert client.concurrency == 1
        with ReadClient(LaunchpadConfig(), concurrency=-3) as client:
            assert client.concurrency == 1


# ---------------------------------------------------------------------------
# Conditional GET
# ---------------------------------------------------------------------------


class TestConditionalGet:
    def test_304_returns_no_payload(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers["If-None-Match"] == '"tag"'
            return httpx.Response(304)

        with _client(1, httpx.MockTransport(handler)) as client:
            got = client.get_conditional("https://lp.test/bugs/1", '"tag"')
        assert got.not_modified
        assert got.payload is None
        assert got.etag == '"tag"'

    def test_304_does_not_attempt_json_parsing(self) -> None:
        """A 304 has an empty body; treating it as a normal response would raise.
        It must be handled before the error branch and before .json()."""
        with _client(1, httpx.MockTransport(lambda r: httpx.Response(304))) as client:
            assert client.get_conditional("https://lp.test/bugs/1", '"t"').not_modified

    def test_200_returns_payload_and_normalised_etag(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"id": 1}, headers={"ETag": '"fresh-gzip"'})

        with _client(1, httpx.MockTransport(handler)) as client:
            got = client.get_conditional("https://lp.test/bugs/1", None)
        assert got.payload == {"id": 1}
        assert got.etag == '"fresh"', "the -gzip suffix must be stripped before storing"

    def test_sends_normalised_etag(self) -> None:
        sent: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request.headers.get("If-None-Match"))
            return httpx.Response(304)

        with _client(1, httpx.MockTransport(handler)) as client:
            client.get_conditional("https://lp.test/bugs/1", '"tag-gzip"')
        assert sent == ['"tag"']

    def test_no_etag_sends_no_header(self) -> None:
        sent: list[str | None] = []

        def handler(request: httpx.Request) -> httpx.Response:
            sent.append(request.headers.get("If-None-Match"))
            return httpx.Response(200, json={"id": 1})

        with _client(1, httpx.MockTransport(handler)) as client:
            client.get_conditional("https://lp.test/bugs/1", None)
        assert sent == [None]

    def test_keeps_old_etag_when_304_omits_one(self) -> None:
        with _client(1, httpx.MockTransport(lambda r: httpx.Response(304))) as client:
            assert client.get_conditional("https://lp.test/x", '"old"').etag == '"old"'


# ---------------------------------------------------------------------------
# total_size typing
# ---------------------------------------------------------------------------


class TestCollectionSize:
    """``ws.show=total_size`` is inconsistently typed: sub-collections such as
    ``/bugs/N/cves`` return an object, while ``searchTasks`` returns a bare
    integer. Assuming an object raises AttributeError on the latter.
    """

    def test_object_form(self) -> None:
        handler = httpx.MockTransport(
            lambda r: httpx.Response(200, json={"total_size": 7, "entries": []})
        )
        with _client(1, handler) as client:
            assert client.collection_size("https://lp.test/c") == 7

    def test_bare_integer_form(self) -> None:
        handler = httpx.MockTransport(lambda r: httpx.Response(200, json=20))
        with _client(1, handler) as client:
            assert client.collection_size("https://lp.test/c") == 20

    def test_count_falls_back_on_unusable_payload(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.params.get("ws.show") == "total_size":
                return httpx.Response(200, json="nonsense")
            return httpx.Response(200, json={"entries": [{}, {}, {}]})

        with _client(1, httpx.MockTransport(handler)) as client:
            assert client.count("https://lp.test/c") == 3

    def test_boolean_is_rejected(self) -> None:
        """``True`` is an int in Python; silently returning 1 would be wrong."""
        handler = httpx.MockTransport(lambda r: httpx.Response(200, json=True))
        with _client(1, handler) as client, pytest.raises(LaunchpadError):
            client.collection_size("https://lp.test/c")


# ---------------------------------------------------------------------------
# Fake Launchpad, for pipeline tests
# ---------------------------------------------------------------------------

FIXTURE_BUGS = (1374898, 717691, 1509299)


class FakeLaunchpad:
    """Replays recorded payloads and counts requests, with no network."""

    def __init__(self, bug_ids: tuple[int, ...] = FIXTURE_BUGS) -> None:
        self.bug_ids = bug_ids
        self.calls: list[str] = []
        self.lock = threading.Lock()
        self.etags: dict[int, str] = {i: f'"etag-{i}"' for i in bug_ids}
        self.honour_if_none_match = True
        self.bump_last_updated = False

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        with self.lock:
            self.calls.append(path)

        if path.endswith("/+source/vim"):
            if request.url.params.get("ws.op") == "searchTasks":
                return httpx.Response(
                    200,
                    json={
                        "entries": [
                            {"bug_link": f"https://lp.test/devel/bugs/{i}"} for i in self.bug_ids
                        ]
                    },
                )
            return httpx.Response(200, json={"name": "vim"})

        if path.endswith("/+archive/primary"):
            return httpx.Response(200, json={"entries": []})

        parts = path.strip("/").split("/")
        if "bugs" in parts:
            index = parts.index("bugs")
            bug_id = int(parts[index + 1])
            tail = parts[index + 2] if len(parts) > index + 2 else ""
            return self._bug_response(request, bug_id, tail)

        return httpx.Response(404)

    def _bug_response(self, request: httpx.Request, bug_id: int, tail: str) -> httpx.Response:
        if not tail:
            inm = request.headers.get("If-None-Match")
            tag = self.etags[bug_id]
            if self.honour_if_none_match and inm == tag:
                return httpx.Response(304)
            payload = load_fixture(f"bug_{bug_id}")
            if self.bump_last_updated:
                payload["date_last_updated"] = "2026-09-01T00:00:00+00:00"
            return httpx.Response(200, json=payload, headers={"ETag": tag})

        name = {"bug_tasks": "bug_tasks", "messages": "messages", "attachments": "attachments",
                "bug_watches": "bug_watches", "cves": "cves",
                "vulnerabilities": "vulnerabilities",
                "linked_merge_proposals": "linked_merge_proposals",
                "linked_branches": "linked_branches"}.get(tail)
        if name is None:
            return httpx.Response(404)
        return httpx.Response(200, json=load_fixture(f"bug_{bug_id}_{name}"))

    # -- helpers -----------------------------------------------------------

    def count_of(self, suffix: str) -> int:
        return sum(1 for c in self.calls if c.endswith(suffix))

    @property
    def bug_entry_calls(self) -> int:
        return sum(1 for c in self.calls if c.rstrip("/").split("/")[-1].isdigit())


def _fetcher(fake: FakeLaunchpad, store: Store, concurrency: int) -> tuple[Fetcher, ReadClient]:
    config = Config.model_validate(
        {"launchpad": {"max_concurrency": concurrency, "distribution": "ubuntu"}}
    )
    client = ReadClient(
        config.launchpad,
        client=httpx.Client(transport=httpx.MockTransport(fake.handler), base_url="https://lp.test"),
        concurrency=concurrency,
    )
    table = SeriesTable.from_api("ubuntu", SERIES_ENTRIES)
    return Fetcher(client, config, store, table), client


class TestFetchRequestBudget:
    """Fetching is latency-bound, so the request count *is* the performance
    contract. These assertions catch a reintroduced round-trip per bug, which no
    behavioural test would notice.
    """

    def test_cold_fetch_request_counts(self, tmp_path: Path) -> None:
        fake = FakeLaunchpad()
        with Store.open(tmp_path) as store:
            fetcher, client = _fetcher(fake, store, 4)
            stats = fetcher.fetch("vim")
            client.close()

        assert stats.bugs_seen == 3
        # 1 entry + 1 bug_tasks per bug, then 7 enrichment calls per survivor.
        expected = 2 * stats.bugs_seen + 7 * stats.enriched
        assert stats.requests == expected, (
            f"expected {expected} requests "
            f"(2 x {stats.bugs_seen} + 7 x {stats.enriched}), got {stats.requests}"
        )

    def test_prefiltered_bugs_cost_only_two_requests(self, tmp_path: Path) -> None:
        """The whole point of the two-phase design: a bug that is obviously
        untouchable must not cost seven extra round-trips."""
        fake = FakeLaunchpad()
        with Store.open(tmp_path) as store:
            fetcher, client = _fetcher(fake, store, 4)
            stats = fetcher.fetch("vim")
            client.close()

        assert stats.prefiltered > 0, "fixtures should include at least one skipped bug"
        assert stats.requests == 2 * stats.bugs_seen + 7 * stats.enriched

    def test_warm_fetch_costs_one_request_per_bug(self, tmp_path: Path) -> None:
        """A revalidated bug needs exactly one conditional GET and nothing else."""
        fake = FakeLaunchpad()
        with Store.open(tmp_path) as store:
            fetcher, client = _fetcher(fake, store, 4)
            fetcher.fetch("vim")
            client.close()

            fake.calls.clear()
            fetcher2, client2 = _fetcher(fake, store, 4)
            stats = fetcher2.fetch("vim")
            client2.close()

        assert stats.from_cache == 3
        assert stats.fetched == 0
        assert stats.requests == 3, f"expected 1 conditional GET per bug, got {stats.requests}"
        assert fake.count_of("/bug_tasks") == 0
        assert fake.count_of("/messages") == 0

    def test_refresh_bypasses_revalidation(self, tmp_path: Path) -> None:
        fake = FakeLaunchpad()
        with Store.open(tmp_path) as store:
            fetcher, client = _fetcher(fake, store, 4)
            fetcher.fetch("vim")
            client.close()

            fetcher2, client2 = _fetcher(fake, store, 4)
            stats = fetcher2.fetch("vim", refresh=True)
            client2.close()

        assert stats.from_cache == 0
        assert stats.fetched == 3

    def test_a_genuinely_changed_bug_is_refetched(self, tmp_path: Path) -> None:
        fake = FakeLaunchpad()
        with Store.open(tmp_path) as store:
            fetcher, client = _fetcher(fake, store, 4)
            first = fetcher.fetch("vim")
            client.close()

            # Simulate the bugs changing upstream: a new ETag *and* a new
            # date_last_updated, which is what a real edit produces.
            fake.etags = {i: f'"new-{i}"' for i in fake.bug_ids}
            fake.bump_last_updated = True
            fetcher2, client2 = _fetcher(fake, store, 4)
            second = fetcher2.fetch("vim")
            client2.close()

        assert second.from_cache == 0
        assert second.fetched == first.fetched

    def test_etag_change_alone_still_reuses_the_snapshot(self, tmp_path: Path) -> None:
        """Deliberate second line of defence.

        An ETag can change for reasons unrelated to the bug's content (a backend
        rebuild, a header change). ``date_last_updated`` is the authoritative
        signal, so when the ETag moves but that timestamp has not, the cached
        snapshot still stands and the extra six requests are skipped.
        """
        fake = FakeLaunchpad()
        with Store.open(tmp_path) as store:
            fetcher, client = _fetcher(fake, store, 4)
            fetcher.fetch("vim")
            client.close()

            fake.etags = {i: f'"rebuilt-{i}"' for i in fake.bug_ids}
            fake.calls.clear()
            fetcher2, client2 = _fetcher(fake, store, 4)
            stats = fetcher2.fetch("vim")
            client2.close()

        assert stats.from_cache == 3
        assert stats.fetched == 0
        assert fake.count_of("/messages") == 0


class TestConcurrentMatchesSequential:
    def test_identical_snapshots_at_any_concurrency(self, tmp_path: Path) -> None:
        """Parallelism must not change a single byte of the result."""
        snapshots: dict[int, list[str]] = {}
        for concurrency in (1, 2, 8):
            directory = tmp_path / f"c{concurrency}"
            fake = FakeLaunchpad()
            with Store.open(directory) as store:
                fetcher, client = _fetcher(fake, store, concurrency)
                fetcher.fetch("vim")
                client.close()
                snapshots[concurrency] = [
                    b.model_dump_json(exclude={"fetched_at"})
                    for b in store.iter_bugs("ubuntu", "vim")
                ]

        assert snapshots[1] == snapshots[2] == snapshots[8]

    def test_ordering_is_deterministic(self, tmp_path: Path) -> None:
        fake = FakeLaunchpad()
        with Store.open(tmp_path) as store:
            fetcher, client = _fetcher(fake, store, 8)
            fetcher.fetch("vim")
            client.close()
            ids = [b.id for b in store.iter_bugs("ubuntu", "vim")]
        assert ids == sorted(ids)


class TestErrorIsolation:
    def test_one_broken_bug_does_not_abort_the_run(self, tmp_path: Path) -> None:
        fake = FakeLaunchpad()
        original = fake.handler

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.rstrip("/").endswith("/717691"):
                return httpx.Response(404)
            return original(request)

        with Store.open(tmp_path) as store:
            config = Config.model_validate({"launchpad": {"max_concurrency": 4}})
            client = ReadClient(
                config.launchpad,
                client=httpx.Client(
                    transport=httpx.MockTransport(handler), base_url="https://lp.test"
                ),
                concurrency=4,
            )
            table = SeriesTable.from_api("ubuntu", SERIES_ENTRIES)
            stats = Fetcher(client, config, store, table).fetch("vim")
            client.close()
            stored = {b.id for b in store.iter_bugs("ubuntu", "vim")}

        assert stats.errors == 1
        assert 717691 not in stored
        assert stored, "the other bugs should still have been fetched"

    def test_failed_enrichment_does_not_store_a_half_enriched_snapshot(
        self, tmp_path: Path
    ) -> None:
        """A snapshot missing its CVE or patch counts would look clean to the
        exclusions, so it must be dropped rather than persisted."""
        fake = FakeLaunchpad()
        original = fake.handler

        def handler(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/cves"):
                return httpx.Response(500)
            return original(request)

        with Store.open(tmp_path) as store:
            config = Config.model_validate(
                {"launchpad": {"max_concurrency": 4, "max_retries": 1}}
            )
            client = ReadClient(
                config.launchpad,
                client=httpx.Client(
                    transport=httpx.MockTransport(handler), base_url="https://lp.test"
                ),
                concurrency=4,
            )
            table = SeriesTable.from_api("ubuntu", SERIES_ENTRIES)
            stats = Fetcher(client, config, store, table).fetch("vim")
            client.close()
            stored = list(store.iter_bugs("ubuntu", "vim"))

        assert stats.errors >= 1
        assert all(not b.enriched for b in stored), (
            "no snapshot should claim to be enriched when enrichment failed"
        )


class TestChunking:
    def test_chunking_produces_the_same_result(self, tmp_path: Path) -> None:
        results: dict[int, list[int]] = {}
        for chunk_size in (1, 2, 100):
            fake = FakeLaunchpad()
            directory = tmp_path / f"k{chunk_size}"
            with Store.open(directory) as store:
                config = Config.model_validate({"launchpad": {"max_concurrency": 4}})
                client = ReadClient(
                    config.launchpad,
                    client=httpx.Client(
                        transport=httpx.MockTransport(fake.handler), base_url="https://lp.test"
                    ),
                    concurrency=4,
                )
                table = SeriesTable.from_api("ubuntu", SERIES_ENTRIES)
                fetcher = Fetcher(client, config, store, table, chunk_size=chunk_size)
                fetcher.fetch("vim")
                client.close()
                results[chunk_size] = [b.id for b in store.iter_bugs("ubuntu", "vim")]

        assert results[1] == results[2] == results[100]


class TestArchiveParallelism:
    def test_one_request_per_live_series(self) -> None:
        calls: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            calls.append(str(request.url.params.get("distro_series")))
            return httpx.Response(200, json={"entries": []})

        table = SeriesTable.from_api("ubuntu", SERIES_ENTRIES)
        client = ReadClient(
            LaunchpadConfig(),
            client=httpx.Client(
                transport=httpx.MockTransport(handler), base_url="https://lp.test"
            ),
            concurrency=4,
        )
        try:
            index = fetch_archive_index(client, table, "vim")
        finally:
            client.close()

        assert len(calls) == len(table.live)
        assert index.queried_series == tuple(s.name for s in table.live)
        assert not index.incomplete

    def test_a_failed_series_lookup_marks_the_index_incomplete(self) -> None:
        """Partial archive data must never drive removed_from_archive."""

        def handler(request: httpx.Request) -> httpx.Response:
            if "noble" in str(request.url.params.get("distro_series")):
                return httpx.Response(500)
            return httpx.Response(200, json={"entries": []})

        table = SeriesTable.from_api("ubuntu", SERIES_ENTRIES)
        client = ReadClient(
            LaunchpadConfig(max_retries=1),
            client=httpx.Client(
                transport=httpx.MockTransport(handler), base_url="https://lp.test"
            ),
            concurrency=4,
        )
        try:
            index = fetch_archive_index(client, table, "vim")
        finally:
            client.close()

        assert index.incomplete


class TestStoreEtags:
    def test_etag_round_trip(self, tmp_path: Path) -> None:
        from pruner.models import BugSnapshot

        with Store.open(tmp_path) as store:
            store.put_bug("ubuntu", "vim", BugSnapshot(id=1, title="x"), etag='"a"')
            assert store.cached_etags("ubuntu", "vim") == {1: '"a"'}

    def test_update_without_an_etag_keeps_the_existing_one(self, tmp_path: Path) -> None:
        """A re-fetch that yields no ETag header must not blank the stored one."""
        from pruner.models import BugSnapshot

        with Store.open(tmp_path) as store:
            store.put_bug("ubuntu", "vim", BugSnapshot(id=1, title="x"), etag='"a"')
            store.put_bug("ubuntu", "vim", BugSnapshot(id=1, title="y"), etag=None)
            assert store.cached_etags("ubuntu", "vim") == {1: '"a"'}

    def test_migrates_a_v1_database(self, tmp_path: Path) -> None:
        """The cache is disposable, but there is no reason to make a user
        re-download a backlog over a new nullable column."""
        import sqlite3

        from pruner.models import BugSnapshot

        database = tmp_path / "cache.db"
        legacy = sqlite3.connect(database)
        legacy.executescript(
            """
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE bugs (
                bug_id INTEGER NOT NULL, distribution TEXT NOT NULL,
                package TEXT NOT NULL, date_last_updated TEXT,
                fetched_at TEXT NOT NULL, payload TEXT NOT NULL,
                PRIMARY KEY (distribution, package, bug_id));
            INSERT INTO meta VALUES ('schema_version', '1');
            """
        )
        legacy.execute(
            "INSERT INTO bugs VALUES (?,?,?,?,?,?)",
            (99, "ubuntu", "vim", None, "2026-01-01T00:00:00+00:00",
             BugSnapshot(id=99, title="legacy").model_dump_json()),
        )
        legacy.commit()
        legacy.close()

        with Store.open(tmp_path) as store:
            preserved = store.get_bug("ubuntu", "vim", 99)
            assert preserved is not None
            assert preserved.title == "legacy"
            assert store.cached_etags("ubuntu", "vim") == {99: None}
            store.put_bug("ubuntu", "vim", BugSnapshot(id=100, title="new"), etag='"b"')
            assert store.cached_etags("ubuntu", "vim")[100] == '"b"'


class TestConditionalDataclass:
    def test_not_modified_is_derived_from_payload(self) -> None:
        assert Conditional(payload=None, etag='"x"').not_modified
        assert not Conditional(payload={"a": 1}, etag='"x"').not_modified
