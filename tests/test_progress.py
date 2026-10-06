"""Progress plumbing.

The callbacks must fire exactly once per unit of work -- including for jobs
that fail -- and must not change what the fetcher actually does. That second
point is the important one: it is the guard on the request-count contract
asserted in ``test_fetch_performance.py``.
"""

from __future__ import annotations

import io
import threading
from pathlib import Path

import httpx
from rich.console import Console

from pruner.config import Config
from pruner.lp.read import ReadClient
from pruner.progress import Reporter, _noop
from pruner.store import Store
from tests.test_fetch_performance import FakeLaunchpad, _fetcher

# ---------------------------------------------------------------------------
# ReadClient.gather(on_done=...)
# ---------------------------------------------------------------------------


class TestGatherOnDone:
    def _client(self, concurrency: int) -> ReadClient:
        config = Config.model_validate(
            {"launchpad": {"max_concurrency": concurrency, "distribution": "ubuntu"}}
        )
        return ReadClient(
            config.launchpad,
            client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(200))),
            concurrency=concurrency,
        )

    def test_fires_once_per_job_pooled(self) -> None:
        client = self._client(4)
        lock = threading.Lock()
        count = 0

        def on_done() -> None:
            nonlocal count
            with lock:
                count += 1

        jobs = [(lambda i=i: i) for i in range(10)]
        results = client.gather(jobs, on_done=on_done)
        client.close()

        assert count == 10
        assert results == list(range(10))

    def test_fires_for_failing_jobs_too(self) -> None:
        client = self._client(4)
        lock = threading.Lock()
        count = 0

        def on_done() -> None:
            nonlocal count
            with lock:
                count += 1

        def ok(i: int) -> int:
            return i

        def boom() -> int:
            raise RuntimeError("nope")

        jobs = [lambda: ok(0), boom, lambda: ok(2), boom]
        results = client.gather(jobs, on_done=on_done)
        client.close()

        assert count == 4
        assert results[0] == 0
        assert isinstance(results[1], RuntimeError)
        assert results[2] == 2
        assert isinstance(results[3], RuntimeError)

    def test_fires_on_inline_path(self) -> None:
        client = self._client(1)
        count = 0

        def on_done() -> None:
            nonlocal count
            count += 1

        results = client.gather([lambda: 1, lambda: 2, lambda: 3], on_done=on_done)
        client.close()

        assert count == 3
        assert results == [1, 2, 3]

    def test_on_done_does_not_perturb_request_count(self, tmp_path: Path) -> None:
        """The important test: threading a callback through ``gather`` must not
        add or remove a single HTTP request."""
        with Store.open(tmp_path / "a") as store:
            fake = FakeLaunchpad()
            fetcher, client = _fetcher(fake, store, 4)
            stats_without = fetcher.fetch("vim")
            client.close()

        seen: list[tuple[str, int, int]] = []

        def stage(done: int, total: int, name: str) -> None:
            seen.append((name, done, total))

        with Store.open(tmp_path / "b") as store2:
            fake2 = FakeLaunchpad()
            fetcher2, client2 = _fetcher(fake2, store2, 4)
            stats_with = fetcher2.fetch("vim", stage=stage)
            client2.close()

        assert stats_with.requests == stats_without.requests
        assert stats_with.requests == 2 * stats_with.bugs_seen + 7 * stats_with.enriched
        assert seen, "stage callback should have fired at least once"


# ---------------------------------------------------------------------------
# Fetcher.fetch(stage=...)
# ---------------------------------------------------------------------------


class TestFetchStageProgress:
    def test_stage_names_and_monotonic_counts(self, tmp_path: Path) -> None:
        fake = FakeLaunchpad()
        events: list[tuple[str, int, int]] = []

        def stage(done: int, total: int, name: str) -> None:
            events.append((name, done, total))

        with Store.open(tmp_path) as store:
            fetcher, client = _fetcher(fake, store, 4)
            stats = fetcher.fetch("vim", stage=stage)
            client.close()

        names = {name for name, _, _ in events}
        assert names <= {"entries", "tasks", "enrich"}
        assert stats.bugs_seen == 3

        by_group: dict[tuple[str, int], list[int]] = {}
        for name, done, total in events:
            by_group.setdefault((name, total), []).append(done)
        for (_name, _total), dones in by_group.items():
            assert dones == sorted(dones)
            assert dones == list(range(1, len(dones) + 1))

        first_entries = min(i for i, (name, _, _) in enumerate(events) if name == "entries")
        enrich_indices = [i for i, (name, _, _) in enumerate(events) if name == "enrich"]
        if enrich_indices:
            assert first_entries < min(enrich_indices)

    def test_progress_reports_true_total_and_leading_zero(self, tmp_path: Path) -> None:
        fake = FakeLaunchpad()
        calls: list[tuple[int, int, str]] = []

        def progress(done: int, total: int, detail: str) -> None:
            calls.append((done, total, detail))

        with Store.open(tmp_path) as store:
            fetcher, client = _fetcher(fake, store, 4)
            stats = fetcher.fetch("vim", progress=progress)
            client.close()

        assert calls[0] == (0, stats.bugs_seen, "")
        assert calls[-1][0] == stats.bugs_seen
        assert calls[-1][1] == stats.bugs_seen


# ---------------------------------------------------------------------------
# Reporter rendering
# ---------------------------------------------------------------------------


class TestReporterNonTty:
    def test_plain_output_has_no_ansi_or_carriage_return(self) -> None:
        buf = io.StringIO()
        console = Console(file=buf, width=100)
        assert console.is_terminal is False

        with Reporter(console) as reporter:
            cb = reporter.task("fetched", total=3)
            cb(1, 3, "detail")
            cb(2, 3, "detail")
            cb(3, 3, "detail")

        out = buf.getvalue()
        assert "3/3" in out
        assert "\x1b[" not in out
        assert "\r" not in out

    def test_stage_is_noop_off_terminal(self) -> None:
        buf = io.StringIO()
        console = Console(file=buf, width=100)

        with Reporter(console) as reporter:
            cb = reporter.stage()
            assert cb is _noop
            cb(1, 5, "entries")  # must not raise, must write nothing

        assert buf.getvalue() == ""


class TestReporterTty:
    def test_stage_resets_between_stage_transitions(self) -> None:
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=100)

        with Reporter(console) as reporter:
            cb = reporter.stage()
            cb(1, 2, "entries")
            cb(2, 2, "entries")
            cb(1, 5, "enrich")

            # The stage task is the second one added (after the primary bar
            # would be, but no primary bar was created here, so it is task 0).
            task = next(iter(reporter._progress.tasks))
            assert task.total == 5
            assert task.completed == 1
            assert task.finished_time is None, (
                "a stage transition must reset the task, or rich freezes "
                "elapsed/ETA/rate at the previous stage's completion"
            )
