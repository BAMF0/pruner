"""Anonymous, read-only Launchpad API client.

This module intentionally uses plain HTTP GETs and holds **no credentials**. The
entire fetch/analyze/report path therefore cannot mutate Launchpad even if it is
buggy -- authenticated access lives solely in :mod:`pruner.lp.write`.

Fetching a backlog is latency-bound, not bandwidth-bound: measured against
production Launchpad, a single request costs around 230ms while returning as
little as 127 bytes, and a fully enriched bug needs nine of them. Launchpad is
HTTP/1.1 only, so there is no multiplexing to exploit; the answer is a bounded
pool of parallel connections. At four workers this measured 2.4x faster
end-to-end than sequential, at eight 5.1x.

So this client:

* runs batches of requests through a bounded thread pool (:meth:`ReadClient.gather`),
  sized to match the HTTP connection pool -- if those two disagree the requests
  silently re-serialise,
* honours ``Retry-After`` on 429/503 and backs off exponentially with jitter
  otherwise, per request, inside whichever worker hit the limit,
* follows ``next_collection_link`` for paging,
* supports conditional GETs so repeat runs revalidate instead of re-downloading.
"""

from __future__ import annotations

import logging
import random
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, TypeVar
from urllib.parse import quote

import httpx

from pruner.config import LaunchpadConfig

log = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

T = TypeVar("T")

#: Apache's ``mod_deflate`` appends ``-gzip`` to the ETag when it compresses a
#: response, but compares ``If-None-Match`` against the *uncompressed* tag. Since
#: httpx requests gzip by default, echoing the tag back verbatim always misses and
#: yields a full 200. Verified against production Launchpad: the suffixed tag
#: returns 200 with 3743 bytes, the stripped tag returns 304 with 0 bytes.
_GZIP_ETAG_SUFFIX = re.compile(r'-gzip(")?$')


def normalise_etag(etag: str | None) -> str | None:
    """Strip ``mod_deflate``'s ``-gzip`` suffix so revalidation actually works."""
    if not etag:
        return None
    return _GZIP_ETAG_SUFFIX.sub(r"\1", etag)


class LaunchpadError(RuntimeError):
    """Non-retryable failure talking to Launchpad."""


class NotFound(LaunchpadError):
    """The requested Launchpad object does not exist (HTTP 404)."""


@dataclass(frozen=True)
class Conditional:
    """Result of a conditional GET."""

    payload: dict[str, Any] | None
    """``None`` when the server answered 304."""

    etag: str | None
    """Normalised ETag to store for next time."""

    @property
    def not_modified(self) -> bool:
        return self.payload is None


class ReadClient:
    """Minimal read-only wrapper over the Launchpad REST API."""

    def __init__(
        self,
        config: LaunchpadConfig | None = None,
        *,
        client: httpx.Client | None = None,
        concurrency: int | None = None,
    ) -> None:
        self.config = config or LaunchpadConfig()
        self.concurrency = max(1, concurrency or self.config.max_concurrency)
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=self.config.timeout,
            headers={
                "User-Agent": self.config.user_agent,
                "Accept": "application/json",
            },
            follow_redirects=True,
            # Must match the worker count. A smaller pool would queue the
            # requests behind each other and undo the parallelism entirely.
            limits=httpx.Limits(
                max_connections=self.concurrency,
                max_keepalive_connections=self.concurrency,
            ),
        )
        self._pool: ThreadPoolExecutor | None = None
        self._counter_lock = threading.Lock()
        self.request_count = 0
        """Requests actually issued. Used by tests to guard against a refactor
        quietly reintroducing a per-bug round-trip."""

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ReadClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- concurrency -------------------------------------------------------

    def gather(
        self,
        jobs: Sequence[Callable[[], T]],
        *,
        on_done: Callable[[], None] | None = None,
    ) -> list[T | BaseException]:
        """Run ``jobs`` in parallel, returning results in the original order.

        Exceptions are captured per job rather than propagated, as
        ``asyncio.gather(return_exceptions=True)`` does, so that one unreachable
        bug cannot abort a whole batch. Callers are expected to inspect the
        results for ``BaseException``.

        With ``concurrency == 1`` this runs inline, which keeps the sequential
        path genuinely sequential (and easy to compare against in tests) rather
        than merely a pool of one.

        ``on_done`` fires exactly once per finished job, successful or not, on
        whichever thread ran it -- callers must make it thread-safe. It exists so
        that a long batch can report progress, and deliberately does not touch
        :attr:`request_count`, which is a performance contract asserted on in
        the tests.
        """
        if not jobs:
            return []

        runner: Callable[[Callable[[], T]], T | BaseException] = (
            _run if on_done is None else partial(_notify, on_done=on_done)
        )

        if self.concurrency == 1 or len(jobs) == 1:
            return [runner(job) for job in jobs]

        if self._pool is None:
            self._pool = ThreadPoolExecutor(
                max_workers=self.concurrency, thread_name_prefix="pruner-lp"
            )
        return list(self._pool.map(runner, jobs))

    # -- low level ---------------------------------------------------------

    def _url(self, path_or_url: str) -> str:
        if path_or_url.startswith(("http://", "https://")):
            return path_or_url
        return f"{self.config.api_root}/{path_or_url.lstrip('/')}"


    def _request(
        self,
        url: str,
        *,
        params: Mapping[str, str] | Sequence[tuple[str, str]] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """Issue one GET, retrying transient failures. The single I/O choke point.

        Returns the response rather than parsed JSON so that callers can
        distinguish a 304 from a body. ``304`` is deliberately handled here
        instead of falling into the ``>= 400`` branch: it is a success, and
        calling ``.json()`` on its empty body would raise.
        """
        # QueryParams rather than a raw list: Launchpad needs repeated keys
        # (``status`` may appear several times), which a plain Mapping cannot
        # express, and QueryParams preserves duplicates. The pairs are rebuilt
        # with httpx's own value type because list is invariant.
        query: httpx.QueryParams | None = None
        if params is not None:
            items = params.items() if isinstance(params, Mapping) else params
            pairs: list[tuple[str, str | int | float | bool | None]] = [
                (key, value) for key, value in items
            ]
            query = httpx.QueryParams(pairs)

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            with self._counter_lock:
                self.request_count += 1
            try:
                response = self._client.get(url, params=query, headers=headers)
            except httpx.HTTPError as exc:  # network-level blip
                last_error = exc
                self._sleep(attempt, None)
                continue

            if response.status_code == 404:
                raise NotFound(f"not found: {url}")
            if response.status_code in RETRY_STATUSES:
                last_error = LaunchpadError(f"HTTP {response.status_code} from {url}")
                self._sleep(attempt, response.headers.get("Retry-After"))
                continue
            if response.status_code == 304:
                return response
            if response.status_code >= 400:
                raise LaunchpadError(
                    f"HTTP {response.status_code} from {url}: {response.text[:300]}"
                )
            return response

        raise LaunchpadError(
            f"giving up on {url} after {self.config.max_retries} attempts"
        ) from last_error

    @staticmethod
    def _json(response: httpx.Response, url: str) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise LaunchpadError(f"non-JSON response from {url}") from exc

    def get(self, path_or_url: str, **params: Any) -> dict[str, Any]:
        """GET a Launchpad resource and return its JSON object."""
        url = self._url(path_or_url)
        query = {k: _param(v) for k, v in params.items() if v is not None}
        payload = self._json(self._request(url, params=query or None), url)
        if not isinstance(payload, dict):
            raise LaunchpadError(f"expected a JSON object from {url}, got {type(payload).__name__}")
        return payload

    def get_conditional(self, path_or_url: str, etag: str | None = None) -> Conditional:
        """GET a resource, revalidating against ``etag`` when one is supplied.

        A hit costs the same round-trip but transfers no body, which keeps repeat
        runs cheap and is markedly kinder to Launchpad's caches.
        """
        url = self._url(path_or_url)
        normalised = normalise_etag(etag)
        headers = {"If-None-Match": normalised} if normalised else None
        response = self._request(url, headers=headers)

        fresh = normalise_etag(response.headers.get("ETag"))
        if response.status_code == 304:
            return Conditional(payload=None, etag=normalised)

        payload = self._json(response, url)
        if not isinstance(payload, dict):
            raise LaunchpadError(f"expected a JSON object from {url}")
        return Conditional(payload=payload, etag=fresh or normalised)

    def _sleep(self, attempt: int, retry_after: str | None) -> None:
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 2.0**attempt
        else:
            # Exponential backoff with jitter, capped so a wedged API does not
            # stall a run for minutes at a time. The jitter also stops parallel
            # workers that hit the same limit from retrying in lockstep.
            delay = min(2.0**attempt + random.uniform(0, 0.5), 30.0)
        log.debug("launchpad backoff: sleeping %.1fs (attempt %d)", delay, attempt + 1)
        time.sleep(delay)

    # -- collections -------------------------------------------------------

    def collection(
        self,
        path_or_url: str,
        *,
        limit: int | None = None,
        page_size: int = 75,
        **params: Any,
    ) -> list[dict[str, Any]]:
        """Fetch all entries of a collection, following pagination."""
        entries: list[dict[str, Any]] = []
        payload = self.get(path_or_url, **{**params, "ws.size": page_size})

        while True:
            entries.extend(payload.get("entries", []))
            if limit is not None and len(entries) >= limit:
                return entries[:limit]
            next_link = payload.get("next_collection_link")
            if not next_link:
                return entries
            payload = self.get(next_link)

    def collection_size(self, path_or_url: str, **params: Any) -> int:
        """Total size of a collection without downloading all of it.

        ``ws.show=total_size`` is not consistently typed: sub-collections such as
        ``/bugs/N/cves`` return an object containing ``total_size``, while
        ``searchTasks`` returns a **bare integer**. Both are handled, because
        assuming an object here raises ``AttributeError`` on the latter.
        """
        url = self._url(path_or_url)
        query = {k: _param(v) for k, v in params.items() if v is not None}
        query.update({"ws.show": "total_size", "ws.size": "1"})
        payload = self._json(self._request(url, params=query), url)
        if isinstance(payload, bool):  # bool is an int; reject it explicitly
            raise LaunchpadError(f"unexpected boolean total_size from {url}")
        if isinstance(payload, int):
            return payload
        if isinstance(payload, dict):
            return int(payload.get("total_size", 0))
        raise LaunchpadError(f"cannot read total_size from {url}")

    def count(self, path_or_url: str, **params: Any) -> int:
        """Cheap count of a sub-collection.

        Falls back to counting a single page for endpoints that ignore
        ``ws.show=total_size``.
        """
        try:
            return self.collection_size(path_or_url, **params)
        except (LaunchpadError, AttributeError, ValueError, TypeError):
            payload = self.get(path_or_url, **{**params, "ws.size": 1})
            if (total := payload.get("total_size")) is not None:
                return int(total)
            return len(payload.get("entries", []))


    # -- domain helpers ----------------------------------------------------

    def distribution(self, name: str) -> dict[str, Any]:
        return self.get(f"/{quote(name)}")

    def series(self, distribution: str) -> list[dict[str, Any]]:
        return self.collection(f"/{quote(distribution)}/series")

    def source_package(self, distribution: str, package: str) -> dict[str, Any]:
        """The ``distribution_source_package`` resource. Raises :class:`NotFound`
        if the package has never existed in the distribution."""
        return self.get(f"/{quote(distribution)}/+source/{quote(package)}")

    def search_tasks(
        self,
        distribution: str,
        package: str,
        *,
        statuses: tuple[str, ...] = (),
        tags: tuple[str, ...] = (),
        limit: int | None = None,
        order_by: str | None = None,
    ) -> list[dict[str, Any]]:
        """Search bug tasks on a distribution source package.

        ``status`` may be repeated, so it is passed as a list rather than through
        the scalar ``params`` path.
        """
        url = self._url(f"/{quote(distribution)}/+source/{quote(package)}")
        params: list[tuple[str, str]] = [("ws.op", "searchTasks")]
        for status in statuses:
            params.append(("status", status))
        for tag in tags:
            params.append(("tags", tag))
        if order_by:
            params.append(("order_by", order_by))
        return self._paged(url, params, limit=limit)

    def _paged(
        self,
        url: str,
        params: list[tuple[str, str]],
        *,
        limit: int | None,
        page_size: int = 75,
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        query = [*params, ("ws.size", str(page_size))]
        payload = self._get_with_params(url, query)
        while True:
            entries.extend(payload.get("entries", []))
            if limit is not None and len(entries) >= limit:
                return entries[:limit]
            next_link = payload.get("next_collection_link")
            if not next_link:
                return entries
            payload = self.get(next_link)

    def _get_with_params(self, url: str, params: list[tuple[str, str]]) -> dict[str, Any]:
        # httpx accepts a list of pairs; Launchpad needs repeated keys (``status``
        # may appear several times), which a Mapping cannot express.
        payload = self._json(self._request(url, params=list(params)), url)
        if not isinstance(payload, dict):
            raise LaunchpadError(f"expected a JSON object from {url}")
        return payload

    def bug(self, bug_id: int) -> dict[str, Any]:
        return self.get(f"/bugs/{bug_id}")

    def bug_conditional(self, bug_id: int, etag: str | None = None) -> Conditional:
        return self.get_conditional(f"/bugs/{bug_id}", etag)

    def bug_tasks(self, bug_id: int) -> list[dict[str, Any]]:
        return self.collection(f"/bugs/{bug_id}/bug_tasks")

    def bug_messages(self, bug_id: int, *, limit: int | None = None) -> list[dict[str, Any]]:
        return self.collection(f"/bugs/{bug_id}/messages", limit=limit)

    def bug_watches(self, bug_id: int) -> list[dict[str, Any]]:
        return self.collection(f"/bugs/{bug_id}/bug_watches")

    def published_sources(
        self,
        distribution: str,
        package: str,
        *,
        series_link: str,
        pocket: str | None = None,
        status: str = "Published",
    ) -> list[dict[str, Any]]:
        """Source publications for ``package`` in one series.

        An empty result across every live series is what the
        ``removed_from_archive`` rule keys on.
        """
        url = self._url(f"/{quote(distribution)}/+archive/primary")
        params: list[tuple[str, str]] = [
            ("ws.op", "getPublishedSources"),
            ("source_name", package),
            ("exact_match", "true"),
            ("status", status),
            ("distro_series", series_link),
        ]
        if pocket:
            params.append(("pocket", pocket))
        return self._paged(url, params, limit=None, page_size=20)


def _param(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _run[R](job: Callable[[], R]) -> R | BaseException:
    """Execute one job, returning rather than raising on failure.

    Catches ``Exception`` only: a ``KeyboardInterrupt`` or ``SystemExit`` must
    still propagate so that Ctrl-C during a long sweep actually stops it. The
    exception is returned, not swallowed -- callers inspect the results.
    """
    try:
        return job()
    except Exception as exc:
        return exc


def _notify[R](job: Callable[[], R], *, on_done: Callable[[], None]) -> R | BaseException:
    """Run one job and signal completion, whether it succeeded or not.

    In a ``finally`` so that a failing job still advances the bar; a batch where
    one bug 404s must not leave progress stalled short of its total.
    """
    try:
        return _run(job)
    finally:
        on_done()
