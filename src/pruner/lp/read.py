"""Anonymous, read-only Launchpad API client.

This module intentionally uses plain HTTP GETs and holds **no credentials**. The
entire fetch/analyze/report path therefore cannot mutate Launchpad even if it is
buggy -- authenticated access lives solely in :mod:`pruner.lp.write`.

Launchpad throttles aggressively on large collections, so this client:

* honours ``Retry-After`` on 429/503 and backs off exponentially otherwise,
* follows ``next_collection_link`` for paging,
* keeps concurrency low by simply being sequential (a backlog sweep is not
  latency-sensitive, and being a good API citizen matters more).
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any
from urllib.parse import quote

import httpx

from pruner.config import LaunchpadConfig

log = logging.getLogger(__name__)

RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})


class LaunchpadError(RuntimeError):
    """Non-retryable failure talking to Launchpad."""


class NotFound(LaunchpadError):
    """The requested Launchpad object does not exist (HTTP 404)."""


class ReadClient:
    """Minimal read-only wrapper over the Launchpad REST API."""

    def __init__(
        self,
        config: LaunchpadConfig | None = None,
        *,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config or LaunchpadConfig()
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=self.config.timeout,
            headers={
                "User-Agent": self.config.user_agent,
                "Accept": "application/json",
            },
            follow_redirects=True,
        )

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> ReadClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- low level ---------------------------------------------------------

    def _url(self, path_or_url: str) -> str:
        if path_or_url.startswith(("http://", "https://")):
            return path_or_url
        return f"{self.config.api_root}/{path_or_url.lstrip('/')}"

    def get(self, path_or_url: str, **params: Any) -> dict[str, Any]:
        """GET a Launchpad resource, retrying transient failures."""
        url = self._url(path_or_url)
        query = {k: _param(v) for k, v in params.items() if v is not None}

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._client.get(url, params=query or None)
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
            if response.status_code >= 400:
                raise LaunchpadError(
                    f"HTTP {response.status_code} from {url}: {response.text[:300]}"
                )

            try:
                payload: dict[str, Any] = response.json()
            except ValueError as exc:
                raise LaunchpadError(f"non-JSON response from {url}") from exc
            return payload

        raise LaunchpadError(
            f"giving up on {url} after {self.config.max_retries} attempts"
        ) from last_error

    def _sleep(self, attempt: int, retry_after: str | None) -> None:
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 2.0 ** attempt
        else:
            # Exponential backoff with jitter, capped so a wedged API does not
            # stall a run for minutes at a time.
            delay = min(2.0 ** attempt + random.uniform(0, 0.5), 30.0)
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
        """Total size of a collection without downloading all of it."""
        payload = self.get(path_or_url, **{**params, "ws.show": "total_size", "ws.size": 1})
        return int(payload.get("total_size", 0))

    def count(self, path_or_url: str, **params: Any) -> int:
        """Cheap count of a sub-collection.

        Falls back to counting a single page for endpoints that ignore
        ``ws.show=total_size``.
        """
        try:
            return self.collection_size(path_or_url, **params)
        except (LaunchpadError, ValueError, TypeError):
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

    def _get_with_params(
        self, url: str, params: list[tuple[str, str]]
    ) -> dict[str, Any]:
        # httpx accepts a list of pairs; Launchpad needs repeated keys (``status``
        # may appear several times), which a Mapping cannot express.
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = self._client.get(url, params=list(params))
            except httpx.HTTPError as exc:
                last_error = exc
                self._sleep(attempt, None)
                continue
            if response.status_code == 404:
                raise NotFound(f"not found: {url}")
            if response.status_code in RETRY_STATUSES:
                last_error = LaunchpadError(f"HTTP {response.status_code}")
                self._sleep(attempt, response.headers.get("Retry-After"))
                continue
            if response.status_code >= 400:
                raise LaunchpadError(
                    f"HTTP {response.status_code} from {url}: {response.text[:300]}"
                )
            payload: dict[str, Any] = response.json()
            return payload
        raise LaunchpadError(f"giving up on {url}") from last_error

    def bug(self, bug_id: int) -> dict[str, Any]:
        return self.get(f"/bugs/{bug_id}")

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
