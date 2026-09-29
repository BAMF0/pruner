"""Source package publication lookups.

Backs two rules:

* ``removed_from_archive`` -- the package has no ``Published`` source in any live
  series, so bugs against it in this distribution are moot.
* ``likely_fixed`` -- the version in the bug's apport metadata is older than what
  every supported series ships, so the report is about long-superseded code.

Uses ``getPublishedSources`` on the primary archive with ``exact_match=true``,
verified live (noble ships ``vim 2:9.1.0016-1ubuntu7`` in the Release pocket).
Version comparison is delegated to ``python-debian`` so epochs and ``~``/``+``
orderings behave exactly like dpkg.
"""

from __future__ import annotations

import logging
from functools import lru_cache

from debian.debian_support import version_compare
from pydantic import BaseModel, ConfigDict

from pruner.lp.read import LaunchpadError, NotFound, ReadClient
from pruner.lp.series import Series, SeriesTable

log = logging.getLogger(__name__)


class Publication(BaseModel):
    model_config = ConfigDict(frozen=True)

    series: str
    version: str
    pocket: str


class ArchiveIndex(BaseModel):
    """Current source versions of one package across the live series."""

    model_config = ConfigDict(frozen=True)

    distribution: str
    package: str
    publications: tuple[Publication, ...] = ()
    queried_series: tuple[str, ...] = ()
    """Series we actually asked about. Distinguishes "no publications" from
    "never looked", which matters because the former drives an Invalid proposal."""

    incomplete: bool = False
    """True if any lookup failed. Suppresses the archive-based rules rather than
    risking a decision on partial data."""

    @property
    def is_published_anywhere(self) -> bool:
        return bool(self.publications)

    def versions_in(self, series: str) -> tuple[str, ...]:
        return tuple(p.version for p in self.publications if p.series == series)

    def lowest_version(self) -> str | None:
        """Lowest version published across the queried series."""
        if not self.publications:
            return None
        return min((p.version for p in self.publications), key=_VersionKey)

    def is_older_than_everything(self, version: str) -> bool:
        """True if ``version`` predates every publication we found."""
        if not self.publications:
            return False
        return all(version_compare(version, p.version) < 0 for p in self.publications)


class _VersionKey:
    """Adapter letting ``min``/``sorted`` use dpkg version ordering."""

    __slots__ = ("value",)

    def __init__(self, value: str) -> None:
        self.value = value

    def __lt__(self, other: _VersionKey) -> bool:
        return version_compare(self.value, other.value) < 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _VersionKey):
            return NotImplemented
        return version_compare(self.value, other.value) == 0

    def __hash__(self) -> int:
        return hash(self.value)


def fetch_archive_index(
    client: ReadClient,
    table: SeriesTable,
    package: str,
    *,
    series: tuple[Series, ...] | None = None,
) -> ArchiveIndex:
    """Look up ``package`` across the live series of a distribution."""
    targets = series if series is not None else table.live
    publications: list[Publication] = []
    queried: list[str] = []
    incomplete = False

    for entry in targets:
        if not entry.self_link:
            continue
        queried.append(entry.name)
        try:
            rows = client.published_sources(
                table.distribution, package, series_link=entry.self_link
            )
        except NotFound:
            continue
        except LaunchpadError:
            log.warning(
                "archive lookup failed for %s in %s; archive rules will be skipped",
                package,
                entry.name,
            )
            incomplete = True
            continue

        publications.extend(
            Publication(
                series=entry.name,
                version=str(row["source_package_version"]),
                pocket=str(row.get("pocket") or ""),
            )
            for row in rows
            if row.get("source_package_version")
        )

    return ArchiveIndex(
        distribution=table.distribution,
        package=package,
        publications=tuple(publications),
        queried_series=tuple(queried),
        incomplete=incomplete,
    )


@lru_cache(maxsize=4096)
def compare_versions(left: str, right: str) -> int:
    """dpkg-semantics comparison, memoised. Returns -1, 0 or 1."""
    result = version_compare(left, right)
    return (result > 0) - (result < 0)
