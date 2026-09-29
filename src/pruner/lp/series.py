"""Distribution series metadata: which releases we will actually fix bugs on.

The EOL rules hinge entirely on this, and the subtlety is worth stating plainly
because getting it wrong is how you either close live bugs or close nothing.

**Launchpad's ``status`` is not the answer on its own.** Verified against live
data, Launchpad reports trusty (14.04), xenial (16.04), bionic (18.04) and focal
(20.04) as ``Supported`` with ``supported: true``, because they remain under
Ubuntu Pro / ESM. It exposes no end-of-life date. If "not Obsolete" were taken to
mean "supported", the EOL rules would decline to fire on precisely the decade-old
bugs a backlog sweep exists to address.

**ESM is not bug-fixing support.** It delivers security updates for a subset of
packages to subscribers. Nobody is going to fix a general functional bug in xterm
on trusty. So for triage purposes an ESM-only release is end-of-life.

Rather than bury that judgement in code, it is an explicit
:class:`SupportPolicy`, recorded on the cached table and therefore in the audit
trail:

``STANDARD`` (default)
    Live if Launchpad does not say ``Obsolete`` **and** standard support (the
    ``eol`` column of ``distro-info``) has not ended. Matches
    ``ubuntu-distro-info --supported``.

``LAUNCHPAD``
    Trust Launchpad alone. ESM releases count as supported. Maximally
    conservative and correspondingly low-reach.

``EXPLICIT``
    The operator names the live series. For teams with their own policy, or for
    distributions with no ``distro-info`` data.

Whenever the supporting data is missing, the code degrades towards treating
releases as *live*, which means pruning less.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime
from typing import Any, Self

from pydantic import BaseModel, ConfigDict

from pruner.distroinfo import load_eol_dates
from pruner.lp.read import ReadClient
from pruner.models import SeriesStatus, SupportPolicy

log = logging.getLogger(__name__)


class Series(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    """Lowercase codename, e.g. ``"noble"``."""

    version: str | None = None
    """Numeric release, e.g. ``"24.04"``."""

    status: SeriesStatus = SeriesStatus.OBSOLETE
    """Launchpad's own status. Counts ESM releases as ``Supported``."""

    supported: bool = False
    """Launchpad's own flag. Also true for ESM releases."""

    active: bool = False
    self_link: str = ""

    standard_eol: date | None = None
    """End of standard support, from ``distro-info``. ``None`` if unknown."""

    triage_live: bool = True
    """Resolved answer to "would anyone fix a bug on this release?".

    Computed once by :meth:`SeriesTable.from_api` under the configured policy.
    Rules read this, never the raw Launchpad fields, so the policy is applied in
    exactly one place.
    """

    @property
    def is_obsolete(self) -> bool:
        """End-of-life *for triage purposes*."""
        return not self.triage_live

    @property
    def is_devel(self) -> bool:
        return self.status in (
            SeriesStatus.DEVELOPMENT,
            SeriesStatus.FROZEN,
            SeriesStatus.EXPERIMENTAL,
        )

    @property
    def esm_only(self) -> bool:
        """Launchpad calls it supported, but standard support has ended."""
        return self.supported and not self.triage_live

    @property
    def label(self) -> str:
        return f"{self.name} ({self.version})" if self.version else self.name


class SeriesTable(BaseModel):
    """Lookup table over a distribution's series, with the support policy applied."""

    model_config = ConfigDict(frozen=True)

    distribution: str
    series: tuple[Series, ...]
    policy: SupportPolicy = SupportPolicy.STANDARD
    evaluated_on: date | None = None
    """The date the policy was evaluated against, so a cached table is auditable."""

    had_eol_data: bool = False
    """False if ``distro-info`` was unavailable and we fell back to Launchpad."""

    # -- construction ------------------------------------------------------

    @classmethod
    def from_api(
        cls,
        distribution: str,
        entries: list[dict[str, Any]],
        *,
        policy: SupportPolicy = SupportPolicy.LAUNCHPAD,
        eol_dates: dict[str, date] | None = None,
        live_series: tuple[str, ...] = (),
        today: date | None = None,
    ) -> Self:
        """Build a table, resolving ``triage_live`` for each series.

        Defaults to :attr:`SupportPolicy.LAUNCHPAD` so that a bare call has no
        hidden dependency on host data; :func:`load_series_table` applies the
        configured policy.
        """
        moment = today or datetime.now(UTC).date()
        eol = {k.lower(): v for k, v in (eol_dates or {}).items()}
        explicit = {s.lower() for s in live_series}

        built: list[Series] = []
        for raw in entries:
            base = _parse(raw)
            series = base.model_copy(
                update={
                    "standard_eol": eol.get(base.name),
                    "triage_live": _resolve_live(
                        base, policy, eol.get(base.name), explicit, moment
                    ),
                }
            )
            built.append(series)

        return cls(
            distribution=distribution.lower(),
            series=tuple(built),
            policy=policy,
            evaluated_on=moment,
            had_eol_data=bool(eol),
        )

    # -- lookups -----------------------------------------------------------

    def get(self, name: str) -> Series | None:
        target = name.lower()
        return next((s for s in self.series if s.name == target), None)

    def by_version(self, version: str) -> Series | None:
        """Resolve a numeric release (``"14.04"``) to its series (``trusty``)."""
        return next((s for s in self.series if s.version == version), None)

    def resolve(self, token: str) -> Series | None:
        """Resolve either a codename or a numeric version."""
        return self.get(token) or self.by_version(token)

    @property
    def obsolete(self) -> tuple[Series, ...]:
        """Series that are end-of-life for triage purposes."""
        return tuple(s for s in self.series if s.is_obsolete)

    @property
    def live(self) -> tuple[Series, ...]:
        """Series we would still fix a bug on."""
        return tuple(s for s in self.series if s.triage_live)

    @property
    def supported(self) -> tuple[Series, ...]:
        """Live series excluding anything still in development."""
        return tuple(s for s in self.live if not s.is_devel)

    @property
    def esm_only(self) -> tuple[Series, ...]:
        """Series Launchpad calls supported but which are past standard support."""
        return tuple(s for s in self.series if s.esm_only)

    @property
    def devel(self) -> Series | None:
        return next((s for s in self.series if s.is_devel), None)

    @property
    def live_names(self) -> frozenset[str]:
        return frozenset(s.name for s in self.live)

    @property
    def obsolete_names(self) -> frozenset[str]:
        return frozenset(s.name for s in self.obsolete)

    def is_obsolete(self, name: str) -> bool:
        series = self.get(name)
        return series is not None and series.is_obsolete

    def oldest_supported(self) -> Series | None:
        """Oldest live, non-development series.

        Used by ``likely_fixed``: if the reported version predates even this, the
        report is about code nobody ships any more.
        """
        candidates = [s for s in self.supported if s.version]
        if not candidates:
            return None
        return min(candidates, key=lambda s: _version_key(s.version or ""))

    def series_tags(self, tags: tuple[str, ...]) -> tuple[Series, ...]:
        """Tags that name a real series of this distribution.

        Ubuntu triagers tag bugs with the codename of the affected release, so
        this is often the only machine-readable release signal on a non-apport bug.
        """
        known = {s.name: s for s in self.series}
        return tuple(known[t.lower()] for t in tags if t.lower() in known)

    def summary(self) -> str:
        parts = [
            f"policy={self.policy}",
            f"{len(self.live)} live",
            f"{len(self.obsolete)} end-of-life",
        ]
        if self.esm_only:
            parts.append(
                "past standard support: " + ", ".join(s.name for s in self.esm_only)
            )
        if self.policy is SupportPolicy.STANDARD and not self.had_eol_data:
            parts.append("WARNING: no distro-info data, fell back to Launchpad status")
        return "; ".join(parts)


def _parse(raw: dict[str, Any]) -> Series:
    status = SeriesStatus.OBSOLETE
    if isinstance(raw.get("status"), str):
        try:
            status = SeriesStatus(raw["status"])
        except ValueError:
            status = SeriesStatus.OBSOLETE
    return Series(
        name=str(raw["name"]).lower(),
        version=str(raw["version"]) if raw.get("version") else None,
        status=status,
        supported=bool(raw.get("supported", False)),
        active=bool(raw.get("active", False)),
        self_link=str(raw.get("self_link") or ""),
    )


def _resolve_live(
    series: Series,
    policy: SupportPolicy,
    eol: date | None,
    explicit: set[str],
    today: date,
) -> bool:
    """Apply the support policy to one series."""
    if policy is SupportPolicy.EXPLICIT:
        return series.name in explicit

    # Launchpad calling it Obsolete is decisive under every policy: it is a strict
    # subset of "end of life" and no policy should resurrect such a series.
    if series.status.is_obsolete:
        return False

    if policy is SupportPolicy.LAUNCHPAD:
        return True

    # STANDARD: also require that standard support has not ended. A series still
    # in development has no EOL date yet and is obviously live.
    if series.is_devel:
        return True
    if eol is None:
        # No data: fall back to Launchpad, which errs towards "live".
        return True
    return eol >= today


def load_series_table(
    client: ReadClient,
    distribution: str,
    *,
    policy: SupportPolicy = SupportPolicy.STANDARD,
    live_series: tuple[str, ...] = (),
    today: date | None = None,
) -> SeriesTable:
    """Fetch and resolve a distribution's series under the configured policy."""
    eol_dates = (
        load_eol_dates(distribution) if policy is SupportPolicy.STANDARD else {}
    )
    return SeriesTable.from_api(
        distribution,
        client.series(distribution),
        policy=policy,
        eol_dates=eol_dates,
        live_series=live_series,
        today=today,
    )


def _version_key(version: str) -> tuple[int, ...]:
    """Sort key for ``"24.04"``-style versions; unparseable parts sort as 0."""
    parts: list[int] = []
    for chunk in version.split("."):
        digits = "".join(c for c in chunk if c.isdigit())
        parts.append(int(digits) if digits else 0)
    return tuple(parts)
