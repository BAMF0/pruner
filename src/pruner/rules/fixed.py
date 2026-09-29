"""``likely_fixed``: the report concerns code nobody ships any more.

Two independent signals:

* **Version drift.** The version in the apport ``Package:`` line is older than
  every source publication in the live series. Comparison is dpkg-exact via
  ``python-debian``, so epochs and ``~`` orderings are handled correctly
  (e.g. ``2:7.4.052-1ubuntu3`` < ``2:9.1.0016-1ubuntu7``).
* **Upstream resolution.** A linked upstream bug watch reports a resolved status
  while the distribution task is still open.

This proposes ``needs-info``, never a close: "the version you reported is ancient"
is a reason to ask for re-verification, not proof of a fix.
"""

from __future__ import annotations

from pruner.lp.archive import compare_versions
from pruner.models import Action, BugSnapshot, RuleClaim, RuleHit
from pruner.rules.base import RuleContext, register


@register("likely_fixed")
def likely_fixed(bug: BugSnapshot, context: RuleContext) -> RuleHit | None:
    if hit := _upstream_resolved(bug, context):
        return hit
    return _version_superseded(bug, context)


def _upstream_resolved(bug: BugSnapshot, context: RuleContext) -> RuleHit | None:
    wanted = {s.casefold() for s in context.config.rules.fixed_upstream_statuses}
    for status in bug.remote_bug_statuses:
        normalised = status.strip().casefold()
        if not normalised:
            continue
        if normalised in wanted or any(w in normalised for w in wanted):
            return RuleHit(
                rule="likely_fixed",
                action=Action.NEEDS_INFO,
                claim=RuleClaim.LIFECYCLE,
                reason=(
                    f"the linked upstream bug is reported as '{status.strip()}', "
                    "so this may already be fixed"
                ),
                evidence={"signal": "upstream_status", "remote_status": status.strip()},
            )
    return None


def _version_superseded(bug: BugSnapshot, context: RuleContext) -> RuleHit | None:
    archive = context.archive
    reported = bug.apport.version
    if archive is None or archive.incomplete or not reported:
        return None
    if not archive.publications:
        # Nothing to compare against; ``removed_from_archive`` handles that case.
        return None

    if not archive.is_older_than_everything(reported):
        return None

    lowest = archive.lowest_version()
    if lowest is None or compare_versions(reported, lowest) >= 0:
        return None

    return RuleHit(
        rule="likely_fixed",
        action=Action.NEEDS_INFO,
        claim=RuleClaim.LIFECYCLE,
        reason=(
            f"it was reported against {bug.apport.package or context.package} "
            f"{reported}, but every supported release now ships {lowest} or newer, "
            "so it may already be fixed"
        ),
        evidence={
            "signal": "version_drift",
            "reported_version": reported,
            "lowest_published": lowest,
            "series_checked": ", ".join(archive.queried_series),
        },
    )
