"""EOL rules: bugs that only ever concerned a release which no longer exists.

Three rules, differing only in where the release evidence comes from, in
descending order of reliability:

1. ``eol_series_tasks``  -- explicit series-nominated bug tasks.
2. ``eol_series_tag``    -- triager-applied series tags.
3. ``eol_apport_release``-- ``DistroRelease:`` in the apport block.

All three propose ``needs-info`` rather than a close. An EOL bug is not
*necessarily* invalid -- the defect may well still exist -- it is just unverifiable
against anything we ship. Asking, and letting Launchpad's janitor expire the
silence, is both accurate and reversible.

Every rule independently re-checks :func:`live_release_evidence`, duplicating the
``affects_live_release`` exclusion on purpose: a rule must be safe in isolation,
including if someone disables that exclusion in their config.
"""

from __future__ import annotations

from pruner.models import Action, BugSnapshot, RuleClaim, RuleHit
from pruner.rules.base import RuleContext, register
from pruner.rules.signals import (
    live_release_evidence,
    obsolete_release_evidence,
    open_target_tasks,
    series_phrase,
    target_tasks,
)


def _blocked(bug: BugSnapshot, context: RuleContext) -> bool:
    return bool(live_release_evidence(bug, context.series, context.package))


@register("eol_series_tasks")
def eol_series_tasks(bug: BugSnapshot, context: RuleContext) -> RuleHit | None:
    """Every open task for this package is nominated to an obsolete series.

    The strongest EOL signal available: somebody explicitly recorded which releases
    this bug applies to, and all of them are gone.
    """
    table = context.series
    open_tasks = open_target_tasks(bug, table.distribution, context.package)
    if not open_tasks:
        return None

    series_tasks = [t for t in open_tasks if t.series]
    # Require that *every* open task is series-nominated. A bare "pkg (Ubuntu)"
    # task means "current development release" and is not EOL evidence.
    if not series_tasks or len(series_tasks) != len(open_tasks):
        return None

    if any(not table.is_obsolete(t.series or "") for t in series_tasks):
        return None
    if _blocked(bug, context):
        return None

    names = sorted({t.series or "" for t in series_tasks})
    resolved = [found for n in names if (found := table.get(n))]
    return RuleHit(
        rule="eol_series_tasks",
        action=Action.NEEDS_INFO,
        claim=RuleClaim.LIFECYCLE,
        reason=(
            "every open task for this package targets "
            f"{series_phrase(resolved)}, and nothing indicates it affects a "
            "release we still support"
        ),
        evidence={"series": ", ".join(names), "task_count": str(len(series_tasks))},
    )


@register("eol_series_tag")
def eol_series_tag(bug: BugSnapshot, context: RuleContext) -> RuleHit | None:
    """The bug carries series tags and all of them name obsolete releases."""
    table = context.series
    tagged = table.series_tags(bug.tags)
    if not tagged:
        return None
    if any(not s.is_obsolete for s in tagged):
        return None
    if _blocked(bug, context):
        return None

    # If a series task exists, ``eol_series_tasks`` is the better-evidenced rule
    # and this one would merely duplicate it.
    if any(t.series for t in target_tasks(bug, table.distribution, context.package)):
        return None

    return RuleHit(
        rule="eol_series_tag",
        action=Action.NEEDS_INFO,
        claim=RuleClaim.LIFECYCLE,
        reason=(
            f"it is tagged only for {series_phrase(tagged)}, and nothing "
            "indicates it affects a release we still support"
        ),
        evidence={"tags": ", ".join(sorted(s.name for s in tagged))},
    )


@register("eol_apport_release")
def eol_apport_release(bug: BugSnapshot, context: RuleContext) -> RuleHit | None:
    """Apport recorded the bug against a release that is now obsolete.

    The weakest of the three signals and the most common by far on old backlogs,
    since most reports carry apport metadata and nothing else.
    """
    table = context.series
    release = bug.apport.distro_release
    if not release:
        return None

    series = table.by_version(release)
    if series is None or not series.is_obsolete:
        return None
    if _blocked(bug, context):
        return None

    # Defer to the better-evidenced rules when they would also fire.
    if table.series_tags(bug.tags):
        return None
    if any(t.series for t in target_tasks(bug, table.distribution, context.package)):
        return None

    return RuleHit(
        rule="eol_apport_release",
        action=Action.NEEDS_INFO,
        claim=RuleClaim.LIFECYCLE,
        reason=(
            f"it was reported against {series_phrase([series])}, and no later "
            "comment mentions a release we still support"
        ),
        evidence={"apport_release": release, "series": series.name},
    )


@register("eol_obsolete_only")
def eol_obsolete_only(bug: BugSnapshot, context: RuleContext) -> RuleHit | None:
    """Catch-all: several independent obsolete-release signals agree and nothing
    points at a live release.

    Not enabled by default -- the three specific rules above cover the same ground
    with clearer reporting. Available for operators who prefer one broad rule.
    """
    obsolete = obsolete_release_evidence(bug, context.series, context.package)
    if len(obsolete) < 1 or _blocked(bug, context):
        return None
    return RuleHit(
        rule="eol_obsolete_only",
        action=Action.NEEDS_INFO,
        claim=RuleClaim.LIFECYCLE,
        reason=(
            f"it is only associated with {series_phrase(obsolete)}, and nothing "
            "indicates it affects a release we still support"
        ),
        evidence={"series": ", ".join(sorted(s.name for s in obsolete))},
    )
