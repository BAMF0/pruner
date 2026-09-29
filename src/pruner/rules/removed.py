"""``removed_from_archive``: the package is gone from the distribution.

If ``getPublishedSources`` returns no ``Published`` source for the package in any
live series, then bugs filed against it in this distribution cannot be fixed --
there is nothing left to fix. This is the one rule that proposes ``invalid``
outright, because the conclusion follows from archive state rather than from a
judgement about the report's content.

Guarded hard against false positives, since "package not found" is exactly what a
typo or a failed API call also looks like:

* the lookup must have covered at least one series,
* no lookup may have failed (``ArchiveIndex.incomplete``),
* and the caller must have confirmed the package is genuinely absent rather than
  merely unqueried.
"""

from __future__ import annotations

from pruner.models import Action, BugSnapshot, RuleClaim, RuleHit
from pruner.rules.base import RuleContext, register


@register("removed_from_archive")
def removed_from_archive(bug: BugSnapshot, context: RuleContext) -> RuleHit | None:
    archive = context.archive
    if archive is None:
        return None

    # Partial data must never produce a close proposal.
    if archive.incomplete or not archive.queried_series:
        return None
    if archive.is_published_anywhere:
        return None

    return RuleHit(
        rule="removed_from_archive",
        action=Action.INVALID,
        claim=RuleClaim.EXISTENCE,
        reason=(
            f"the '{archive.package}' source package is no longer published in any "
            f"supported {context.distribution.title()} release "
            f"(checked: {', '.join(archive.queried_series)}), so this bug can no "
            "longer be acted on in the distribution"
        ),
        evidence={
            "package": archive.package,
            "series_checked": ", ".join(archive.queried_series),
        },
    )
