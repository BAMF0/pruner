"""``empty_report``: there is not enough here to act on.

Measures the *human-written* part of the description, with the apport block
stripped, because an otherwise contentless report can still carry several
kilobytes of machine metadata.

This is the rule that leans hardest on the LLM. "Short" and "unactionable" are not
the same thing -- "gedit segfaults when opening any file over 2GB" is a perfectly
good bug report in 47 characters. So the rule requires several corroborating
signals (nobody ever followed up, no attachments, nobody else affected) and the
LLM retains its veto for the terse-but-valid case.
"""

from __future__ import annotations

from pruner.apport import prose_length
from pruner.models import Action, BugSnapshot, RuleClaim, RuleHit
from pruner.rules.base import RuleContext, register


@register("empty_report")
def empty_report(bug: BugSnapshot, context: RuleContext) -> RuleHit | None:
    threshold = context.config.rules.min_desc_chars
    length = prose_length(bug.description)
    if length >= threshold:
        return None

    # Corroborating signals: a discussion, an attachment, or other affected users
    # all indicate there is something real here despite the thin description.
    if bug.message_count > 1:
        return None
    if bug.attachment_count > 0:
        return None
    if bug.users_affected_count > 1:
        return None

    return RuleHit(
        rule="empty_report",
        action=Action.NEEDS_INFO,
        claim=RuleClaim.QUALITY,
        reason=(
            f"the report contains only {length} characters of description "
            f"(threshold {threshold}), has no attachments and no follow-up, so "
            "there is not enough information to reproduce or triage it"
        ),
        evidence={
            "prose_chars": str(length),
            "threshold": str(threshold),
            "message_count": str(bug.message_count),
        },
    )
