"""Prompt construction.

Two things matter here and both are about resisting a failure mode rather than
extracting cleverness.

**The model must not rubber-stamp.** It is told the rule findings, because they
are genuinely useful context, but it is also told explicitly that its job is to
catch cases where those rules are *wrong*. Without that framing a model shown
"rule says close this" will agree essentially always, which would quietly turn a
veto into a rubber stamp and destroy the safety property the whole design rests on.

**Prompts must stay small.** Apport blocks run to kilobytes of dependency
listings; they are stripped, and comments are truncated and capped.
"""

from __future__ import annotations

from pruner.apport import strip_apport
from pruner.config import LlmConfig
from pruner.models import BugSnapshot, RuleHit

SYSTEM_PROMPT = """\
You are assisting a human maintainer with triaging the bug backlog of a single \
Ubuntu source package. You assess bug reports; you do not close them.

Your assessment is used in exactly two ways:

1. To BLOCK automated action when a report is a genuine, actionable defect. This \
is your most important function. An automated rule has already flagged this bug \
on mechanical grounds (such as the Ubuntu release it was filed against being \
end-of-life). Those rules cannot read; you can. If the report describes a real \
and clearly-described defect, say so plainly and recommend "keep", even though a \
rule flagged it.
2. To distinguish reports that are not defects at all (support questions, feature \
requests, spam) from reports that merely lack detail.

Guidelines:
- Judge the report on its own merits. Do not assume the rule finding is correct.
- Being old, or filed against an obsolete release, does NOT make a report invalid. \
Assess only what the report says.
- IMPORTANT: if the description or any comment indicates the problem occurs on a \
specific Ubuntu release, list every such release in "releases_mentioned" \
(codename or version, e.g. "noble" or "24.04"). This field is used as a safety \
check, so err on the side of listing a release you are unsure about.
- A short report can still be a good report. "Application X crashes when opening \
any file larger than 2GB" is actionable. Do not demand boilerplate.
- A request for new behaviour is a feature-request, not a defect.
- A user asking how to configure something is a support-question, not a defect.
- Set confidence honestly. Use a low value when the report is ambiguous; low \
confidence is respected and results in no action being taken on your account.
- Reply with a single JSON object matching the required schema and nothing else.
"""


def build_user_prompt(
    bug: BugSnapshot,
    hits: tuple[RuleHit, ...],
    config: LlmConfig,
    *,
    package: str,
) -> str:
    """Render the per-bug prompt body."""
    sections: list[str] = [
        f"Source package: {package}",
        f"Bug #{bug.id}: {bug.title}",
    ]

    facts: list[str] = []
    if bug.date_created:
        facts.append(f"reported {bug.date_created.date()}")
    if bug.date_last_updated:
        facts.append(f"last activity {bug.date_last_updated.date()}")
    if bug.apport.distro_release:
        facts.append(f"filed against Ubuntu {bug.apport.distro_release}")
    if bug.apport.version:
        facts.append(f"package version {bug.apport.version}")
    facts.append(f"{bug.message_count} message(s)")
    facts.append(f"{bug.users_affected_count} user(s) marked affected")
    if bug.tags:
        facts.append(f"tags: {', '.join(bug.tags)}")
    sections.append("Metadata: " + "; ".join(facts) + ".")

    description = strip_apport(bug.description)
    if not description:
        description = "(the reporter wrote no description)"
    sections.append(
        "--- Description (machine-generated apport metadata removed) ---\n"
        + _truncate(description, config.max_description_chars)
    )

    if bug.comment_texts:
        shown = bug.comment_texts[: config.max_comments]
        rendered = "\n\n".join(
            f"[comment {index}] {_truncate(strip_apport(text) or text, config.max_comment_chars)}"
            for index, text in enumerate(shown, start=1)
        )
        omitted = len(bug.comment_texts) - len(shown)
        if omitted > 0:
            rendered += f"\n\n({omitted} further comment(s) not shown)"
        sections.append("--- Comments ---\n" + rendered)
    else:
        sections.append("--- Comments ---\n(nobody ever followed up on this report)")

    if hits:
        findings = "\n".join(f"- {hit.rule}: {hit.reason}" for hit in hits)
        sections.append(
            "--- Automated rule findings (mechanical; may well be wrong) ---\n"
            + findings
            + "\n\nAssess the report independently of the above."
        )

    sections.append(
        "Now produce your JSON assessment. Recommend 'keep' if this is a genuine, "
        "actionable defect."
    )
    return "\n\n".join(sections)


def _truncate(text: str, limit: int) -> str:
    collapsed = text.strip()
    if len(collapsed) <= limit:
        return collapsed
    return collapsed[:limit].rstrip() + f"\n[... truncated, {len(collapsed) - limit} more chars]"
