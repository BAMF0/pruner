"""Composing the comment posted to Launchpad.

Design constraints, all learned from how bot comments are received in practice:

* **Say what happened and why, in specifics.** "Reported against Ubuntu 14.04
  (trusty), which reached end of life" is checkable. "Automated triage" is not.
* **Say how to undo it, first.** The reporter's most likely question is "how do I
  get my bug back?", so it is answered before anything else.
* **Name the rule.** If the tool is wrong, a maintainer should be able to find out
  which rule was wrong without reading the source.
* **Carry a stable marker.** Lets the tool recognise its own comments and never
  nag the same bug twice.
"""

from __future__ import annotations

from pruner.config import Config
from pruner.models import Action, BugSnapshot, Decision

_NEEDS_INFO_INTRO = (
    "Thank you for taking the time to report this bug. This is an automated "
    "message from a backlog triage pass over the {package} package."
)

_INVALID_INTRO = (
    "Thank you for reporting this bug. This is an automated message from a "
    "backlog triage pass over the {package} package."
)


def compose_comment(
    bug: BugSnapshot, decision: Decision, config: Config, *, package: str
) -> str:
    """Render the comment body for an actionable decision."""
    if decision.action is Action.NEEDS_INFO:
        body = _needs_info(bug, decision, config, package=package)
    elif decision.action is Action.INVALID:
        body = _invalid(bug, decision, config, package=package)
    elif decision.action is Action.WONT_FIX:
        body = _wont_fix(bug, decision, config, package=package)
    else:
        raise ValueError(f"no comment defined for action {decision.action}")

    return "\n\n".join(part for part in body if part.strip())


def _needs_info(
    bug: BugSnapshot, decision: Decision, config: Config, *, package: str
) -> list[str]:
    parts = [
        _NEEDS_INFO_INTRO.format(package=package),
        f"This bug is being set to Incomplete because {decision.reason}.",
    ]

    asks = _information_requests(decision)
    parts.append(
        "If you can still reproduce this on a currently supported Ubuntu release, "
        "please let us know by commenting with:"
        + "".join(f"\n  * {ask}" for ask in asks)
    )

    parts.append(
        "Setting the status back to New (or Confirmed) along with that information "
        "is all that is needed to keep this report open, and any reply will bring "
        "it back to our attention. If we do not hear anything, Launchpad will "
        "expire the report automatically after about 60 days. That is not a "
        "judgement on the original issue -- it just keeps the backlog focused on "
        "reports we are able to act on."
    )

    parts.append(_footer(decision, config))
    return parts


def _invalid(
    bug: BugSnapshot, decision: Decision, config: Config, *, package: str
) -> list[str]:
    parts = [
        _INVALID_INTRO.format(package=package),
        f"This bug is being closed as Invalid because {decision.reason}.",
    ]

    if decision.llm_reclassified:
        parts.append(
            "If this was intended as a bug report rather than a question, please "
            "reply describing what you expected to happen and what happened "
            "instead, and set the status back to New. If you are looking for help "
            "using Ubuntu, https://askubuntu.com/ and "
            "https://discourse.ubuntu.com/ are better places to get an answer."
        )
    else:
        parts.append(
            "If you believe this is wrong, please reply explaining why and set the "
            "status back to New."
        )

    parts.append(_footer(decision, config))
    return parts


def _wont_fix(
    bug: BugSnapshot, decision: Decision, config: Config, *, package: str
) -> list[str]:
    """Age escalation: closed as Won't Fix because the report is too old to verify.

    Distinct from ``_invalid`` in one essential way: Won't Fix does not claim the
    report was never a real bug, so the body must not imply it was. The reason
    (the rule's clause plus the age) is already in ``decision.reason``.
    """
    parts = [
        _INVALID_INTRO.format(package=package),
        f"This bug is being closed as Won't Fix because {decision.reason}.",
    ]

    parts.append(
        "Won't Fix rather than Invalid: this is not a judgement on the original "
        "report, which may well describe a real defect. It is closed because the "
        "report is too old to verify against anything currently shipped. If you "
        "still see this problem on a supported Ubuntu release, a fresh report "
        "against a current package is more useful than reopening this one -- but "
        "replying here and setting the status back to New also works."
    )

    parts.append(_footer(decision, config))
    return parts


def _information_requests(decision: Decision) -> list[str]:
    """What to ask for: the model's specifics when it offered any, else a default.

    The model's ``missing_info`` is usually more useful than a generic checklist
    because it is grounded in what this particular report left out.
    """
    asks: list[str] = []
    verdict = decision.verdict
    if verdict is not None and not verdict.failed:
        asks.extend(item.strip().rstrip(".") for item in verdict.missing_info if item.strip())
        asks.extend(
            item.strip().rstrip(".") for item in verdict.suggested_comment_points if item.strip()
        )

    if not asks:
        asks = [
            "the Ubuntu release you are seeing this on",
            "the version of the package (`apt policy <package>`)",
            "the exact steps that trigger the problem",
        ]

    # De-duplicate while preserving order, and keep the list short enough to read.
    seen: dict[str, None] = {}
    for ask in asks:
        seen.setdefault(ask, None)
    return list(seen)[:5]


def _footer(decision: Decision, config: Config) -> str:
    lines = ["--", config.comment.signature]
    if config.comment.include_rule_names and decision.rule_names:
        lines.append(f"Triage rule(s): {', '.join(decision.rule_names)}.")
    lines.append(config.comment.marker)
    return "\n".join(lines)
