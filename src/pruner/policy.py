"""Fusing deterministic rule findings with the advisory LLM verdict.

The invariant this module exists to enforce:

    **No LLM output, of any kind, can cause a bug to be actioned that the
    deterministic rules did not already make eligible.**

The LLM has exactly three powers, all of which either reduce or redirect action:

1. **Veto** -- turn a proposed action into ``keep``.
2. **Live-release veto** -- as above, triggered by the model noticing a supported
   release mentioned in prose that the text-matching exclusion missed.
3. **Reclassification** -- change an already-eligible ``needs-info`` into
   ``invalid`` when the report is not a bug at all. This changes the *kind* of an
   action the rules already authorised; it cannot create one.

Veto power is scoped by :class:`~pruner.models.RuleClaim`, which matters more than
it might appear. See that class for the reasoning; briefly, letting "this is a
genuine reproducible defect" veto a *lifecycle* rule would mean the best-written
end-of-life reports are precisely the ones never pruned.
"""

from __future__ import annotations

from pruner.config import Config
from pruner.lp.series import SeriesTable
from pruner.models import (
    Action,
    BugSnapshot,
    Decision,
    Exclusion,
    IsABug,
    LlmVerdict,
    RuleClaim,
    RuleHit,
)

#: Action precedence when several rules fire.
#:
#: ``invalid`` outranks ``needs-info`` because the only rule that proposes it,
#: ``removed_from_archive``, makes a stronger claim: if the package is gone from
#: the archive there is no point asking the reporter to re-verify against it.
_ACTION_PRECEDENCE: dict[Action, int] = {
    Action.INVALID: 3,
    Action.NEEDS_INFO: 2,
    Action.ESCALATE: 1,
    Action.KEEP: 0,
}


def choose_rule_action(hits: tuple[RuleHit, ...]) -> tuple[Action, RuleHit | None]:
    """Highest-precedence action among the rules that fired, and its hit."""
    if not hits:
        return Action.KEEP, None
    best = max(hits, key=lambda h: _ACTION_PRECEDENCE.get(h.action, 0))
    return best.action, best


def decide(
    bug: BugSnapshot,
    *,
    hits: tuple[RuleHit, ...],
    exclusions: tuple[Exclusion, ...],
    verdict: LlmVerdict | None,
    config: Config,
    series: SeriesTable,
) -> Decision:
    """Produce the final decision for one bug."""
    rule_action, primary = choose_rule_action(hits)

    def outcome(
        action: Action,
        reason: str,
        branch: str,
        *,
        vetoed: bool = False,
        reclassified: bool = False,
    ) -> Decision:
        """Build a Decision with the shared provenance fields already filled in.

        A typed builder rather than a ``**dict`` spread so that mypy actually
        checks every construction on this path.
        """
        return Decision(
            bug_id=bug.id,
            action=action,
            reason=reason,
            exclusions=exclusions,
            rule_hits=hits,
            verdict=verdict,
            rule_action=rule_action,
            llm_vetoed=vetoed,
            llm_reclassified=reclassified,
            policy_branch=branch,
        )

    # 1. Hard exclusions win over everything, including the rules themselves.
    if exclusions:
        return outcome(
            Action.KEEP, f"protected: {exclusions[0].reason}", "excluded"
        )

    # 2. Eligibility comes only from rules.
    if rule_action is Action.KEEP or primary is None:
        return outcome(Action.KEEP, "no prune rule matched", "no_rule_hit")

    # 3. A failed or absent verdict is silence, never consent. The rules stand.
    if verdict is None or verdict.failed:
        note = " (no LLM assessment available)" if verdict else ""
        return outcome(rule_action, primary.reason + note, "rules_only")

    # 4. The model spotted a live release in prose that our matching missed.
    if live := _live_releases(verdict, series):
        return outcome(
            Action.KEEP,
            "the report or its comments reference the still-supported release "
            f"{', '.join(live)}, so it is not an end-of-life bug",
            "llm_live_release_veto",
            vetoed=True,
        )

    # 5. Scoped veto: only against rules whose claim the model can contradict.
    if _can_veto(primary.claim) and (why := _veto_reason(verdict, config)):
        return outcome(Action.KEEP, why, "llm_veto", vetoed=True)

    # 6. Reclassification, strictly within the eligibility the rules granted.
    if rule_action is Action.NEEDS_INFO and _should_reclassify(verdict, config):
        return outcome(
            Action.INVALID,
            f"not a bug report ({verdict.bug_kind}): {verdict.rationale.strip()} "
            f"[rule: {primary.rule}]",
            "llm_reclassified",
            reclassified=True,
        )

    # 7. Rules stand, with the model having had its say and not objected.
    return outcome(rule_action, primary.reason, "rules_with_llm_concurrence")


def _can_veto(claim: RuleClaim) -> bool:
    """Whether a general "this is a real, actionable bug" veto applies.

    Only quality claims are contradicted by that statement. A lifecycle claim
    ("the release is dead") and an existence claim ("the package is gone") are
    facts about the distribution, not about the report, so the model has no
    standing to overrule them on the basis of report quality.
    """
    return claim is RuleClaim.QUALITY


def _veto_reason(verdict: LlmVerdict, config: Config) -> str | None:
    threshold = config.llm.veto_threshold
    if verdict.confidence < threshold:
        return None

    if verdict.is_actually_a_bug is IsABug.YES and verdict.reproducible_from_report:
        return (
            "assessed as a genuine, reproducible defect despite being flagged: "
            + (verdict.rationale.strip() or "no rationale given")
        )
    if verdict.recommendation is Action.KEEP:
        return "assessed as worth keeping: " + (
            verdict.rationale.strip() or "no rationale given"
        )
    return None


def _should_reclassify(verdict: LlmVerdict, config: Config) -> bool:
    if not config.llm.allow_llm_reclassify_to_invalid:
        return False
    if verdict.is_actually_a_bug is not IsABug.NO:
        return False
    if verdict.bug_kind not in config.llm.reclassify_kinds:
        return False
    return verdict.confidence >= config.llm.reclassify_threshold


def _live_releases(verdict: LlmVerdict, series: SeriesTable) -> tuple[str, ...]:
    """Releases the model reported that resolve to a series still alive.

    Applied irrespective of confidence: a hallucinated release name here causes a
    bug to be spared, which is the harmless direction.
    """
    found: list[str] = []
    for token in verdict.releases_mentioned:
        cleaned = token.strip().lower().removeprefix("ubuntu").strip()
        if not cleaned:
            continue
        resolved = series.resolve(cleaned)
        if resolved is not None and not resolved.is_obsolete:
            found.append(resolved.label)
    return tuple(dict.fromkeys(found))
