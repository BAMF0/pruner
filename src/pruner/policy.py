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

Separately from the LLM, the rules' own proposal can be hardened one step:
**age escalation** turns an already-eligible ``needs-info`` into ``wont-fix``
when the bug is older than ``[age].wont_fix_after_days``. Like the rules this is
deterministic, so it sits on the rule side of the invariant -- the LLM neither
enables nor disables it -- and like reclassification it can only ever *harden*
an action, never create one. Ordering against the model's powers is deliberate:

* An LLM **veto still wins over age** (branches 4 and 5 are checked first): if
  the model is confident the bug is live-release or genuinely reproducible,
  old age does not close it.
* LLM **reclassification takes precedence on the reason** (branch 6 checked
  before age is applied in branch 7): "not a bug report (support-question)"
  says more than "old" does.

Age escalation is scoped by claim exactly like the veto, defaulting to
``lifecycle`` only: "old AND on a dead release" is the airtight case. A
``quality`` hit on an old bug still just asks for information.
"""

from __future__ import annotations

from datetime import datetime

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
#:
#: ``wont-fix`` is unreachable today -- no rule proposes it; :mod:`pruner.policy`
#: adds it as a rule-side escalation of an existing ``needs-info``. It is listed
#: anyway so that a future ``wont-fix`` rule does not silently fall to the
#: ``.get(…, 0)`` default and rank as ``keep``.
_ACTION_PRECEDENCE: dict[Action, int] = {
    Action.INVALID: 4,
    Action.WONT_FIX: 3,
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
    now: datetime,
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
        escalated: bool = False,
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
            age_escalated=escalated,
            policy_branch=branch,
        )

    def stand(branch: str, note: str = "") -> Decision:
        """The rules' proposal stands. Applies rule-side age escalation.

        Used by the two branches where the LLM has had nothing (or nothing
        conclusive) to say. Escalation belongs here, not in a rule, because it
        never *creates* eligibility -- it only hardens an existing ``needs-info``
        -- and not in ``actions.py``, because the approvals file a human reviewed
        must say what will actually happen.
        """
        assert primary is not None
        age = (
            _escalation_age(bug, primary, config, now)
            if rule_action is Action.NEEDS_INFO
            else None
        )
        if age is not None:
            return outcome(
                Action.WONT_FIX,
                f"{primary.reason}, and the report is {age / 365.25:.0f} years old "
                "with no resolution",
                "age_escalated",
                escalated=True,
            )
        return outcome(rule_action, primary.reason + note, branch)

    # 1. Hard exclusions win over everything, including the rules themselves.
    if exclusions:
        return outcome(
            Action.KEEP, f"protected: {exclusions[0].reason}", "excluded"
        )

    # 2. Eligibility comes only from rules. Age escalation happens below, on the
    #    branches where the rules' proposal stands -- never here, because no rule
    #    hit means nothing to escalate.
    if rule_action is Action.KEEP or primary is None:
        return outcome(Action.KEEP, "no prune rule matched", "no_rule_hit")

    # 3. A failed or absent verdict is silence, never consent. The rules stand.
    if verdict is None or verdict.failed:
        note = " (no LLM assessment available)" if verdict else ""
        return stand("rules_only", note)

    # 4. The model spotted a live release in prose that our matching missed.
    #    Outranks age escalation: a live release is a live release.
    if live := _live_releases(verdict, series):
        return outcome(
            Action.KEEP,
            "the report or its comments reference the still-supported release "
            f"{', '.join(live)}, so it is not an end-of-life bug",
            "llm_live_release_veto",
            vetoed=True,
        )

    # 5. Scoped veto: only against rules whose claim the model can contradict.
    #    Also outranks age escalation.
    if _can_veto(primary.claim) and (why := _veto_reason(verdict, config)):
        return outcome(Action.KEEP, why, "llm_veto", vetoed=True)

    # 6. Reclassification, strictly within the eligibility the rules granted.
    #    Checked before age escalation because its reason is more informative:
    #    "not a bug report" says more than "old" does.
    if rule_action is Action.NEEDS_INFO and _should_reclassify(verdict, config):
        return outcome(
            Action.INVALID,
            f"not a bug report ({verdict.bug_kind}): {verdict.rationale.strip()} "
            f"[rule: {primary.rule}]",
            "llm_reclassified",
            reclassified=True,
        )

    # 7. Rules stand, with the model having had its say and not objected.
    return stand("rules_with_llm_concurrence")


def _escalation_age(
    bug: BugSnapshot, primary: RuleHit, config: Config, now: datetime
) -> float | None:
    """Age in days if a rule-eligible ``needs-info`` should be ``wont-fix``, else None.

    Fails closed at every step: disabled config, an out-of-scope claim, and above
    all an unknown ``date_created`` each mean no escalation. A bug whose age we
    cannot establish is never old enough to close.
    """
    threshold = config.age.wont_fix_after_days
    if threshold <= 0:
        return None
    if primary.claim not in config.age.claims:
        return None
    age = bug.age_days(now=now)
    if age is None or age < threshold:
        return None
    return age


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
