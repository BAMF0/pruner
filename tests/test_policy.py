"""Policy fusion: the safety invariant.

The central property under test:

    No LLM output, of any kind, can cause a bug to be actioned that the
    deterministic rules did not already make eligible.

:class:`TestLlmCannotCreateAction` asserts this exhaustively across the full
cross-product of model outputs, which is the test that would catch a future
refactor quietly handing the model more authority.
"""

from __future__ import annotations

import itertools

import pytest

from pruner.config import Config
from pruner.lp.series import SeriesTable
from pruner.models import (
    Action,
    BugKind,
    BugSnapshot,
    Exclusion,
    IsABug,
    LlmVerdict,
    RuleClaim,
    RuleHit,
)
from pruner.policy import choose_rule_action, decide
from tests.conftest import NOW, make_bug


def hit(
    rule: str = "eol_apport_release",
    action: Action = Action.NEEDS_INFO,
    claim: RuleClaim = RuleClaim.LIFECYCLE,
) -> RuleHit:
    return RuleHit(rule=rule, action=action, claim=claim, reason=f"{rule} fired")


def verdict(**kwargs) -> LlmVerdict:
    kwargs.setdefault("model", "test:model")
    return LlmVerdict(**kwargs)


def run(
    config: Config,
    series: SeriesTable,
    *,
    hits: tuple[RuleHit, ...] = (),
    exclusions: tuple[Exclusion, ...] = (),
    llm: LlmVerdict | None = None,
    bug: BugSnapshot | None = None,
):
    return decide(
        bug or make_bug(),
        hits=hits,
        exclusions=exclusions,
        verdict=llm,
        config=config,
        series=series,
        now=NOW,
    )


def young_bug() -> BugSnapshot:
    """A bug old enough to be quiet, far too young for age escalation.

    Tests whose premise is the LLM's behaviour (veto, reclassification) use this
    so the outcome is not also shaped by ``[age].wont_fix_after_days``; age
    escalation has its own dedicated test class.
    """
    return make_bug(quiet_days=400)


# ---------------------------------------------------------------------------
# The invariant
# ---------------------------------------------------------------------------


class TestLlmCannotCreateAction:
    """No combination of model output may action a bug the rules did not flag."""

    @pytest.mark.parametrize(
        ("is_bug", "kind", "recommendation", "confidence", "quiet_days"),
        list(
            itertools.product(
                list(IsABug),
                list(BugKind),
                list(Action),
                [0.0, 0.5, 0.9, 1.0],
                [10, 3000, 30000],
            )
        ),
    )
    def test_no_rule_hit_always_keeps(
        self,
        config: Config,
        series: SeriesTable,
        is_bug: IsABug,
        kind: BugKind,
        recommendation: Action,
        confidence: float,
        quiet_days: int,
    ) -> None:
        """Neither model output nor extreme age can create eligibility."""
        decision = run(
            config,
            series,
            hits=(),
            bug=make_bug(quiet_days=quiet_days),
            llm=verdict(
                is_actually_a_bug=is_bug,
                bug_kind=kind,
                recommendation=recommendation,
                confidence=confidence,
                needs_more_info=True,
                reproducible_from_report=False,
            ),
        )
        assert decision.action is Action.KEEP
        assert decision.policy_branch == "no_rule_hit"

    @pytest.mark.parametrize("recommendation", list(Action))
    def test_exclusion_always_wins(
        self, config: Config, series: SeriesTable, recommendation: Action
    ) -> None:
        """Including against age escalation: the bug here is 3000 days old."""
        decision = run(
            config,
            series,
            hits=(hit(),),
            exclusions=(Exclusion(rule="security", reason="flagged security_related"),),
            llm=verdict(recommendation=recommendation, confidence=1.0),
        )
        assert decision.action is Action.KEEP
        assert decision.policy_branch == "excluded"
        assert "security_related" in decision.reason


# ---------------------------------------------------------------------------
# Rules alone
# ---------------------------------------------------------------------------


class TestRulesOnly:
    def test_no_verdict_means_rules_stand(self, config: Config, series: SeriesTable) -> None:
        decision = run(config, series, hits=(hit(),), bug=young_bug(), llm=None)
        assert decision.action is Action.NEEDS_INFO
        assert decision.policy_branch == "rules_only"

    def test_failed_verdict_is_silence_not_consent(
        self, config: Config, series: SeriesTable
    ) -> None:
        """A provider outage must not change the outcome, and must be visible."""
        decision = run(
            config,
            series,
            hits=(hit(),),
            bug=young_bug(),
            llm=LlmVerdict.no_opinion(failed=True),
        )
        assert decision.action is Action.NEEDS_INFO
        assert decision.policy_branch == "rules_only"
        assert "no LLM assessment" in decision.reason

    def test_failed_verdict_cannot_veto(self, config: Config, series: SeriesTable) -> None:
        decision = run(
            config,
            series,
            hits=(hit(claim=RuleClaim.QUALITY),),
            llm=LlmVerdict.no_opinion(failed=True),
        )
        assert decision.action is Action.NEEDS_INFO
        assert not decision.llm_vetoed


class TestActionPrecedence:
    def test_invalid_outranks_needs_info(self) -> None:
        """If the package is gone, asking the reporter to re-verify is pointless."""
        action, primary = choose_rule_action(
            (
                hit("eol_apport_release", Action.NEEDS_INFO),
                hit("removed_from_archive", Action.INVALID, RuleClaim.EXISTENCE),
            )
        )
        assert action is Action.INVALID
        assert primary is not None
        assert primary.rule == "removed_from_archive"

    def test_no_hits_is_keep(self) -> None:
        assert choose_rule_action(()) == (Action.KEEP, None)


# ---------------------------------------------------------------------------
# Veto scoping
# ---------------------------------------------------------------------------


class TestQualityVeto:
    """A quality claim ("not enough information") *is* contradicted by
    "a triager could reproduce this", so the model gets a full veto."""

    def test_genuine_reproducible_bug_vetoes_empty_report(
        self, config: Config, series: SeriesTable
    ) -> None:
        decision = run(
            config,
            series,
            hits=(hit("empty_report", claim=RuleClaim.QUALITY),),
            llm=verdict(
                is_actually_a_bug=IsABug.YES,
                bug_kind=BugKind.DEFECT,
                reproducible_from_report=True,
                confidence=0.9,
                rationale="Clear crash with a specific trigger.",
            ),
        )
        assert decision.action is Action.KEEP
        assert decision.llm_vetoed
        assert decision.policy_branch == "llm_veto"
        assert "Clear crash" in decision.reason

    def test_explicit_keep_recommendation_vetoes(
        self, config: Config, series: SeriesTable
    ) -> None:
        decision = run(
            config,
            series,
            hits=(hit("empty_report", claim=RuleClaim.QUALITY),),
            llm=verdict(recommendation=Action.KEEP, confidence=0.75),
        )
        assert decision.action is Action.KEEP
        assert decision.llm_vetoed

    def test_low_confidence_does_not_veto(self, config: Config, series: SeriesTable) -> None:
        decision = run(
            config,
            series,
            hits=(hit("empty_report", claim=RuleClaim.QUALITY),),
            llm=verdict(
                is_actually_a_bug=IsABug.YES,
                reproducible_from_report=True,
                confidence=0.2,
            ),
        )
        assert decision.action is Action.NEEDS_INFO
        assert not decision.llm_vetoed

    def test_not_reproducible_does_not_veto(self, config: Config, series: SeriesTable) -> None:
        decision = run(
            config,
            series,
            hits=(hit("empty_report", claim=RuleClaim.QUALITY),),
            llm=verdict(
                is_actually_a_bug=IsABug.YES,
                reproducible_from_report=False,
                recommendation=Action.NEEDS_INFO,
                confidence=0.95,
            ),
        )
        assert decision.action is Action.NEEDS_INFO


class TestLifecycleVetoScoping:
    def test_genuine_bug_does_not_veto_a_lifecycle_claim(
        self, config: Config, series: SeriesTable
    ) -> None:
        """The crux of the veto-scoping design.

        A real, well-described, reproducible defect against an end-of-life release
        is still unverifiable against anything we ship. If "it's a genuine bug"
        vetoed lifecycle rules, the best-written EOL reports would be precisely
        the ones never pruned, which is backwards and would make the EOL rules
        useless in practice.
        """
        decision = run(
            config,
            series,
            hits=(hit("eol_apport_release", claim=RuleClaim.LIFECYCLE),),
            bug=young_bug(),
            llm=verdict(
                is_actually_a_bug=IsABug.YES,
                bug_kind=BugKind.DEFECT,
                reproducible_from_report=True,
                confidence=1.0,
            ),
        )
        assert decision.action is Action.NEEDS_INFO
        assert not decision.llm_vetoed
        assert decision.policy_branch == "rules_with_llm_concurrence"

    def test_existence_claim_cannot_be_vetoed_on_quality(
        self, config: Config, series: SeriesTable
    ) -> None:
        decision = run(
            config,
            series,
            hits=(hit("removed_from_archive", Action.INVALID, RuleClaim.EXISTENCE),),
            llm=verdict(
                is_actually_a_bug=IsABug.YES,
                reproducible_from_report=True,
                confidence=1.0,
            ),
        )
        assert decision.action is Action.INVALID
        assert not decision.llm_vetoed


class TestLiveReleaseVeto:
    """The model reading prose our text matching missed. Applies to every rule."""

    def test_supported_release_mention_vetoes(
        self, config: Config, series: SeriesTable
    ) -> None:
        decision = run(
            config,
            series,
            hits=(hit("eol_apport_release"),),
            llm=verdict(releases_mentioned=("24.04",), confidence=0.1),
        )
        assert decision.action is Action.KEEP
        assert decision.llm_vetoed
        assert decision.policy_branch == "llm_live_release_veto"

    def test_codename_mention_vetoes(self, config: Config, series: SeriesTable) -> None:
        decision = run(
            config,
            series,
            hits=(hit("eol_apport_release"),),
            llm=verdict(releases_mentioned=("Ubuntu noble",)),
        )
        assert decision.action is Action.KEEP

    def test_applies_regardless_of_confidence(
        self, config: Config, series: SeriesTable
    ) -> None:
        """A hallucinated release here spares a bug, which is the harmless
        direction, so this check is deliberately not confidence-gated."""
        decision = run(
            config,
            series,
            hits=(hit("eol_apport_release"),),
            llm=verdict(releases_mentioned=("noble",), confidence=0.0),
        )
        assert decision.action is Action.KEEP

    def test_obsolete_release_mention_does_not_veto(
        self, config: Config, series: SeriesTable
    ) -> None:
        decision = run(
            config,
            series,
            hits=(hit("eol_apport_release"),),
            bug=young_bug(),
            llm=verdict(releases_mentioned=("14.04", "focal"), confidence=0.9),
        )
        assert decision.action is Action.NEEDS_INFO

    def test_nonsense_release_does_not_veto(self, config: Config, series: SeriesTable) -> None:
        decision = run(
            config,
            series,
            hits=(hit("eol_apport_release"),),
            bug=young_bug(),
            llm=verdict(releases_mentioned=("", "banana", "99.04")),
        )
        assert decision.action is Action.NEEDS_INFO

    def test_live_release_veto_beats_existence_claim(
        self, config: Config, series: SeriesTable
    ) -> None:
        decision = run(
            config,
            series,
            hits=(hit("removed_from_archive", Action.INVALID, RuleClaim.EXISTENCE),),
            llm=verdict(releases_mentioned=("noble",)),
        )
        assert decision.action is Action.KEEP


# ---------------------------------------------------------------------------
# Reclassification
# ---------------------------------------------------------------------------


class TestReclassification:
    def test_support_question_becomes_invalid(
        self, config: Config, series: SeriesTable
    ) -> None:
        decision = run(
            config,
            series,
            hits=(hit(),),
            llm=verdict(
                is_actually_a_bug=IsABug.NO,
                bug_kind=BugKind.SUPPORT_QUESTION,
                confidence=0.95,
                rationale="The reporter is asking how to configure syntax highlighting.",
            ),
        )
        assert decision.action is Action.INVALID
        assert decision.llm_reclassified
        assert decision.policy_branch == "llm_reclassified"
        assert "support-question" in decision.reason

    def test_spam_becomes_invalid(self, config: Config, series: SeriesTable) -> None:
        decision = run(
            config,
            series,
            hits=(hit(),),
            llm=verdict(
                is_actually_a_bug=IsABug.NO, bug_kind=BugKind.SPAM, confidence=0.99
            ),
        )
        assert decision.action is Action.INVALID

    def test_feature_request_is_not_reclassified_by_default(
        self, config: Config, series: SeriesTable
    ) -> None:
        """Ubuntu convention keeps wishlist items open rather than closing them as
        Invalid, so feature-request is excluded from the default kinds."""
        decision = run(
            config,
            series,
            hits=(hit(),),
            bug=young_bug(),
            llm=verdict(
                is_actually_a_bug=IsABug.NO,
                bug_kind=BugKind.FEATURE_REQUEST,
                confidence=1.0,
            ),
        )
        assert decision.action is Action.NEEDS_INFO
        assert not decision.llm_reclassified

    def test_feature_request_reclassified_when_opted_in(self, series: SeriesTable) -> None:
        config = Config.model_validate(
            {"llm": {"reclassify_kinds": ["feature-request"]}}
        )
        decision = run(
            config,
            series,
            hits=(hit(),),
            llm=verdict(
                is_actually_a_bug=IsABug.NO,
                bug_kind=BugKind.FEATURE_REQUEST,
                confidence=1.0,
            ),
        )
        assert decision.action is Action.INVALID

    def test_below_reclassify_threshold_stays_needs_info(
        self, config: Config, series: SeriesTable
    ) -> None:
        decision = run(
            config,
            series,
            hits=(hit(),),
            bug=young_bug(),
            llm=verdict(
                is_actually_a_bug=IsABug.NO,
                bug_kind=BugKind.SUPPORT_QUESTION,
                confidence=0.7,
            ),
        )
        assert decision.action is Action.NEEDS_INFO

    def test_unclear_is_not_reclassified(self, config: Config, series: SeriesTable) -> None:
        decision = run(
            config,
            series,
            hits=(hit(),),
            bug=young_bug(),
            llm=verdict(
                is_actually_a_bug=IsABug.UNCLEAR,
                bug_kind=BugKind.SUPPORT_QUESTION,
                confidence=1.0,
            ),
        )
        assert decision.action is Action.NEEDS_INFO

    def test_disabled_by_config(self, series: SeriesTable) -> None:
        config = Config.model_validate({"llm": {"allow_llm_reclassify_to_invalid": False}})
        decision = run(
            config,
            series,
            hits=(hit(),),
            bug=young_bug(),
            llm=verdict(
                is_actually_a_bug=IsABug.NO,
                bug_kind=BugKind.SUPPORT_QUESTION,
                confidence=1.0,
            ),
        )
        assert decision.action is Action.NEEDS_INFO

    def test_cannot_reclassify_an_invalid_into_something_else(
        self, config: Config, series: SeriesTable
    ) -> None:
        """Reclassification only ever upgrades needs-info, never re-opens the
        question for a rule that already proposed invalid."""
        decision = run(
            config,
            series,
            hits=(hit("removed_from_archive", Action.INVALID, RuleClaim.EXISTENCE),),
            llm=verdict(
                is_actually_a_bug=IsABug.NO,
                bug_kind=BugKind.SUPPORT_QUESTION,
                confidence=1.0,
            ),
        )
        assert decision.action is Action.INVALID
        assert not decision.llm_reclassified


class TestDecisionProvenance:
    def test_decision_records_everything_needed_to_audit_it(
        self, config: Config, series: SeriesTable
    ) -> None:
        hits = (hit("eol_series_tag"), hit("likely_fixed"))
        llm = verdict(confidence=0.4, rationale="unsure")
        decision = run(config, series, hits=hits, llm=llm)

        assert decision.rule_names == ("eol_series_tag", "likely_fixed")
        assert decision.rule_action is Action.NEEDS_INFO
        assert decision.verdict is llm
        assert decision.policy_branch
        assert decision.actionable


# ---------------------------------------------------------------------------
# Age escalation
# ---------------------------------------------------------------------------


class TestAgeEscalation:
    """Rule-side hardening of ``needs-info`` into ``wont-fix`` for very old bugs.

    Distinct from the LLM's powers: deterministic, no model involvement, and
    strictly incapable of creating eligibility -- that direction is asserted
    exhaustively by ``TestLlmCannotCreateAction`` via its ``quiet_days`` axis.
    """

    def test_old_lifecycle_bug_becomes_wont_fix(
        self, config: Config, series: SeriesTable
    ) -> None:
        decision = run(config, series, hits=(hit(),), llm=None)
        assert decision.action is Action.WONT_FIX
        assert decision.policy_branch == "age_escalated"
        assert decision.age_escalated

    def test_provenance_keeps_the_original_rule_action(
        self, config: Config, series: SeriesTable
    ) -> None:
        decision = run(config, series, hits=(hit(),), llm=None)
        assert decision.rule_action is Action.NEEDS_INFO
        assert not decision.llm_reclassified
        assert decision.reason.startswith("eol_apport_release fired")
        assert "8 years old" in decision.reason  # the default bug is 3000 days old

    def test_applies_with_llm_concurrence(self, config: Config, series: SeriesTable) -> None:
        """Deterministic, so it does not need the LLM to have spoken -- but a
        concurring model does not block it either."""
        decision = run(config, series, hits=(hit(),), llm=verdict(confidence=0.1))
        assert decision.action is Action.WONT_FIX
        assert decision.policy_branch == "age_escalated"

    def test_live_release_veto_beats_age(self, config: Config, series: SeriesTable) -> None:
        decision = run(
            config,
            series,
            hits=(hit(),),
            llm=verdict(releases_mentioned=("noble",)),
        )
        assert decision.action is Action.KEEP
        assert not decision.age_escalated

    def test_scoped_veto_beats_age(self, series: SeriesTable) -> None:
        """Reachable only when quality claims are opted into escalation: branch
        order pins that even then, the veto is checked first."""
        config = Config.model_validate({"age": {"claims": ["lifecycle", "quality"]}})
        decision = run(
            config,
            series,
            hits=(hit("empty_report", claim=RuleClaim.QUALITY),),
            llm=verdict(recommendation=Action.KEEP, confidence=0.9),
        )
        assert decision.action is Action.KEEP
        assert decision.llm_vetoed

    def test_reclassification_beats_age(self, config: Config, series: SeriesTable) -> None:
        """"Not a bug report" says more than "old" does."""
        decision = run(
            config,
            series,
            hits=(hit(),),
            llm=verdict(
                is_actually_a_bug=IsABug.NO,
                bug_kind=BugKind.SUPPORT_QUESTION,
                confidence=0.95,
            ),
        )
        assert decision.action is Action.INVALID
        assert decision.llm_reclassified
        assert not decision.age_escalated

    def test_unknown_creation_date_never_escalates(
        self, config: Config, series: SeriesTable
    ) -> None:
        """``age_days`` fails closed, deliberately unlike ``quiet_days``: an
        unknown date must never authorise a close."""
        bug = make_bug(date_created=None)
        decision = run(config, series, hits=(hit(),), bug=bug, llm=None)
        assert decision.action is Action.NEEDS_INFO
        assert not decision.age_escalated

    def test_disabled_when_threshold_zero(self, series: SeriesTable) -> None:
        config = Config.model_validate({"age": {"wont_fix_after_days": 0}})
        decision = run(config, series, hits=(hit(),), llm=None)
        assert decision.action is Action.NEEDS_INFO

    def test_threshold_boundary(self, config: Config, series: SeriesTable) -> None:
        over = run(config, series, hits=(hit(),), bug=make_bug(quiet_days=2555), llm=None)
        under = run(config, series, hits=(hit(),), bug=make_bug(quiet_days=2554), llm=None)
        assert over.action is Action.WONT_FIX
        assert under.action is Action.NEEDS_INFO

    def test_quality_claim_not_escalated_by_default(
        self, config: Config, series: SeriesTable
    ) -> None:
        """A thin report being old is not itself a reason to close it."""
        decision = run(
            config,
            series,
            hits=(hit("empty_report", claim=RuleClaim.QUALITY),),
            llm=None,
        )
        assert decision.action is Action.NEEDS_INFO
        assert not decision.age_escalated

    def test_quality_claim_escalates_when_configured(self, series: SeriesTable) -> None:
        config = Config.model_validate({"age": {"claims": ["lifecycle", "quality"]}})
        decision = run(
            config,
            series,
            hits=(hit("empty_report", claim=RuleClaim.QUALITY),),
            llm=None,
        )
        assert decision.action is Action.WONT_FIX

    def test_invalid_hit_is_untouched(self, config: Config, series: SeriesTable) -> None:
        """Escalation only hardens needs-info; an existence finding already
        proposes the stronger action."""
        decision = run(
            config,
            series,
            hits=(hit("removed_from_archive", Action.INVALID, RuleClaim.EXISTENCE),),
            llm=None,
        )
        assert decision.action is Action.INVALID
        assert not decision.age_escalated
