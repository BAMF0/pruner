"""The ``analyze`` stage: snapshots in, decisions out.

Entirely offline with respect to Launchpad -- it reads cached snapshots and the
cached series table, and the only network traffic is to the LLM provider. That
means you can iterate on thresholds all afternoon without touching the API.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable
from datetime import UTC, datetime

from pydantic import BaseModel, Field

from pruner.config import Config
from pruner.llm.base import Analyzer
from pruner.lp.archive import ArchiveIndex
from pruner.lp.series import SeriesTable
from pruner.models import Action, BugSnapshot, Decision
from pruner.policy import choose_rule_action, decide
from pruner.rules import RuleContext, evaluate_rules
from pruner.rules.exclusions import evaluate_exclusions

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int, str], None]


class AnalysisStats(BaseModel):
    """Counters for ``pruner stats``: enough to tune a policy on real data."""

    package: str
    bugs: int = 0
    actions: dict[str, int] = Field(default_factory=dict)
    rule_hits: dict[str, int] = Field(default_factory=dict)
    exclusions: dict[str, int] = Field(default_factory=dict)
    policy_branches: dict[str, int] = Field(default_factory=dict)
    llm_calls: int = 0
    llm_failures: int = 0
    llm_vetoes: int = 0
    llm_reclassifications: int = 0
    eligible_before_llm: int = 0
    """Bugs the rules made eligible, before the LLM had a say. The gap between
    this and the actioned count is exactly the LLM's contribution."""

    @staticmethod
    def _bump(counter: dict[str, int], key: str) -> None:
        counter[key] = counter.get(key, 0) + 1

    def record(self, decision: Decision) -> None:
        self.bugs += 1
        self._bump(self.actions, str(decision.action))
        self._bump(self.policy_branches, decision.policy_branch)
        for hit in decision.rule_hits:
            self._bump(self.rule_hits, hit.rule)
        for exclusion in decision.exclusions:
            self._bump(self.exclusions, exclusion.rule)
        if decision.rule_action is not Action.KEEP and not decision.exclusions:
            self.eligible_before_llm += 1
        if decision.verdict is not None:
            self.llm_calls += 1
            if decision.verdict.failed:
                self.llm_failures += 1
        if decision.llm_vetoed:
            self.llm_vetoes += 1
        if decision.llm_reclassified:
            self.llm_reclassifications += 1


class AnalysisResult(BaseModel):
    decisions: list[Decision]
    stats: AnalysisStats

    @property
    def actionable(self) -> list[Decision]:
        return [d for d in self.decisions if d.actionable]


def analyze(
    bugs: Iterable[BugSnapshot],
    *,
    config: Config,
    series: SeriesTable,
    package: str,
    analyzer: Analyzer,
    archive: ArchiveIndex | None = None,
    now: datetime | None = None,
    progress: ProgressCallback | None = None,
    use_cache: bool = True,
) -> AnalysisResult:
    moment = now or datetime.now(UTC)
    context = RuleContext(
        config=config, series=series, package=package, archive=archive, now=moment
    )
    stats = AnalysisStats(package=package)
    decisions: list[Decision] = []

    snapshots = list(bugs)
    for index, bug in enumerate(snapshots, start=1):
        if progress:
            progress(index, len(snapshots), f"bug #{bug.id}")

        exclusions = evaluate_exclusions(
            bug, config, series, package, now=moment
        )
        hits = evaluate_rules(bug, context)

        # The LLM is consulted only where it can actually matter: a bug that is
        # protected, or that no rule flagged, is going to be kept regardless. This
        # is both a large cost saving and a guarantee that the model is never even
        # shown the majority of the backlog.
        rule_action, _ = choose_rule_action(hits)
        verdict = None
        if analyzer.enabled and not exclusions and rule_action is not Action.KEEP:
            verdict = analyzer.assess(bug, hits, package=package, use_cache=use_cache)

        decision = decide(
            bug,
            hits=hits,
            exclusions=exclusions,
            verdict=verdict,
            config=config,
            series=series,
        )
        decisions.append(decision)
        stats.record(decision)

    return AnalysisResult(decisions=decisions, stats=stats)
