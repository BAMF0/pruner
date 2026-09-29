"""Rule plumbing: the context object, the registry, and the evaluation entry point.

A rule is a pure function ``(BugSnapshot, RuleContext) -> RuleHit | None``. Purity
is the point: every rule is unit-testable against a recorded snapshot with no
network, and a rule cannot reach out and mutate anything.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from pruner.config import Config
from pruner.lp.archive import ArchiveIndex
from pruner.lp.series import SeriesTable
from pruner.models import BugSnapshot, RuleHit

RuleFunc = Callable[[BugSnapshot, "RuleContext"], RuleHit | None]


class RuleContext(BaseModel):
    """Everything a rule may look at besides the bug itself.

    Passing this explicitly (rather than letting rules fetch things) is what keeps
    them pure and keeps ``analyze`` fully offline.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    config: Config
    series: SeriesTable
    package: str
    archive: ArchiveIndex | None = None
    now: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def distribution(self) -> str:
        return self.config.launchpad.distribution


_REGISTRY: dict[str, RuleFunc] = {}


def register(name: str) -> Callable[[RuleFunc], RuleFunc]:
    """Decorator registering a prune rule under a stable name.

    The name is what appears in ``[rules].enabled``, in the report, and in the
    comment posted to Launchpad, so it is part of the tool's public contract.
    """

    def decorator(func: RuleFunc) -> RuleFunc:
        if name in _REGISTRY:
            raise ValueError(f"duplicate rule name: {name}")
        _REGISTRY[name] = func
        func.__rule_name__ = name  # type: ignore[attr-defined]
        return func

    return decorator


def registered_rules() -> dict[str, RuleFunc]:
    return dict(_REGISTRY)


def all_rule_names() -> frozenset[str]:
    return frozenset(_REGISTRY)


def get_rule(name: str) -> RuleFunc:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise KeyError(f"no such rule: {name}") from exc


def unknown_rule_names(names: Iterable[str]) -> set[str]:
    return {n for n in names if n not in _REGISTRY}


def evaluate_rules(
    bug: BugSnapshot,
    context: RuleContext,
    *,
    enabled: Iterable[str] | None = None,
) -> tuple[RuleHit, ...]:
    """Run the enabled prune rules against a bug and collect every hit.

    All enabled rules run even after the first hit: the report is more useful when
    it shows every reason a bug looks prunable, and :mod:`pruner.policy` picks the
    action from the full set.
    """
    names = list(enabled if enabled is not None else context.config.rules.enabled)
    hits: list[RuleHit] = []
    for name in names:
        rule = _REGISTRY.get(name)
        if rule is None:
            continue
        if hit := rule(bug, context):
            hits.append(hit)
    return tuple(hits)
