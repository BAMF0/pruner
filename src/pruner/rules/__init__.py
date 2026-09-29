"""Deterministic rules.

Two distinct kinds, and the distinction is the core safety property of the tool:

* **Exclusions** (:mod:`pruner.rules.exclusions`) are vetoes. If any fires, the bug
  is untouchable and the final action is forced to ``keep``.
* **Prune rules** (everything else) propose an action. A bug is only ever
  *eligible* for action because a prune rule fired on concrete evidence.

The LLM appears nowhere in this package. It can later block or re-kind an action
(see :mod:`pruner.policy`) but it cannot create eligibility.
"""

from __future__ import annotations

# Importing these modules is what populates the registry.
from pruner.rules import content, eol, fixed, removed  # noqa: F401  (side effect)
from pruner.rules.base import (
    RuleContext,
    all_rule_names,
    evaluate_rules,
    get_rule,
    register,
    registered_rules,
    unknown_rule_names,
)

__all__ = [
    "RuleContext",
    "all_rule_names",
    "evaluate_rules",
    "get_rule",
    "register",
    "registered_rules",
    "unknown_rule_names",
]
