"""The structured response we require from the model, and lenient parsing of it.

The schema is hand-written rather than generated from Pydantic because Ollama and
the OpenAI-compatible APIs are happiest with a flat schema containing no ``$ref``
indirection.

Parsing is deliberately forgiving about *shape* (fenced code blocks, surrounding
prose, unknown enum values) and strict about *meaning*: anything we cannot
interpret becomes "no opinion", which the policy treats as silence rather than as
consent to act.
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pruner.models import Action, BugKind, IsABug, LlmVerdict

#: Actions the model is allowed to recommend. Notably excludes anything the model
#: could use to invent a novel action; policy maps these onto real behaviour.
_ALLOWED_RECOMMENDATIONS = [
    Action.KEEP.value,
    Action.NEEDS_INFO.value,
    Action.INVALID.value,
    Action.ESCALATE.value,
]

RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "is_actually_a_bug",
        "bug_kind",
        "needs_more_info",
        "reproducible_from_report",
        "recommendation",
        "confidence",
        "rationale",
    ],
    "properties": {
        "is_actually_a_bug": {
            "type": "string",
            "enum": [e.value for e in IsABug],
            "description": "Does this report describe a genuine software defect?",
        },
        "bug_kind": {
            "type": "string",
            "enum": [e.value for e in BugKind],
            "description": "What kind of report this actually is.",
        },
        "needs_more_info": {
            "type": "boolean",
            "description": "True if a triager could not act without asking the reporter.",
        },
        "missing_info": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific missing facts, e.g. 'exact steps to reproduce'.",
        },
        "reproducible_from_report": {
            "type": "boolean",
            "description": "Could a competent triager attempt a reproduction from this alone?",
        },
        "releases_mentioned": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Ubuntu releases referenced anywhere in the report or comments.",
        },
        "recommendation": {
            "type": "string",
            "enum": _ALLOWED_RECOMMENDATIONS,
        },
        "confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "rationale": {
            "type": "string",
            "description": "Two or three sentences justifying the assessment.",
        },
        "suggested_comment_points": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Bullet points to include when asking the reporter for more.",
        },
    },
}


class LlmResponse(BaseModel):
    """Validated model output, before it becomes an advisory verdict."""

    model_config = ConfigDict(extra="ignore")

    is_actually_a_bug: IsABug = IsABug.UNCLEAR
    bug_kind: BugKind = BugKind.UNCLEAR
    needs_more_info: bool = False
    missing_info: list[str] = Field(default_factory=list)
    reproducible_from_report: bool = False
    releases_mentioned: list[str] = Field(default_factory=list)
    recommendation: Action = Action.KEEP
    confidence: float = 0.0
    rationale: str = ""
    suggested_comment_points: list[str] = Field(default_factory=list)

    # Small models routinely emit near-miss enum values ("feature request",
    # "not a bug", "needs info"). Normalising is much better than discarding an
    # otherwise useful assessment -- but anything genuinely unrecognised falls
    # back to the neutral value, never to an actionable one.

    @field_validator("is_actually_a_bug", mode="before")
    @classmethod
    def _norm_is_bug(cls, value: Any) -> Any:
        return _normalise(value, IsABug, IsABug.UNCLEAR, {"true": "yes", "false": "no"})

    @field_validator("bug_kind", mode="before")
    @classmethod
    def _norm_kind(cls, value: Any) -> Any:
        return _normalise(
            value,
            BugKind,
            BugKind.UNCLEAR,
            {
                "feature": "feature-request",
                "featurerequest": "feature-request",
                "enhancement": "feature-request",
                "wishlist": "feature-request",
                "question": "support-question",
                "support": "support-question",
                "supportquestion": "support-question",
                "userror": "support-question",
                "bug": "defect",
                "docs": "documentation",
                "doc": "documentation",
            },
        )

    @field_validator("recommendation", mode="before")
    @classmethod
    def _norm_recommendation(cls, value: Any) -> Any:
        return _normalise(
            value,
            Action,
            Action.KEEP,
            {
                "needsinfo": "needs-info",
                "needinfo": "needs-info",
                "incomplete": "needs-info",
                "moreinfo": "needs-info",
                "close": "invalid",
                "wontfix": "invalid",
                "notabug": "invalid",
                "open": "keep",
                "leaveopen": "keep",
                "nothing": "keep",
            },
        )

    @field_validator("confidence", mode="before")
    @classmethod
    def _norm_confidence(cls, value: Any) -> Any:
        """Clamp to [0, 1] and accept percentages, which models emit constantly."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return 0.0
        if number > 1.0:
            number = number / 100.0 if number <= 100.0 else 1.0
        return min(max(number, 0.0), 1.0)

    @field_validator(
        "missing_info", "releases_mentioned", "suggested_comment_points", mode="before"
    )
    @classmethod
    def _norm_list(cls, value: Any) -> Any:
        if value is None:
            return []
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        return []

    def to_verdict(self, *, model: str) -> LlmVerdict:
        return LlmVerdict(
            is_actually_a_bug=self.is_actually_a_bug,
            bug_kind=self.bug_kind,
            needs_more_info=self.needs_more_info,
            missing_info=tuple(self.missing_info[:8]),
            reproducible_from_report=self.reproducible_from_report,
            releases_mentioned=tuple(self.releases_mentioned[:8]),
            recommendation=self.recommendation,
            confidence=self.confidence,
            rationale=self.rationale.strip()[:1200],
            suggested_comment_points=tuple(self.suggested_comment_points[:6]),
            model=model,
            failed=False,
        )


def _normalise[E](value: Any, enum: type[Any], default: E, aliases: dict[str, str]) -> Any:
    if not isinstance(value, str):
        return default
    raw = value.strip()
    try:
        return enum(raw)
    except ValueError:
        pass
    key = re.sub(r"[^a-z]", "", raw.lower())
    if mapped := aliases.get(key):
        return enum(mapped)
    # Try a slug form: "Feature Request" -> "feature-request"
    slug = re.sub(r"[^a-z]+", "-", raw.lower()).strip("-")
    try:
        return enum(slug)
    except ValueError:
        return default


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a model response.

    Handles bare JSON, fenced blocks, and JSON surrounded by chatty prose, all of
    which local models produce even when asked not to.
    """
    if not text:
        return None

    candidates: list[str] = []
    if match := _FENCE_RE.search(text):
        candidates.append(match.group(1))
    candidates.append(text)

    for candidate in candidates:
        stripped = candidate.strip()
        try:
            parsed = json.loads(stripped)
        except ValueError:
            parsed = None
        if isinstance(parsed, dict):
            return parsed
        if block := _balanced_object(stripped):
            try:
                parsed = json.loads(block)
            except ValueError:
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def _balanced_object(text: str) -> str | None:
    """First brace-balanced ``{...}`` span, ignoring braces inside strings."""
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def parse_response(text: str, *, model: str) -> LlmVerdict | None:
    """Parse raw model output into a verdict, or ``None`` if unusable."""
    payload = extract_json(text)
    if payload is None:
        return None
    try:
        return LlmResponse.model_validate(payload).to_verdict(model=model)
    except ValueError:
        return None
