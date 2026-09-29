"""LLM response parsing.

Small local models are sloppy: they wrap JSON in prose, use fenced blocks, emit
percentages for confidence, and invent near-miss enum values. Recovering a usable
assessment from that is worth real effort, but anything genuinely unrecognisable
must degrade to "no opinion" rather than to something actionable.
"""

from __future__ import annotations

import json

import pytest

from pruner.config import LlmConfig
from pruner.llm.base import Analyzer, Provider, ProviderError
from pruner.llm.prompt import build_user_prompt
from pruner.llm.schema import RESPONSE_SCHEMA, extract_json, parse_response
from pruner.models import Action, ApportInfo, BugKind, IsABug
from tests.conftest import make_bug

VALID = {
    "is_actually_a_bug": "yes",
    "bug_kind": "defect",
    "needs_more_info": False,
    "missing_info": [],
    "reproducible_from_report": True,
    "releases_mentioned": ["24.04"],
    "recommendation": "keep",
    "confidence": 0.85,
    "rationale": "Clear reproduction steps and a specific failure.",
    "suggested_comment_points": [],
}


class TestExtractJson:
    def test_bare_object(self) -> None:
        assert extract_json('{"a": 1}') == {"a": 1}

    def test_fenced_block(self) -> None:
        assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_unlabelled_fence(self) -> None:
        assert extract_json('```\n{"a": 1}\n```') == {"a": 1}

    def test_surrounded_by_prose(self) -> None:
        text = 'Sure! Here is my assessment:\n{"a": 1}\nHope that helps.'
        assert extract_json(text) == {"a": 1}

    def test_braces_inside_strings_do_not_confuse_the_scanner(self) -> None:
        text = 'Here: {"rationale": "the config had a } in it", "a": 1} done'
        parsed = extract_json(text)
        assert parsed is not None
        assert parsed["a"] == 1

    def test_escaped_quote_inside_string(self) -> None:
        payload = {"rationale": 'he said "it crashed"', "a": 2}
        assert extract_json("noise " + json.dumps(payload)) == payload

    def test_nested_objects(self) -> None:
        assert extract_json('{"a": {"b": {"c": 1}}}') == {"a": {"b": {"c": 1}}}

    def test_no_json_returns_none(self) -> None:
        assert extract_json("I cannot help with that.") is None
        assert extract_json("") is None

    def test_json_array_is_not_accepted(self) -> None:
        assert extract_json("[1, 2, 3]") is None


class TestParseResponse:
    def test_valid_response(self) -> None:
        got = parse_response(json.dumps(VALID), model="m")
        assert got is not None
        assert got.is_actually_a_bug is IsABug.YES
        assert got.bug_kind is BugKind.DEFECT
        assert got.recommendation is Action.KEEP
        assert got.confidence == pytest.approx(0.85)
        assert got.model == "m"
        assert not got.failed

    def test_unparseable_returns_none(self) -> None:
        assert parse_response("nope", model="m") is None

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("feature request", BugKind.FEATURE_REQUEST),
            ("Feature Request", BugKind.FEATURE_REQUEST),
            ("enhancement", BugKind.FEATURE_REQUEST),
            ("wishlist", BugKind.FEATURE_REQUEST),
            ("question", BugKind.SUPPORT_QUESTION),
            ("support question", BugKind.SUPPORT_QUESTION),
            ("bug", BugKind.DEFECT),
            ("docs", BugKind.DOCUMENTATION),
            ("total gibberish", BugKind.UNCLEAR),
        ],
    )
    def test_bug_kind_normalisation(self, given: str, expected: BugKind) -> None:
        payload = {**VALID, "bug_kind": given}
        got = parse_response(json.dumps(payload), model="m")
        assert got is not None
        assert got.bug_kind is expected

    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("needs info", Action.NEEDS_INFO),
            ("needs_info", Action.NEEDS_INFO),
            ("incomplete", Action.NEEDS_INFO),
            ("not a bug", Action.INVALID),
            ("wontfix", Action.INVALID),
            ("leave open", Action.KEEP),
            ("gibberish", Action.KEEP),
        ],
    )
    def test_recommendation_normalisation(self, given: str, expected: Action) -> None:
        got = parse_response(json.dumps({**VALID, "recommendation": given}), model="m")
        assert got is not None
        assert got.recommendation is expected

    def test_unrecognised_recommendation_defaults_to_keep(self) -> None:
        """The neutral fallback must be the *least* destructive value."""
        got = parse_response(json.dumps({**VALID, "recommendation": "obliterate"}), model="m")
        assert got is not None
        assert got.recommendation is Action.KEEP

    @pytest.mark.parametrize(
        ("given", "expected"),
        [(85, 0.85), (0.85, 0.85), ("0.5", 0.5), (150, 1.0), (-3, 0.0), ("junk", 0.0), (None, 0.0)],
    )
    def test_confidence_coercion(self, given: object, expected: float) -> None:
        got = parse_response(json.dumps({**VALID, "confidence": given}), model="m")
        assert got is not None
        assert got.confidence == pytest.approx(expected)

    def test_string_instead_of_list(self) -> None:
        got = parse_response(json.dumps({**VALID, "missing_info": "steps"}), model="m")
        assert got is not None
        assert got.missing_info == ("steps",)

    def test_null_list_becomes_empty(self) -> None:
        got = parse_response(json.dumps({**VALID, "releases_mentioned": None}), model="m")
        assert got is not None
        assert got.releases_mentioned == ()

    def test_missing_optional_fields_are_defaulted(self) -> None:
        got = parse_response('{"is_actually_a_bug": "no"}', model="m")
        assert got is not None
        assert got.is_actually_a_bug is IsABug.NO
        assert got.recommendation is Action.KEEP
        assert got.confidence == 0.0

    def test_extra_fields_ignored(self) -> None:
        got = parse_response(json.dumps({**VALID, "extra": "stuff"}), model="m")
        assert got is not None

    def test_boolean_for_is_actually_a_bug(self) -> None:
        got = parse_response(json.dumps({**VALID, "is_actually_a_bug": "true"}), model="m")
        assert got is not None
        assert got.is_actually_a_bug is IsABug.YES

    def test_long_rationale_truncated(self) -> None:
        got = parse_response(json.dumps({**VALID, "rationale": "x" * 5000}), model="m")
        assert got is not None
        assert len(got.rationale) <= 1200


class TestSchema:
    def test_schema_is_flat(self) -> None:
        """No ``$ref`` indirection: Ollama and OpenAI-compatible proxies handle a
        flat schema far more reliably."""
        assert "$defs" not in RESPONSE_SCHEMA
        assert "$ref" not in json.dumps(RESPONSE_SCHEMA)

    def test_required_fields_present_in_properties(self) -> None:
        props = set(RESPONSE_SCHEMA["properties"])
        assert set(RESPONSE_SCHEMA["required"]) <= props


# ---------------------------------------------------------------------------
# Analyzer failure handling
# ---------------------------------------------------------------------------


class FakeProvider(Provider):
    name = "fake"

    def __init__(self, config: LlmConfig, responses: list[str | Exception]) -> None:
        super().__init__(config)
        self.responses = responses
        self.calls = 0

    def complete(self, system: str, user: str, *, schema: dict) -> str:
        self.calls += 1
        item = self.responses[min(self.calls - 1, len(self.responses) - 1)]
        if isinstance(item, Exception):
            raise item
        return item


class TestAnalyzerFailSafe:
    def test_provider_error_yields_no_opinion(self) -> None:
        config = LlmConfig(max_attempts=2)
        provider = FakeProvider(config, [ProviderError("down")])
        got = Analyzer(provider, config).assess(make_bug(), (), package="vim")
        assert got.failed
        assert got.recommendation is Action.KEEP
        assert got.confidence == 0.0

    def test_retries_then_succeeds(self) -> None:
        config = LlmConfig(max_attempts=2)
        provider = FakeProvider(config, ["garbage", json.dumps(VALID)])
        got = Analyzer(provider, config).assess(make_bug(), (), package="vim")
        assert not got.failed
        assert provider.calls == 2

    def test_exhausted_attempts_yields_no_opinion(self) -> None:
        config = LlmConfig(max_attempts=2)
        provider = FakeProvider(config, ["garbage"])
        got = Analyzer(provider, config).assess(make_bug(), (), package="vim")
        assert got.failed
        assert provider.calls == 2

    def test_disabled_analyzer_never_calls_out(self) -> None:
        analyzer = Analyzer(None, LlmConfig(provider="none"))
        assert not analyzer.enabled
        got = analyzer.assess(make_bug(), (), package="vim")
        assert got.model == "none"
        assert got.recommendation is Action.KEEP


class TestVerdictCaching:
    def test_verdict_cached_by_text_fingerprint(self, tmp_path) -> None:
        from pruner.store import Store

        config = LlmConfig(max_attempts=1)
        provider = FakeProvider(config, [json.dumps(VALID)])
        with Store.open(tmp_path) as store:
            analyzer = Analyzer(provider, config, store)
            bug = make_bug()
            analyzer.assess(bug, (), package="vim")
            analyzer.assess(bug, (), package="vim")
            assert provider.calls == 1, "second identical bug should hit the cache"

    def test_changed_text_invalidates_cache(self, tmp_path) -> None:
        from pruner.store import Store

        config = LlmConfig(max_attempts=1)
        provider = FakeProvider(config, [json.dumps(VALID)])
        with Store.open(tmp_path) as store:
            analyzer = Analyzer(provider, config, store)
            analyzer.assess(make_bug(), (), package="vim")
            analyzer.assess(
                make_bug(comment_texts=("a new comment",)), (), package="vim"
            )
            assert provider.calls == 2

    def test_failed_verdicts_are_not_cached(self, tmp_path) -> None:
        """A transient outage must not poison later runs with a permanent
        "no opinion"."""
        from pruner.store import Store

        config = LlmConfig(max_attempts=1)
        provider = FakeProvider(config, [ProviderError("down"), json.dumps(VALID)])
        with Store.open(tmp_path) as store:
            analyzer = Analyzer(provider, config, store)
            bug = make_bug()
            assert analyzer.assess(bug, (), package="vim").failed
            assert not analyzer.assess(bug, (), package="vim").failed


class TestPrompt:
    def test_apport_block_excluded_from_prompt(self) -> None:
        description = (
            "vim segfaults on a large file.\n\n"
            "ProblemType: Bug\nDistroRelease: Ubuntu 14.04\n"
            "Dependencies: libc6 2.19-0ubuntu6\n"
        )
        bug = make_bug(
            description=description,
            apport=ApportInfo(distro_release="14.04", package="vim"),
        )
        prompt = build_user_prompt(bug, (), LlmConfig(), package="vim")
        assert "segfaults" in prompt
        assert "Dependencies:" not in prompt
        assert "ProblemType:" not in prompt
        # Apport facts still reach the model via the metadata line.
        assert "filed against Ubuntu 14.04" in prompt

    def test_description_truncated(self) -> None:
        bug = make_bug(description="z" * 20000)
        prompt = build_user_prompt(
            bug, (), LlmConfig(max_description_chars=500), package="vim"
        )
        assert "truncated" in prompt
        assert len(prompt) < 5000

    def test_comments_capped(self) -> None:
        bug = make_bug(comment_texts=tuple(f"comment {i}" for i in range(20)))
        prompt = build_user_prompt(bug, (), LlmConfig(max_comments=3), package="vim")
        assert "17 further comment(s) not shown" in prompt

    def test_rule_findings_included_with_independence_instruction(self) -> None:
        from pruner.models import Action as A
        from pruner.models import RuleClaim, RuleHit

        hits = (
            RuleHit(
                rule="eol_apport_release",
                action=A.NEEDS_INFO,
                claim=RuleClaim.LIFECYCLE,
                reason="dead release",
            ),
        )
        prompt = build_user_prompt(make_bug(), hits, LlmConfig(), package="vim")
        assert "eol_apport_release" in prompt
        # Without this framing the model rubber-stamps the rule.
        assert "independently" in prompt
        assert "may well be wrong" in prompt

    def test_no_comments_is_stated_explicitly(self) -> None:
        prompt = build_user_prompt(make_bug(), (), LlmConfig(), package="vim")
        assert "nobody ever followed up" in prompt
