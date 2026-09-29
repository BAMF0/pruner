"""Structural safety properties.

These test the *architecture* rather than behaviour. They exist because the
guarantees they cover are easy to erode accidentally in a later refactor, and
because a reviewer should be able to verify the claims mechanically rather than
by reading every module.
"""

from __future__ import annotations

import subprocess
import sys

from pruner.rules.exclusions import EXCLUSIONS, PREFILTER_EXCLUSIONS


class TestWritePathIsolation:
    def test_read_path_does_not_import_launchpadlib(self) -> None:
        """Importing the fetch/analyze/report path must not pull in OAuth
        machinery. Credentials should be unreachable until ``apply`` asks for them.
        """
        code = (
            "import pruner.cli, pruner.fetcher, pruner.analysis, pruner.report, "
            "pruner.policy, pruner.rules, pruner.llm\n"
            "import sys\n"
            "leaked = [m for m in sys.modules if m.split('.')[0] "
            "in {'launchpadlib', 'lazr', 'oauthlib'}]\n"
            "print(','.join(sorted(leaked)))\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        assert result.stdout.strip() == "", f"write deps leaked: {result.stdout.strip()}"

    def test_only_write_module_imports_launchpadlib(self) -> None:
        """Checked via the AST, not a text search, so prose mentioning the library
        in a docstring does not count -- only a real import does."""
        import ast
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "src" / "pruner"
        write_deps = {"launchpadlib", "lazr", "oauthlib"}
        offenders: list[str] = []

        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            imported: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            if imported & write_deps:
                offenders.append(path.relative_to(root).as_posix())

        assert offenders == ["lp/write.py"], f"unexpected write-dep imports: {offenders}"


class TestPrefilterSoundness:
    def test_prefilter_is_a_subset_of_all_exclusions(self) -> None:
        assert {name for name, _ in EXCLUSIONS} >= PREFILTER_EXCLUSIONS

    def test_prefilter_excludes_checks_needing_enrichment(self) -> None:
        """``fetch`` must not prefilter on data it has not fetched yet."""
        needs_enrichment = {
            "incomplete_snapshot",
            "dev_activity",
            "affects_live_release",
            "already_triaged",
        }
        assert not (PREFILTER_EXCLUSIONS & needs_enrichment)

    def test_incomplete_snapshot_check_runs_first(self) -> None:
        """It must be evaluated before anything could conclude a bug is clean."""
        assert EXCLUSIONS[0][0] == "incomplete_snapshot"


class TestActionSemantics:
    def test_only_reversible_or_justified_actions_mutate_status(self) -> None:
        from pruner.models import Action

        mutating = {a for a in Action if a.mutates_status}
        assert mutating == {Action.NEEDS_INFO, Action.INVALID}
        assert not Action.KEEP.mutates_status
        assert not Action.ESCALATE.mutates_status

    def test_rule_claim_is_required_not_defaulted(self) -> None:
        """Enforced by the model rather than by convention: a rule author cannot
        forget to say what kind of claim they are making, because omitting it is a
        validation error. This matters because the claim governs LLM veto scoping.
        """
        import pytest

        from pruner.models import Action, RuleHit

        with pytest.raises(ValueError):
            RuleHit(rule="x", action=Action.NEEDS_INFO, reason="y")

    def test_every_registered_rule_produces_a_claim_when_it_fires(self) -> None:
        """Complements the above: exercised for real in tests/test_rules.py, which
        asserts the specific claim of each rule's hit."""
        from pruner.models import RuleClaim
        from pruner.rules import all_rule_names

        assert all_rule_names(), "registry should not be empty"
        assert set(RuleClaim) == {
            RuleClaim.LIFECYCLE,
            RuleClaim.QUALITY,
            RuleClaim.EXISTENCE,
        }


class TestDefaultsAreConservative:
    def test_shipped_config_matches_code_defaults(self) -> None:
        """``pruner.toml`` documents the defaults; drift between the two would make
        the file misleading."""
        from pathlib import Path

        from pruner.config import Config, load_config

        path = Path(__file__).resolve().parent.parent / "pruner.toml"
        from_file = load_config(path)
        built_in = Config()

        assert from_file.launchpad.support_policy == built_in.launchpad.support_policy
        assert from_file.launchpad.distribution == built_in.launchpad.distribution
        assert from_file.safety.min_quiet_days == built_in.safety.min_quiet_days
        assert from_file.safety.protect_importances == built_in.safety.protect_importances
        assert from_file.safety.protect_tags == built_in.safety.protect_tags
        assert from_file.safety.max_actions_per_run == built_in.safety.max_actions_per_run
        assert from_file.rules.enabled == built_in.rules.enabled
        assert from_file.rules.min_desc_chars == built_in.rules.min_desc_chars
        assert from_file.llm.reclassify_kinds == built_in.llm.reclassify_kinds
        assert from_file.llm.veto_threshold == built_in.llm.veto_threshold
        assert from_file.comment.marker == built_in.comment.marker

    def test_dry_run_is_the_default_for_apply(self) -> None:
        """``--commit`` must be opt-in."""
        import inspect

        from pruner.cli import apply

        assert inspect.signature(apply).parameters["commit"].default is False

    def test_rollback_dry_run_is_also_the_default(self) -> None:
        import inspect

        from pruner.cli import rollback

        assert inspect.signature(rollback).parameters["commit"].default is False


class TestReasonPhrasing:
    """Rule reasons are interpolated into a posted comment after the word
    "because", so they must read as a clause.

    This caught a real bug: ``eol_series_tag`` originally produced "tagged only
    for ...", yielding "This bug is being set to Incomplete because tagged only
    for lucid (10.04)."
    """

    ACCEPTABLE_OPENINGS = ("it ", "the ", "every ", "nothing ", "a ", "there ")

    def _all_reasons(self) -> list[tuple[str, str]]:
        import datetime

        from pruner.config import Config
        from pruner.lp.archive import ArchiveIndex
        from pruner.lp.series import SeriesTable
        from pruner.models import ApportInfo, SupportPolicy
        from pruner.rules import RuleContext, registered_rules
        from tests.conftest import NOW, SERIES_ENTRIES, make_bug, make_task

        series = SeriesTable.from_api(
            "ubuntu",
            SERIES_ENTRIES,
            policy=SupportPolicy.STANDARD,
            eol_dates={"trusty": datetime.date(2019, 4, 25)},
            today=NOW.date(),
        )
        config = Config()

        # One bug per rule, crafted so that the rule fires.
        cases = {
            "eol_series_tasks": (
                make_bug(tasks=(make_task(target="vim (Ubuntu Trusty)", series_name="trusty"),)),
                None,
            ),
            "eol_series_tag": (make_bug(tags=("trusty",)), None),
            "eol_apport_release": (
                make_bug(apport=ApportInfo(distro_release="14.04")),
                None,
            ),
            "eol_obsolete_only": (make_bug(tags=("trusty",)), None),
            "likely_fixed": (
                make_bug(apport=ApportInfo(version="1.0")),
                ArchiveIndex(
                    distribution="ubuntu",
                    package="vim",
                    publications=(),
                    queried_series=("noble",),
                ).model_copy(
                    update={
                        "publications": (
                            __import__(
                                "pruner.lp.archive", fromlist=["Publication"]
                            ).Publication(series="noble", version="2.0", pocket="Release"),
                        )
                    }
                ),
            ),
            "removed_from_archive": (
                make_bug(),
                ArchiveIndex(
                    distribution="ubuntu",
                    package="gone",
                    publications=(),
                    queried_series=("noble",),
                ),
            ),
            "empty_report": (make_bug(description="broken"), None),
        }

        found: list[tuple[str, str]] = []
        for name, func in registered_rules().items():
            assert name in cases, f"no phrasing case for rule {name}"
            bug, archive = cases[name]
            context = RuleContext(
                config=config,
                series=series,
                package="vim",
                archive=archive,
                now=NOW,
            )
            hit = func(bug, context)
            assert hit is not None, f"rule {name} did not fire for its phrasing case"
            found.append((name, hit.reason))
        return found

    def test_reasons_read_as_a_clause(self) -> None:
        for name, reason in self._all_reasons():
            assert reason[0].islower(), f"{name}: reason should not start capitalised"
            assert reason.startswith(self.ACCEPTABLE_OPENINGS), (
                f"{name}: reason {reason!r} does not read as a clause after "
                f'"because"; expected it to start with one of {self.ACCEPTABLE_OPENINGS}'
            )
            assert not reason.endswith("."), f"{name}: reason should not end with a period"

    def test_reason_renders_in_a_comment(self, config) -> None:
        from pruner.comment import compose_comment
        from pruner.models import Action, Decision, RuleClaim, RuleHit
        from tests.conftest import make_bug

        for name, reason in self._all_reasons():
            hit = RuleHit(
                rule=name,
                action=Action.NEEDS_INFO,
                claim=RuleClaim.LIFECYCLE,
                reason=reason,
            )
            body = compose_comment(
                make_bug(),
                Decision(
                    bug_id=1,
                    action=Action.NEEDS_INFO,
                    reason=reason,
                    rule_action=Action.NEEDS_INFO,
                    rule_hits=(hit,),
                ),
                config,
                package="vim",
            )
            assert f"because {reason}." in body
            assert "because tagged" not in body
