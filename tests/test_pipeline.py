"""Store, approvals and report rendering, plus an end-to-end pipeline run.

The approvals tests are the review gate: if ``apply`` could act on an unreviewed
file, or on decisions from a different analysis, the whole human-in-the-loop
design would be decorative.
"""

from __future__ import annotations

from pathlib import Path

from pruner.analysis import AnalysisResult, AnalysisStats, analyze
from pruner.config import Config
from pruner.llm.base import Analyzer
from pruner.lp.archive import ArchiveIndex
from pruner.lp.series import SeriesTable
from pruner.models import (
    Action,
    ApportInfo,
    BugTaskStatus,
    Decision,
    RuleClaim,
    RuleHit,
)
from pruner.report import read_approvals, render_csv, render_markdown, write_approvals
from pruner.store import Store, verdict_fingerprint
from tests.conftest import NOW, make_bug, make_task


def _approve_entry(path: Path, bug_id: int) -> None:
    """Flip the approve flag for one entry, the way a human editing the file would.

    Scoped to the entry rather than a blind string replace, because the file's
    header comment also mentions the flag.
    """
    lines = path.read_text().splitlines()
    in_entry = False
    for index, line in enumerate(lines):
        if line.strip() == f"id = {bug_id}":
            in_entry = True
        elif line.startswith("[[bug]]"):
            in_entry = False
        elif in_entry and line.startswith("approve = "):
            lines[index] = "approve = true"
            break
    path.write_text("\n".join(lines) + "\n")


def result_from(decisions: list[Decision]) -> AnalysisResult:
    stats = AnalysisStats(package="vim")
    for item in decisions:
        stats.record(item)
    return AnalysisResult(decisions=decisions, stats=stats)


def a_decision(bug_id: int, action: Action = Action.NEEDS_INFO) -> Decision:
    return Decision(
        bug_id=bug_id,
        action=action,
        reason="reported against Ubuntu 14.04 (trusty), which reached end of life",
        rule_action=action,
        rule_hits=(
            RuleHit(
                rule="eol_apport_release",
                action=action,
                claim=RuleClaim.LIFECYCLE,
                reason="dead release",
            ),
        ),
        policy_branch="rules_only",
    )


class TestStore:
    def test_round_trip_snapshot(self, tmp_path: Path) -> None:
        bug = make_bug(bug_id=42, tags=("trusty",))
        with Store.open(tmp_path) as store:
            store.put_bug("ubuntu", "vim", bug)
            got = store.get_bug("ubuntu", "vim", 42)
        assert got is not None
        assert got.id == 42
        assert got.tags == ("trusty",)

    def test_upsert_does_not_duplicate(self, tmp_path: Path) -> None:
        with Store.open(tmp_path) as store:
            store.put_bug("ubuntu", "vim", make_bug(bug_id=1))
            store.put_bug("ubuntu", "vim", make_bug(bug_id=1, title="changed"))
            assert store.bug_count("ubuntu", "vim") == 1
            got = store.get_bug("ubuntu", "vim", 1)
            assert got is not None and got.title == "changed"

    def test_packages_are_isolated(self, tmp_path: Path) -> None:
        with Store.open(tmp_path) as store:
            store.put_bug("ubuntu", "vim", make_bug(bug_id=1))
            store.put_bug("ubuntu", "nano", make_bug(bug_id=2))
            assert store.bug_count("ubuntu", "vim") == 1
            assert store.get_bug("ubuntu", "vim", 2) is None

    def test_cached_last_updated_for_freshness_check(self, tmp_path: Path) -> None:
        bug = make_bug(bug_id=7)
        with Store.open(tmp_path) as store:
            store.put_bug("ubuntu", "vim", bug)
            cached = store.cached_last_updated("ubuntu", "vim")
        assert bug.date_last_updated is not None
        assert cached[7] == bug.date_last_updated.isoformat()

    def test_series_round_trip(self, tmp_path: Path, series: SeriesTable) -> None:
        with Store.open(tmp_path) as store:
            store.put_series(series)
            got = store.get_series("ubuntu")
        assert got is not None
        assert got.is_obsolete("trusty")
        assert not got.is_obsolete("noble")

    def test_archive_round_trip(self, tmp_path: Path, archive: ArchiveIndex) -> None:
        with Store.open(tmp_path) as store:
            store.put_archive(archive)
            got = store.get_archive("ubuntu", "vim")
        assert got is not None
        assert got.is_published_anywhere

    def test_decisions_scoped_to_run(self, tmp_path: Path) -> None:
        with Store.open(tmp_path) as store:
            store.start_run("run-a", "analyze", "ubuntu", "vim", {})
            store.start_run("run-b", "analyze", "ubuntu", "vim", {})
            store.put_decision("run-a", a_decision(1))
            store.put_decision("run-b", a_decision(2))
            assert [d.bug_id for d in store.get_decisions("run-a")] == [1]
            assert [d.bug_id for d in store.get_decisions("run-b")] == [2]

    def test_latest_run(self, tmp_path: Path) -> None:
        with Store.open(tmp_path) as store:
            store.start_run("old", "analyze", "ubuntu", "vim", {})
            store.start_run("new", "analyze", "ubuntu", "vim", {})
            assert store.latest_run("analyze", "ubuntu", "vim") in ("old", "new")
            assert store.latest_run("analyze", "ubuntu", "nano") is None

    def test_reopening_is_safe(self, tmp_path: Path) -> None:
        with Store.open(tmp_path) as store:
            store.put_bug("ubuntu", "vim", make_bug(bug_id=1))
        with Store.open(tmp_path) as store:
            assert store.bug_count("ubuntu", "vim") == 1


class TestVerdictFingerprint:
    def test_stable_for_identical_text(self) -> None:
        assert verdict_fingerprint(make_bug()) == verdict_fingerprint(make_bug())

    def test_changes_with_description(self) -> None:
        assert verdict_fingerprint(make_bug()) != verdict_fingerprint(
            make_bug(description="different")
        )

    def test_changes_with_new_comment(self) -> None:
        assert verdict_fingerprint(make_bug()) != verdict_fingerprint(
            make_bug(comment_texts=("new",))
        )

    def test_unaffected_by_status_only_change(self) -> None:
        """A status flip does not change what the model would see, so a cached
        verdict is still valid and should not be recomputed."""
        plain = make_bug()
        flipped = make_bug(tasks=(make_task(status=BugTaskStatus.CONFIRMED),))
        assert verdict_fingerprint(plain) == verdict_fingerprint(flipped)


class TestApprovals:
    def test_nothing_approved_by_default(self, tmp_path: Path, config: Config) -> None:
        """The review gate: an unedited approvals file must authorise nothing."""
        path = tmp_path / "approvals.toml"
        result = result_from([a_decision(1), a_decision(2)])
        write_approvals(
            path, result, run_id="r1", config=config, package="vim", bugs={}
        )
        approvals = read_approvals(path)
        assert approvals.approved == frozenset()
        assert not approvals.allows(1)

    def test_approving_one_entry(self, tmp_path: Path, config: Config) -> None:
        path = tmp_path / "approvals.toml"
        write_approvals(
            path,
            result_from([a_decision(1), a_decision(2)]),
            run_id="r1",
            config=config,
            package="vim",
            bugs={},
        )
        _approve_entry(path, 1)

        approvals = read_approvals(path)
        assert approvals.allows(1)
        assert not approvals.allows(2)
        assert approvals.unapproved == frozenset({2})

    def test_approve_all_flag_covers_listed_entries(
        self, tmp_path: Path, config: Config
    ) -> None:
        """--approve-all is a CLI flag, not a file setting: a bulk-approve switch
        stored in a file is too easy to set once and forget."""
        path = tmp_path / "approvals.toml"
        write_approvals(
            path,
            result_from([a_decision(1), a_decision(2)]),
            run_id="r1",
            config=config,
            package="vim",
            bugs={},
        )
        approvals = read_approvals(path)
        assert not approvals.allows(1)
        assert approvals.allows(1, approve_all=True)
        assert approvals.allows(2, approve_all=True)

    def test_approve_all_overrides_per_entry_flags(
        self, tmp_path: Path, config: Config
    ) -> None:
        path = tmp_path / "approvals.toml"
        write_approvals(
            path,
            result_from([a_decision(1), a_decision(2)]),
            run_id="r1",
            config=config,
            package="vim",
            bugs={},
        )
        approvals = read_approvals(path)
        assert approvals.allows(2, approve_all=True)

    def test_approve_all_cannot_action_an_unlisted_bug(
        self, tmp_path: Path, config: Config
    ) -> None:
        path = tmp_path / "approvals.toml"
        write_approvals(
            path,
            result_from([a_decision(1)]),
            run_id="r1",
            config=config,
            package="vim",
            bugs={},
        )
        assert not read_approvals(path).allows(999, approve_all=True)

    def test_run_id_recorded(self, tmp_path: Path, config: Config) -> None:
        """Binds an approvals file to the analysis it was reviewed against, so it
        cannot be replayed onto a different set of decisions."""
        path = tmp_path / "approvals.toml"
        write_approvals(
            path,
            result_from([a_decision(1)]),
            run_id="analyze-xyz",
            config=config,
            package="vim",
            bugs={},
        )
        assert read_approvals(path).run_id == "analyze-xyz"
        assert read_approvals(path).package == "vim"

    def test_only_actionable_decisions_listed(self, tmp_path: Path, config: Config) -> None:
        path = tmp_path / "approvals.toml"
        count = write_approvals(
            path,
            result_from([a_decision(1), a_decision(2, Action.KEEP)]),
            run_id="r1",
            config=config,
            package="vim",
            bugs={},
        )
        assert count == 1

    def test_quotes_in_titles_do_not_break_the_file(
        self, tmp_path: Path, config: Config
    ) -> None:
        path = tmp_path / "approvals.toml"
        bug = make_bug(bug_id=1, title='crash with "quotes" and \\ backslash')
        write_approvals(
            path,
            result_from([a_decision(1)]),
            run_id="r1",
            config=config,
            package="vim",
            bugs={1: bug},
        )
        assert read_approvals(path).unapproved == frozenset({1})
        assert read_approvals(path).listed == frozenset({1})


class TestReport:
    def test_markdown_contains_the_essentials(self, config: Config) -> None:
        bug = make_bug(bug_id=1, apport=ApportInfo(distro_release="14.04"))
        text = render_markdown(
            result_from([a_decision(1)]),
            run_id="r1",
            config=config,
            package="vim",
            bugs={1: bug},
            analyzer_model="ollama:qwen2.5:7b",
        )
        assert "# Backlog triage report: ubuntu/vim" in text
        assert "r1" in text
        assert "ollama:qwen2.5:7b" in text
        assert "#1" in text
        assert "eol_apport_release" in text
        assert "end of life" in text

    def test_spared_section_explains_why(self, config: Config) -> None:
        from pruner.models import Exclusion

        spared = Decision(
            bug_id=5,
            action=Action.KEEP,
            reason="protected: flagged security_related",
            rule_action=Action.NEEDS_INFO,
            rule_hits=(
                RuleHit(
                    rule="eol_apport_release",
                    action=Action.NEEDS_INFO,
                    claim=RuleClaim.LIFECYCLE,
                    reason="dead",
                ),
            ),
            exclusions=(Exclusion(rule="security", reason="flagged security_related"),),
            policy_branch="excluded",
        )
        text = render_markdown(
            result_from([spared]),
            run_id="r1",
            config=config,
            package="vim",
            bugs={5: make_bug(bug_id=5)},
            analyzer_model="none",
        )
        assert "Flagged but spared" in text
        assert "security_related" in text

    def test_pipe_in_text_does_not_break_tables(self, config: Config) -> None:
        spared = Decision(
            bug_id=5,
            action=Action.KEEP,
            reason="protected: tag a|b matched",
            rule_action=Action.NEEDS_INFO,
            rule_hits=(
                RuleHit(
                    rule="r",
                    action=Action.NEEDS_INFO,
                    claim=RuleClaim.LIFECYCLE,
                    reason="x",
                ),
            ),
            policy_branch="excluded",
        )
        text = render_markdown(
            result_from([spared]),
            run_id="r1",
            config=config,
            package="vim",
            bugs={5: make_bug(bug_id=5)},
            analyzer_model="none",
        )
        assert "a\\|b" in text

    def test_csv_has_a_row_per_decision(self) -> None:
        csv_text = render_csv(
            result_from([a_decision(1), a_decision(2, Action.KEEP)]),
            {1: make_bug(bug_id=1), 2: make_bug(bug_id=2)},
        )
        lines = [line for line in csv_text.splitlines() if line.strip()]
        assert len(lines) == 3  # header + 2
        assert "bug_id,action" in lines[0]

    def test_empty_result_renders(self, config: Config) -> None:
        text = render_markdown(
            result_from([]),
            run_id="r1",
            config=config,
            package="vim",
            bugs={},
            analyzer_model="none",
        )
        assert "No changes proposed." in text


class TestEndToEnd:
    """Full pipeline on synthetic snapshots with the LLM disabled."""

    def _bugs(self):
        return [
            # 1: classic EOL apport bug -> needs-info
            make_bug(
                bug_id=1,
                apport=ApportInfo(distro_release="14.04", package="vim"),
                description="vim segfaults when the swap directory is unwritable. " * 5,
            ),
            # 2: same, but confirmed on a supported release -> protected
            make_bug(
                bug_id=2,
                apport=ApportInfo(distro_release="14.04", package="vim"),
                description="vim segfaults on startup. " * 10,
                comment_texts=("Still happens on 24.04.",),
            ),
            # 3: security -> protected
            make_bug(
                bug_id=3,
                apport=ApportInfo(distro_release="14.04"),
                security_related=True,
            ),
            # 4: recently active -> protected
            make_bug(
                bug_id=4,
                apport=ApportInfo(distro_release="14.04"),
                quiet_days=5,
            ),
            # 5: obsolete series tag -> needs-info
            make_bug(bug_id=5, tags=("focal",), description="crash on exit. " * 20),
            # 6: on a supported release -> nothing fires
            make_bug(
                bug_id=6,
                apport=ApportInfo(distro_release="24.04"),
                description="crash on exit. " * 20,
            ),
        ]

    def _run(self, config: Config, series: SeriesTable, archive: ArchiveIndex):
        return analyze(
            self._bugs(),
            config=config,
            series=series,
            package="vim",
            analyzer=Analyzer(None, config.llm),
            archive=archive,
            now=NOW,
        )

    def test_expected_outcomes(
        self, config: Config, series: SeriesTable, archive: ArchiveIndex
    ) -> None:
        result = self._run(config, series, archive)
        actions = {d.bug_id: d.action for d in result.decisions}
        assert actions[1] is Action.NEEDS_INFO
        assert actions[2] is Action.KEEP
        assert actions[3] is Action.KEEP
        assert actions[4] is Action.KEEP
        assert actions[5] is Action.NEEDS_INFO
        assert actions[6] is Action.KEEP

    def test_protection_reasons_are_specific(
        self, config: Config, series: SeriesTable, archive: ArchiveIndex
    ) -> None:
        by_id = {d.bug_id: d for d in self._run(config, series, archive).decisions}
        assert "supported release" in by_id[2].reason
        assert "security" in by_id[3].reason
        assert "quiet period" in by_id[4].reason

    def test_stats_add_up(
        self, config: Config, series: SeriesTable, archive: ArchiveIndex
    ) -> None:
        stats = self._run(config, series, archive).stats
        assert stats.bugs == 6
        assert stats.actions.get("needs-info") == 2
        assert stats.actions.get("keep") == 4
        assert stats.eligible_before_llm == 2

    def test_llm_disabled_means_no_verdicts(
        self, config: Config, series: SeriesTable, archive: ArchiveIndex
    ) -> None:
        result = self._run(config, series, archive)
        assert all(d.verdict is None for d in result.decisions)
        assert result.stats.llm_calls == 0

    def test_llm_only_consulted_for_eligible_bugs(
        self, config: Config, series: SeriesTable, archive: ArchiveIndex
    ) -> None:
        """Cost control and a privacy property: protected bugs and bugs no rule
        flagged are never sent to the model at all."""
        import json

        from pruner.config import LlmConfig
        from pruner.llm.base import Provider

        seen: list[str] = []

        class Recorder(Provider):
            name = "rec"

            def complete(self, system: str, user: str, *, schema: dict) -> str:
                seen.append(user)
                return json.dumps(
                    {
                        "is_actually_a_bug": "unclear",
                        "bug_kind": "unclear",
                        "needs_more_info": True,
                        "reproducible_from_report": False,
                        "recommendation": "needs-info",
                        "confidence": 0.3,
                        "rationale": "thin",
                    }
                )

        llm_config = LlmConfig(max_attempts=1)
        analyze(
            self._bugs(),
            config=config,
            series=series,
            package="vim",
            analyzer=Analyzer(Recorder(llm_config), llm_config),
            archive=archive,
            now=NOW,
        )
        assert len(seen) == 2, "only the two rule-eligible bugs should be sent"
        assert all("Bug #1:" in s or "Bug #5:" in s for s in seen)
