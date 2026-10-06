"""Applying decisions, re-checking live state, and rolling back.

The tests that matter most here are the ones asserting we *decline* to act:
between ``analyze`` and ``apply`` a human may have triaged a bug, and clobbering
that is the failure mode most likely to make people distrust the tool.
"""

from __future__ import annotations

from pathlib import Path

from pruner.actions import apply_decisions, rollback_run
from pruner.audit import AuditLog
from pruner.comment import compose_comment
from pruner.config import Config
from pruner.lp.write import DryRunWriter
from pruner.models import Action, BugTaskStatus, Decision, RuleClaim, RuleHit
from tests.conftest import make_bug, make_task

TASK_LINK = "https://api.launchpad.net/devel/ubuntu/+source/vim/+bug/1/task"


def decision(
    bug_id: int = 1,
    action: Action = Action.NEEDS_INFO,
    *,
    rule: str = "eol_apport_release",
    claim: RuleClaim = RuleClaim.LIFECYCLE,
    reason: str = "it was reported against Ubuntu 14.04 (trusty), which reached end of life",
) -> Decision:
    return Decision(
        bug_id=bug_id,
        action=action,
        reason=reason,
        rule_action=action,
        rule_hits=(RuleHit(rule=rule, action=action, claim=claim, reason=reason),),
        policy_branch="rules_only",
    )


def bug_with_task(status: BugTaskStatus = BugTaskStatus.NEW, bug_id: int = 1):
    return make_bug(
        bug_id=bug_id,
        tasks=(make_task(status=status, link=TASK_LINK),),
    )


def setup(tmp_path: Path, status: BugTaskStatus = BugTaskStatus.NEW):
    bug = bug_with_task(status)
    writer = DryRunWriter({TASK_LINK: str(status)})
    audit = AuditLog(tmp_path / "audit.jsonl")
    return bug, writer, audit


class TestDryRun:
    def test_dry_run_makes_no_mutations(self, tmp_path: Path, config: Config) -> None:
        bug, writer, audit = setup(tmp_path)
        stats = apply_decisions(
            [decision()],
            {1: bug},
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=True,
        )
        assert stats.applied == 0
        assert writer.status_changes == []
        assert writer.comments == []
        assert writer.tags == []

    def test_dry_run_still_records_the_plan(self, tmp_path: Path, config: Config) -> None:
        bug, writer, audit = setup(tmp_path)
        apply_decisions(
            [decision()],
            {1: bug},
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=True,
        )
        records = audit.records()
        assert len(records) == 1
        assert records[0].dry_run
        assert records[0].outcome == "skipped"
        assert records[0].task_changes[0].new_status == "Incomplete"
        assert records[0].comment_body, "the exact comment should be previewable"

    def test_dry_run_records_are_not_rollback_candidates(
        self, tmp_path: Path, config: Config
    ) -> None:
        bug, writer, audit = setup(tmp_path)
        apply_decisions(
            [decision()],
            {1: bug},
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=True,
        )
        assert audit.applied_for_run("r1") == []


class TestApply:
    def test_needs_info_sets_incomplete_and_comments(
        self, tmp_path: Path, config: Config
    ) -> None:
        bug, writer, audit = setup(tmp_path)
        stats = apply_decisions(
            [decision()],
            {1: bug},
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=False,
        )
        assert stats.applied == 1
        assert writer.status_changes == [(TASK_LINK, "Incomplete")]
        assert len(writer.comments) == 1
        assert writer.tags == [(1, ["pruner-triaged"])]

    def test_invalid_sets_invalid(self, tmp_path: Path, config: Config) -> None:
        bug, writer, audit = setup(tmp_path)
        apply_decisions(
            [decision(action=Action.INVALID, rule="removed_from_archive")],
            {1: bug},
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=False,
        )
        assert writer.status_changes == [(TASK_LINK, "Invalid")]

    def test_wont_fix_sets_wont_fix(self, tmp_path: Path, config: Config) -> None:
        bug, writer, audit = setup(tmp_path)
        stats = apply_decisions(
            [decision(action=Action.WONT_FIX, reason="it is ancient")],
            {1: bug},
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=False,
        )
        assert stats.applied == 1
        assert writer.status_changes == [(TASK_LINK, "Won't Fix")]
        assert len(writer.comments) == 1

    def test_wont_fix_applies_from_incomplete(self, tmp_path: Path, config: Config) -> None:
        """An escalation decision against a task already at Incomplete still
        applies: Incomplete is actionable, Won't Fix is a different target."""
        bug, writer, audit = setup(tmp_path, BugTaskStatus.INCOMPLETE)
        stats = apply_decisions(
            [decision(action=Action.WONT_FIX, reason="it is ancient")],
            {1: bug},
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=False,
        )
        assert stats.applied == 1
        assert writer.status_changes == [(TASK_LINK, "Won't Fix")]

    def test_actor_is_recorded(self, tmp_path: Path, config: Config) -> None:
        """With a bot account in play, "who did this" must be in the audit log."""
        bug, writer, audit = setup(tmp_path)
        apply_decisions(
            [decision()],
            {1: bug},
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=False,
        )
        assert audit.records()[0].actor == writer.actor

    def test_comment_is_posted_before_status_change(
        self, tmp_path: Path, config: Config
    ) -> None:
        """An unexplained status change from a bot is the worst outcome, so the
        explanation goes first."""
        bug, _, audit = setup(tmp_path)
        order: list[str] = []

        class OrderRecorder(DryRunWriter):
            def add_comment(self, bug_id: int, body: str, *, subject: str = "") -> None:
                order.append("comment")
                super().add_comment(bug_id, body, subject=subject)

            def set_task_status(self, task_link: str, status: BugTaskStatus) -> None:
                order.append("status")
                super().set_task_status(task_link, status)

        apply_decisions(
            [decision()],
            {1: bug},
            config=config,
            package="vim",
            writer=OrderRecorder({TASK_LINK: "New"}),
            audit=audit,
            run_id="r1",
            dry_run=False,
        )
        assert order == ["comment", "status"]

    def test_audit_record_captures_previous_status(
        self, tmp_path: Path, config: Config
    ) -> None:
        bug, writer, audit = setup(tmp_path, BugTaskStatus.CONFIRMED)
        apply_decisions(
            [decision()],
            {1: bug},
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=False,
        )
        change = audit.records()[0].task_changes[0]
        assert change.previous_status == "Confirmed"
        assert change.new_status == "Incomplete"


class TestLiveStateRecheck:
    def test_human_triage_since_analysis_is_not_clobbered(
        self, tmp_path: Path, config: Config
    ) -> None:
        """Snapshot says New; Launchpad now says In Progress. Somebody picked it
        up, so we must leave it alone."""
        bug = bug_with_task(BugTaskStatus.NEW)
        writer = DryRunWriter({TASK_LINK: "In Progress"})
        audit = AuditLog(tmp_path / "audit.jsonl")
        stats = apply_decisions(
            [decision()],
            {1: bug},
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=False,
        )
        assert stats.applied == 0
        assert stats.skipped == 1
        assert writer.status_changes == []
        assert writer.comments == []
        assert "changed since analysis" in audit.records()[0].error

    def test_already_closed_is_skipped(self, tmp_path: Path, config: Config) -> None:
        bug = bug_with_task(BugTaskStatus.NEW)
        writer = DryRunWriter({TASK_LINK: "Won't Fix"})
        audit = AuditLog(tmp_path / "audit.jsonl")
        stats = apply_decisions(
            [decision()],
            {1: bug},
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=False,
        )
        assert stats.applied == 0
        assert writer.comments == []

    def test_already_in_target_status_is_skipped(
        self, tmp_path: Path, config: Config
    ) -> None:
        bug = bug_with_task(BugTaskStatus.INCOMPLETE)
        writer = DryRunWriter({TASK_LINK: "Incomplete"})
        audit = AuditLog(tmp_path / "audit.jsonl")
        stats = apply_decisions(
            [decision()],
            {1: bug},
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=False,
        )
        assert stats.applied == 0
        assert writer.comments == [], "must not comment when nothing changes"

    def test_unrecognised_live_status_is_skipped(
        self, tmp_path: Path, config: Config
    ) -> None:
        bug = bug_with_task(BugTaskStatus.NEW)
        writer = DryRunWriter({TASK_LINK: "Some Future Status"})
        audit = AuditLog(tmp_path / "audit.jsonl")
        stats = apply_decisions(
            [decision()],
            {1: bug},
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=False,
        )
        assert stats.applied == 0


class TestCircuitBreaker:
    def test_max_actions_per_run_enforced(self, tmp_path: Path) -> None:
        config = Config.model_validate(
            {"safety": {"max_actions_per_run": 2, "action_delay_seconds": 0}}
        )
        bugs = {}
        decisions = []
        statuses = {}
        for bug_id in range(1, 11):
            link = f"{TASK_LINK}/{bug_id}"
            bugs[bug_id] = make_bug(
                bug_id=bug_id, tasks=(make_task(status=BugTaskStatus.NEW, link=link),)
            )
            statuses[link] = "New"
            decisions.append(decision(bug_id=bug_id))

        writer = DryRunWriter(statuses)
        audit = AuditLog(tmp_path / "audit.jsonl")
        stats = apply_decisions(
            decisions,
            bugs,
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=False,
        )
        assert stats.applied == 2

    def test_limit_flag_tightens_further(self, tmp_path: Path) -> None:
        config = Config.model_validate(
            {"safety": {"max_actions_per_run": 50, "action_delay_seconds": 0}}
        )
        bugs, decisions, statuses = {}, [], {}
        for bug_id in range(1, 6):
            link = f"{TASK_LINK}/{bug_id}"
            bugs[bug_id] = make_bug(
                bug_id=bug_id, tasks=(make_task(status=BugTaskStatus.NEW, link=link),)
            )
            statuses[link] = "New"
            decisions.append(decision(bug_id=bug_id))
        stats = apply_decisions(
            decisions,
            bugs,
            config=config,
            package="vim",
            writer=DryRunWriter(statuses),
            audit=AuditLog(tmp_path / "audit.jsonl"),
            run_id="r1",
            dry_run=False,
            limit=1,
        )
        assert stats.applied == 1

    def test_keep_decisions_are_never_applied(self, tmp_path: Path, config: Config) -> None:
        bug, writer, audit = setup(tmp_path)
        stats = apply_decisions(
            [decision(action=Action.KEEP)],
            {1: bug},
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=False,
        )
        assert stats.applied == 0
        assert writer.comments == []


class TestRollback:
    def _applied_run(self, tmp_path: Path, config: Config):
        bug, writer, audit = setup(tmp_path)
        apply_decisions(
            [decision()],
            {1: bug},
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=False,
        )
        return audit

    def test_restores_previous_status(self, tmp_path: Path) -> None:
        config = Config.model_validate({"safety": {"action_delay_seconds": 0}})
        audit = self._applied_run(tmp_path, config)
        writer = DryRunWriter({TASK_LINK: "Incomplete"})
        stats = rollback_run(
            "r1",
            config=config,
            writer=writer,
            audit=audit,
            new_run_id="rb1",
            dry_run=False,
        )
        assert stats.applied == 1
        assert writer.status_changes == [(TASK_LINK, "New")]
        assert len(writer.comments) == 1

    def test_restores_from_wont_fix(self, tmp_path: Path) -> None:
        config = Config.model_validate({"safety": {"action_delay_seconds": 0}})
        bug, writer, audit = setup(tmp_path)
        apply_decisions(
            [decision(action=Action.WONT_FIX, reason="it is ancient")],
            {1: bug},
            config=config,
            package="vim",
            writer=writer,
            audit=audit,
            run_id="r1",
            dry_run=False,
        )
        reverter = DryRunWriter({TASK_LINK: "Won't Fix"})
        stats = rollback_run(
            "r1",
            config=config,
            writer=reverter,
            audit=audit,
            new_run_id="rb1",
            dry_run=False,
        )
        assert stats.applied == 1
        assert reverter.status_changes == [(TASK_LINK, "New")]

    def test_refuses_when_state_changed_since(self, tmp_path: Path) -> None:
        """If a human moved the bug on after our change, their state wins."""
        config = Config.model_validate({"safety": {"action_delay_seconds": 0}})
        audit = self._applied_run(tmp_path, config)
        writer = DryRunWriter({TASK_LINK: "Fix Released"})
        stats = rollback_run(
            "r1",
            config=config,
            writer=writer,
            audit=audit,
            new_run_id="rb1",
            dry_run=False,
        )
        assert stats.applied == 0
        assert stats.skipped == 1
        assert writer.status_changes == []

    def test_double_rollback_is_a_no_op(self, tmp_path: Path) -> None:
        config = Config.model_validate({"safety": {"action_delay_seconds": 0}})
        audit = self._applied_run(tmp_path, config)
        rollback_run(
            "r1",
            config=config,
            writer=DryRunWriter({TASK_LINK: "Incomplete"}),
            audit=audit,
            new_run_id="rb1",
            dry_run=False,
        )
        assert audit.applied_for_run("r1") == [], "already reverted"

    def test_dry_run_rollback_changes_nothing(self, tmp_path: Path) -> None:
        config = Config.model_validate({"safety": {"action_delay_seconds": 0}})
        audit = self._applied_run(tmp_path, config)
        writer = DryRunWriter({TASK_LINK: "Incomplete"})
        rollback_run(
            "r1",
            config=config,
            writer=writer,
            audit=audit,
            new_run_id="rb1",
            dry_run=True,
        )
        assert writer.status_changes == []
        assert audit.applied_for_run("r1"), "still rollback-able afterwards"


class TestComment:
    def test_needs_info_comment_content(self, config: Config) -> None:
        body = compose_comment(bug_with_task(), decision(), config, package="vim")
        assert "Incomplete" in body
        assert "14.04" in body, "must state the specific reason, not just 'automated'"
        assert "New" in body, "must say how to reopen"
        assert "60 days" in body
        assert config.comment.marker in body
        assert "eol_apport_release" in body, "must name the rule that fired"

    def test_invalid_comment_content(self, config: Config) -> None:
        body = compose_comment(
            bug_with_task(),
            decision(action=Action.INVALID, rule="removed_from_archive"),
            config,
            package="vim",
        )
        assert "Invalid" in body
        assert "set the status back to New" in body

    def test_wont_fix_comment_content(self, config: Config) -> None:
        body = compose_comment(
            bug_with_task(),
            decision(
                action=Action.WONT_FIX,
                reason="it was reported against Ubuntu 14.04 (trusty), which "
                "reached end of life, and it was reported 12 years ago",
            ),
            config,
            package="vim",
        )
        assert "Won't Fix" in body
        assert "12 years ago" in body, "must state the age, the point of the action"
        assert "not a judgement" in body, "must not imply the report was never a bug"
        assert "back to New" in body, "must say how to undo it"
        assert config.comment.marker in body

    def test_reclassified_comment_points_elsewhere(self, config: Config) -> None:
        from pruner.models import BugKind, IsABug, LlmVerdict

        reclassified = decision(action=Action.INVALID).model_copy(
            update={
                "llm_reclassified": True,
                "verdict": LlmVerdict(
                    is_actually_a_bug=IsABug.NO,
                    bug_kind=BugKind.SUPPORT_QUESTION,
                    confidence=0.9,
                ),
            }
        )
        body = compose_comment(bug_with_task(), reclassified, config, package="vim")
        assert "askubuntu.com" in body

    def test_marker_makes_comment_self_recognising(self, config: Config) -> None:
        """Posting a comment then re-analysing must trip ``already_triaged``."""
        from pruner.rules.exclusions import evaluate_exclusions
        from tests.conftest import NOW

        body = compose_comment(bug_with_task(), decision(), config, package="vim")
        bug = make_bug(comment_texts=(body,))
        fired = {
            e.rule for e in evaluate_exclusions(bug, config, _series_table(), "vim", now=NOW)
        }
        assert "already_triaged" in fired

    def test_uses_llm_missing_info_when_available(self, config: Config) -> None:
        from pruner.models import LlmVerdict

        with_verdict = decision().model_copy(
            update={
                "verdict": LlmVerdict(
                    missing_info=("the exact vim configuration used",),
                    confidence=0.5,
                )
            }
        )
        body = compose_comment(bug_with_task(), with_verdict, config, package="vim")
        assert "the exact vim configuration used" in body

    def test_falls_back_to_generic_asks(self, config: Config) -> None:
        body = compose_comment(bug_with_task(), decision(), config, package="vim")
        assert "the Ubuntu release you are seeing this on" in body


def _series_table():
    from pruner.lp.series import SeriesTable
    from tests.conftest import SERIES_ENTRIES

    return SeriesTable.from_api("ubuntu", SERIES_ENTRIES)
