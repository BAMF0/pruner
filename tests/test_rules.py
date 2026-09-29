"""Prune rules.

Rules are pure functions, so every test here is a direct statement of the form
"given this bug, this rule fires (or does not) for this reason".
"""

from __future__ import annotations

from pruner.config import Config
from pruner.lp.archive import ArchiveIndex
from pruner.lp.series import SeriesTable
from pruner.models import Action, ApportInfo, BugTaskStatus, RuleClaim
from pruner.rules import RuleContext, evaluate_rules
from pruner.rules.base import get_rule
from tests.conftest import NOW, make_bug, make_task


def ctx(
    config: Config,
    series: SeriesTable,
    *,
    archive: ArchiveIndex | None = None,
    package: str = "vim",
) -> RuleContext:
    return RuleContext(
        config=config, series=series, package=package, archive=archive, now=NOW
    )


def run(rule: str, bug, context: RuleContext):
    return get_rule(rule)(bug, context)


class TestEolSeriesTasks:
    def test_all_tasks_on_obsolete_series(self, config: Config, series: SeriesTable) -> None:
        bug = make_bug(
            tasks=(
                make_task(target="vim (Ubuntu Trusty)", series_name="trusty"),
                make_task(target="vim (Ubuntu Focal)", series_name="focal"),
            )
        )
        hit = run("eol_series_tasks", bug, ctx(config, series))
        assert hit is not None
        assert hit.action is Action.NEEDS_INFO
        assert hit.claim is RuleClaim.LIFECYCLE
        assert "trusty" in hit.evidence["series"]
        assert "focal" in hit.evidence["series"]

    def test_one_live_series_task_blocks(self, config: Config, series: SeriesTable) -> None:
        bug = make_bug(
            tasks=(
                make_task(target="vim (Ubuntu Trusty)", series_name="trusty"),
                make_task(target="vim (Ubuntu Noble)", series_name="noble"),
            )
        )
        assert run("eol_series_tasks", bug, ctx(config, series)) is None

    def test_bare_distro_task_is_not_eol_evidence(
        self, config: Config, series: SeriesTable
    ) -> None:
        """A plain ``vim (Ubuntu)`` task means the development release, not an
        obsolete one, so it must never be read as EOL evidence."""
        bug = make_bug(tasks=(make_task(),))
        assert run("eol_series_tasks", bug, ctx(config, series)) is None

    def test_mixed_bare_and_obsolete_tasks_does_not_fire(
        self, config: Config, series: SeriesTable
    ) -> None:
        bug = make_bug(
            tasks=(
                make_task(),
                make_task(target="vim (Ubuntu Trusty)", series_name="trusty"),
            )
        )
        assert run("eol_series_tasks", bug, ctx(config, series)) is None

    def test_closed_obsolete_tasks_are_ignored(
        self, config: Config, series: SeriesTable
    ) -> None:
        bug = make_bug(
            tasks=(
                make_task(
                    target="vim (Ubuntu Trusty)",
                    series_name="trusty",
                    status=BugTaskStatus.WONT_FIX,
                ),
            )
        )
        assert run("eol_series_tasks", bug, ctx(config, series)) is None


class TestEolSeriesTag:
    def test_obsolete_tag_only(self, config: Config, series: SeriesTable) -> None:
        bug = make_bug(tags=("trusty", "amd64"))
        hit = run("eol_series_tag", bug, ctx(config, series))
        assert hit is not None
        assert hit.evidence["tags"] == "trusty"

    def test_live_tag_blocks(self, config: Config, series: SeriesTable) -> None:
        bug = make_bug(tags=("trusty", "noble"))
        assert run("eol_series_tag", bug, ctx(config, series)) is None

    def test_no_series_tags_does_not_fire(self, config: Config, series: SeriesTable) -> None:
        bug = make_bug(tags=("amd64", "apport-bug"))
        assert run("eol_series_tag", bug, ctx(config, series)) is None

    def test_defers_to_series_task_rule(self, config: Config, series: SeriesTable) -> None:
        """Avoids reporting the same conclusion twice with weaker evidence."""
        bug = make_bug(
            tags=("trusty",),
            tasks=(make_task(target="vim (Ubuntu Trusty)", series_name="trusty"),),
        )
        assert run("eol_series_tag", bug, ctx(config, series)) is None
        assert run("eol_series_tasks", bug, ctx(config, series)) is not None


class TestEolApportRelease:
    def test_obsolete_apport_release(self, config: Config, series: SeriesTable) -> None:
        bug = make_bug(apport=ApportInfo(distro_release="14.04", package="vim"))
        hit = run("eol_apport_release", bug, ctx(config, series))
        assert hit is not None
        assert hit.evidence == {"apport_release": "14.04", "series": "trusty"}

    def test_supported_apport_release_does_not_fire(
        self, config: Config, series: SeriesTable
    ) -> None:
        bug = make_bug(apport=ApportInfo(distro_release="24.04", package="vim"))
        assert run("eol_apport_release", bug, ctx(config, series)) is None

    def test_unknown_release_does_not_fire(self, config: Config, series: SeriesTable) -> None:
        """An unrecognised release must mean "no evidence", never "assume dead"."""
        bug = make_bug(apport=ApportInfo(distro_release="99.04", package="vim"))
        assert run("eol_apport_release", bug, ctx(config, series)) is None

    def test_no_apport_data_does_not_fire(self, config: Config, series: SeriesTable) -> None:
        assert run("eol_apport_release", make_bug(), ctx(config, series)) is None

    def test_live_release_comment_blocks(self, config: Config, series: SeriesTable) -> None:
        bug = make_bug(
            apport=ApportInfo(distro_release="14.04", package="vim"),
            comment_texts=("Confirmed on 24.04 too.",),
        )
        assert run("eol_apport_release", bug, ctx(config, series)) is None


class TestLikelyFixed:
    def test_version_predating_everything_shipped(
        self, config: Config, series: SeriesTable, archive: ArchiveIndex
    ) -> None:
        bug = make_bug(apport=ApportInfo(distro_release="14.04", version="2:7.4.052-1ubuntu3"))
        hit = run("likely_fixed", bug, ctx(config, series, archive=archive))
        assert hit is not None
        assert hit.evidence["signal"] == "version_drift"
        assert hit.evidence["reported_version"] == "2:7.4.052-1ubuntu3"

    def test_current_version_does_not_fire(
        self, config: Config, series: SeriesTable, archive: ArchiveIndex
    ) -> None:
        bug = make_bug(apport=ApportInfo(version="2:9.1.1600-1ubuntu1"))
        assert run("likely_fixed", bug, ctx(config, series, archive=archive)) is None

    def test_epoch_aware_comparison(
        self, config: Config, series: SeriesTable, archive: ArchiveIndex
    ) -> None:
        """A naive string compare would call ``2:10.0`` older than ``2:9.1``."""
        bug = make_bug(apport=ApportInfo(version="2:10.0.0-1ubuntu1"))
        assert run("likely_fixed", bug, ctx(config, series, archive=archive)) is None

    def test_no_archive_data_does_not_fire(self, config: Config, series: SeriesTable) -> None:
        bug = make_bug(apport=ApportInfo(version="2:7.4.052-1ubuntu3"))
        assert run("likely_fixed", bug, ctx(config, series, archive=None)) is None

    def test_incomplete_archive_data_does_not_fire(
        self, config: Config, series: SeriesTable, archive: ArchiveIndex
    ) -> None:
        """Partial lookups must never drive a conclusion."""
        broken = archive.model_copy(update={"incomplete": True})
        bug = make_bug(apport=ApportInfo(version="2:7.4.052-1ubuntu3"))
        assert run("likely_fixed", bug, ctx(config, series, archive=broken)) is None

    def test_upstream_resolved_status(
        self, config: Config, series: SeriesTable, archive: ArchiveIndex
    ) -> None:
        bug = make_bug(remote_bug_statuses=("RESOLVED FIXED",))
        hit = run("likely_fixed", bug, ctx(config, series, archive=archive))
        assert hit is not None
        assert hit.evidence["signal"] == "upstream_status"

    def test_upstream_open_status_does_not_fire(
        self, config: Config, series: SeriesTable, archive: ArchiveIndex
    ) -> None:
        bug = make_bug(remote_bug_statuses=("NEW", "CONFIRMED"))
        assert run("likely_fixed", bug, ctx(config, series, archive=archive)) is None


class TestRemovedFromArchive:
    def test_no_publications_anywhere(self, config: Config, series: SeriesTable) -> None:
        empty = ArchiveIndex(
            distribution="ubuntu",
            package="gone",
            publications=(),
            queried_series=("noble", "jammy"),
        )
        hit = run("removed_from_archive", make_bug(), ctx(config, series, archive=empty))
        assert hit is not None
        assert hit.action is Action.INVALID
        assert hit.claim is RuleClaim.EXISTENCE

    def test_still_published_does_not_fire(
        self, config: Config, series: SeriesTable, archive: ArchiveIndex
    ) -> None:
        assert run("removed_from_archive", make_bug(), ctx(config, series, archive=archive)) is None

    def test_failed_lookup_never_proposes_invalid(
        self, config: Config, series: SeriesTable
    ) -> None:
        """"Package not found" is also what a failed API call looks like. Getting
        this wrong would close an entire backlog."""
        broken = ArchiveIndex(
            distribution="ubuntu",
            package="vim",
            publications=(),
            queried_series=("noble",),
            incomplete=True,
        )
        assert run("removed_from_archive", make_bug(), ctx(config, series, archive=broken)) is None

    def test_unqueried_never_proposes_invalid(
        self, config: Config, series: SeriesTable
    ) -> None:
        never = ArchiveIndex(distribution="ubuntu", package="vim", queried_series=())
        assert run("removed_from_archive", make_bug(), ctx(config, series, archive=never)) is None


class TestEmptyReport:
    def test_thin_report(self, config: Config, series: SeriesTable) -> None:
        bug = make_bug(description="it broke")
        hit = run("empty_report", bug, ctx(config, series))
        assert hit is not None
        assert hit.claim is RuleClaim.QUALITY, "the LLM must be able to veto this"

    def test_apport_boilerplate_does_not_count_as_content(
        self, config: Config, series: SeriesTable
    ) -> None:
        """A contentless report can still be kilobytes of machine metadata."""
        description = "broken\n\n" + "ProblemType: Bug\nDistroRelease: Ubuntu 14.04\n" + (
            "SomeKey: " + "x" * 500 + "\n"
        )
        bug = make_bug(description=description)
        assert run("empty_report", bug, ctx(config, series)) is not None

    def test_long_report_does_not_fire(self, config: Config, series: SeriesTable) -> None:
        assert run("empty_report", make_bug(description="y" * 500), ctx(config, series)) is None

    def test_discussion_means_there_is_something_here(
        self, config: Config, series: SeriesTable
    ) -> None:
        bug = make_bug(description="it broke", message_count=4)
        assert run("empty_report", bug, ctx(config, series)) is None

    def test_attachment_means_there_is_something_here(
        self, config: Config, series: SeriesTable
    ) -> None:
        bug = make_bug(description="it broke", attachment_count=1)
        assert run("empty_report", bug, ctx(config, series)) is None

    def test_other_affected_users_mean_something_here(
        self, config: Config, series: SeriesTable
    ) -> None:
        bug = make_bug(description="it broke", users_affected_count=2)
        assert run("empty_report", bug, ctx(config, series)) is None


class TestRuleRegistry:
    def test_only_enabled_rules_run(self, config: Config, series: SeriesTable) -> None:
        bug = make_bug(tags=("trusty",))
        assert evaluate_rules(bug, ctx(config, series), enabled=[]) == ()
        assert evaluate_rules(bug, ctx(config, series), enabled=["eol_series_tag"])

    def test_unknown_rule_name_is_ignored_not_fatal(
        self, config: Config, series: SeriesTable
    ) -> None:
        assert evaluate_rules(make_bug(), ctx(config, series), enabled=["nope"]) == ()

    def test_config_rejects_unknown_rule_names(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="unknown rule"):
            Config.model_validate({"rules": {"enabled": ["not_a_real_rule"]}})

    def test_every_enabled_default_rule_exists(self, config: Config) -> None:
        from pruner.rules import all_rule_names

        assert set(config.rules.enabled) <= all_rule_names()
