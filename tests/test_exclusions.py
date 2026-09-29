"""Hard exclusions.

Each test isolates one protection by starting from a bug that would otherwise be
pruned and changing only the field under test.
"""

from __future__ import annotations

import pytest

from pruner.config import Config
from pruner.lp.series import SeriesTable
from pruner.models import ApportInfo, BugTaskStatus, Importance
from pruner.rules.exclusions import (
    PREFILTER_EXCLUSIONS,
    evaluate_exclusions,
    prefilter,
)
from tests.conftest import NOW, make_bug, make_task


def fired(bug, config: Config, series: SeriesTable, package: str = "vim") -> set[str]:
    return {e.rule for e in evaluate_exclusions(bug, config, series, package, now=NOW)}


# A baseline EOL-ish bug that nothing should protect.
def baseline(**kwargs):
    kwargs.setdefault("apport", ApportInfo(distro_release="14.04", package="vim"))
    return make_bug(**kwargs)


class TestBaselineIsNotProtected:
    def test_no_exclusions_on_a_plain_stale_eol_bug(
        self, config: Config, series: SeriesTable
    ) -> None:
        assert fired(baseline(), config, series) == set()


class TestSafetyCritical:
    def test_private_bug_protected(self, config: Config, series: SeriesTable) -> None:
        bug = baseline(private=True, information_type="Private")
        assert "private" in fired(bug, config, series)

    def test_security_related_protected(self, config: Config, series: SeriesTable) -> None:
        assert "security" in fired(baseline(security_related=True), config, series)

    def test_linked_cve_protected(self, config: Config, series: SeriesTable) -> None:
        assert "security" in fired(baseline(cve_count=1), config, series)

    def test_public_security_information_type_protected(
        self, config: Config, series: SeriesTable
    ) -> None:
        assert "security" in fired(baseline(information_type="Public Security"), config, series)

    def test_vulnerability_record_protected(self, config: Config, series: SeriesTable) -> None:
        assert "security" in fired(baseline(vulnerability_count=1), config, series)

    def test_patch_attachment_protected(self, config: Config, series: SeriesTable) -> None:
        assert "has_patch" in fired(baseline(patch_attachment_count=1), config, series)

    def test_merge_proposal_protected(self, config: Config, series: SeriesTable) -> None:
        assert "dev_activity" in fired(baseline(linked_mp_count=1), config, series)

    def test_linked_branch_protected(self, config: Config, series: SeriesTable) -> None:
        assert "dev_activity" in fired(baseline(linked_branch_count=1), config, series)

    def test_assigned_protected(self, config: Config, series: SeriesTable) -> None:
        bug = baseline(tasks=(make_task(assignee="someone"),))
        assert "assigned" in fired(bug, config, series)

    def test_milestoned_protected(self, config: Config, series: SeriesTable) -> None:
        bug = baseline(tasks=(make_task(milestone="ubuntu-26.10"),))
        assert "milestoned" in fired(bug, config, series)

    @pytest.mark.parametrize("importance", [Importance.CRITICAL, Importance.HIGH])
    def test_high_importance_protected(
        self, config: Config, series: SeriesTable, importance: Importance
    ) -> None:
        bug = baseline(tasks=(make_task(importance=importance),))
        assert "protected_importance" in fired(bug, config, series)

    def test_low_importance_not_protected(self, config: Config, series: SeriesTable) -> None:
        bug = baseline(tasks=(make_task(importance=Importance.LOW),))
        assert "protected_importance" not in fired(bug, config, series)

    @pytest.mark.parametrize(
        "status",
        [BugTaskStatus.IN_PROGRESS, BugTaskStatus.FIX_COMMITTED, BugTaskStatus.FIX_RELEASED],
    )
    def test_work_in_progress_protected(
        self, config: Config, series: SeriesTable, status: BugTaskStatus
    ) -> None:
        bug = baseline(tasks=(make_task(status=status),))
        assert "progressing" in fired(bug, config, series)

    def test_duplicate_protected(self, config: Config, series: SeriesTable) -> None:
        assert "duplicate" in fired(baseline(duplicate_of=999), config, series)

    def test_popular_by_affected_users(self, config: Config, series: SeriesTable) -> None:
        assert "popular" in fired(baseline(users_affected_count=5), config, series)

    def test_popular_by_duplicates(self, config: Config, series: SeriesTable) -> None:
        assert "popular" in fired(baseline(number_of_duplicates=3), config, series)

    def test_just_below_popularity_threshold_not_protected(
        self, config: Config, series: SeriesTable
    ) -> None:
        assert "popular" not in fired(baseline(users_affected_count=4), config, series)

    @pytest.mark.parametrize(
        "tag", ["regression-release", "rls-nn-incoming", "block-proposed-noble", "patch"]
    )
    def test_protected_tags(self, config: Config, series: SeriesTable, tag: str) -> None:
        assert "protected_tag" in fired(baseline(tags=(tag,)), config, series)

    def test_unprotected_tag_is_fine(self, config: Config, series: SeriesTable) -> None:
        assert "protected_tag" not in fired(baseline(tags=("apport-bug",)), config, series)

    def test_recent_activity_protected(self, config: Config, series: SeriesTable) -> None:
        assert "recently_active" in fired(baseline(quiet_days=10), config, series)

    def test_quiet_period_boundary(self, config: Config, series: SeriesTable) -> None:
        assert "recently_active" not in fired(baseline(quiet_days=181), config, series)
        assert "recently_active" in fired(baseline(quiet_days=179), config, series)

    def test_no_open_task_protected(self, config: Config, series: SeriesTable) -> None:
        bug = baseline(tasks=(make_task(status=BugTaskStatus.WONT_FIX),))
        assert "no_open_task" in fired(bug, config, series)

    def test_other_package_task_is_not_ours(self, config: Config, series: SeriesTable) -> None:
        bug = baseline(tasks=(make_task(target="nano (Ubuntu)", package="nano"),))
        assert "no_open_task" in fired(bug, config, series)


class TestIncompleteSnapshot:
    def test_unenriched_snapshot_is_never_actionable(
        self, config: Config, series: SeriesTable
    ) -> None:
        """The fetch-time prefilter is only safe because an un-enriched snapshot is
        treated as untouchable. Without this, a bug whose CVE/patch counts were
        never fetched would look clean."""
        bug = baseline(enriched=False, prefilter_reason="recently_active: too fresh")
        assert "incomplete_snapshot" in fired(bug, config, series)

    def test_reason_is_surfaced(self, config: Config, series: SeriesTable) -> None:
        bug = baseline(enriched=False, prefilter_reason="security: flagged")
        found = evaluate_exclusions(bug, config, series, "vim", now=NOW)
        assert any("security: flagged" in e.reason for e in found)


class TestAffectsLiveRelease:
    """The single most important exclusion."""

    def test_open_task_on_supported_series(self, config: Config, series: SeriesTable) -> None:
        bug = baseline(
            tasks=(make_task(target="vim (Ubuntu Noble)", series_name="noble"),),
        )
        assert "affects_live_release" in fired(bug, config, series)

    def test_supported_series_tag(self, config: Config, series: SeriesTable) -> None:
        assert "affects_live_release" in fired(baseline(tags=("noble",)), config, series)

    def test_apport_release_still_supported(self, config: Config, series: SeriesTable) -> None:
        bug = make_bug(apport=ApportInfo(distro_release="24.04", package="vim"))
        assert "affects_live_release" in fired(bug, config, series)

    def test_comment_confirming_on_current_release(
        self, config: Config, series: SeriesTable
    ) -> None:
        """The scenario that matters most: an EOL-filed bug that somebody later
        confirmed still happens on a supported release."""
        bug = baseline(comment_texts=("I still see this on Ubuntu 24.04, FWIW.",))
        assert "affects_live_release" in fired(bug, config, series)

    def test_comment_mentioning_codename(self, config: Config, series: SeriesTable) -> None:
        bug = baseline(comment_texts=("Still reproducible on noble.",))
        assert "affects_live_release" in fired(bug, config, series)

    def test_obsolete_release_mention_does_not_protect(
        self, config: Config, series: SeriesTable
    ) -> None:
        bug = baseline(comment_texts=("Also happens on 14.04 and focal.",))
        assert "affects_live_release" not in fired(bug, config, series)

    def test_version_match_is_not_a_substring_match(
        self, config: Config, series: SeriesTable
    ) -> None:
        """'124.04' or '24.043' must not read as Ubuntu 24.04."""
        bug = baseline(comment_texts=("The value was 124.045 in my config.",))
        assert "affects_live_release" not in fired(bug, config, series)

    def test_apport_metadata_versions_do_not_trigger(
        self, config: Config, series: SeriesTable
    ) -> None:
        """Apport dependency listings are full of version numbers. Scanning them
        would protect nearly every bug and make the EOL rules useless."""
        description = (
            "vim crashes on startup when the config is large.\n\n"
            "ProblemType: Bug\n"
            "DistroRelease: Ubuntu 14.04\n"
            "Package: vim 2:7.4.052\n"
            "Dependencies:\n"
            " libc6 2.24.04\n"
            " libtinfo5 22.04-1\n"
        )
        bug = make_bug(
            description=description,
            apport=ApportInfo(distro_release="14.04", package="vim"),
        )
        assert "affects_live_release" not in fired(bug, config, series)


class TestAlreadyTriaged:
    def test_our_own_marker_prevents_a_second_pass(
        self, config: Config, series: SeriesTable
    ) -> None:
        bug = baseline(
            comment_texts=(f"Setting to Incomplete.\n--\n{config.comment.marker}",),
        )
        assert "already_triaged" in fired(bug, config, series)


class TestPrefilter:
    def test_prefilter_only_uses_phase_one_fields(self) -> None:
        """Every prefilter exclusion must be computable from the bug entry plus
        tasks alone, or ``fetch`` would be deciding on data it has not fetched."""
        needs_enrichment = {
            "incomplete_snapshot",
            "dev_activity",
            "affects_live_release",
            "already_triaged",
        }
        assert not (PREFILTER_EXCLUSIONS & needs_enrichment)

    def test_prefilter_catches_recent_activity(
        self, config: Config, series: SeriesTable
    ) -> None:
        skip = prefilter(baseline(quiet_days=5), config, series, "vim", now=NOW)
        assert skip is not None
        assert skip.rule == "recently_active"

    def test_prefilter_passes_a_candidate(self, config: Config, series: SeriesTable) -> None:
        assert prefilter(baseline(), config, series, "vim", now=NOW) is None


class TestConfigurableSafety:
    def test_relaxing_importance_protection(self, series: SeriesTable) -> None:
        config = Config.model_validate({"safety": {"protect_importances": []}})
        bug = baseline(tasks=(make_task(importance=Importance.CRITICAL),))
        assert "protected_importance" not in fired(bug, config, series)

    def test_tightening_quiet_period(self, series: SeriesTable) -> None:
        config = Config.model_validate({"safety": {"min_quiet_days": 4000}})
        assert "recently_active" in fired(baseline(), config, series)
