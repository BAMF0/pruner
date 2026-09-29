"""Parsing of real recorded Launchpad payloads into snapshots.

These tests exist to pin down the API's actual quirks, each of which was verified
against live Launchpad while building the tool.
"""

from __future__ import annotations

from pruner.apport import parse_apport, prose_length, strip_apport
from pruner.fetcher import _comment_texts, _unique_bug_ids
from pruner.lp.series import SeriesTable
from pruner.models import BugSnapshot, BugTaskStatus, Importance
from tests.conftest import load_fixture


def _snapshot(bug_id: int) -> BugSnapshot:
    raw = load_fixture(f"bug_{bug_id}")
    tasks = load_fixture(f"bug_{bug_id}_bug_tasks")["entries"]
    attachments = load_fixture(f"bug_{bug_id}_attachments")["entries"]
    messages = load_fixture(f"bug_{bug_id}_messages")["entries"]
    watches = load_fixture(f"bug_{bug_id}_bug_watches")["entries"]
    return BugSnapshot.from_api(
        raw,
        tasks=tasks,
        attachment_count=len(attachments),
        patch_attachment_count=sum(1 for a in attachments if a.get("type") == "Patch"),
        comment_texts=_comment_texts(messages),
        remote_bug_statuses=tuple(
            str(w.get("remote_status") or "") for w in watches if w.get("remote_status")
        ),
        enriched=True,
    )


class TestApportBug:
    """Bug #1374898: apport metadata, trusty tag, and a Patch attachment."""

    def test_core_fields(self) -> None:
        bug = _snapshot(1374898)
        assert bug.id == 1374898
        assert "trusty" in bug.tags
        assert bug.comment_texts, "should have recovered comment bodies"

    def test_apport_metadata_parsed(self) -> None:
        bug = _snapshot(1374898)
        assert bug.apport.distro_release == "14.04"
        assert bug.apport.package == "vim-gtk"
        assert bug.apport.version == "2:7.4.052-1ubuntu3"
        assert bug.apport.problem_type == "Bug"

    def test_patch_detected_from_both_signals(self) -> None:
        """This bug reports ``latest_patch_uploaded`` *and* carries a ``Patch``
        attachment, so both detection paths should agree. ``has_patch`` accepts
        either, so an unset field alone cannot cause a patch to be missed."""
        bug = _snapshot(1374898)
        assert bug.latest_patch_uploaded is not None
        assert bug.patch_attachment_count == 1
        assert bug.has_patch

    def test_has_patch_from_attachment_alone(self) -> None:
        bug = _snapshot(1374898).model_copy(update={"latest_patch_uploaded": None})
        assert bug.has_patch, "a Patch attachment alone must be enough"

    def test_has_patch_false_when_neither_signal(self) -> None:
        bug = _snapshot(717691)
        assert bug.latest_patch_uploaded is None
        assert bug.patch_attachment_count == 0
        assert not bug.has_patch

    def test_non_ubuntu_task_is_not_a_distro_task(self) -> None:
        """This bug also has a bare ``debian`` task. It must not be mistaken for
        an Ubuntu task, or exclusions would read the wrong statuses."""
        bug = _snapshot(1374898)
        assert {t.target_name for t in bug.tasks} == {"vim (Ubuntu)", "debian"}
        assert {t.target_name for t in bug.distro_tasks("ubuntu")} == {"vim (Ubuntu)"}

    def test_apport_block_stripped_from_prose(self) -> None:
        bug = _snapshot(1374898)
        stripped = strip_apport(bug.description)
        assert "ProblemType:" not in stripped
        assert "DistroRelease:" not in stripped
        assert "InstallationMedia:" not in stripped
        # The human-written part survives.
        assert "log files" in stripped
        assert prose_length(bug.description) < len(bug.description)


class TestLegacyBug:
    """Bug #717691: pre-apport report with a ``Binary package hint:`` preamble."""

    def test_no_apport_metadata(self) -> None:
        bug = _snapshot(717691)
        assert bug.apport.is_empty

    def test_binary_package_hint_removed(self) -> None:
        bug = _snapshot(717691)
        assert "Binary package hint" in bug.description
        assert "Binary package hint" not in strip_apport(bug.description)

    def test_upstream_and_distro_tasks_separated(self) -> None:
        """This bug has both an upstream ``vim`` project task and ``vim (Ubuntu)``.
        Only the latter is ours to act on."""
        bug = _snapshot(717691)
        assert {t.target_name for t in bug.tasks} == {"vim", "vim (Ubuntu)"}

        distro = bug.distro_tasks("ubuntu")
        assert len(distro) == 1
        assert distro[0].package == "vim"
        assert distro[0].series is None, "a bare 'vim (Ubuntu)' task is not series-nominated"


class TestHighImportanceBug:
    def test_importance_parsed(self) -> None:
        bug = _snapshot(1509299)
        assert bug.tasks[0].importance is Importance.HIGH
        assert bug.tasks[0].status is BugTaskStatus.NEW


class TestTaskTargetParsing:
    def test_generic_distro_task(self) -> None:
        raw = {"bug_target_name": "vim (Ubuntu)", "status": "New"}
        task = __import__(
            "pruner.models", fromlist=["TaskSnapshot"]
        ).TaskSnapshot.from_api(raw)
        assert task.package == "vim"
        assert task.distribution == "ubuntu"
        assert task.series is None

    def test_series_nominated_task(self) -> None:
        from pruner.models import TaskSnapshot

        task = TaskSnapshot.from_api(
            {"bug_target_name": "vim (Ubuntu Jammy)", "status": "Incomplete"}
        )
        assert task.package == "vim"
        assert task.distribution == "ubuntu"
        assert task.series == "jammy"
        assert task.status is BugTaskStatus.INCOMPLETE

    def test_upstream_project_task(self) -> None:
        from pruner.models import TaskSnapshot

        task = TaskSnapshot.from_api({"bug_target_name": "vim", "status": "Unknown"})
        assert task.package == "vim"
        assert task.distribution is None
        assert task.series is None

    def test_unknown_status_does_not_raise(self) -> None:
        """Launchpad may grow new status values; that must not crash a run."""
        from pruner.models import TaskSnapshot

        task = TaskSnapshot.from_api(
            {"bug_target_name": "vim (Ubuntu)", "status": "Something New", "importance": "??"}
        )
        assert task.status is BugTaskStatus.UNKNOWN
        assert task.importance is Importance.UNDECIDED


class TestSeriesTable:
    def test_parses_recorded_payload(self) -> None:
        table = SeriesTable.from_api("ubuntu", load_fixture("ubuntu_series")["entries"])
        assert table.get("noble") is not None
        assert table.by_version("24.04") is not None
        assert table.by_version("24.04").name == "noble"

    def test_obsolete_and_live_partition(self, series: SeriesTable) -> None:
        assert series.is_obsolete("trusty")
        assert series.is_obsolete("focal")
        assert not series.is_obsolete("noble")
        assert "noble" in series.live_names
        assert "trusty" in series.obsolete_names

    def test_oldest_supported(self, series: SeriesTable) -> None:
        oldest = series.oldest_supported()
        assert oldest is not None
        assert oldest.name == "jammy"

    def test_resolve_accepts_codename_or_version(self, series: SeriesTable) -> None:
        assert series.resolve("noble") is series.resolve("24.04")

    def test_series_tags_ignores_unrelated_tags(self, series: SeriesTable) -> None:
        found = series.series_tags(("trusty", "amd64", "apport-bug", "noble"))
        assert {s.name for s in found} == {"trusty", "noble"}


class TestApportParser:
    def test_version_noise_stripped(self) -> None:
        info = parse_apport("Package: vim 2:8.0-1ubuntu1 [modified: usr/bin/vim]")
        assert info.version == "2:8.0-1ubuntu1"

    def test_not_installed_yields_no_version(self) -> None:
        info = parse_apport("ProblemType: Bug\nPackage: vim (not installed)")
        assert info.package == "vim"
        assert info.version is None

    def test_release_with_lts_suffix(self) -> None:
        info = parse_apport("DistroRelease: Ubuntu 14.04.1 LTS")
        assert info.distro_release == "14.04"

    def test_no_metadata_is_empty_not_an_error(self) -> None:
        assert parse_apport("just some prose").is_empty
        assert parse_apport("").is_empty

    def test_prose_only_description_is_unchanged(self) -> None:
        text = "Steps: open a file\nExpected: works\nActual: crashes"
        # "Steps:"/"Expected:" look like keys but are not apport anchors, so the
        # text must survive intact.
        assert strip_apport(text) == text


class TestFetcherHelpers:
    def test_message_zero_is_the_description(self) -> None:
        """``/messages`` returns the description as message 0; including it would
        double-count it in every text scan and in the LLM prompt."""
        messages = load_fixture("bug_1374898_messages")["entries"]
        comments = _comment_texts(messages)
        assert len(comments) < len(messages)
        description = load_fixture("bug_1374898")["description"]
        assert not any(c and c in description for c in comments)

    def test_empty_comments_dropped(self) -> None:
        assert _comment_texts([{"content": "desc"}, {"content": "  "}, {"content": "x"}]) == ("x",)

    def test_bug_ids_deduplicated(self) -> None:
        tasks = [
            {"bug_link": "https://api.launchpad.net/devel/bugs/42"},
            {"bug_link": "https://api.launchpad.net/devel/bugs/42"},
            {"bug_link": "https://api.launchpad.net/devel/bugs/7"},
            {"bug_link": "nonsense"},
        ]
        assert _unique_bug_ids(tasks) == [42, 7]
