"""Shared signal derivation used by several rules.

Kept in one place so that "which release does this bug actually concern?" is
answered identically by the safety exclusions and by the EOL rules. Getting those
two out of sync would be the most dangerous class of bug in this tool.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from fnmatch import fnmatch

from pruner.apport import strip_apport
from pruner.lp.series import Series, SeriesTable
from pruner.models import BugSnapshot, TaskSnapshot


def target_tasks(bug: BugSnapshot, distribution: str, package: str) -> tuple[TaskSnapshot, ...]:
    """Tasks for the package we are triaging, in the distribution we are triaging."""
    return tuple(
        t
        for t in bug.distro_tasks(distribution)
        if t.package is None or t.package == package
    )


def open_target_tasks(
    bug: BugSnapshot, distribution: str, package: str
) -> tuple[TaskSnapshot, ...]:
    open_ids = {t.self_link for t in bug.open_distro_tasks(distribution)}
    return tuple(t for t in target_tasks(bug, distribution, package) if t.self_link in open_ids)


def matches_any(value: str, patterns: tuple[str, ...]) -> bool:
    lowered = value.lower()
    return any(fnmatch(lowered, p.lower()) for p in patterns)


def matched_tags(tags: tuple[str, ...], patterns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(t for t in tags if matches_any(t, patterns))


def searchable_text(bug: BugSnapshot) -> str:
    """Description (apport block removed) plus all comment bodies.

    The apport block is stripped because it is full of version numbers from
    dependency listings that would otherwise read as release mentions.
    """
    return "\n".join((strip_apport(bug.description), *bug.comment_texts))


def _version_pattern(version: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\d.]){re.escape(version)}(?![\d])")


def _codename_pattern(name: str) -> re.Pattern[str]:
    return re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)


def mentions_series(text: str, series: Series) -> str | None:
    """Return the matched token if ``text`` appears to reference ``series``.

    Codenames are matched on a plain word boundary, which will occasionally fire on
    ordinary English ("a noble effort", "resolute"). That is deliberate: this
    function guards the safety exclusion, where a false positive merely means a bug
    is left for a human, while a false negative could mean closing a bug that
    somebody confirmed on a current release. The matched token is returned so the
    report can show *why* a bug was skipped and spurious hits stay visible.
    """
    if series.version and _version_pattern(series.version).search(text):
        return series.version
    if _codename_pattern(series.name).search(text):
        return series.name
    return None


def series_phrase(series: Iterable[Series]) -> str:
    """Render a list of series for a posted comment, accurately.

    Two things this gets right that a bare join would not:

    * **Precision about ESM.** Calling trusty flatly "end of life" is wrong
      enough to draw a correction from anyone on Ubuntu Pro, and being sloppy
      about it undermines trust in the rest of the message. Where we know the
      standard-support end date, we say exactly that instead.
    * **Grammar.** These phrases are interpolated after "because", so they have
      to read as a clause.
    """
    parts: list[str] = []
    for item in sorted(series, key=lambda s: s.name):
        if item.esm_only and item.standard_eol:
            parts.append(
                f"{item.label}, which left standard support on "
                f"{item.standard_eol:%-d %B %Y}"
            )
        elif item.standard_eol:
            parts.append(
                f"{item.label}, which reached end of life on "
                f"{item.standard_eol:%-d %B %Y}"
            )
        else:
            parts.append(f"{item.label}, which is end of life")
    if len(parts) == 1:
        return parts[0]
    return "; ".join(parts)


def live_release_evidence(bug: BugSnapshot, table: SeriesTable, package: str) -> tuple[str, ...]:
    """Evidence that the bug concerns a release that is still alive.

    Any evidence at all makes the bug ineligible for EOL-based pruning.
    """
    evidence: list[str] = []

    if release := bug.apport.distro_release:
        series = table.by_version(release)
        if series and not series.is_obsolete:
            evidence.append(f"apport DistroRelease {series.label}")

    for series in table.series_tags(bug.tags):
        if not series.is_obsolete:
            evidence.append(f"tag '{series.name}'")

    for task in open_target_tasks(bug, table.distribution, package):
        if task.series and not table.is_obsolete(task.series):
            evidence.append(f"open task on {task.series}")

    text = searchable_text(bug)
    if text:
        for series in table.live:
            if token := mentions_series(text, series):
                evidence.append(f"text mentions '{token}'")

    # Preserve order but drop duplicates.
    return tuple(dict.fromkeys(evidence))


def obsolete_release_evidence(
    bug: BugSnapshot, table: SeriesTable, package: str
) -> tuple[Series, ...]:
    """Obsolete series this bug is positively associated with."""
    found: dict[str, Series] = {}

    if release := bug.apport.distro_release:
        series = table.by_version(release)
        if series and series.is_obsolete:
            found[series.name] = series

    for series in table.series_tags(bug.tags):
        if series.is_obsolete:
            found[series.name] = series

    for task in target_tasks(bug, table.distribution, package):
        if task.series and (series := table.get(task.series)) and series.is_obsolete:
            found[series.name] = series

    return tuple(found.values())
