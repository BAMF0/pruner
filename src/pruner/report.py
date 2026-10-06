"""Report and approvals-file generation.

The review gate. ``analyze`` writes two artefacts:

* a **Markdown report** meant to be read (and shared with other triagers),
* an **approvals TOML file** meant to be edited.

``apply`` acts only on bugs marked approved. Flipping the default from "approved"
to "not approved" is a one-line config change in the file itself, and the file
records the run ID so an approvals file can never be applied against a different
analysis than the one it was reviewed against.
"""

from __future__ import annotations

import tomllib
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from pruner.analysis import AnalysisResult
from pruner.config import Config
from pruner.models import Action, BugSnapshot, Decision

# ---------------------------------------------------------------------------
# Approvals
# ---------------------------------------------------------------------------


class Approvals(BaseModel):
    run_id: str
    package: str
    distribution: str
    approved: frozenset[int]
    listed: frozenset[int]
    """Every bug in the file, approved or not."""

    def allows(self, bug_id: int, *, approve_all: bool = False) -> bool:
        """Whether ``bug_id`` may be actioned.

        ``approve_all`` comes from an explicit ``--approve-all`` flag on ``apply``
        rather than from the file: a bulk-approve switch stored in a file is too
        easy to set once and forget, whereas a flag is a deliberate act each run.

        It overrides the per-entry flag for everything *listed in this file*, and
        nothing else -- a bug absent from the file can never be actioned. If you
        want to reject specific entries, either leave ``--approve-all`` off and
        approve individually, or delete those entries from the file.
        """
        if bug_id in self.approved:
            return True
        return approve_all and bug_id in self.listed

    @property
    def unapproved(self) -> frozenset[int]:
        return self.listed - self.approved


def write_approvals(
    path: Path,
    result: AnalysisResult,
    *,
    run_id: str,
    config: Config,
    package: str,
    bugs: dict[int, BugSnapshot],
    default_approved: bool = False,
) -> int:
    """Write the editable approvals file. Returns the number of entries."""
    actionable = result.actionable
    lines: list[str] = [
        "# pruner approvals",
        "#",
        "# Review each entry below and flip its approve flag to true for the changes",
        "# you want applied. `pruner apply` ignores everything else, so an unreviewed",
        "# file results in no changes at all.",
        "#",
        "# If you have read the report and want every entry here, pass --approve-all",
        "# to `pruner apply` instead of editing each one. To reject specific entries",
        "# while using that flag, delete them from this file.",
        "",
        "[meta]",
        f'run_id = "{run_id}"',
        f'package = "{package}"',
        f'distribution = "{config.launchpad.distribution}"',
        f'generated = "{datetime.now(UTC).isoformat()}"',
        "",
    ]

    for decision in actionable:
        bug = bugs.get(decision.bug_id)
        title = (bug.title if bug else "").replace("\n", " ").strip()
        lines.extend(
            [
                "[[bug]]",
                f"id = {decision.bug_id}",
                f"action = {_toml_str(str(decision.action))}",
                f"title = {_toml_str(title)}",
                f"url = {_toml_str(bug.web_link if bug else '')}",
                f"rules = {_toml_str(', '.join(decision.rule_names))}",
                f"reason = {_toml_str(decision.reason)}",
                f"approve = {'true' if default_approved else 'false'}",
                "",
            ]
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return len(actionable)


def read_approvals(path: Path) -> Approvals:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)

    meta = raw.get("meta") or {}
    entries = raw.get("bug") or []

    approved: set[int] = set()
    listed: set[int] = set()
    for entry in entries:
        bug_id = entry.get("id")
        if not isinstance(bug_id, int):
            continue
        listed.add(bug_id)
        if entry.get("approve") is True:
            approved.add(bug_id)

    return Approvals(
        run_id=str(meta.get("run_id") or ""),
        package=str(meta.get("package") or ""),
        distribution=str(meta.get("distribution") or ""),
        approved=frozenset(approved),
        listed=frozenset(listed),
    )


def _toml_str(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    escaped = escaped.replace("\n", " ").replace("\r", " ")
    return f'"{escaped}"'


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def render_markdown(
    result: AnalysisResult,
    *,
    run_id: str,
    config: Config,
    package: str,
    bugs: dict[int, BugSnapshot],
    analyzer_model: str,
    include_kept: bool = True,
) -> str:
    stats = result.stats
    out: list[str] = [
        f"# Backlog triage report: {config.launchpad.distribution}/{package}",
        "",
        f"- Run ID: `{run_id}`",
        f"- Generated: {datetime.now(UTC).strftime('%Y-%m-%d %H:%M UTC')}",
        f"- Launchpad service: `{config.launchpad.service}`",
        f"- LLM: `{analyzer_model}`",
        f"- Bugs analysed: **{stats.bugs}**",
        "",
        "## Summary",
        "",
        "| Outcome | Bugs |",
        "| --- | --- |",
    ]

    for action in (
        Action.NEEDS_INFO,
        Action.INVALID,
        Action.WONT_FIX,
        Action.ESCALATE,
        Action.KEEP,
    ):
        out.append(f"| {action.value} | {stats.actions.get(action.value, 0)} |")

    out.extend(
        [
            "",
            f"Rules made **{stats.eligible_before_llm}** bug(s) eligible for action. "
            f"The LLM then vetoed **{stats.llm_vetoes}** and reclassified "
            f"**{stats.llm_reclassifications}**. Age escalation hardened "
            f"**{stats.age_escalations}** to Won't Fix.",
            "",
        ]
    )

    if stats.llm_failures:
        out.extend(
            [
                f"> {stats.llm_failures} LLM call(s) failed or returned unusable output. "
                "Those bugs were decided on rules alone (the model's silence is never "
                "treated as agreement, but it also cannot veto).",
                "",
            ]
        )

    out.extend(_counter_section("Rule hits", stats.rule_hits))
    out.extend(_counter_section("Exclusions (bugs protected)", stats.exclusions))
    out.extend(_counter_section("Policy branches", stats.policy_branches))

    proposed = result.actionable
    out.extend(["## Proposed changes", ""])
    if not proposed:
        out.append("No changes proposed.")
        out.append("")
    else:
        out.append(f"{len(proposed)} bug(s) proposed for action. Review each before applying.")
        out.append("")
        for decision in proposed:
            out.extend(_render_decision(decision, bugs.get(decision.bug_id)))

    if include_kept:
        near = [
            d
            for d in result.decisions
            if not d.actionable and d.rule_action is not Action.KEEP
        ]
        out.extend(["## Flagged but spared", ""])
        if not near:
            out.append("None.")
            out.append("")
        else:
            out.append(
                f"{len(near)} bug(s) matched a prune rule but were protected or vetoed. "
                "These are worth skimming: a systematic pattern here usually means a "
                "threshold needs adjusting."
            )
            out.append("")
            out.append("| Bug | Rule(s) | Would have been | Spared because |")
            out.append("| --- | --- | --- | --- |")
            for decision in near:
                bug = bugs.get(decision.bug_id)
                out.append(
                    f"| [#{decision.bug_id}]({bug.web_link if bug else ''}) "
                    f"| {', '.join(decision.rule_names) or '-'} "
                    f"| {decision.rule_action.value} "
                    f"| {_cell(decision.reason)} |"
                )
            out.append("")

    return "\n".join(out)


def _render_decision(decision: Decision, bug: BugSnapshot | None) -> list[str]:
    title = bug.title if bug else f"bug {decision.bug_id}"
    link = bug.web_link if bug else ""
    out = [
        f"### [#{decision.bug_id}]({link}) — {title}",
        "",
        f"**Action:** `{decision.action.value}`  ",
        f"**Rules:** {', '.join(decision.rule_names) or '-'}  ",
    ]

    if bug:
        facts = []
        if bug.date_created:
            facts.append(f"reported {bug.date_created.date()}")
        if bug.date_last_updated:
            facts.append(f"last activity {bug.date_last_updated.date()}")
        if bug.apport.distro_release:
            facts.append(f"filed against {bug.apport.distro_release}")
        facts.append(f"{bug.users_affected_count} affected")
        facts.append(f"{bug.message_count} message(s)")
        out.append(f"**Bug:** {'; '.join(facts)}  ")
        if bug.tags:
            out.append(f"**Tags:** `{'` `'.join(bug.tags)}`  ")

    out.extend(["", f"**Why:** {decision.reason}", ""])

    for hit in decision.rule_hits:
        evidence = ", ".join(f"{k}={v}" for k, v in hit.evidence.items())
        out.append(f"- `{hit.rule}` ({hit.claim.value}): {hit.reason}")
        if evidence:
            out.append(f"  - evidence: {evidence}")
    if decision.rule_hits:
        out.append("")

    verdict = decision.verdict
    if verdict is not None and not verdict.failed:
        out.extend(
            [
                "<details><summary>LLM assessment</summary>",
                "",
                f"- is a bug: **{verdict.is_actually_a_bug.value}** "
                f"(kind: {verdict.bug_kind.value}, confidence: {verdict.confidence:.2f})",
                f"- needs more info: {verdict.needs_more_info}",
                f"- reproducible from report: {verdict.reproducible_from_report}",
                f"- recommendation: `{verdict.recommendation.value}`",
            ]
        )
        if verdict.releases_mentioned:
            out.append(f"- releases mentioned: {', '.join(verdict.releases_mentioned)}")
        if verdict.missing_info:
            out.append(f"- missing: {'; '.join(verdict.missing_info)}")
        if verdict.rationale:
            out.extend(["", f"> {verdict.rationale}"])
        out.extend(["", "</details>", ""])
    elif verdict is not None and verdict.failed:
        out.extend(["_LLM assessment unavailable; decided on rules alone._", ""])

    out.append("")
    return out


def _counter_section(title: str, counter: dict[str, int]) -> list[str]:
    if not counter:
        return []
    out = [f"### {title}", "", "| Name | Count |", "| --- | --- |"]
    for name, count in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        out.append(f"| `{name}` | {count} |")
    out.append("")
    return out


def _cell(text: str, limit: int = 140) -> str:
    flat = " ".join(text.split())
    flat = flat.replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def render_csv(result: AnalysisResult, bugs: dict[int, BugSnapshot]) -> str:
    """Flat CSV of every decision, for spreadsheet review or diffing runs."""
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "bug_id",
            "action",
            "rule_action",
            "rules",
            "policy_branch",
            "llm_vetoed",
            "llm_reclassified",
            "title",
            "url",
            "reason",
        ]
    )
    for decision in result.decisions:
        bug = bugs.get(decision.bug_id)
        writer.writerow(
            [
                decision.bug_id,
                decision.action.value,
                decision.rule_action.value,
                ";".join(decision.rule_names),
                decision.policy_branch,
                int(decision.llm_vetoed),
                int(decision.llm_reclassified),
                (bug.title if bug else ""),
                (bug.web_link if bug else ""),
                " ".join(decision.reason.split()),
            ]
        )
    return buffer.getvalue()

