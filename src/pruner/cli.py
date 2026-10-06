"""Command line interface.

Pipeline: ``fetch`` -> ``analyze`` -> ``report`` -> edit approvals -> ``apply``.

The commands are separate rather than one do-everything invocation so that the
expensive, the reviewable, and the irreversible steps are distinct. In particular
``apply`` requires three independent things to line up before it writes to
production: ``--commit``, ``--service production``, and an approvals file whose
run ID matches the analysis.
"""

from __future__ import annotations

import logging
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

from pruner.analysis import analyze as run_analysis
from pruner.audit import AuditLog
from pruner.config import Config, load_config
from pruner.fetcher import Fetcher, load_series
from pruner.llm import build_analyzer
from pruner.llm.base import Analyzer, ProviderError
from pruner.lp.archive import fetch_archive_index
from pruner.lp.read import NotFound, ReadClient
from pruner.models import DEFAULT_FETCH_STATUSES, BugTaskStatus
from pruner.progress import Reporter, spinner
from pruner.report import read_approvals, render_csv, render_markdown, write_approvals
from pruner.store import Store

if TYPE_CHECKING:
    # Type-only import: keeps the annotation checked without loading the write
    # path (and therefore launchpadlib) during fetch/analyze/report.
    from pruner.lp.write import BugWriter

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Prune a Launchpad bug backlog for one source package, carefully.",
)
console = Console()
err_console = Console(stderr=True)

# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------

PackageArg = Annotated[str, typer.Option("--package", "-p", help="Source package name.")]
ConfigOpt = Annotated[
    Path | None, typer.Option("--config", "-c", help="Path to pruner.toml.", exists=True)
]
StateOpt = Annotated[
    Path, typer.Option("--state-dir", help="Where to keep the cache and audit log.")
]
ServiceOpt = Annotated[
    str | None,
    typer.Option("--service", help="Launchpad service: production or staging."),
]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Debug logging.")]
ConcurrencyOpt = Annotated[
    int | None,
    typer.Option(
        "--concurrency",
        "-j",
        min=1,
        max=32,
        help="Parallel Launchpad requests (default 4). Fetching is latency-bound, "
        "so this is the main speed knob.",
    ),
]


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=err_console, rich_tracebacks=True, show_path=verbose)],
    )
    # launchpadlib and httpx are chatty at DEBUG.
    for noisy in ("httpx", "httpcore", "launchpadlib", "lazr"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _load(config_path: Path | None, service: str | None) -> Config:
    try:
        config = load_config(config_path)
    except (ValueError, OSError) as exc:
        err_console.print(f"[red]configuration error:[/red] {exc}")
        raise typer.Exit(2) from exc

    if service:
        if service not in ("production", "staging"):
            err_console.print("[red]--service must be 'production' or 'staging'[/red]")
            raise typer.Exit(2)
        config = config.model_copy(
            update={"launchpad": config.launchpad.model_copy(update={"service": service})}
        )
    return config


def _new_run_id(kind: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    return f"{kind}-{stamp}-{uuid.uuid4().hex[:6]}"


def _build_analyzer(config: Config, store: Store, override: str | None) -> Analyzer:
    llm_config = config.llm
    if override:
        if override == "none":
            llm_config = llm_config.model_copy(update={"provider": "none"})
        elif ":" in override:
            provider, model = override.split(":", 1)
            llm_config = llm_config.model_copy(
                update={"provider": provider, "model": model}
            )
        else:
            llm_config = llm_config.model_copy(update={"provider": override})
    try:
        return build_analyzer(llm_config, store)
    except ProviderError as exc:
        err_console.print(f"[red]LLM provider unavailable:[/red] {exc}")
        raise typer.Exit(2) from exc


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------


@app.command()
def fetch(
    package: PackageArg,
    statuses: Annotated[
        list[str] | None,
        typer.Option("--status", help="Bug task status to include (repeatable)."),
    ] = None,
    limit: Annotated[int | None, typer.Option("--limit", help="Cap bugs fetched.")] = None,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Ignore the snapshot cache.")
    ] = False,
    refresh_series: Annotated[
        bool, typer.Option("--refresh-series", help="Re-read the distro series table.")
    ] = False,
    concurrency: ConcurrencyOpt = None,
    config_path: ConfigOpt = None,
    state_dir: StateOpt = Path(".pruner"),
    service: ServiceOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Download the package's open bugs into the local cache. Read-only."""
    _setup_logging(verbose)
    config = _load(config_path, service)

    selected: tuple[BugTaskStatus, ...]
    if statuses:
        try:
            selected = tuple(BugTaskStatus(s) for s in statuses)
        except ValueError as exc:
            err_console.print(f"[red]invalid status:[/red] {exc}")
            raise typer.Exit(2) from exc
    else:
        selected = DEFAULT_FETCH_STATUSES

    with (
        ReadClient(config.launchpad, concurrency=concurrency) as client,
        Store.open(state_dir) as store,
    ):
        with spinner(err_console, "Reading the distro series table..."):
            table = load_series(client, config, store, refresh=refresh_series)
        console.print(f"Series: {table.summary()}")
        console.print(
            f"  live: {', '.join(s.name for s in table.live)}", style="dim"
        )
        if table.esm_only:
            console.print(
                "  treated as end-of-life despite Launchpad reporting them as "
                f"Supported (ESM only): {', '.join(s.name for s in table.esm_only)}",
                style="dim",
            )

        fetcher = Fetcher(client, config, store, table)
        try:
            fetcher.ensure_package_exists(package)
        except NotFound as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc

        console.print(
            f"Fetching [bold]{package}[/bold] bugs ({', '.join(selected)}) "
            f"with {client.concurrency} parallel request(s)..."
        )
        with Reporter(err_console) as reporter:
            stats = fetcher.fetch(
                package,
                statuses=selected,
                limit=limit,
                refresh=refresh,
                progress=reporter.task("fetched", total=None),
                stage=reporter.stage(),
            )

        # The archive index feeds likely_fixed and removed_from_archive.
        with spinner(err_console, f"Checking the archive for {package}..."):
            index = fetch_archive_index(client, table, package)
        store.put_archive(index)

    table_out = Table(title=f"fetch: {package}", show_header=False, box=None)
    table_out.add_row("bug tasks found", str(stats.tasks_found))
    table_out.add_row("distinct bugs", str(stats.bugs_seen))
    table_out.add_row("newly fetched", str(stats.fetched))
    table_out.add_row("reused from cache", str(stats.from_cache))
    table_out.add_row("fully enriched", str(stats.enriched))
    table_out.add_row("skipped by prefilter", str(stats.prefiltered))
    table_out.add_row("errors", str(stats.errors))
    table_out.add_row("http requests", str(stats.requests))
    table_out.add_row(
        "archive publications",
        f"{len(index.publications)} across {len(index.queried_series)} series",
    )
    console.print(table_out)

    if stats.prefilter_reasons:
        console.print("\n[dim]Prefilter skips (cheap exclusions, not enriched):[/dim]")
        for rule, count in sorted(stats.prefilter_reasons.items(), key=lambda kv: -kv[1]):
            console.print(f"  [dim]{rule}: {count}[/dim]")

    console.print(f"\nNext: [bold]pruner analyze --package {package}[/bold]")


# ---------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------


@app.command()
def analyze(
    package: PackageArg,
    llm: Annotated[
        str | None,
        typer.Option(
            "--llm",
            help="Override provider, e.g. 'none', 'ollama', 'ollama:qwen2.5:7b', 'anthropic'.",
        ),
    ] = None,
    no_llm_cache: Annotated[
        bool, typer.Option("--no-llm-cache", help="Re-query the model for every bug.")
    ] = False,
    out_dir: Annotated[
        Path, typer.Option("--out", "-o", help="Directory for the report and approvals.")
    ] = Path("out"),
    config_path: ConfigOpt = None,
    state_dir: StateOpt = Path(".pruner"),
    service: ServiceOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Run rules and the LLM over cached bugs, then write a report to review."""
    _setup_logging(verbose)
    config = _load(config_path, service)
    distribution = config.launchpad.distribution

    with Store.open(state_dir) as store:
        table = store.get_series(distribution)
        if table is None:
            err_console.print("[red]no cached series data; run `pruner fetch` first[/red]")
            raise typer.Exit(2)

        bugs = list(store.iter_bugs(distribution, package))
        if not bugs:
            err_console.print(
                f"[red]no cached bugs for {distribution}/{package}; run `pruner fetch` first[/red]"
            )
            raise typer.Exit(2)

        archive = store.get_archive(distribution, package)
        analyzer = _build_analyzer(config, store, llm)
        run_id = _new_run_id("analyze")

        console.print(
            f"Analysing {len(bugs)} bug(s) for [bold]{package}[/bold] "
            f"(LLM: {analyzer.model_id})..."
        )
        try:
            with Reporter(err_console) as reporter:
                result = run_analysis(
                    bugs,
                    config=config,
                    series=table,
                    package=package,
                    analyzer=analyzer,
                    archive=archive,
                    progress=reporter.task("analysed", total=len(bugs)),
                    use_cache=not no_llm_cache,
                )
        finally:
            analyzer.close()

        store.start_run(
            run_id,
            "analyze",
            distribution,
            package,
            {
                "llm": analyzer.model_id,
                "rules": list(config.rules.enabled),
                "config": str(config.source_path) if config.source_path else "defaults",
            },
        )
        store.put_decisions(run_id, result.decisions)

        by_id = {b.id: b for b in bugs}
        out_dir.mkdir(parents=True, exist_ok=True)
        report_path = out_dir / f"{package}-report.md"
        approvals_path = out_dir / f"{package}-approvals.toml"
        csv_path = out_dir / f"{package}-decisions.csv"

        report_path.write_text(
            render_markdown(
                result,
                run_id=run_id,
                config=config,
                package=package,
                bugs=by_id,
                analyzer_model=analyzer.model_id,
            ),
            encoding="utf-8",
        )
        count = write_approvals(
            approvals_path,
            result,
            run_id=run_id,
            config=config,
            package=package,
            bugs=by_id,
        )
        csv_path.write_text(render_csv(result, by_id), encoding="utf-8")

    _print_stats(result.stats.model_dump())
    console.print(f"\nRun ID: [bold]{run_id}[/bold]")
    console.print(f"Report:    {report_path}")
    console.print(f"Approvals: {approvals_path} ({count} entr{'y' if count == 1 else 'ies'})")
    console.print(f"CSV:       {csv_path}")
    console.print(
        f"\nNext: read the report, set [bold]approve = true[/bold] in the approvals "
        f"file, then\n  [bold]pruner apply --package {package} "
        f"--approvals {approvals_path}[/bold]"
    )


# ---------------------------------------------------------------------------
# report / stats
# ---------------------------------------------------------------------------


@app.command()
def report(
    package: PackageArg,
    run_id: Annotated[
        str | None, typer.Option("--run-id", help="Analysis run to render (default: latest).")
    ] = None,
    out: Annotated[Path | None, typer.Option("--out", "-o", help="Write here.")] = None,
    config_path: ConfigOpt = None,
    state_dir: StateOpt = Path(".pruner"),
    verbose: VerboseOpt = False,
) -> None:
    """Re-render the report for a stored analysis run."""
    _setup_logging(verbose)
    config = _load(config_path, None)
    distribution = config.launchpad.distribution

    with Store.open(state_dir) as store:
        resolved = run_id or store.latest_run("analyze", distribution, package)
        if not resolved:
            err_console.print("[red]no analysis runs found; run `pruner analyze` first[/red]")
            raise typer.Exit(2)
        decisions = store.get_decisions(resolved)
        if not decisions:
            err_console.print(f"[red]no decisions stored for run {resolved}[/red]")
            raise typer.Exit(2)

        bugs = {b.id: b for b in store.iter_bugs(distribution, package)}
        details = store.run_details(resolved) or {}

    from pruner.analysis import AnalysisResult, AnalysisStats

    stats = AnalysisStats(package=package)
    for decision in decisions:
        stats.record(decision)
    result = AnalysisResult(decisions=decisions, stats=stats)

    text = render_markdown(
        result,
        run_id=resolved,
        config=config,
        package=package,
        bugs=bugs,
        analyzer_model=str((details.get("details") or {}).get("llm", "unknown")),
    )
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        console.print(f"Wrote {out}")
    else:
        sys.stdout.write(text)


@app.command()
def stats(
    package: PackageArg,
    run_id: Annotated[str | None, typer.Option("--run-id")] = None,
    config_path: ConfigOpt = None,
    state_dir: StateOpt = Path(".pruner"),
    verbose: VerboseOpt = False,
) -> None:
    """Show rule/exclusion/action counters for an analysis run."""
    _setup_logging(verbose)
    config = _load(config_path, None)
    distribution = config.launchpad.distribution

    with Store.open(state_dir) as store:
        resolved = run_id or store.latest_run("analyze", distribution, package)
        if not resolved:
            err_console.print("[red]no analysis runs found[/red]")
            raise typer.Exit(2)
        decisions = store.get_decisions(resolved)

    from pruner.analysis import AnalysisStats

    counters = AnalysisStats(package=package)
    for decision in decisions:
        counters.record(decision)

    console.print(f"Run [bold]{resolved}[/bold]")
    _print_stats(counters.model_dump())


def _print_stats(data: dict[str, object]) -> None:
    summary = Table(title="Outcomes", box=None, show_header=False)
    summary.add_row("bugs analysed", str(data.get("bugs", 0)))
    summary.add_row("eligible by rules", str(data.get("eligible_before_llm", 0)))
    summary.add_row("LLM calls", str(data.get("llm_calls", 0)))
    summary.add_row("LLM failures", str(data.get("llm_failures", 0)))
    summary.add_row("LLM vetoes", str(data.get("llm_vetoes", 0)))
    summary.add_row("LLM reclassifications", str(data.get("llm_reclassifications", 0)))
    summary.add_row("age escalations", str(data.get("age_escalations", 0)))
    console.print(summary)

    for title, key in (
        ("Actions", "actions"),
        ("Rule hits", "rule_hits"),
        ("Exclusions", "exclusions"),
        ("Policy branches", "policy_branches"),
    ):
        counter = data.get(key) or {}
        if not isinstance(counter, dict) or not counter:
            continue
        rendered = Table(title=title, box=None)
        rendered.add_column("name")
        rendered.add_column("count", justify="right")
        for name, count in sorted(counter.items(), key=lambda kv: (-int(kv[1]), kv[0])):
            rendered.add_row(str(name), str(count))
        console.print(rendered)


# ---------------------------------------------------------------------------
# apply
# ---------------------------------------------------------------------------


@app.command()
def apply(
    package: PackageArg,
    approvals: Annotated[
        Path, typer.Option("--approvals", help="Approvals file from `analyze`.", exists=True)
    ],
    commit: Annotated[
        bool,
        typer.Option("--commit", help="Actually write to Launchpad. Without this, dry run."),
    ] = False,
    approve_all: Annotated[
        bool,
        typer.Option(
            "--approve-all",
            help="Treat every entry in the approvals file as approved, except those "
            "explicitly left at approve = false. Read the report first.",
        ),
    ] = False,
    limit: Annotated[int | None, typer.Option("--limit", help="Cap actions this run.")] = None,
    credentials: Annotated[
        Path | None,
        typer.Option(
            "--credentials",
            help="launchpadlib credentials file (chmod 600). Overrides the "
            "environment variable named by [auth].token_env, which is how a bot "
            "account is normally supplied.",
        ),
    ] = None,
    config_path: ConfigOpt = None,
    state_dir: StateOpt = Path(".pruner"),
    service: ServiceOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Apply approved decisions. Dry run unless --commit is given."""
    _setup_logging(verbose)
    config = _load(config_path, service)
    distribution = config.launchpad.distribution
    dry_run = not commit

    approved = read_approvals(approvals)
    if approved.package and approved.package != package:
        err_console.print(
            f"[red]approvals file is for package '{approved.package}', not '{package}'[/red]"
        )
        raise typer.Exit(2)

    with Store.open(state_dir) as store:
        decisions = store.get_decisions(approved.run_id)
        if not decisions:
            err_console.print(
                f"[red]no stored decisions for run '{approved.run_id}'. The approvals file "
                "must come from an analysis in this state directory.[/red]"
            )
            raise typer.Exit(2)

        selected = [
            d
            for d in decisions
            if d.actionable and approved.allows(d.bug_id, approve_all=approve_all)
        ]
        bugs = {b.id: b for b in store.iter_bugs(distribution, package)}

    if not selected:
        console.print(
            "[yellow]Nothing approved.[/yellow] Set `approve = true` on the entries you "
            "want applied, or pass --approve-all once you have read the report."
        )
        raise typer.Exit(0)

    if approve_all:
        console.print(
            f"[yellow]--approve-all[/yellow]: treating all {len(selected)} listed "
            "entr(ies) as approved."
        )

    console.print(
        f"Run [bold]{approved.run_id}[/bold]: {len(selected)} approved of "
        f"{sum(1 for d in decisions if d.actionable)} proposed."
    )

    writer: BugWriter
    if dry_run:
        from pruner.lp.write import DryRunWriter

        statuses = {
            task.self_link: str(task.status)
            for bug in bugs.values()
            for task in bug.tasks
            if task.self_link
        }
        writer = DryRunWriter(statuses)
    else:
        from pruner.lp.write import LaunchpadWriter, WriteError
        from pruner.secrets import CredentialError, resolve_credential

        # The writer is built before the confirmation prompt so the prompt can
        # name the account. Thinking you are the bot when you are actually
        # yourself is the main failure mode a bot account introduces, and the
        # cheapest place to catch it is the one prompt a human always reads.
        try:
            source = resolve_credential(config, cli_path=credentials)
            writer = LaunchpadWriter(config, source=source)
        except (CredentialError, WriteError) as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc

    if dry_run:
        console.print("[green]DRY RUN[/green] — no changes will be made. Add --commit to apply.")
    else:
        account = f"~{writer.actor}" if writer.actor else "(unknown account)"
        console.print(
            f"[red bold]LIVE[/red bold] — writing to "
            f"[bold]{config.launchpad.service}[/bold] as [bold]{account}[/bold] "
            f"(credential: {source.origin})."
        )
        if config.launchpad.service == "production":
            typer.confirm(
                f"Modify {len(selected)} bug(s) on PRODUCTION Launchpad as {account}?",
                abort=True,
            )

    from pruner.actions import apply_decisions

    audit = AuditLog.open(state_dir)
    apply_run_id = _new_run_id("apply")
    with Reporter(err_console) as reporter:
        result = apply_decisions(
            selected,
            bugs,
            config=config,
            package=package,
            writer=writer,
            audit=audit,
            run_id=apply_run_id,
            dry_run=dry_run,
            limit=limit,
            progress=reporter.task("applied"),
        )

    console.print(
        f"\napplied={result.applied} skipped={result.skipped} failed={result.failed}"
    )
    for reason, count in sorted(result.reasons.items(), key=lambda kv: -kv[1]):
        console.print(f"  [dim]{reason}: {count}[/dim]")
    console.print(f"Audit log: {audit.path}")
    if not dry_run:
        console.print(f"Apply run ID: [bold]{apply_run_id}[/bold]")
        console.print(f"To undo: [bold]pruner rollback --run-id {apply_run_id}[/bold]")


# ---------------------------------------------------------------------------
# rollback
# ---------------------------------------------------------------------------


@app.command()
def rollback(
    run_id: Annotated[str, typer.Option("--run-id", help="Apply run to undo.")],
    commit: Annotated[bool, typer.Option("--commit", help="Actually revert.")] = False,
    credentials: Annotated[Path | None, typer.Option("--credentials")] = None,
    config_path: ConfigOpt = None,
    state_dir: StateOpt = Path(".pruner"),
    service: ServiceOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Restore statuses changed by a previous apply run."""
    _setup_logging(verbose)
    config = _load(config_path, service)
    dry_run = not commit

    audit = AuditLog.open(state_dir)
    records = audit.applied_for_run(run_id)
    if not records:
        err_console.print(
            f"[red]no applied, un-reverted records for run '{run_id}'[/red]\n"
            f"known runs: {', '.join(audit.run_ids()) or 'none'}"
        )
        raise typer.Exit(2)

    console.print(f"Reverting {len(records)} bug(s) from run [bold]{run_id}[/bold].")
    if dry_run:
        console.print("[green]DRY RUN[/green] — add --commit to actually revert.")

    writer: BugWriter
    if dry_run:
        from pruner.lp.write import DryRunWriter

        writer = DryRunWriter(
            {
                change.task_link: change.new_status
                for record in records
                for change in record.task_changes
            }
        )
    else:
        from pruner.lp.write import LaunchpadWriter, WriteError
        from pruner.secrets import CredentialError, resolve_credential

        try:
            source = resolve_credential(config, cli_path=credentials)
            writer = LaunchpadWriter(config, source=source)
        except (CredentialError, WriteError) as exc:
            err_console.print(f"[red]{exc}[/red]")
            raise typer.Exit(2) from exc

    from pruner.actions import rollback_run

    with Reporter(err_console) as reporter:
        result = rollback_run(
            run_id,
            config=config,
            writer=writer,
            audit=audit,
            new_run_id=_new_run_id("rollback"),
            dry_run=dry_run,
            progress=reporter.task("reverted"),
        )
    console.print(
        f"\nreverted={result.applied} skipped={result.skipped} failed={result.failed}"
    )
    for reason, count in sorted(result.reasons.items(), key=lambda kv: -kv[1]):
        console.print(f"  [dim]{reason}: {count}[/dim]")


# ---------------------------------------------------------------------------
# whoami
# ---------------------------------------------------------------------------


@app.command()
def whoami(
    credentials: Annotated[
        Path | None,
        typer.Option("--credentials", help="launchpadlib credentials file (chmod 600)."),
    ] = None,
    config_path: ConfigOpt = None,
    service: ServiceOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Show which Launchpad account writes would come from.

    Verifies the resolved credential against Launchpad and prints the account.
    Run this before a production ``apply`` whenever a bot account is in play:
    the audit log and every bug's history will name whoever this prints.
    """
    _setup_logging(verbose)
    config = _load(config_path, service)

    from pruner.lp.write import LaunchpadWriter, WriteError
    from pruner.secrets import CredentialError, resolve_credential

    try:
        source = resolve_credential(config, cli_path=credentials)
        writer = LaunchpadWriter(config, source=source)
    except (CredentialError, WriteError) as exc:
        err_console.print(f"[red]{exc}[/red]")
        raise typer.Exit(2) from exc

    console.print(f"service:    {config.launchpad.service}")
    console.print(f"credential: {source.origin}")
    console.print(f"acting as:  ~{writer.actor}" if writer.actor else "acting as:  (unknown)")


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------


@app.command("rules")
def list_rules(
    config_path: ConfigOpt = None,
) -> None:
    """List available rules and which are enabled."""
    config = _load(config_path, None)
    from pruner.rules import all_rule_names
    from pruner.rules.exclusions import EXCLUSIONS

    enabled = set(config.rules.enabled)
    table = Table(title="Prune rules")
    table.add_column("rule")
    table.add_column("enabled", justify="center")
    for name in sorted(all_rule_names()):
        table.add_row(name, "[green]yes[/green]" if name in enabled else "[dim]no[/dim]")
    console.print(table)

    protections = Table(title="Hard exclusions (always active)")
    protections.add_column("exclusion")
    for name, _ in EXCLUSIONS:
        protections.add_row(name)
    console.print(protections)


if __name__ == "__main__":
    app()
