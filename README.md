# pruner

Prune the Launchpad bug backlog of a single source package, using deterministic
rules for eligibility and an LLM for judgement — with a human approving every
change.

## What it actually does

A stale distro bug backlog is mostly reports that can no longer be acted on:
filed against a release that no longer exists, against a package version nobody
ships, or with too little information to reproduce. `pruner` finds those,
explains why, and — once you approve — sets them to **Incomplete** with a comment
asking the reporter to confirm, or **Invalid** when the report is not a bug at all.

`Incomplete` is the default because it is reversible: any reply reopens the bug,
and Launchpad's janitor expires unanswered Incomplete bugs after ~60 days. The
tool asks a question rather than pronouncing a verdict.

## Safety model

This writes to a public bug tracker on other people's reports, so the design is
built around a single invariant:

> **No LLM output, of any kind, can cause a bug to be actioned that the
> deterministic rules did not already make eligible.**

Concretely:

| Layer | Guarantee |
| --- | --- |
| **Eligibility** | Only deterministic rules, on concrete evidence, can flag a bug. |
| **The LLM** | Can *veto* an action, or change an eligible `needs-info` into `invalid`. It can never create one. A failed or unparseable response is silence, never consent. |
| **Hard exclusions** | 16 vetoes (security, patches, assignees, popularity, recent activity, …) override rules *and* the LLM. |
| **Read/write split** | `fetch`/`analyze`/`report` use anonymous HTTP GETs and hold no credentials. `launchpadlib` is imported only by `lp/write.py`. Enforced by a test. |
| **Review gate** | `apply` acts only on bugs approved in a file bound to a specific analysis run. |
| **Live re-check** | Every task's status is re-read immediately before mutation, so a human's triage since the analysis is never clobbered. |
| **Circuit breaker** | `max_actions_per_run` (default 50) caps the damage a bad policy can do. |
| **Reversibility** | Every mutation is logged with its prior status; `pruner rollback` restores it. |

The invariant is tested exhaustively across the cross-product of every possible
model output (`tests/test_policy.py::TestLlmCannotCreateAction`).

## Install

```bash
uv sync                  # read/analyze only — cannot write to Launchpad
uv sync --extra write    # adds launchpadlib, needed for `apply`/`rollback`
```

## Use

```bash
# 1. Download the backlog (read-only, no credentials)
pruner fetch --package xterm

# 2. Run rules + LLM, write a report and an approvals file
pruner analyze --package xterm

# 3. Read it
less out/xterm-report.md

# 4. Approve what you agree with (edit approve = true), then rehearse
pruner apply --package xterm --approvals out/xterm-approvals.toml

# 5. Rehearse for real against Launchpad's staging copy
pruner apply --package xterm --approvals out/xterm-approvals.toml \
             --service staging --commit

# 6. Do it
pruner apply --package xterm --approvals out/xterm-approvals.toml --commit

# If it went wrong
pruner rollback --run-id apply-20260928T153523-ab12ef --commit
```

`apply` is a dry run unless `--commit` is passed, and prompts for confirmation
before touching production.

Start with `--llm none` to tune the rules against your real backlog for free
before spending anything on inference:

```bash
pruner analyze --package xterm --llm none
pruner stats --package xterm
```

## The rules

Run `pruner rules` to list them. Six are enabled by default.

**End-of-life** — the bug only ever concerned a release that is gone. Proposes
`needs-info`.
- `eol_series_tasks` — every open task is nominated to a dead series (strongest evidence)
- `eol_series_tag` — all series tags name dead releases
- `eol_apport_release` — apport's `DistroRelease:` is a dead release (most common)

**Superseded** — `likely_fixed`: the version in the apport metadata predates
everything in the archive (compared with dpkg semantics, so epochs and `~` sort
correctly), or upstream reports the bug resolved. Proposes `needs-info`.

**Gone** — `removed_from_archive`: no `Published` source in any live series.
The only rule that proposes `invalid`, because the conclusion follows from
archive state rather than judgement. Refuses to fire on incomplete lookup data.

**Thin** — `empty_report`: under 120 characters of actual prose once the apport
block is stripped, no attachments, no follow-up, nobody else affected. This is
the rule the LLM most often overrules.

### What counts as "end of life"

Not what Launchpad says. Launchpad reports trusty (14.04), xenial, bionic and
focal as **`Supported`**, because they remain under Ubuntu Pro / ESM, and it
exposes no end-of-life date at all. Trusting that field would make the EOL rules
fire on almost nothing — precisely the decade-old bugs a sweep exists to clear.

ESM is not bug-fixing support: it ships security updates for a subset of packages
to subscribers. Nobody is going to fix a functional xterm bug on trusty. So by
default `pruner` treats a release as end-of-life once *standard* support ends,
reading the `eol` column of `/usr/share/distro-info/ubuntu.csv` — the same column
`ubuntu-distro-info --supported` uses.

This is an explicit, recorded policy rather than a buried assumption:

```toml
[launchpad]
support_policy = "standard"   # standard | launchpad | explicit
```

- `standard` — dead once standard support ends (default)
- `launchpad` — trust Launchpad; ESM counts as supported. Very low reach.
- `explicit` — you name the live series in `live_series`

`pruner fetch` prints which releases it is treating as dead and why. If
`distro-info` data is unavailable the tool falls back to Launchpad's status,
which errs towards pruning *less*.

## The LLM's role

It answers two questions, as you'd expect from the name: **is this actually a
bug**, and **does it need more information**. It returns a schema-validated
assessment (`is_actually_a_bug`, `bug_kind`, `needs_more_info`, `missing_info`,
`reproducible_from_report`, `releases_mentioned`, `confidence`, `rationale`).

Its three powers, all of which reduce or redirect action:

1. **Veto** — turn a proposed action into `keep`.
2. **Live-release veto** — it reads a comment saying "still happens on 24.04" that
   the text matcher missed. Applied regardless of confidence, because a
   hallucinated release name here merely spares a bug.
3. **Reclassify** — turn an eligible `needs-info` into `invalid` when the report
   is a support question or spam (high confidence required).

### Veto scoping

Veto power depends on *what kind of claim* a rule makes, and this distinction is
load-bearing rather than cosmetic.

A **quality** claim ("not enough information here") is directly contradicted by
"a triager could reproduce this", so the model gets a full veto.

A **lifecycle** claim ("the release is dead") is not. A real, well-described,
reproducible defect on a dead release is *still* unverifiable against anything we
ship. If "it's a genuine bug" could veto a lifecycle rule, the best-written EOL
reports would be exactly the ones never pruned — backwards, and enough to make
the EOL rules useless.

The model is also told explicitly that its job is to catch cases where the rules
are *wrong*. Without that framing, a model shown "rule says close this" agrees
essentially always, which turns the veto into a rubber stamp.

Feature requests are **not** reclassified to `invalid` by default: Ubuntu
convention keeps them open at Wishlist importance, and a bot mass-closing
wishlist items would be both wrong and unpopular.

### Providers

```toml
[llm]
provider = "ollama"      # ollama | anthropic | openai | openrouter | none
model = "qwen2.5:7b"
```

Local Ollama by default (free, uses structured outputs so small models return
valid JSON). Cloud providers read a key from the environment. `none` disables
inference entirely.

The model is only shown bugs that are **rule-eligible and unprotected** — around
a tenth of a typical backlog. That is a large cost saving and means most reports
are never sent anywhere. Verdicts are cached against a fingerprint of exactly the
text the model saw, so re-running after a threshold change costs nothing.

## Configuration

`pruner.toml` ships with the defaults written out and commented; a test asserts
it matches the code so it cannot drift. The conservative ones:

| Setting | Default | Why |
| --- | --- | --- |
| `min_quiet_days` | 180 | Nothing touched in the last two release cycles is acted on |
| `protect_importances` | Critical, High | Always get a human |
| `protect_users_affected` | 5 | Lots of people care |
| `protect_duplicates` | 3 | Ditto |
| `protect_tags` | `regression-*`, `rls-*`, `sru-*`, `verification-*`, `patch`, … | In a pipeline, or someone contributed a fix |
| `max_actions_per_run` | 50 | Circuit breaker |
| `min_desc_chars` | 120 | Prose, apport metadata excluded |
| `veto_threshold` | 0.6 | Confidence for the model to block |
| `reclassify_threshold` | 0.8 | Confidence for needs-info → invalid |

## Comments posted

Specific, checkable, and reversible — in that order:

```
Thank you for taking the time to report this bug. This is an automated message
from a backlog triage pass over the xterm package.

This bug is being set to Incomplete because it was tagged only for end-of-life
release(s) lucid (10.04), with no indication it affects anything still supported.

If you can still reproduce this on a currently supported Ubuntu release, please
let us know by commenting with:
  * the Ubuntu release you are seeing this on
  * the version of the package (`apt policy <package>`)
  * the exact steps that trigger the problem

Setting the status back to New (or Confirmed) along with that information is all
that is needed to keep this report open, and any reply will bring it back to our
attention. If we do not hear anything, Launchpad will expire the report
automatically after about 60 days. That is not a judgement on the original issue
-- it just keeps the backlog focused on reports we are able to act on.

--
[signature]
Triage rule(s): eol_series_tag.
[pruner-automated-triage]
```

The rule name is included so that if the tool is wrong, a maintainer can tell
*which rule* was wrong without reading the source. The marker line lets the tool
recognise its own comments and never nag the same bug twice.

## Development

```bash
uv run pytest              # 644 tests, no network
uv run ruff check src tests
uv run mypy src/pruner     # strict
uv run python -m tests.record_fixtures   # refresh recorded Launchpad payloads
```

Tests replay real recorded Launchpad payloads rather than hand-written dicts,
because the parsing code exists to cope with the API's actual quirks. "Now" and
the series table are pinned, since the EOL rules are functions of the calendar.

## Before your first production run

- Rehearse against `--service staging`, a real copy of production whose writes
  are discarded.
- Run with `--llm none` first and read `pruner stats` to see which rules fire and
  how often.
- Read the "Flagged but spared" section of the report: a systematic pattern there
  usually means a threshold needs adjusting.
- Tell the package's triage team what you are about to do. This tool is careful,
  but it is still a bot commenting on other people's bug reports.
- Start with a small `--limit`.
