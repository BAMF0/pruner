"""Progress rendering for the CLI.

One renderer for every command, so ``fetch``, ``analyze``, ``apply`` and
``rollback`` all look the same. Two design points worth keeping:

* The ``Progress`` shares its ``Console`` with the logging handler (both on
  stderr). Rich only renders log lines *above* an active live region when they
  share a console; two consoles on one tty corrupt each other. That sharing is
  the whole reason this module exists rather than a bare ``Progress`` per
  command.
* Off a terminal (a pipe, CI) the live region is disabled entirely and progress
  degrades to a throttled newline-terminated line, so piped logs stay
  greppable. Nothing here writes to stdout.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from time import monotonic

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

#: ``(completed, total, detail) -> None``. Deliberately identical to the
#: callback already accepted by :meth:`pruner.fetcher.Fetcher.fetch`,
#: :func:`pruner.analysis.analyze`, :func:`pruner.actions.apply_decisions` and
#: :func:`pruner.actions.rollback_run`.
ProgressCallback = Callable[[int, int, str], None]

#: Seconds between plain-text updates when not attached to a terminal.
_PLAIN_INTERVAL = 5.0


def _noop(completed: int, total: int, detail: str) -> None:
    return None


class RateColumn(ProgressColumn):
    """Throughput; rich has no built-in equivalent.

    Worth a custom column because it is the one number that shows whether
    ``analyze`` is actually calling the model or coasting on the verdict cache,
    and whether ``fetch`` is being throttled.
    """

    def render(self, task: Task) -> Text:
        speed = task.speed
        if not speed:
            return Text("      ", style="progress.data.speed")
        if speed >= 1:
            return Text(f"{speed:>5.1f}/s", style="progress.data.speed")
        return Text(f"{speed * 60:>5.1f}/m", style="progress.data.speed")


class Reporter:
    """Live progress on stderr for the duration of one command step."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._live = console.is_terminal
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=None),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            RateColumn(),
            console=console,
            transient=True,
            disable=not self._live,
        )
        self._lock = threading.Lock()

    def __enter__(self) -> Reporter:
        self._progress.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._progress.stop()

    def task(self, label: str, *, total: int | None = None) -> ProgressCallback:
        """A bar for the primary unit of work, counted in one consistent unit.

        ``total`` may be ``None`` when the count is not known until the work
        starts; it is corrected on the first call.
        """
        task_id = self._progress.add_task(label, total=total)
        plain = {"last": 0.0}

        def callback(completed: int, total_now: int, detail: str) -> None:
            description = " ".join(part for part in (label, detail) if part)
            with self._lock:
                self._progress.update(
                    task_id,
                    completed=completed,
                    total=total_now or None,
                    description=description,
                )
                if not self._live:
                    self._plain(description, completed, total_now, plain)

        return callback

    def stage(self) -> ProgressCallback:
        """A secondary bar whose unit and total change as work moves on.

        ``detail`` names the current stage. The task is **reset** whenever the
        stage or its total changes, because rich freezes a task's clock and
        speed once ``completed >= total`` (it sets ``finished_time``); reusing
        one task across stages without a reset leaves elapsed, ETA and rate
        stuck at the first stage's values.

        Silent off a terminal: the primary bar already prints there, and a
        second stream of stage lines would only add noise.
        """
        if not self._live:
            return _noop

        task_id = self._progress.add_task("", total=None, visible=False)
        seen: dict[str, tuple[str, int] | None] = {"key": None}

        def callback(completed: int, total: int, detail: str) -> None:
            key = (detail, total)
            with self._lock:
                if seen["key"] != key:
                    seen["key"] = key
                    self._progress.reset(
                        task_id,
                        total=total,
                        completed=completed,
                        description=detail,
                        visible=True,
                    )
                else:
                    self._progress.update(task_id, completed=completed)

        return callback

    def _plain(
        self, description: str, completed: int, total: int, state: dict[str, float]
    ) -> None:
        """Throttled fallback. Newline-terminated, never ``\\r``."""
        now = monotonic()
        if completed != total and now - state["last"] < _PLAIN_INTERVAL:
            return
        state["last"] = now
        percent = f" ({completed * 100 // total}%)" if total else ""
        self.console.print(
            f"  {description} {completed}/{total}{percent}", style="dim", highlight=False
        )


@contextmanager
def spinner(console: Console, message: str) -> Iterator[None]:
    """Indeterminate progress for work with no countable unit.

    A module-level function rather than a :class:`Reporter` method so that it is
    structurally obvious this never nests inside a bar: rich < 14 raises
    ``LiveError`` on a second live region for one console, and ``pyproject.toml``
    only requires ``rich>=13.7``.
    """
    if not console.is_terminal:
        console.print(f"  {message}", style="dim", highlight=False)
        yield
        return
    with console.status(message, spinner="dots"):
        yield
