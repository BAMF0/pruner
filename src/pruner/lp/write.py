"""The only module that authenticates to Launchpad and mutates anything.

Isolated on purpose. ``launchpadlib`` (and therefore OAuth credentials) is
imported lazily inside this module, so the fetch/analyze/report path never even
loads the ability to write.

Credentials are handled by launchpadlib's own store: ``login_with`` opens a
browser for a one-time authorisation and caches the token under
``~/.launchpadlib``. Pass ``--service staging`` to authorise against
``api.staging.launchpad.net``, which is a real copy of production whose writes are
periodically discarded -- the right place to rehearse a batch.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

from pruner.config import Config
from pruner.models import BugTaskStatus

log = logging.getLogger(__name__)

APPLICATION_NAME = "pruner-backlog-triage"


class WriteError(RuntimeError):
    """A mutation could not be performed."""


class BugWriter(Protocol):
    """The mutation surface ``apply`` depends on.

    A Protocol rather than a concrete class so tests can substitute a recorder and
    assert on intended mutations without any credentials or network.
    """

    def task_status(self, task_link: str) -> str: ...

    def set_task_status(self, task_link: str, status: BugTaskStatus) -> None: ...

    def add_comment(self, bug_id: int, body: str, *, subject: str = "") -> None: ...

    def add_tags(self, bug_id: int, tags: list[str]) -> None: ...


class LaunchpadWriter:
    """Authenticated writer backed by launchpadlib."""

    def __init__(self, config: Config, *, credentials_file: Path | None = None) -> None:
        self.config = config
        self._launchpad = self._login(config, credentials_file)
        self._bugs: dict[int, Any] = {}

    @staticmethod
    def _login(config: Config, credentials_file: Path | None) -> Any:
        try:
            from launchpadlib.launchpad import Launchpad
        except ImportError as exc:  # pragma: no cover
            raise WriteError(
                "launchpadlib is required to apply changes. Install the optional "
                "dependency group: `uv sync --extra write`."
            ) from exc

        service = config.launchpad.service
        log.info("authenticating to Launchpad (%s)", service)
        try:
            return Launchpad.login_with(
                APPLICATION_NAME,
                service,
                version="devel",
                credentials_file=str(credentials_file) if credentials_file else None,
            )
        except Exception as exc:  # launchpadlib raises a wide variety
            raise WriteError(f"Launchpad authentication failed: {exc}") from exc

    # -- helpers -----------------------------------------------------------

    def _bug(self, bug_id: int) -> Any:
        if bug_id not in self._bugs:
            try:
                self._bugs[bug_id] = self._launchpad.bugs[bug_id]
            except Exception as exc:
                raise WriteError(f"cannot load bug #{bug_id}: {exc}") from exc
        return self._bugs[bug_id]

    def _task(self, task_link: str) -> Any:
        try:
            return self._launchpad.load(task_link)
        except Exception as exc:
            raise WriteError(f"cannot load task {task_link}: {exc}") from exc

    # -- mutations ---------------------------------------------------------

    def task_status(self, task_link: str) -> str:
        """Live status, re-read immediately before mutation.

        Deliberately not taken from the cached snapshot: between ``analyze`` and
        ``apply`` a human may have triaged the bug, and we must not clobber that.
        """
        return str(self._task(task_link).status)

    def set_task_status(self, task_link: str, status: BugTaskStatus) -> None:
        task = self._task(task_link)
        task.status = str(status)
        try:
            task.lp_save()
        except Exception as exc:
            raise WriteError(f"cannot set {task_link} to {status}: {exc}") from exc

    def add_comment(self, bug_id: int, body: str, *, subject: str = "") -> None:
        bug = self._bug(bug_id)
        try:
            bug.newMessage(content=body, subject=subject or None)
        except Exception as exc:
            raise WriteError(f"cannot comment on bug #{bug_id}: {exc}") from exc

    def add_tags(self, bug_id: int, tags: list[str]) -> None:
        if not tags:
            return
        bug = self._bug(bug_id)
        existing = list(bug.tags or [])
        merged = existing + [t for t in tags if t not in existing]
        if merged == existing:
            return
        bug.tags = merged
        try:
            bug.lp_save()
        except Exception as exc:
            raise WriteError(f"cannot tag bug #{bug_id}: {exc}") from exc


class DryRunWriter:
    """Records intended mutations instead of performing them.

    Used by ``--dry-run`` (the default) and by the test suite. Reads still need to
    come from somewhere; statuses are served from the analysed snapshots, which is
    exactly the data the dry-run report was built from.
    """

    def __init__(self, statuses: dict[str, str] | None = None) -> None:
        self.statuses = statuses or {}
        self.status_changes: list[tuple[str, str]] = []
        self.comments: list[tuple[int, str]] = []
        self.tags: list[tuple[int, list[str]]] = []

    def task_status(self, task_link: str) -> str:
        return self.statuses.get(task_link, str(BugTaskStatus.NEW))

    def set_task_status(self, task_link: str, status: BugTaskStatus) -> None:
        self.status_changes.append((task_link, str(status)))

    def add_comment(self, bug_id: int, body: str, *, subject: str = "") -> None:
        self.comments.append((bug_id, body))

    def add_tags(self, bug_id: int, tags: list[str]) -> None:
        self.tags.append((bug_id, list(tags)))
