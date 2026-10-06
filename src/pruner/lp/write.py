"""The only module that authenticates to Launchpad and mutates anything.

Isolated on purpose. ``launchpadlib`` (and therefore OAuth credentials) is
imported lazily inside this module, so the fetch/analyze/report path never even
loads the ability to write.

Which account writes come from is decided by :mod:`pruner.secrets`, which is
launchpadlib-free. Three shapes, in increasing order of convenience and
decreasing order of auditability:

* **An environment variable** holding a serialised credential (the bot case).
  It is never persisted anywhere: :class:`_StaticCredentialStore` refuses to
  copy it into the caller's keyring or onto disk, and a revoked or expired
  token raises instead of opening a browser, via :class:`_NonInteractiveEngine`.
* **A credentials file** (``--credentials`` or ``[auth].credentials_file``).
  :mod:`pruner.secrets` refuses group/world-readable files before we get here.
* **launchpadlib's default store**: keyring lookup, falling back to a one-time
  browser authorisation. Still the default for a human on a laptop, and still
  gated by ``[auth].allow_interactive``.

Pass ``--service staging`` to authorise against ``api.staging.launchpad.net``,
which is a real copy of production whose writes are periodically discarded --
the right place to rehearse a batch.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from pruner.config import Config
from pruner.models import BugTaskStatus
from pruner.secrets import CredentialSource

log = logging.getLogger(__name__)

APPLICATION_NAME = "pruner-backlog-triage"


class WriteError(RuntimeError):
    """A mutation could not be performed."""


class BugWriter(Protocol):
    """The mutation surface ``apply`` depends on.

    A Protocol rather than a concrete class so tests can substitute a recorder and
    assert on intended mutations without any credentials or network.
    """

    @property
    def actor(self) -> str:
        """Launchpad username writes will come from. Recorded in the audit log
        and shown before a production run, because "who did this" is no longer
        implicit once a bot account may be acting."""
        ...

    def task_status(self, task_link: str) -> str: ...

    def set_task_status(self, task_link: str, status: BugTaskStatus) -> None: ...

    def add_comment(self, bug_id: int, body: str, *, subject: str = "") -> None: ...

    def add_tags(self, bug_id: int, tags: list[str]) -> None: ...


def _static_credential_store(credentials: Any) -> Any:
    """A CredentialStore serving one fixed credential for any key, saving nothing.

    ``do_save`` is deliberately a no-op: a credential that arrived via the
    environment must not be copied into the caller's keyring or written to disk.
    A non-``None`` ``do_load`` means ``login_with`` never invokes the
    authorization engine, so no browser is ever opened for this credential.

    Defined as a factory, not a module-level class, because subclassing
    launchpadlib's ``CredentialStore`` at module level would import launchpadlib
    eagerly -- and the read path must never load it.
    """
    from launchpadlib.credentials import CredentialStore

    class StaticStore(CredentialStore):  # type: ignore[misc]  # lazy import: base is Any
        def __init__(self, held: Any) -> None:
            super().__init__()
            self._held = held

        def do_load(self, unique_key: str) -> Any:
            return self._held

        def do_save(self, credentials: Any, unique_consumer_id: str) -> Any:
            return credentials

    return StaticStore(credentials)


def _non_interactive_engine(service: str) -> Any:
    """An authorization engine that raises instead of asking a human.

    launchpadlib's default engines open a browser when a token is missing,
    expired, or revoked -- correct for a human at a keyboard, exactly wrong for
    a bot credential in automation or an unattended ``apply``. ``__call__`` is
    the whole authorisation flow, so overriding it covers both the initial login
    and the mid-session re-authorisation ``LaunchpadOAuthAwareHttp`` attempts
    after a 401.
    """
    from launchpadlib.credentials import RequestTokenAuthorizationEngine

    class NonInteractiveEngine(RequestTokenAuthorizationEngine):  # type: ignore[misc]
        def __call__(self, credentials: Any, credential_store: Any) -> Any:
            raise WriteError(
                "Launchpad credentials are missing, expired or revoked, and this "
                "credential source does not allow interactive re-authorisation, "
                "so no browser will be opened. Re-authorise the account and "
                "update the credential."
            )

    return NonInteractiveEngine(service, application_name=APPLICATION_NAME)


class LaunchpadWriter:
    """Authenticated writer backed by launchpadlib."""

    def __init__(self, config: Config, *, source: CredentialSource) -> None:
        self.config = config
        self._launchpad = self._login(config, source)
        self._bugs: dict[int, Any] = {}
        self._actor = self._resolve_actor()

    @staticmethod
    def _login(config: Config, source: CredentialSource) -> Any:
        try:
            from launchpadlib.credentials import Credentials
            from launchpadlib.launchpad import Launchpad
        except ImportError as exc:  # pragma: no cover
            raise WriteError(
                "launchpadlib is required to apply changes. Install the optional "
                "dependency group: `uv sync --extra write`."
            ) from exc

        service = config.launchpad.service
        log.info("authenticating to Launchpad (%s) via %s", service, source.origin)

        # Bot credentials never get an interactive engine, no matter what
        # allow_interactive says: a browser flow would re-authorise as whoever
        # happens to be logged in next, silently replacing the bot with a human.
        # File and default sources may fall back to launchpadlib's browser flow
        # only when interactive auth is allowed; otherwise the raising engine
        # makes a missing or revoked token fail fast.
        try:
            if source.kind == "env":
                assert source.blob is not None
                credentials = Credentials.from_string(source.blob.get_secret_value())
                return Launchpad.login_with(
                    APPLICATION_NAME,
                    service,
                    version="devel",
                    credential_store=_static_credential_store(credentials),
                    authorization_engine=_non_interactive_engine(service),
                )
            engine = (
                None
                if config.auth.allow_interactive
                else _non_interactive_engine(service)
            )
            if source.kind == "file":
                assert source.path is not None
                return Launchpad.login_with(
                    APPLICATION_NAME,
                    service,
                    version="devel",
                    credentials_file=str(source.path),
                    authorization_engine=engine,
                )
            return Launchpad.login_with(
                APPLICATION_NAME,
                service,
                version="devel",
                authorization_engine=engine,
            )
        except WriteError:
            raise
        except Exception as exc:  # launchpadlib raises a wide variety
            raise WriteError(f"Launchpad authentication failed: {exc}") from exc

    def _resolve_actor(self) -> str:
        """The username this writer will act as, for the audit log and the
        production confirmation. Best-effort: a failure to read it must not
        block an otherwise working credential."""
        try:
            return str(self._launchpad.me.name)
        except Exception as exc:  # lazr raises a wide variety
            log.warning("could not resolve the authenticated Launchpad user: %s", exc)
            return ""

    @property
    def actor(self) -> str:
        return self._actor

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

    @property
    def actor(self) -> str:
        return "dry-run"

    def task_status(self, task_link: str) -> str:
        return self.statuses.get(task_link, str(BugTaskStatus.NEW))

    def set_task_status(self, task_link: str, status: BugTaskStatus) -> None:
        self.status_changes.append((task_link, str(status)))

    def add_comment(self, bug_id: int, body: str, *, subject: str = "") -> None:
        self.comments.append((bug_id, body))

    def add_tags(self, bug_id: int, tags: list[str]) -> None:
        self.tags.append((bug_id, list(tags)))
