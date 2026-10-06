"""Resolving which Launchpad credential the write path acts as.

Deliberately free of any ``launchpadlib`` import. ``tests/test_invariants.py``
AST-audits every module and allows ``launchpadlib`` only in ``pruner.lp.write``,
and credential *resolution* -- env vars, file paths, permission checks -- needs
none of it. This module decides **where the credential comes from**;
``pruner.lp.write`` turns that into launchpadlib objects.

Secret-handling rules, all load-bearing:

* **The credential never appears in ``pruner.toml``.** Config names an
  environment variable (``[auth].token_env``), exactly as ``llm.api_key_env``
  already does. ``AuthConfig`` even rejects literal token-shaped keys with a
  message that says so.
* **The credential is held in a** :class:`~pydantic.SecretStr`, so it cannot
  leak through a repr, a log line, or a ``model_dump_json``.
* **Logs name the source, never the material**: the env var name, the resolved
  file path, or the authenticated username.
"""

from __future__ import annotations

import logging
import os
import stat
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, SecretStr

from pruner.config import Config

log = logging.getLogger(__name__)


class CredentialError(RuntimeError):
    """The credential source is unusable: missing, empty, or unsafely stored."""


class CredentialSource(BaseModel):
    """Where the write credential comes from, and the material if we hold it."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["env", "file", "default"]
    """``env``: serialised credential from an environment variable (bot/CI).
    ``file``: a credentials file launchpadlib loads. ``default``: launchpadlib's
    own store (keyring, with interactive browser authorisation if allowed)."""

    origin: str
    """Human-readable provenance for logs: an env var name or a file path.
    Never the credential material."""

    blob: SecretStr | None = None
    """The serialised credential, present only for ``kind == "env"``."""

    path: Path | None = None
    """The credentials file, present only for ``kind == "file"``."""


def resolve_credential(config: Config, *, cli_path: Path | None = None) -> CredentialSource:
    """Choose the credential source, most explicit first:

    1. ``--credentials <path>``
    2. the environment variable named by ``[auth].token_env``
    3. ``[auth].credentials_file``
    4. ``LP_CREDENTIALS_FILE`` (launchpadlib's own convention, honoured by
       ``login_with`` itself, so it just becomes the default)
    5. launchpadlib's keyring/browser flow, if ``[auth].allow_interactive``
    """
    auth = config.auth

    if cli_path is not None:
        path = _checked_file(cli_path.expanduser(), label="--credentials")
        return CredentialSource(kind="file", origin=str(path), path=path)

    if auth.token_env:
        value = os.environ.get(auth.token_env)
        if value is not None:
            if not value.strip():
                raise CredentialError(
                    f"${auth.token_env} is set but empty. Unset it, or give it a "
                    "serialised launchpadlib credential (see `pruner whoami --help`)."
                )
            return CredentialSource(
                kind="env", origin=f"${auth.token_env}", blob=SecretStr(value)
            )

    if auth.credentials_file is not None:
        path = _checked_file(auth.credentials_file.expanduser(), label="[auth].credentials_file")
        return CredentialSource(kind="file", origin=str(path), path=path)

    if not auth.allow_interactive and not os.environ.get("LP_CREDENTIALS_FILE"):
        raise CredentialError(
            "no credential supplied and [auth].allow_interactive is false. Set the "
            f"environment variable ${auth.token_env or 'PRUNER_LP_CREDENTIALS'}, or "
            "point [auth].credentials_file at a chmod 600 credentials file."
        )

    return CredentialSource(kind="default", origin="launchpadlib store (keyring/browser)")


def _checked_file(path: Path, *, label: str) -> Path:
    """A credentials file must exist, be a regular file, and be unreadable by
    anyone but its owner.

    ``launchpadlib``'s own ``UnencryptedFileCredentialStore`` checks none of
    this. The permission refusal is not pedantry: this file is write access to
    a shared public bug tracker, and a group- or world-readable copy failing
    silently is exactly how a bot credential ends up exfiltrated by another
    local user or a stray backup.
    """
    if not path.exists():
        raise CredentialError(f"credentials file from {label} does not exist: {path}")
    st = path.stat()
    if not stat.S_ISREG(st.st_mode):
        raise CredentialError(f"credentials file from {label} is not a regular file: {path}")
    if st.st_size == 0:
        raise CredentialError(f"credentials file from {label} is empty: {path}")
    if st.st_mode & 0o077:
        raise CredentialError(
            f"credentials file from {label} is readable by others: {path}\n"
            f"Fix with: chmod 600 {path}"
        )
    return path
