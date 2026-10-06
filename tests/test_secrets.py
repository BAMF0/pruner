"""Credential resolution: precedence, file hygiene, and secret containment.

The properties that matter here are about what does *not* happen: no literal
secret in config, no group-readable credential file used silently, no secret
material in a repr or a log-shaped dump.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pruner.config import Config
from pruner.secrets import CredentialError, resolve_credential

BLOB = "[1]\nconsumer_key = pruner\naccess_token = tok\naccess_secret = sec\n"


def config_with(auth: dict[str, object]) -> Config:
    return Config.model_validate({"auth": auth})


@pytest.fixture
def cred_file(tmp_path: Path) -> Path:
    path = tmp_path / "bot.credentials"
    path.write_text(BLOB, encoding="utf-8")
    path.chmod(0o600)
    return path


class TestPrecedence:
    def test_cli_path_beats_env(
        self, monkeypatch: pytest.MonkeyPatch, cred_file: Path
    ) -> None:
        monkeypatch.setenv("PRUNER_LP_CREDENTIALS", BLOB)
        source = resolve_credential(Config(), cli_path=cred_file)
        assert source.kind == "file"
        assert source.path == cred_file

    def test_env_beats_config_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setenv("PRUNER_LP_CREDENTIALS", BLOB)
        config = config_with({"credentials_file": str(tmp_path / "unused.credentials")})
        source = resolve_credential(config)
        assert source.kind == "env"
        assert source.origin == "$PRUNER_LP_CREDENTIALS"
        assert source.blob is not None

    def test_custom_env_var_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_BOT_TOKEN", BLOB)
        config = config_with({"token_env": "MY_BOT_TOKEN"})
        source = resolve_credential(config)
        assert source.kind == "env"
        assert source.origin == "$MY_BOT_TOKEN"

    def test_config_file_used_when_no_env(
        self, monkeypatch: pytest.MonkeyPatch, cred_file: Path
    ) -> None:
        monkeypatch.delenv("PRUNER_LP_CREDENTIALS", raising=False)
        config = config_with({"credentials_file": str(cred_file)})
        source = resolve_credential(config)
        assert source.kind == "file"

    def test_default_when_nothing_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("PRUNER_LP_CREDENTIALS", raising=False)
        source = resolve_credential(Config())
        assert source.kind == "default"

    def test_interactive_forbidden_with_no_credential(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("PRUNER_LP_CREDENTIALS", raising=False)
        monkeypatch.delenv("LP_CREDENTIALS_FILE", raising=False)
        config = config_with({"allow_interactive": False})
        with pytest.raises(CredentialError, match="allow_interactive is false"):
            resolve_credential(config)

    def test_empty_env_var_is_an_error_not_a_login(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty export is a classic CI misconfiguration; silently falling
        through to the keyring would authenticate as the wrong account."""
        monkeypatch.setenv("PRUNER_LP_CREDENTIALS", "   ")
        with pytest.raises(CredentialError, match="empty"):
            resolve_credential(Config())


class TestFileHygiene:
    def test_group_readable_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, cred_file: Path
    ) -> None:
        monkeypatch.delenv("PRUNER_LP_CREDENTIALS", raising=False)
        cred_file.chmod(0o640)
        with pytest.raises(CredentialError, match="chmod 600"):
            resolve_credential(Config(), cli_path=cred_file)

    def test_world_readable_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, cred_file: Path
    ) -> None:
        monkeypatch.delenv("PRUNER_LP_CREDENTIALS", raising=False)
        cred_file.chmod(0o604)
        with pytest.raises(CredentialError, match="chmod 600"):
            resolve_credential(Config(), cli_path=cred_file)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(CredentialError, match="does not exist"):
            resolve_credential(Config(), cli_path=tmp_path / "nope.credentials")

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.credentials"
        path.write_text("", encoding="utf-8")
        path.chmod(0o600)
        with pytest.raises(CredentialError, match="empty"):
            resolve_credential(Config(), cli_path=path)

    def test_directory_is_not_a_credential_file(self, tmp_path: Path) -> None:
        with pytest.raises(CredentialError, match="not a regular file"):
            resolve_credential(Config(), cli_path=tmp_path)

    def test_tilde_is_expanded(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.delenv("PRUNER_LP_CREDENTIALS", raising=False)
        home = tmp_path / "home"
        home.mkdir()
        path = home / "bot.credentials"
        path.write_text(BLOB, encoding="utf-8")
        path.chmod(0o600)
        monkeypatch.setenv("HOME", str(home))
        config = config_with({"credentials_file": "~/bot.credentials"})
        source = resolve_credential(config)
        assert source.path == path


class TestSecretContainment:
    def test_blob_does_not_leak_through_repr(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("PRUNER_LP_CREDENTIALS", BLOB)
        source = resolve_credential(Config())
        rendered = repr(source) + str(source) + source.model_dump_json()
        assert "access_secret" not in rendered
        assert "sec" not in rendered.split("origin")[1]  # after the origin field

    def test_literal_token_in_config_is_rejected_with_guidance(self) -> None:
        for key in ("token", "access_token", "secret", "consumer_secret"):
            with pytest.raises(ValueError, match="must not contain credential"):
                config_with({key: "oauth:whatever"})


class TestEnvIsolation:
    """The tests above mutate the process environment; make sure a real
    developer machine's variable cannot leak in and flip a result."""

    @pytest.fixture(autouse=True)
    def _clean_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in ("PRUNER_LP_CREDENTIALS", "LP_CREDENTIALS_FILE"):
            monkeypatch.delenv(name, raising=False)
        assert os.environ.get("PRUNER_LP_CREDENTIALS") is None
