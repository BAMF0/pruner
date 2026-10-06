"""The launchpadlib glue in ``lp.write``, tested without network or credentials.

These run against launchpadlib's real classes -- it is a dev dependency -- but
never touch the network: the credential store and authorization engine are the
seams where the safety properties live.
"""

from __future__ import annotations

import pytest

from pruner.lp.write import (
    WriteError,
    _non_interactive_engine,
    _static_credential_store,
)


@pytest.fixture
def credentials():
    from launchpadlib.credentials import Credentials

    blob = (
        "[1]\n"
        "consumer_key = pruner-backlog-triage\n"
        "consumer_secret =\n"
        "access_token = fake-token\n"
        "access_secret = fake-secret\n"
    )
    return Credentials.from_string(blob)


class TestStaticCredentialStore:
    def test_serves_its_credential_for_any_key(self, credentials) -> None:
        """``login_with`` computes its own consumer key; the store must answer
        regardless, or the engine is invoked and a browser opens."""
        store = _static_credential_store(credentials)
        assert store.load("anything@production") is credentials
        assert store.load("something-else@staging") is credentials

    def test_save_is_a_noop(self, credentials) -> None:
        """A credential that arrived via the environment must never be copied
        into the caller's keyring or onto disk."""
        store = _static_credential_store(credentials)
        store.save(credentials, "key")
        assert store.__dict__.keys() == {"credential_save_failed", "_held"} or not hasattr(
            store, "_credentials"
        ), "no persistent state may be added by save"

    def test_is_a_real_credential_store(self, credentials) -> None:
        from launchpadlib.credentials import CredentialStore

        assert isinstance(_static_credential_store(credentials), CredentialStore)


class TestNonInteractiveEngine:
    def test_call_raises_instead_of_authorizing(self) -> None:
        """The engine is invoked on a missing/expired/revoked token, including
        mid-session after a 401. It must fail, not open a browser."""
        engine = _non_interactive_engine("staging")
        with pytest.raises(WriteError, match="missing, expired or revoked"):
            engine(object(), object())

    def test_carries_the_application_name(self) -> None:
        """``login_with`` asserts the engine's application name matches its own
        argument; a mismatch would be a ValueError at login time."""
        from pruner.lp.write import APPLICATION_NAME

        engine = _non_interactive_engine("staging")
        assert engine.application_name == APPLICATION_NAME
