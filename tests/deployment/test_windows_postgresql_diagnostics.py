"""Connection diagnostics publish fixed hints, never libpq text or connection data."""

from __future__ import annotations

import json
from pathlib import Path

import psycopg
import pytest

from deployment.windows_installer import provision
from deployment import windows_stage9_clean_install as clean_install


@pytest.mark.parametrize(
    ("message", "expected"),
    (
        ("fe_sendauth: no password supplied", "PASSWORD_UNAVAILABLE"),
        ("password authentication failed", "PASSWORD_REJECTED"),
        ("could not open password file", "PASSFILE_UNREADABLE"),
        ("root certificate file does not exist", "TLS_ROOT_CERTIFICATE_UNREADABLE"),
        ("could not load private key file", "TLS_CLIENT_CERTIFICATE_UNREADABLE"),
        ("could not load SSL client certificate", "TLS_CLIENT_CERTIFICATE_UNREADABLE"),
        ("server certificate does not match host name", "TLS_SERVER_NAME_MISMATCH"),
        ("SSL error: certificate verify failed", "TLS_SERVER_CERTIFICATE_REJECTED"),
        ("server does not support SSL", "TLS_UNAVAILABLE"),
        ("connection refused", "LISTENER_REFUSED"),
        ("target machine actively refused it", "LISTENER_REFUSED"),
        ("timeout expired", "CONNECTION_TIMEOUT"),
        ("Permission denied", "PERMISSION_DENIED"),
        ("Access is denied", "PERMISSION_DENIED"),
        ("unrecognized or localized provider detail", "UNCLASSIFIED"),
    ),
)
def test_postgresql_diagnostic_never_publishes_provider_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys, message: str, expected: str
) -> None:
    monkeypatch.setenv("ProgramData", str(tmp_path))
    secret = "synthetic-private-connection-material"
    error = psycopg.OperationalError(message + ": " + secret)
    outer = RuntimeError(secret)
    outer.__cause__ = error
    provision.write_safe_failure("install", "POSTGRES_CREATE_DATABASE", outer)
    value = provision.read_safe_failure()
    assert value is not None
    assert value["first_exception"] == "OperationalError"
    assert value["postgresql_connection_failure"] == expected
    logs = tmp_path / "logs"
    logs.mkdir()
    clean_install._capture_provision_failure(logs)
    output = capsys.readouterr().out
    assert expected in output
    assert secret not in output
    assert secret not in provision.safe_failure_path().read_text()
    assert json.loads((logs / "provision-failure.json").read_text()) == value


def test_connection_hint_refuses_unknown_fields_values_and_wrong_stage() -> None:
    value = {
        "schema_version": 1,
        "operation": "install",
        "stage": "POSTGRES_CREATE_DATABASE",
        "first_exception": "OperationalError",
        "postgresql_connection_failure": "PASSWORD_UNAVAILABLE",
    }
    assert provision._valid_safe_failure(value)
    for update in (
        {"postgresql_connection_failure": "provider-secret"},
        {"postgresql_connection_failure": []},
        {"stage": "POSTGRES_INITDB"},
        {"dsn": "provider-secret"},
    ):
        assert not provision._valid_safe_failure({**value, **update})


def test_connection_hint_handles_exception_cycles_and_unrenderable_provider() -> None:
    class BrokenError(psycopg.OperationalError):
        def __str__(self) -> str:
            raise ValueError("synthetic secret")

    assert provision._postgresql_connection_failure(BrokenError()) == "UNCLASSIFIED"
    error = RuntimeError("synthetic secret")
    error.__cause__ = error
    assert provision._postgresql_connection_failure(error) == "UNCLASSIFIED"
