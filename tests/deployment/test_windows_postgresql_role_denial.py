from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import threading
from typing import Any

import psycopg
import pytest

from deployment.windows_installer import backend_service, verifier_service
from deployment.windows_postgresql_auth_contract import probe_postgresql_role_denial


PRIVATE_DIAGNOSTIC = "private certificate C:\\Security\\client.key DSN password=secret"


def server_error(sqlstate: str | None) -> psycopg.Error:
    if sqlstate is None:
        return psycopg.OperationalError(PRIVATE_DIAGNOSTIC)
    return psycopg.errors.lookup(sqlstate)(PRIVATE_DIAGNOSTIC)


@pytest.mark.parametrize("sqlstate", ["28000", "28P01"])
def test_role_probe_accepts_only_explicit_server_authentication_denial(sqlstate: str) -> None:
    attempted_roles: list[str] = []

    def connect(**options: Any) -> Any:
        attempted_roles.append(options["user"])
        raise server_error(sqlstate)

    assert probe_postgresql_role_denial(connect, role="postgres") == sqlstate
    assert attempted_roles == ["postgres"]


@pytest.mark.parametrize(
    "sqlstate",
    [None, "08001", "08004", "08006", "08P01", "42501", "57014", "57P01", "53300"],
)
def test_connection_failure_is_not_authentication_denial_and_diagnostics_stay_private(
    sqlstate: str | None,
) -> None:
    def connect(**_: Any) -> Any:
        raise server_error(sqlstate)

    with pytest.raises(RuntimeError, match="authentication denial is unproven") as caught:
        probe_postgresql_role_denial(connect, role="postgres")
    assert PRIVATE_DIAGNOSTIC not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__suppress_context__ is True


def test_timeout_is_not_authentication_denial() -> None:
    def connect(**_: Any) -> Any:
        raise TimeoutError("connection timeout")

    with pytest.raises(TimeoutError):
        probe_postgresql_role_denial(connect, role="postgres")


def test_successful_forbidden_connection_is_closed_and_rejected() -> None:
    closed: list[bool] = []
    connection = SimpleNamespace(close=lambda: closed.append(True))
    with pytest.raises(RuntimeError, match="certificate entered a forbidden role"):
        probe_postgresql_role_denial(lambda **_: connection, role="postgres")
    assert closed == [True]


class OwnRoleConnection:
    def __init__(self, role: str) -> None:
        self.role = role

    def __enter__(self) -> OwnRoleConnection:
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def execute(self, query: str) -> OwnRoleConnection:
        self.query = query
        return self

    def fetchone(self) -> tuple[str, ...]:
        if "inet_server_addr" in self.query:
            return self.role, self.role, "127.0.0.1"
        return self.role, self.role


@pytest.mark.parametrize("service", [backend_service, verifier_service])
@pytest.mark.parametrize("failed_role", ["opposite", "postgres"])
@pytest.mark.parametrize("sqlstate", [None, "08006", "57014"])
def test_service_never_publishes_readiness_for_inconclusive_cross_role_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    service: Any,
    failed_role: str,
    sqlstate: str | None,
) -> None:
    own_role = "freshness_runtime" if service is backend_service else "freshness_crypto_verifier"
    opposite_role = (
        "freshness_crypto_verifier" if service is backend_service else "freshness_runtime"
    )
    failure_role = opposite_role if failed_role == "opposite" else "postgres"
    attempted_roles: list[str] = []

    def connect(**options: Any) -> Any:
        role = options["user"]
        attempted_roles.append(role)
        if role == own_role:
            return OwnRoleConnection(role)
        assert options["sslmode"] == "verify-full"
        assert options["connect_timeout"] == 5
        raise server_error(sqlstate if role == failure_role else "28000")

    monkeypatch.setattr(service, "machine_root", lambda: tmp_path)
    monkeypatch.setattr(service.psycopg, "connect", connect)
    if service is backend_service:
        monkeypatch.setattr(service, "service_logger", lambda _: SimpleNamespace())
    reported: list[int] = []
    with pytest.raises(RuntimeError, match="authentication denial is unproven"):
        service._run(threading.Event(), reported.append)
    assert attempted_roles[-1] == failure_role
    assert reported == []
    assert list(tmp_path.rglob("*readiness*")) == []


@pytest.mark.parametrize("service", [backend_service, verifier_service])
def test_service_never_publishes_readiness_if_forbidden_role_connects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, service: Any
) -> None:
    own_role = "freshness_runtime" if service is backend_service else "freshness_crypto_verifier"
    closed: list[bool] = []

    def connect(**options: Any) -> Any:
        if options["user"] == own_role:
            return OwnRoleConnection(own_role)
        return SimpleNamespace(close=lambda: closed.append(True))

    monkeypatch.setattr(service, "machine_root", lambda: tmp_path)
    monkeypatch.setattr(service.psycopg, "connect", connect)
    if service is backend_service:
        monkeypatch.setattr(service, "service_logger", lambda _: SimpleNamespace())
    reported: list[int] = []
    with pytest.raises(RuntimeError, match="certificate entered a forbidden role"):
        service._run(threading.Event(), reported.append)
    assert closed == [True]
    assert reported == []
    assert list(tmp_path.rglob("*readiness*")) == []
