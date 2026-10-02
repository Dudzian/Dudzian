"""Demand-start SCM boundary for the frozen semantic verifier role."""

from __future__ import annotations
import json
import os
import threading
from dataclasses import dataclass
from typing import Any
import psycopg
from bot_core.freshness_semantic_verifier import (
    FreshnessSemanticVerifier,
    PostgreSQLPreparationWriter,
    PostgreSQLRetainedCredentialResolver,
)
from deployment.windows_installer.contract import CONTRACT
from deployment.windows_installer.service_base import build_smoke_requested, machine_root, run_service

SERVICE_NAME = "CryptoHunterFreshnessVerifier"


@dataclass(frozen=True)
class WindowsVerifierConnection:
    def connect(self) -> Any:
        security = machine_root() / "Security"
        client = security / "verifier"
        connection = psycopg.connect(
            host=CONTRACT.postgresql_host,
            port=CONTRACT.postgresql_port,
            dbname="freshness_gate",
            user="freshness_crypto_verifier",
            sslmode="verify-full",
            sslrootcert=str(security / "ca.crt"),
            sslcert=str(client / "client.crt"),
            sslkey=str(client / "client.key"),
        )
        facts = connection.execute(
            "SELECT session_user,current_user,inet_server_addr()::text"
        ).fetchone()
        if facts != ("freshness_crypto_verifier", "freshness_crypto_verifier", "127.0.0.1"):
            connection.close()
            raise RuntimeError("verifier database identity differs")
        return connection


def _run(stop: threading.Event, report: object) -> None:
    connection = WindowsVerifierConnection()
    with connection.connect():
        pass
    security = machine_root() / "Security"
    client = security / "verifier"
    for forbidden_role in ("freshness_runtime", "postgres"):
        try:
            psycopg.connect(
                host=CONTRACT.postgresql_host,
                port=CONTRACT.postgresql_port,
                dbname="freshness_gate",
                user=forbidden_role,
                sslmode="verify-full",
                sslrootcert=str(security / "ca.crt"),
                sslcert=str(client / "client.crt"),
                sslkey=str(client / "client.key"),
                connect_timeout=5,
            )
        except psycopg.Error:
            pass
        else:
            raise RuntimeError(f"verifier certificate entered forbidden role: {forbidden_role}")
    verifier = FreshnessSemanticVerifier(
        PostgreSQLRetainedCredentialResolver(connection),
        PostgreSQLPreparationWriter(connection),
        "semantic-verifier-v1",
    )  # type: ignore[arg-type]
    if not isinstance(verifier, FreshnessSemanticVerifier):
        raise RuntimeError("verifier graph unavailable")
    try:
        (security / "runtime" / "client.key").read_bytes()
    except OSError:
        pass
    else:
        raise RuntimeError("verifier can read runtime private key")
    import win32service  # type: ignore[import-not-found]

    report(win32service.SERVICE_RUNNING)  # type: ignore[operator]
    readiness = machine_root() / "Runtime" / "Verifier" / "verifier-readiness.json"
    temporary = readiness.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "db_role": "freshness_crypto_verifier",
                "cross_key_denied": True,
                "cross_roles_denied": True,
            }
        ),
        encoding="utf-8",
    )
    os.replace(temporary, readiness)
    stop.wait()


def main() -> None:
    if build_smoke_requested():
        return
    run_service(SERVICE_NAME, _run)


if __name__ == "__main__":
    main()
