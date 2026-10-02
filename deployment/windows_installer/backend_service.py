"""Production SCM entrypoint for the continuously running CoreHost boundary."""

from __future__ import annotations
import json
import os
import threading
import psycopg
from deployment.windows_installer.service_base import (
    build_smoke_requested, machine_root, run_service, service_logger
)
from deployment.windows_installer.contract import CONTRACT
from deployment.windows_installer.corehost_composition import (
    build_production_core_host,
    load_windows_external_provisioning_handoff,
)

SERVICE_NAME = "CryptoHunterBackend"


def _run(stop: threading.Event, report: object) -> None:
    root = machine_root()
    log = service_logger(SERVICE_NAME)
    security = root / "Security"
    client = security / "runtime"
    with psycopg.connect(
        host=CONTRACT.postgresql_host,
        port=CONTRACT.postgresql_port,
        dbname="freshness_gate",
        user="freshness_runtime",
        sslmode="verify-full",
        sslrootcert=str(security / "ca.crt"),
        sslcert=str(client / "client.crt"),
        sslkey=str(client / "client.key"),
    ) as connection:
        if connection.execute("SELECT session_user,current_user").fetchone() != (
            "freshness_runtime",
            "freshness_runtime",
        ):
            raise RuntimeError("backend database identity differs")
    for forbidden_role in ("freshness_crypto_verifier", "postgres"):
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
            raise RuntimeError(f"runtime certificate entered forbidden role: {forbidden_role}")
    try:
        (security / "verifier" / "client.key").read_bytes()
    except OSError:
        pass
    else:
        raise RuntimeError("backend can read verifier private key")
    state_path = root / "State" / "corehost.sqlite"
    provisioning = load_windows_external_provisioning_handoff()
    host = build_production_core_host(state_path, provisioning)
    with host:
        import win32service  # type: ignore[import-not-found]

        report(win32service.SERVICE_RUNNING)  # type: ignore[operator]
        readiness = root / "Runtime" / "backend-readiness.json"
        temporary = readiness.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "db_role": "freshness_runtime",
                    "corehost_lock": host.owns_process_lock,
                    "cross_key_denied": True,
                    "cross_roles_denied": True,
                }
            ),
            encoding="utf-8",
        )
        os.replace(temporary, readiness)
        log.info("CoreHost production service started")
        while not stop.wait(1.0):
            pass
        log.info("CoreHost production service stopping")


def main() -> None:
    if build_smoke_requested():
        return
    run_service(SERVICE_NAME, _run)


if __name__ == "__main__":
    main()
