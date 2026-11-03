"""Disposable native Windows PostgreSQL bootstrap proof; emits only fixed diagnostics.

Run through deployment/windows_postgresql_bootstrap_regression.ps1. This script
does not install an MSI, create services, enroll a host, or qualify Stage 9/10.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import struct
import sys


class ProofFailure(RuntimeError):
    """An assertion failed; exception text is never published."""


def require(value: bool) -> None:
    if not value:
        raise ProofFailure


def failure_category(exc: BaseException) -> str:
    """Inspect locally but never export provider messages or arbitrary attributes."""
    if isinstance(exc, PermissionError):
        return "WINDOWS_ACCESS_DENIED"
    if isinstance(exc, ImportError):
        return "DEPENDENCY_UNAVAILABLE"
    if isinstance(exc, ProofFailure):
        return "PROOF_ASSERTION_FAILED"
    if getattr(exc, "sqlstate", None) == "28P01":
        return "SCRAM_PASSWORD_REJECTED"
    try:
        message = str(exc).lower()
    except Exception:
        return "UNCLASSIFIED_FAILURE"
    if "no password supplied" in message:
        return "LIBPQ_NO_PASSWORD"
    if "password authentication failed" in message:
        return "SCRAM_PASSWORD_REJECTED"
    if "certificate verify failed" in message or "root certificate" in message:
        return "TLS_CERTIFICATE_REJECTED"
    if "does not match host name" in message:
        return "TLS_HOSTNAME_REJECTED"
    if "connection refused" in message or "actively refused" in message:
        return "LISTENER_UNAVAILABLE"
    if "pg_hba.conf rejects connection" in message or "no pg_hba.conf entry" in message:
        return "HBA_REJECTED"
    if "permission denied" in message or "access is denied" in message:
        return "WINDOWS_ACCESS_DENIED"
    return "UNCLASSIFIED_FAILURE"


def probe(psycopg, dsn: str, **overrides) -> str:
    try:
        with psycopg.connect(dsn, autocommit=True, **overrides) as connection:
            require(connection.execute("SELECT session_user").fetchone() == ("postgres",))
    except Exception as exc:
        return failure_category(exc)
    return "CONNECTED"


def prepare(root: Path, report: dict[str, object]) -> None:
    from deployment.windows_installer import build

    require(not root.exists())
    root.mkdir(parents=True)
    pins = json.loads((Path(build.__file__).parent / "pins.json").read_text(encoding="utf-8"))
    archive = build.acquire_postgresql(pins, None, root / "download")
    build.normalize_postgresql_archive(archive, root / "payload" / "PostgreSQL")
    report["archive_sha256_verified"] = True
    report["result"] = "PREPARED"


def run(root: Path, report: dict[str, object]) -> None:
    import psycopg
    from psycopg.conninfo import make_conninfo

    from deployment.windows_installer import provision

    require(sys.platform == "win32" and struct.calcsize("P") == 8)
    require(platform.machine().upper() in {"AMD64", "X86_64"})
    provision._require_bootstrap_system()
    report["effective_local_system"] = True
    for name in tuple(os.environ):
        if name.startswith("PG"):
            del os.environ[name]
    with socket.socket() as reserve:
        reserve.bind(("127.0.0.1", 0))
        port = reserve.getsockname()[1]
    provision.CONTRACT = replace(provision.CONTRACT, postgresql_port=port)
    grants = {
        provision.ADMINISTRATORS_SID: provision.FILE_ALL_ACCESS,
        provision.SYSTEM_SID: provision.FILE_ALL_ACCESS,
    }
    # Distinct syntactically valid service SIDs model the grants without SCM or
    # account creation. They must not collide with SYSTEM's full-access entry.
    sids = {
        name: f"S-1-5-80-111111111-222222222-333333333-444444444-{index}"
        for index, name in enumerate(provision.SERVICES, 1)
    }
    data, security = root / "data", root / "security"
    provision._protect(root, grants, inherit=True)
    provision._qualify_protected(root, grants, inherit=True)
    data.mkdir()
    security.mkdir()
    provision._protect(data, grants, inherit=True)
    provision._protect(security, grants, inherit=True)

    def stage(value: str) -> None:
        # Only production constants can reach this callback; export no paths.
        report["stage"] = value if value in STAGES else "UNEXPECTED_STAGE"
        if value != "POSTGRES_CREATE_DATABASE":
            return
        with socket.create_connection(("127.0.0.1", port), timeout=5):
            report["listener_available"] = True
        passfiles = list(security.glob(".postgres-bootstrap-*/pgpass"))
        require(len(passfiles) == 1)
        passfile = passfiles[0]
        # Confirm the actual native credential boundary before reading in the
        # already authorized SYSTEM process. No credential leaves this process.
        provision._qualify_bootstrap_directory(passfile.parent)
        report["bootstrap_directory_qualified"] = True
        password = passfile.read_text(encoding="utf-8").rstrip("\n").split(":")[-1]
        dsn = make_conninfo(
            host="127.0.0.1",
            port=port,
            user="postgres",
            dbname="postgres",
            sslmode="verify-full",
            sslrootcert=str(security / "ca.crt"),
            passfile=str(passfile),
            connect_timeout=5,
        )
        report["passfile_connection"] = probe(psycopg, dsn)
        report["memory_password_connection"] = probe(psycopg, dsn, password=password)
        report["missing_password"] = probe(
            psycopg, dsn, passfile=str(root / "absent-passfile"), password=""
        )
        report["wrong_password"] = probe(psycopg, dsn, password=provision.secrets.token_urlsafe(48))
        # The leaf certificate is deliberately not a trust anchor for the CA.
        report["wrong_ca"] = probe(
            psycopg,
            dsn,
            password=password,
            sslrootcert=str(security / "server" / "server.crt"),
        )
        report["tls_disabled"] = probe(psycopg, dsn, password=password, sslmode="disable")

    provision._pki(security, sids, stage)
    bootstrap_error = None
    try:
        provision._configure_database(root / "payload", data, security, stage)
        report["final_hba_matches"] = (data / "pg_hba.conf").read_text(
            encoding="utf-8"
        ).splitlines() == list(provision.FINAL_HBA)
    except Exception as exc:
        bootstrap_error = exc
        report["bootstrap_failure"] = failure_category(exc)
    finally:
        report["credentials_removed"] = not list(security.glob(".postgres-bootstrap-*"))
        report["postmaster_stopped"] = not (data / "postmaster.pid").exists()
    if bootstrap_error is not None:
        raise bootstrap_error
    require(report.get("passfile_connection") == "CONNECTED")
    require(report.get("memory_password_connection") == "CONNECTED")
    require(report.get("missing_password") == "LIBPQ_NO_PASSWORD")
    require(report.get("wrong_password") == "SCRAM_PASSWORD_REJECTED")
    require(report.get("wrong_ca") == "TLS_CERTIFICATE_REJECTED")
    require(report.get("tls_disabled") == "HBA_REJECTED")
    require(report.get("final_hba_matches") is True)
    require(report.get("credentials_removed") is True)
    require(report.get("postmaster_stopped") is True)
    report["result"] = "PASS"


STAGES = frozenset(
    {
        "CREATE_LOCAL_PKI",
        "QUALIFY_LOCAL_PKI",
        "POSTGRES_BOOTSTRAP_CREDENTIALS",
        "POSTGRES_INITDB",
        "POSTGRES_BOOTSTRAP_START",
        "POSTGRES_CREATE_DATABASE",
        "POSTGRES_PROVISION_FRESHNESS_AUTHORITY",
        "POSTGRES_QUALIFY_FRESHNESS_AUTHORITY",
        "POSTGRES_FINAL_HBA_WRITE",
        "POSTGRES_FINAL_HBA_QUALIFY",
        "POSTGRES_FINAL_ADMIN_REJECTION",
        "POSTGRES_BOOTSTRAP_STOP",
    }
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--prepare", action="store_true")
    args = parser.parse_args()
    report: dict[str, object] = {"schema_version": 1, "result": "FAIL", "stage": "SETUP"}
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    code = 1
    job = None
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            try:
                require(sys.platform == "win32")
                if args.prepare:
                    prepare(args.root, report)
                else:
                    import win32api
                    import win32job

                    # Every disposable postmaster stays in this kill-on-close
                    # job, including after abrupt scheduled-task termination.
                    job = win32job.CreateJobObject(None, None)
                    limits = win32job.QueryInformationJobObject(
                        job, win32job.JobObjectExtendedLimitInformation
                    )
                    limits["BasicLimitInformation"]["LimitFlags"] |= (
                        win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                    )
                    win32job.SetInformationJobObject(
                        job, win32job.JobObjectExtendedLimitInformation, limits
                    )
                    win32job.AssignProcessToJobObject(job, win32api.GetCurrentProcess())
                    run(args.root, report)
                code = 0
            except Exception as exc:
                report["failure"] = failure_category(exc)
            finally:
                if not args.prepare and args.root.exists():
                    try:
                        shutil.rmtree(args.root)
                        report["temporary_root_removed"] = True
                    except Exception:
                        report["temporary_root_removed"] = False
                        report["result"] = "FAIL"
                        code = 1
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    # Keep the job handle alive until exit: closing a job containing this Python
    # process early would terminate it before the sanitized report is written.
    if job is not None:
        job.Detach()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
