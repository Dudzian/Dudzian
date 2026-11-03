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
import re
import shutil
import socket
import stat
import struct
import subprocess
import sys
from xml.etree import ElementTree


class ProofFailure(RuntimeError):
    """An assertion failed; exception text is never published."""


def require(value: bool) -> None:
    if not value:
        raise ProofFailure


OWNERSHIP_MARKER = ".cryptohunter-isolated-postgresql-bootstrap-owner"
OWNERSHIP_CONTENT = "cryptohunter-isolated-postgresql-bootstrap-v1\n"


def _reject_reparse(path: Path) -> None:
    metadata = path.lstat()
    require(not stat.S_ISLNK(metadata.st_mode))
    require(not getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def temporary_root(root: Path) -> Path:
    """Accept only an ordinary strict descendant of the runner's temp storage."""
    temporary = Path(os.environ["RUNNER_TEMP"])
    require(temporary.is_absolute() and root.is_absolute())
    require(".." not in temporary.parts and ".." not in root.parts)
    for path in (temporary, *temporary.parents):
        _reject_reparse(path)
    require(temporary.is_dir())
    require(root != temporary and temporary in root.parents)
    require(root.parent.is_dir())
    for path in (root, *root.parents):
        try:
            _reject_reparse(path)
        except FileNotFoundError:
            require(path == root)
        if path == temporary:
            break
    return root


def owned_root(root: Path) -> Path:
    root = temporary_root(root)
    require(root.is_dir())
    marker = root / OWNERSHIP_MARKER
    _reject_reparse(marker)
    require(marker.is_file())
    require(marker.read_text(encoding="ascii") == OWNERSHIP_CONTENT)
    return root


def cleanup_owned_root(root: Path) -> None:
    """Remove only this harness's marked directory, never an arbitrary CLI path."""
    root = owned_root(root)
    for directory, names, filenames in os.walk(root, followlinks=False):
        for name in (*names, *filenames):
            _reject_reparse(Path(directory) / name)
    shutil.rmtree(root)


def failure_category(exc: BaseException) -> str:
    """Inspect locally but never export provider messages or arbitrary attributes."""
    if isinstance(exc, PermissionError) or getattr(exc, "winerror", None) == 5:
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


def failure_type(exc: BaseException) -> str:
    """Publish only a fixed type category, never an arbitrary class name."""
    if isinstance(exc, ImportError):
        return "IMPORT_ERROR"
    if isinstance(exc, AttributeError):
        return "ATTRIBUTE_ERROR"
    if isinstance(exc, KeyError):
        return "KEY_ERROR"
    if isinstance(exc, TypeError):
        return "TYPE_ERROR"
    if isinstance(exc, ValueError):
        return "VALUE_ERROR"
    if isinstance(exc, ProofFailure):
        return "PROOF_FAILURE"
    if type(exc).__module__ == "pywintypes" and type(exc).__name__ == "error":
        return "PYWIN32_ERROR"
    if type(exc).__name__ == "ProvisionError":
        return "PROVISION_ERROR"
    return "OTHER_ERROR"


KNOWN_WINERRORS = frozenset({5, 6, 87, 1008, 1300, 1314})


def qualify_endpoint(connection, endpoint: tuple[str, int]) -> None:
    """Read the real libpq destination and the authenticated server's transport."""
    host, port = endpoint
    require(connection.pgconn.hostaddr == host.encode("ascii"))
    require(connection.pgconn.used_password)
    parameters = {option.keyword: option.val for option in connection.pgconn.info}
    require(parameters[b"sslmode"] == b"verify-full")
    require(parameters[b"gssencmode"] == b"disable")
    # inet::text includes a /32 mask even for a single IPv4 address. host()
    # returns only the address, allowing the same exact endpoint comparison.
    address, actual_port, ssl, version = connection.execute(
        "SELECT pg_catalog.host(inet_server_addr()),inet_server_port(),ssl,version "
        "FROM pg_stat_ssl WHERE pid=pg_backend_pid()"
    ).fetchone()
    require((address, actual_port) == endpoint)
    require(ssl is True and version in {"TLSv1.2", "TLSv1.3"})
    verifier = connection.execute(
        "SELECT rolpassword FROM pg_authid WHERE rolname='postgres'"
    ).fetchone()[0]
    require(verifier.startswith("SCRAM-SHA-256$"))
    # Observing a fresh production connection must not leave a transaction open.
    if not connection.autocommit:
        connection.rollback()


def probe(psycopg, dsn: str, *, endpoint: tuple[str, int] | None = None, **overrides) -> str:
    try:
        with psycopg.connect(dsn, autocommit=True, **overrides) as connection:
            require(connection.execute("SELECT session_user").fetchone() == ("postgres",))
            if endpoint is not None:
                qualify_endpoint(connection, endpoint)
    except Exception as exc:
        return failure_category(exc)
    return "CONNECTED"


def prove_unpinned_libpq_route(psycopg, root: Path) -> None:
    """Show actual ambient routing using a decoy that never receives a secret."""
    from psycopg.conninfo import make_conninfo

    foreign = "127.0.0.2"
    require(os.environ.get("PGHOSTADDR") == foreign)
    with socket.socket() as listener:
        listener.bind((foreign, 0))
        listener.listen(1)
        listener.settimeout(5)
        # No real passfile or password is supplied to the unpinned control.
        # connect_start is never polled, so no authentication can take place.
        dsn = make_conninfo(
            host="127.0.0.1",
            port=listener.getsockname()[1],
            user="postgres",
            dbname="postgres",
            password="",
            passfile=str(root / "absent-route-control-passfile"),
            sslmode="verify-full",
            sslrootcert=str(root / "absent-route-control-ca"),
            gssencmode="disable",
            connect_timeout=5,
        )
        connection = psycopg.pq.PGconn.connect_start(dsn.encode("utf-8"))
        try:
            require(connection.hostaddr == foreign.encode("ascii"))
            require(not connection.password)
            with listener.accept()[0] as accepted:
                require(accepted.getsockname()[0] == foreign)
        finally:
            connection.finish()


def prepare(root: Path, report: dict[str, object]) -> None:
    report["stage"] = "VALIDATE_TEMPORARY_ROOT"
    root = temporary_root(root)
    require(not root.exists())
    root.mkdir()
    with (root / OWNERSHIP_MARKER).open("x", encoding="ascii", newline="\n") as marker:
        marker.write(OWNERSHIP_CONTENT)
    from deployment.windows_installer import build

    pins = json.loads((Path(build.__file__).parent / "pins.json").read_text(encoding="utf-8"))
    report["stage"] = "ACQUIRE_PINNED_POSTGRESQL"
    archive = build.acquire_postgresql(pins, None, root / "download")
    report["stage"] = "EXTRACT_PINNED_POSTGRESQL"
    build.normalize_postgresql_archive(archive, root / "payload" / "PostgreSQL")
    report["archive_sha256_verified"] = True
    report["result"] = "PREPARED"


NATIVE_CREDENTIAL_ACL_TEST = (
    "test_native_windows_private_credentials_deny_anonymous_and_survive_crash_privately"
)


def record_native_credential_case_diagnostics(document, report: dict[str, object]) -> None:
    """Publish only known case labels, fixed result classes and source line numbers."""
    cases = list(document.iter("testcase"))
    for value, field in ((False, "native_acl_normal"), (True, "native_acl_crash")):
        selected = [
            case for case in cases if case.get("name") == NATIVE_CREDENTIAL_ACL_TEST + f"[{value}]"
        ]
        if len(selected) != 1:
            report[field + "_status"] = "CASE_MISSING" if not selected else "CASE_DUPLICATE"
            continue
        case = selected[0]
        error = case.find("error")
        failure = case.find("failure")
        if case.find("skipped") is not None:
            status = "CASE_SKIPPED"
        elif error is not None:
            status = "CASE_ERROR"
        elif failure is not None:
            status = "CASE_FAILED"
        else:
            status = "CASE_PASSED"
        report[field + "_status"] = status
        problem = error if error is not None else failure
        if problem is None:
            continue
        detail = problem.text or ""
        last_line = detail.rstrip().split("\n")[-1]
        kind = "OTHER_ERROR"
        for pattern, category in (
            (r"(?:^|:\s*)PermissionError\s*$", "PERMISSION_ERROR"),
            (r"(?:^|:\s*)AssertionError\s*$", "ASSERTION_ERROR"),
            (r"(?:^|:\s*)(?:[\w.]+\.)?ProvisionError\s*$", "PROVISION_ERROR"),
            (r"(?:^|:\s*)pywintypes\.error\s*$", "PYWIN32_ERROR"),
        ):
            if re.search(pattern, last_line):
                kind = category
                break
        report[field + "_failure_type"] = kind
        source_lines = re.findall(r"test_windows_bootstrap_credentials\.py:(\d+):", detail)
        if source_lines:
            line = int(source_lines[-1])
            if 1 <= line <= 1000:
                report[field + "_failure_line"] = line


def qualify_native_credential_acl_crash(root: Path, report: dict[str, object]) -> None:
    """Require both actual foreign-token/crash cases; skipped tests are not proof."""
    root = owned_root(root)
    report["stage"] = "NATIVE_CREDENTIAL_ACL_CRASH"
    results = root / "credential-acl-results.xml"
    environment = {
        **os.environ,
        "CRYPTOHUNTER_RUN_WINDOWS_BOOTSTRAP_ACL_INTEGRATION": "1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    }
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--confcutdir=tests/deployment",
            "-q",
            "-o",
            "tmp_path_retention_policy=failed",
            "tests/deployment/test_windows_bootstrap_credentials.py::" + NATIVE_CREDENTIAL_ACL_TEST,
            "--basetemp",
            str(root / "credential-acl"),
            "--junitxml",
            str(results),
        ],
        cwd=Path(__file__).resolve().parents[2],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        timeout=90,
        check=False,
    )
    if type(completed.returncode) is int and completed.returncode in range(6):
        report["native_acl_pytest_exit"] = completed.returncode
    document = ElementTree.parse(results).getroot()
    record_native_credential_case_diagnostics(document, report)
    require(completed.returncode == 0)
    require(document.tag in {"testsuite", "testsuites"})
    cases = list(document.iter("testcase"))
    require(len(cases) == 2)
    require(
        {case.get("name") for case in cases}
        == {NATIVE_CREDENTIAL_ACL_TEST + "[False]", NATIVE_CREDENTIAL_ACL_TEST + "[True]"}
    )
    for suite in document.iter():
        if suite.tag in {"testsuite", "testsuites"}:
            for attribute in ("errors", "failures", "skipped"):
                require(int(suite.get(attribute, "0")) == 0)
            if "tests" in suite.attrib:
                require(int(suite.attrib["tests"]) == len(list(suite.iter("testcase"))))
    for case in cases:
        require(not any(case.find(tag) is not None for tag in ("skipped", "error", "failure")))
    report["native_credential_acl_crash"] = True


def run(root: Path, report: dict[str, object]) -> None:
    report["stage"] = "VALIDATE_ROOT"
    root = owned_root(root)
    report["stage"] = "IMPORT_POSTGRESQL_DRIVER"
    import psycopg
    from psycopg.conninfo import make_conninfo

    report["stage"] = "IMPORT_BOOTSTRAP"
    from deployment.windows_installer import provision

    report["stage"] = "VALIDATE_WINDOWS_ARCH"
    require(sys.platform == "win32" and struct.calcsize("P") == 8)
    require(platform.machine().upper() in {"AMD64", "X86_64"})
    report["stage"] = "REQUIRE_LOCAL_SYSTEM"
    provision._require_bootstrap_system()
    report["effective_local_system"] = True
    qualify_native_credential_acl_crash(root, report)
    ambient_password = os.environ.get("PGPASSWORD")
    report["ambient_password_present"] = ambient_password is not None
    report["ambient_password_synthetic"] = not bool(ambient_password)
    # This dedicated single-threaded proof process owns its test environment;
    # the production installer never mutates global PG* variables.
    for name in tuple(os.environ):
        if name.startswith("PG"):
            del os.environ[name]
    # Keep the real machine credential when present. Otherwise use a controlled
    # mismatching value to exercise the same libpq precedence without exporting
    # either password. The colon cannot occur in token_urlsafe bootstrap values.
    wrong_ambient_password = ambient_password or (provision.secrets.token_urlsafe(48) + ":")
    os.environ["PGPASSWORD"] = wrong_ambient_password
    # GSS must not replace verified TLS even when explicitly requested by env.
    os.environ["PGGSSENCMODE"] = "require"
    report["stage"] = "SELECT_LOOPBACK_PORT"
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
    report["stage"] = "PROTECT_TEMPORARY_ROOT"
    provision._protect(root, grants, inherit=True)
    provision._qualify_protected(root, grants, inherit=True)
    report["stage"] = "PREPARE_TEMPORARY_DIRECTORIES"
    data.mkdir()
    security.mkdir()
    provision._protect(data, grants, inherit=True)
    provision._protect(security, grants, inherit=True)

    observe_production = False
    production_endpoint_checks = 0
    secrets_to_check = [wrong_ambient_password]

    def stage(value: str) -> None:
        nonlocal observe_production
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
        secrets_to_check.append(password)
        dsn = make_conninfo(
            host="127.0.0.1",
            hostaddr=provision.CONTRACT.postgresql_host,
            port=port,
            user="postgres",
            dbname="postgres",
            sslmode="verify-full",
            sslrootcert=str(security / "ca.crt"),
            gssencmode="disable",
            passfile=str(passfile),
            connect_timeout=5,
        )
        report["ambient_password_connection"] = probe(psycopg, dsn)
        # An explicit empty password suppresses the environment fallback while
        # still asking libpq to obtain the credential from its explicit passfile.
        endpoint = (provision.CONTRACT.postgresql_host, port)
        report["passfile_connection"] = probe(psycopg, dsn, password="", endpoint=endpoint)
        for label, with_password, with_hostaddr in (
            ("pgpassword", True, False),
            ("pghostaddr", False, True),
            ("combined", True, True),
        ):
            if with_password:
                os.environ["PGPASSWORD"] = wrong_ambient_password
            else:
                os.environ.pop("PGPASSWORD", None)
            if with_hostaddr:
                os.environ["PGHOSTADDR"] = "127.0.0.2"
            else:
                os.environ.pop("PGHOSTADDR", None)
            report[label + "_passfile_connection"] = probe(
                psycopg, dsn, password="", endpoint=endpoint
            )
            require(report[label + "_passfile_connection"] == "CONNECTED")
        # Both poisoned values stay set for every actual production SQL call.
        prove_unpinned_libpq_route(psycopg, root)
        report["libpq_unpinned_foreign_endpoint_observed"] = True
        report["libpq_pinned_endpoint_verified"] = True
        report["scram_verifier_verified"] = True
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
        observe_production = True

    real_connect = psycopg.connect

    def production_connect(*args, **kwargs):
        nonlocal production_endpoint_checks
        connection = real_connect(*args, **kwargs)
        if observe_production:
            try:
                qualify_endpoint(connection, (provision.CONTRACT.postgresql_host, port))
            except BaseException:
                connection.close()
                raise
            production_endpoint_checks += 1
        return connection

    provision._pki(security, sids, stage)
    bootstrap_error = None
    try:
        # Instrumentation delegates every operation to real psycopg/libpq; it
        # checks the production DSN's destination, rather than simulating it.
        psycopg.connect = production_connect
        provision._configure_database(root / "payload", data, security, stage)
        report["final_hba_matches"] = (data / "pg_hba.conf").read_text(
            encoding="utf-8"
        ).splitlines() == list(provision.FINAL_HBA)
    except Exception as exc:
        bootstrap_error = exc
        report["bootstrap_failure"] = failure_category(exc)
    finally:
        psycopg.connect = real_connect
        report["credentials_removed"] = not list(security.glob(".postgres-bootstrap-*"))
        report["postmaster_stopped"] = not (data / "postmaster.pid").exists()
    if bootstrap_error is not None:
        raise bootstrap_error
    require(report.get("ambient_password_connection") == "SCRAM_PASSWORD_REJECTED")
    require(report.get("passfile_connection") == "CONNECTED")
    require(report.get("memory_password_connection") == "CONNECTED")
    require(report.get("missing_password") == "LIBPQ_NO_PASSWORD")
    require(report.get("wrong_password") == "SCRAM_PASSWORD_REJECTED")
    require(report.get("wrong_ca") == "TLS_CERTIFICATE_REJECTED")
    require(report.get("tls_disabled") == "HBA_REJECTED")
    require(production_endpoint_checks >= 4)
    report["production_libpq_endpoint_verified"] = True
    require(report.get("final_hba_matches") is True)
    require(report.get("credentials_removed") is True)
    require(report.get("postmaster_stopped") is True)
    require(all(value not in json.dumps(report) for value in secrets_to_check))
    report["report_secrets_absent"] = True
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
    parser.add_argument("--runner-temp", type=Path)
    args = parser.parse_args()
    report: dict[str, object] = {"schema_version": 1, "result": "FAIL", "stage": "SETUP"}
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    code = 1
    job = None
    with open(os.devnull, "w", encoding="utf-8") as sink:
        with redirect_stdout(sink), redirect_stderr(sink):
            try:
                report["stage"] = "VALIDATE_PLATFORM"
                require(sys.platform == "win32")
                if args.runner_temp is not None:
                    os.environ["RUNNER_TEMP"] = str(args.runner_temp)
                if args.prepare:
                    prepare(args.root, report)
                else:
                    report["stage"] = "IMPORT_WIN32_JOB"
                    import win32api
                    import win32job

                    # Every disposable postmaster stays in this kill-on-close
                    # job, including after abrupt scheduled-task termination.
                    report["stage"] = "CREATE_JOB_OBJECT"
                    job = win32job.CreateJobObject(None, "")
                    report["stage"] = "QUERY_JOB_OBJECT"
                    limits = win32job.QueryInformationJobObject(
                        job, win32job.JobObjectExtendedLimitInformation
                    )
                    limits["BasicLimitInformation"]["LimitFlags"] |= (
                        win32job.JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
                    )
                    report["stage"] = "CONFIGURE_JOB_OBJECT"
                    win32job.SetInformationJobObject(
                        job, win32job.JobObjectExtendedLimitInformation, limits
                    )
                    report["stage"] = "ASSIGN_JOB_OBJECT"
                    win32job.AssignProcessToJobObject(job, win32api.GetCurrentProcess())
                    run(args.root, report)
                code = 0
            except Exception as exc:
                report["failure"] = failure_category(exc)
                report["failure_type"] = failure_type(exc)
                winerror = getattr(exc, "winerror", None)
                if type(winerror) is int and winerror in KNOWN_WINERRORS:
                    report["failure_winerror"] = winerror
            finally:
                if not args.prepare and report.get("effective_local_system") is True:
                    try:
                        cleanup_owned_root(args.root)
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
