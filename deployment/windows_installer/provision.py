"""Elevated, non-interactive Stage-9 installation transaction provisioner."""

from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import ipaddress
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
from typing import Any, Callable
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID, ExtendedKeyUsageOID
from bot_core.postgresql_freshness_authority import (
    PostgreSQLConnectionConfig,
    provision_postgresql_freshness_authority,
    qualify_postgresql_freshness_authority,
)
from deployment.windows_installer.contract import CONTRACT
from deployment.windows_installer.postgresql_service import (
    SERVICE_NAME as POSTGRESQL_SERVICE_NAME,
    STARTUP_DIAGNOSTIC,
    startup_diagnostic_path,
)
from deployment.windows_installer.corehost_composition import (
    load_windows_external_provisioning_handoff,
    materialize_canonical_pre_state,
    resolve_accepted_handoff,
)
from deployment.windows_dacl_qualification import (
    ADMINISTRATORS_SID,
    FILE_ALL_ACCESS,
    FILE_GENERIC_EXECUTE,
    FILE_GENERIC_READ,
    FILE_GENERIC_WRITE,
    DELETE,
    SYSTEM_SID,
    expected_aces,
)

TRANSACTION_JOURNAL = ".CryptoHunter.stage9-transaction.json"
SAFE_FAILURE_DIAGNOSTIC = ".CryptoHunter.stage9-install-failure.json"
OWNERSHIP_RECORD = ".stage9-install-ownership.json"
ENROLLMENT_FINALIZATION = "enrollment-finalization.json"
SCHEMA = 1
SAFE_FAILURE_FIELDS = frozenset(
    {"schema_version", "operation", "stage", "first_exception"}
)
SERVICES = (CONTRACT.postgresql_service, CONTRACT.backend_service, CONTRACT.verifier_service)
from deployment.windows_postgresql_auth_contract import (
    final_hba_lines,
    ident_lines,
    qualify_hba_rows,
)

FINAL_HBA = final_hba_lines("freshness_gate")
IDENT = ident_lines()


class ProvisionError(RuntimeError):
    pass


def _ignore_stage(stage: str) -> None:
    del stage


def _first_exception_type(exc: BaseException) -> str:
    """Return the oldest chained failure type without exposing sensitive text."""
    seen: set[int] = set()
    current = exc
    while id(current) not in seen:
        seen.add(id(current))
        earlier = current.__cause__ or current.__context__
        if earlier is None:
            break
        current = earlier
    return type(current).__name__


def _write(path: Path, value: dict[str, Any]) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    os.replace(temp, path)


def safe_failure_path() -> Path:
    """Return the fixed machine-level diagnostic path; CLI paths cannot redirect it."""
    return Path(os.environ["ProgramData"]) / SAFE_FAILURE_DIAGNOSTIC


def _valid_safe_failure(value: object) -> bool:
    return (
        isinstance(value, dict)
        and set(value) == SAFE_FAILURE_FIELDS
        and value.get("schema_version") == SCHEMA
        and value.get("operation") == "install"
        and all(
            isinstance(value.get(field), str) and bool(value[field])
            for field in ("stage", "first_exception")
        )
    )


def read_safe_failure(path: Path | None = None) -> dict[str, Any] | None:
    """Read only the strict, secret-free schema from the canonical path."""
    canonical = safe_failure_path()
    if path is not None and path.resolve() != canonical.resolve():
        raise ProvisionError("safe diagnostic path is not canonical")
    try:
        value = json.loads(canonical.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return None
    return value if _valid_safe_failure(value) else None


def clear_safe_failure() -> None:
    """Remove only a canonical diagnostic which has the exact approved schema."""
    path = safe_failure_path()
    if read_safe_failure() is not None:
        path.unlink(missing_ok=True)


def write_safe_failure(operation: str, stage: str, exc: BaseException) -> None:
    """Atomically persist allowlisted failure metadata, never exception text."""
    _write(
        safe_failure_path(),
        {
            "schema_version": SCHEMA,
            "operation": operation,
            "stage": stage,
            "first_exception": _first_exception_type(exc),
        },
    )


def transaction_path(program_data: Path) -> Path:
    return program_data.parent / TRANSACTION_JOURNAL


def create_journal(program_files: Path, program_data: Path) -> tuple[Path, dict[str, Any]]:
    path = transaction_path(program_data)
    if path.exists() or program_data.exists():
        raise ProvisionError("clean-install target pre-exists")
    value = {
        "schema_version": SCHEMA,
        "transaction_id": secrets.token_hex(32),
        "program_files": str(program_files.resolve()),
        "program_data": str(program_data.resolve()),
        "service_names": list(SERVICES),
        "resources_created": [],
        "state": "PROVISIONING",
    }
    _write(path, value)
    if os.name == "nt":
        grants = {ADMINISTRATORS_SID: FILE_ALL_ACCESS, SYSTEM_SID: FILE_ALL_ACCESS}
        _protect(path, grants, inherit=False)
        _qualify_protected(path, grants, inherit=False)
    return path, value


def checkpoint(path: Path, journal: dict[str, Any], resource: Path) -> None:
    journal["resources_created"].append(str(resource.resolve()))
    _write(path, journal)


def _service_sids() -> dict[str, str]:
    import win32security  # type: ignore[import-not-found]

    result = {}
    for name in SERVICES:
        sid, _, _ = win32security.LookupAccountName(None, rf"NT SERVICE\{name}")
        result[name] = win32security.ConvertSidToStringSid(sid)
    return result


def _protect(path: Path, grants: dict[str, int], *, inherit: bool) -> None:
    import win32security  # type: ignore[import-not-found]

    acl = win32security.ACL()
    flags = 3 if inherit else 0
    for principal, mask in grants.items():
        sid = (
            win32security.ConvertStringSidToSid(principal)
            if principal.startswith("S-")
            else win32security.LookupAccountName(None, principal)[0]
        )
        acl.AddAccessAllowedAceEx(win32security.ACL_REVISION_DS, flags, mask, sid)
    owner = win32security.ConvertStringSidToSid("S-1-5-32-544")
    win32security.SetNamedSecurityInfo(
        str(path),
        win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION
        | win32security.DACL_SECURITY_INFORMATION
        | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
        owner,
        None,
        acl,
        None,
    )


def _qualify_protected(path: Path, grants: dict[str, int], *, inherit: bool) -> None:
    import win32security  # type: ignore[import-not-found]

    descriptor = win32security.GetNamedSecurityInfo(
        str(path),
        win32security.SE_FILE_OBJECT,
        win32security.OWNER_SECURITY_INFORMATION | win32security.DACL_SECURITY_INFORMATION,
    )
    control, _ = descriptor.GetSecurityDescriptorControl()
    if not control & win32security.SE_DACL_PROTECTED:
        raise ProvisionError(f"unprotected DACL: {path}")
    dacl = descriptor.GetSecurityDescriptorDacl()
    expected_flags = 3 if inherit else 0
    observed = {}
    for index in range(dacl.GetAceCount()):
        header, mask, sid = dacl.GetAce(index)
        if header[0] != win32security.ACCESS_ALLOWED_ACE_TYPE or header[1] != expected_flags:
            raise ProvisionError(f"unexpected ACE: {path}")
        observed[win32security.ConvertSidToStringSid(sid)] = mask
    expected = {
        (
            principal
            if principal.startswith("S-")
            else win32security.ConvertSidToStringSid(
                win32security.LookupAccountName(None, principal)[0]
            )
        ): mask
        for principal, mask in grants.items()
    }
    if observed != expected or dacl.GetAceCount() != len(expected):
        raise ProvisionError(f"DACL differs: {path}")


def _certificate(
    common_name: str, issuer: x509.Name, issuer_key: Any, *, server: bool = False
) -> tuple[bytes, bytes]:
    key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    now = datetime.now(timezone.utc)
    subject = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)])
    builder = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=825))
    )
    eku = ExtendedKeyUsageOID.SERVER_AUTH if server else ExtendedKeyUsageOID.CLIENT_AUTH
    builder = builder.add_extension(x509.ExtendedKeyUsage([eku]), critical=True)
    if server:
        builder = builder.add_extension(
            x509.SubjectAlternativeName(
                [x509.IPAddress(ipaddress.ip_address(CONTRACT.postgresql_host))]
            ),
            critical=False,
        )
    cert = builder.sign(issuer_key, hashes.SHA256())
    return (
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        ),
        cert.public_bytes(serialization.Encoding.PEM),
    )


def _pki(
    security: Path, sids: dict[str, str], set_stage: Callable[[str], None] = _ignore_stage
) -> None:
    set_stage("CREATE_LOCAL_PKI")
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=3072)
    now = datetime.now(timezone.utc)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "CryptoHunter Local CA")])
    ca = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=3650))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    (security / "ca.crt").write_bytes(ca.public_bytes(serialization.Encoding.PEM))
    (security / "ca.key").write_bytes(
        ca_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    for folder, identity, service, server in (
        ("server", "127.0.0.1", CONTRACT.postgresql_service, True),
        ("runtime", CONTRACT.backend_service, CONTRACT.backend_service, False),
        ("verifier", CONTRACT.verifier_service, CONTRACT.verifier_service, False),
    ):
        set_stage("CREATE_LOCAL_PKI")
        target = security / folder
        target.mkdir()
        key, cert = _certificate(identity, ca_name, ca_key, server=server)
        stem = "server" if server else "client"
        (target / f"{stem}.key").write_bytes(key)
        (target / f"{stem}.crt").write_bytes(cert)
        set_stage("QUALIFY_LOCAL_PKI")
        directory_grants = {
            ADMINISTRATORS_SID: FILE_ALL_ACCESS,
            SYSTEM_SID: FILE_ALL_ACCESS,
            sids[service]: FILE_GENERIC_READ | FILE_GENERIC_EXECUTE,
        }
        _protect(target, directory_grants, inherit=True)
        _qualify_protected(target, directory_grants, inherit=True)
        certificate_grants = {
            ADMINISTRATORS_SID: FILE_ALL_ACCESS,
            SYSTEM_SID: FILE_ALL_ACCESS,
            sids[service]: FILE_GENERIC_READ,
        }
        _protect(target / f"{stem}.crt", certificate_grants, inherit=False)
        _qualify_protected(target / f"{stem}.crt", certificate_grants, inherit=False)
        _protect(
            target / f"{stem}.key",
            {SYSTEM_SID: FILE_ALL_ACCESS, sids[service]: FILE_GENERIC_READ},
            inherit=False,
        )
        _qualify_protected(
            target / f"{stem}.key",
            {SYSTEM_SID: FILE_ALL_ACCESS, sids[service]: FILE_GENERIC_READ},
            inherit=False,
        )
    _protect(security / "ca.key", {SYSTEM_SID: FILE_ALL_ACCESS}, inherit=False)
    _qualify_protected(security / "ca.key", {SYSTEM_SID: FILE_ALL_ACCESS}, inherit=False)


def _configure_database(
    program_files: Path,
    data: Path,
    security: Path,
    set_stage: Callable[[str], None] = _ignore_stage,
) -> None:
    bindir = program_files / "PostgreSQL" / "bin"
    initdb = bindir / "initdb.exe"
    pg_ctl = bindir / "pg_ctl.exe"
    set_stage("POSTGRES_INITDB")
    subprocess.run(
        [
            str(initdb),
            "-D",
            str(data),
            "--encoding=UTF8",
            "--auth-local=reject",
            "--auth-host=reject",
            "--username=postgres",
        ],
        check=True,
        timeout=120,
    )
    ca = (security / "ca.crt").as_posix()
    cert = (security / "server/server.crt").as_posix()
    key = (security / "server/server.key").as_posix()
    (data / "postgresql.conf").write_text(
        (data / "postgresql.conf").read_text()
        + f"\nlisten_addresses='127.0.0.1'\nport={CONTRACT.postgresql_port}\nssl=on\nssl_ca_file='{ca}'\nssl_cert_file='{cert}'\nssl_key_file='{key}'\n",
        encoding="utf-8",
    )
    # Bootstrap exists only while the postmaster is installer-owned and is replaced before final start.
    (data / "pg_hba.conf").write_text("host all postgres 127.0.0.1/32 trust\n", encoding="utf-8")
    set_stage("POSTGRES_BOOTSTRAP_START")
    subprocess.run(
        [str(pg_ctl), "start", "-D", str(data), "-w", "-t", "60"], check=True, timeout=70
    )
    primary_failure: BaseException | None = None
    try:
        dsn = f"host=127.0.0.1 port={CONTRACT.postgresql_port} dbname=postgres user=postgres"
        import psycopg

        set_stage("POSTGRES_CREATE_DATABASE")
        with psycopg.connect(dsn, autocommit=True) as connection:
            connection.execute("CREATE DATABASE freshness_gate")
        authority = PostgreSQLConnectionConfig(
            f"host=127.0.0.1 port={CONTRACT.postgresql_port} dbname=freshness_gate user=postgres"
        )
        set_stage("POSTGRES_PROVISION_FRESHNESS_AUTHORITY")
        provision_postgresql_freshness_authority(authority)
        set_stage("POSTGRES_QUALIFY_FRESHNESS_AUTHORITY")
        qualify_postgresql_freshness_authority(authority)
        # Open the final parser-qualification session before removing bootstrap
        # access. PostgreSQL applies a replaced HBA immediately to new sessions,
        # while this already-authenticated session remains valid.
        with psycopg.connect(dsn, autocommit=True) as connection:
            identity = connection.execute(
                "SELECT session_user,current_database()"
            ).fetchone()
            if tuple(identity) != ("postgres", "postgres"):
                raise ProvisionError("bootstrap administrator session qualification failed")
            set_stage("POSTGRES_FINAL_HBA_WRITE")
            (data / "pg_hba.conf").write_text(
                "\n".join(FINAL_HBA) + "\n", encoding="utf-8"
            )
            (data / "pg_ident.conf").write_text(
                "\n".join(IDENT) + "\n", encoding="utf-8"
            )
            set_stage("POSTGRES_FINAL_HBA_QUALIFY")
            rows = connection.execute(
                "SELECT line_number,type,database,user_name,address,netmask,auth_method,options,error "
                "FROM pg_catalog.pg_hba_file_rules ORDER BY line_number"
            ).fetchall()
            try:
                qualify_hba_rows(rows, "freshness_gate")
            except ValueError as exc:
                raise ProvisionError("final HBA parser qualification failed") from exc
            set_stage("POSTGRES_FINAL_ADMIN_REJECTION")
            try:
                unexpected = psycopg.connect(dsn)
            except psycopg.OperationalError as exc:
                rejection = str(exc).lower()
                if getattr(exc, "sqlstate", None) != "28000" and not any(
                    marker in rejection
                    for marker in (
                        "no pg_hba.conf entry",
                        "pg_hba.conf rejects connection",
                        "requires a valid client certificate",
                        "certificate authentication failed",
                    )
                ):
                    raise ProvisionError(
                        "new administrator connection did not prove HBA rejection"
                    ) from exc
                # Distinguish an authentication rejection from server unavailability.
                if connection.execute("SELECT 1").fetchone() != (1,):
                    raise ProvisionError("bootstrap PostgreSQL liveness proof failed")
            else:
                # This path is a contract violation, but do not leak the
                # unexpectedly accepted privileged session while reporting it.
                unexpected.close()
                raise ProvisionError("final HBA accepted a new administrator connection")
    except BaseException as exc:
        primary_failure = exc
        raise
    finally:
        if primary_failure is None:
            set_stage("POSTGRES_BOOTSTRAP_STOP")
        try:
            subprocess.run(
                [str(pg_ctl), "stop", "-D", str(data), "-m", "fast", "-w", "-t", "30"],
                check=True,
                timeout=40,
            )
        except BaseException:
            if primary_failure is None:
                raise


def _recovery() -> None:
    import win32service  # type: ignore[import-not-found]

    manager = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
    service = win32service.OpenService(
        manager,
        CONTRACT.backend_service,
        win32service.SERVICE_CHANGE_CONFIG | win32service.SERVICE_QUERY_CONFIG,
    )
    try:
        policy = {
            "ResetPeriod": 86400,
            "RebootMsg": "",
            "Command": "",
            "Actions": [(win32service.SC_ACTION_RESTART, d) for d in (1000, 5000, 30000)]
            + [(win32service.SC_ACTION_NONE, 0)],
        }
        win32service.ChangeServiceConfig2(
            service, win32service.SERVICE_CONFIG_FAILURE_ACTIONS, policy
        )
        observed = win32service.QueryServiceConfig2(
            service, win32service.SERVICE_CONFIG_FAILURE_ACTIONS
        )
        if observed["ResetPeriod"] != 86400 or observed["Actions"] != policy["Actions"]:
            raise ProvisionError("recovery read-back mismatch")
    finally:
        win32service.CloseServiceHandle(service)
        win32service.CloseServiceHandle(manager)


def _commit_backend_start_type() -> None:
    """Idempotently set and read back the enrolled backend start type."""
    import win32service  # type: ignore[import-not-found]

    manager = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
    service = win32service.OpenService(
        manager,
        CONTRACT.backend_service,
        win32service.SERVICE_CHANGE_CONFIG | win32service.SERVICE_QUERY_CONFIG,
    )
    try:
        if win32service.QueryServiceConfig(service)[1] != win32service.SERVICE_AUTO_START:
            win32service.ChangeServiceConfig(
                service,
                win32service.SERVICE_NO_CHANGE,
                win32service.SERVICE_AUTO_START,
                win32service.SERVICE_NO_CHANGE,
                None,
                None,
                0,
                None,
                None,
                None,
                None,
            )
        if win32service.QueryServiceConfig(service)[1] != win32service.SERVICE_AUTO_START:
            raise ProvisionError("backend start type read-back mismatch")
    finally:
        win32service.CloseServiceHandle(service)
        win32service.CloseServiceHandle(manager)


def _ensure_backend_running() -> None:
    import win32service  # type: ignore[import-not-found]

    manager = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
    service = win32service.OpenService(
        manager,
        CONTRACT.backend_service,
        win32service.SERVICE_QUERY_STATUS | win32service.SERVICE_START,
    )
    try:
        if win32service.QueryServiceStatus(service)[1] != win32service.SERVICE_RUNNING:
            win32service.StartService(service, None)
    finally:
        win32service.CloseServiceHandle(service)
        win32service.CloseServiceHandle(manager)
    _wait_service(CONTRACT.backend_service, win32service.SERVICE_RUNNING)


def _enrollment_record_path(program_data: Path) -> Path:
    return program_data / "State" / ENROLLMENT_FINALIZATION


def _write_enrollment_phase(path: Path, record: dict[str, Any], phase: str) -> None:
    _write(path, {**record, "phase": phase})


def _verify_backend_activation() -> None:
    import win32service  # type: ignore[import-not-found]

    manager = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
    service = win32service.OpenService(
        manager,
        CONTRACT.backend_service,
        win32service.SERVICE_QUERY_CONFIG | win32service.SERVICE_QUERY_STATUS,
    )
    try:
        if win32service.QueryServiceConfig(service)[1] != win32service.SERVICE_AUTO_START:
            raise ProvisionError("backend start type read-back mismatch")
        if win32service.QueryServiceStatus(service)[1] != win32service.SERVICE_RUNNING:
            raise ProvisionError("backend running read-back mismatch")
    finally:
        win32service.CloseServiceHandle(service)
        win32service.CloseServiceHandle(manager)
    _recovery()


def install(
    program_files: Path,
    program_data: Path,
    set_stage: Callable[[str], None] = _ignore_stage,
) -> None:
    """Install machine-scoped resources without creating product authority."""
    set_stage("CREATE_JOURNAL")
    journal_path, journal = create_journal(program_files, program_data)
    set_stage("RESOLVE_SERVICE_SIDS")
    sids = _service_sids()
    set_stage("PREPARE_POSTGRES_SERVICE_DIAGNOSTIC")
    diagnostic = startup_diagnostic_path()
    diagnostic.write_text(
        json.dumps(
            {
                "service": POSTGRESQL_SERVICE_NAME,
                "stage": "SERVICE_ENTRY",
                "first_exception": "NONE",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    diagnostic_grants = {
        ADMINISTRATORS_SID: FILE_ALL_ACCESS,
        SYSTEM_SID: FILE_ALL_ACCESS,
        sids[CONTRACT.postgresql_service]: FILE_GENERIC_READ | FILE_GENERIC_WRITE,
    }
    _protect(diagnostic, diagnostic_grants, inherit=False)
    _qualify_protected(diagnostic, diagnostic_grants, inherit=False)
    set_stage("CREATE_PROGRAMDATA_LAYOUT")
    program_data.mkdir()
    checkpoint(journal_path, journal, program_data)
    for name in ("Config", "State", "Logs", "Runtime", "Updates", "PostgreSQL", "Security"):
        path = program_data / name
        path.mkdir()
        checkpoint(journal_path, journal, path)
    data = program_data / "PostgreSQL" / "Data"
    data.mkdir()
    checkpoint(journal_path, journal, data)
    verifier_runtime = program_data / "Runtime" / "Verifier"
    verifier_runtime.mkdir()
    checkpoint(journal_path, journal, verifier_runtime)
    set_stage("APPLY_BASE_DACL")
    base = {ADMINISTRATORS_SID: FILE_ALL_ACCESS, SYSTEM_SID: FILE_ALL_ACCESS}
    policies = {}
    for role in ("CONFIG", "STATE", "RUNTIME", "LOGS"):
        policies[role.title()] = {
            sid: mask for sid, mask, _ in expected_aces(role, sids[CONTRACT.backend_service])
        }
    policies["Updates"] = base
    security_readers = {
        sids[CONTRACT.backend_service]: FILE_GENERIC_READ | FILE_GENERIC_EXECUTE,
        sids[CONTRACT.verifier_service]: FILE_GENERIC_READ | FILE_GENERIC_EXECUTE,
        sids[CONTRACT.postgresql_service]: FILE_GENERIC_READ | FILE_GENERIC_EXECUTE,
    }
    policies["Security"] = {**base, **security_readers}
    for name, grants in policies.items():
        _protect(program_data / name, grants, inherit=True)
        _qualify_protected(program_data / name, grants, inherit=True)
    verifier_runtime_grants = {
        ADMINISTRATORS_SID: FILE_ALL_ACCESS,
        SYSTEM_SID: FILE_ALL_ACCESS,
        sids[CONTRACT.verifier_service]: (
            FILE_GENERIC_READ | FILE_GENERIC_EXECUTE | FILE_GENERIC_WRITE | DELETE
        ),
    }
    _protect(verifier_runtime, verifier_runtime_grants, inherit=True)
    _qualify_protected(verifier_runtime, verifier_runtime_grants, inherit=True)
    pg_grants = {**base, sids[CONTRACT.postgresql_service]: FILE_ALL_ACCESS}
    _protect(program_data / "PostgreSQL", pg_grants, inherit=True)
    _protect(data, pg_grants, inherit=True)
    _qualify_protected(data, pg_grants, inherit=True)
    _pki(program_data / "Security", sids, set_stage)
    ca_public = {**base, **{sid: FILE_GENERIC_READ for sid in security_readers}}
    _protect(program_data / "Security" / "ca.crt", ca_public, inherit=False)
    _qualify_protected(program_data / "Security" / "ca.crt", ca_public, inherit=False)
    _configure_database(program_files, data, program_data / "Security", set_stage)
    set_stage("FINALIZE_PROVISIONING_JOURNAL")
    journal["state"] = "PROVISIONED"
    _write(journal_path, journal)


def enroll(program_files: Path, program_data: Path) -> None:
    """Resume durable local finalization of accepted production enrollment."""
    _committed_record(program_data)
    state = program_data / "State" / "corehost.sqlite"
    provisioning = load_windows_external_provisioning_handoff()
    reference, claim, _ = resolve_accepted_handoff(provisioning)
    path = _enrollment_record_path(program_data)
    identity = {
        "schema_version": 1,
        "claim_reference": reference,
        "account_id": claim.account_id,
        "device_installation_id": claim.device_installation_id,
    }
    if path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        if any(record.get(key) != value for key, value in identity.items()):
            raise ProvisionError("enrollment finalization identity conflict")
    else:
        record = {**identity, "phase": "AUTHORITY_ACCEPTED"}
        _write_enrollment_phase(path, record, "AUTHORITY_ACCEPTED")
    phase = record.get("phase")
    phases = (
        "AUTHORITY_ACCEPTED",
        "PRE_MATERIALIZED",
        "BACKEND_START_TYPE_COMMITTED",
        "RECOVERY_POLICY_COMMITTED",
        "BACKEND_RUNNING_VERIFIED",
        "COMMITTED",
    )
    if phase not in phases:
        raise ProvisionError("enrollment finalization phase invalid")
    materialize_canonical_pre_state(state, provisioning)
    if phases.index(str(phase)) < phases.index("PRE_MATERIALIZED"):
        _write_enrollment_phase(path, record, "PRE_MATERIALIZED")
    _commit_backend_start_type()
    if phases.index(str(phase)) < phases.index("BACKEND_START_TYPE_COMMITTED"):
        _write_enrollment_phase(path, record, "BACKEND_START_TYPE_COMMITTED")
    _recovery()
    if phases.index(str(phase)) < phases.index("RECOVERY_POLICY_COMMITTED"):
        _write_enrollment_phase(path, record, "RECOVERY_POLICY_COMMITTED")
    _ensure_backend_running()
    if phases.index(str(phase)) < phases.index("BACKEND_RUNNING_VERIFIED"):
        _write_enrollment_phase(path, record, "BACKEND_RUNNING_VERIFIED")
    _verify_backend_activation()
    _write_enrollment_phase(path, record, "COMMITTED")


def rollback(program_files: Path, program_data: Path) -> None:
    path = transaction_path(program_data)
    if not path.is_file():
        return
    value = json.loads(path.read_text())
    expected = {
        "schema_version": SCHEMA,
        "program_files": str(program_files.resolve()),
        "program_data": str(program_data.resolve()),
        "service_names": list(SERVICES),
    }
    if any(value.get(k) != v for k, v in expected.items()) or value.get("state") not in {
        "PROVISIONING",
        "PROVISIONED",
    }:
        return
    roots = [Path(p) for p in value.get("resources_created", [])]
    if any(
        program_data.resolve() not in p.resolve().parents and p.resolve() != program_data.resolve()
        for p in roots
    ):
        raise ProvisionError("rollback journal escaped ProgramData")
    for resource in reversed(roots):
        if resource.exists():
            shutil.rmtree(resource) if resource.is_dir() else resource.unlink()
    path.unlink(missing_ok=True)


def commit(program_files: Path, program_data: Path) -> None:
    path = transaction_path(program_data)
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("state") != "PROVISIONED" or value.get("program_files") != str(
        program_files.resolve()
    ):
        raise ProvisionError("only a provisioned transaction may commit")
    value["state"] = "COMMITTED"
    ownership = program_data / OWNERSHIP_RECORD
    _write(ownership, value)
    if os.name == "nt":
        grants = {ADMINISTRATORS_SID: FILE_ALL_ACCESS, SYSTEM_SID: FILE_ALL_ACCESS}
        _protect(ownership, grants, inherit=False)
        _qualify_protected(ownership, grants, inherit=False)
    path.unlink()
    (program_data.parent / STARTUP_DIAGNOSTIC).unlink(missing_ok=True)


def _committed_record(program_data: Path) -> dict[str, Any]:
    journal = json.loads((program_data / OWNERSHIP_RECORD).read_text(encoding="utf-8"))
    if journal.get("state") != "COMMITTED" or journal.get("program_data") != str(
        program_data.resolve()
    ):
        raise ProvisionError("committed ownership record absent")
    return journal


def qualify_dacl(program_files: Path, program_data: Path) -> None:
    _committed_record(program_data)
    data = program_data / "PostgreSQL" / "Data"
    sids = _service_sids()
    for role in ("CONFIG", "STATE", "RUNTIME", "LOGS"):
        grants = {sid: mask for sid, mask, _ in expected_aces(role, sids[CONTRACT.backend_service])}
        _qualify_protected(program_data / role.title(), grants, inherit=True)
    base = {ADMINISTRATORS_SID: FILE_ALL_ACCESS, SYSTEM_SID: FILE_ALL_ACCESS}
    _qualify_protected(
        data, {**base, sids[CONTRACT.postgresql_service]: FILE_ALL_ACCESS}, inherit=True
    )
    security = program_data / "Security"
    readers = {sids[name]: FILE_GENERIC_READ | FILE_GENERIC_EXECUTE for name in SERVICES}
    _qualify_protected(security, {**base, **readers}, inherit=True)
    _qualify_protected(
        security / "ca.crt", {**base, **{sid: FILE_GENERIC_READ for sid in readers}}, inherit=False
    )
    for folder, stem, service in (
        ("runtime", "client", CONTRACT.backend_service),
        ("verifier", "client", CONTRACT.verifier_service),
        ("server", "server", CONTRACT.postgresql_service),
    ):
        _qualify_protected(
            security / folder,
            {**base, sids[service]: FILE_GENERIC_READ | FILE_GENERIC_EXECUTE},
            inherit=True,
        )
        _qualify_protected(
            security / folder / f"{stem}.crt",
            {**base, sids[service]: FILE_GENERIC_READ},
            inherit=False,
        )
        _qualify_protected(
            security / folder / f"{stem}.key",
            {SYSTEM_SID: FILE_ALL_ACCESS, sids[service]: FILE_GENERIC_READ},
            inherit=False,
        )
    _qualify_protected(security / "ca.key", {SYSTEM_SID: FILE_ALL_ACCESS}, inherit=False)


def qualify(program_files: Path, program_data: Path) -> None:
    """Aggregate read-only final qualification; proof authorities call narrower functions."""
    qualify_dacl(program_files, program_data)
    qualify_postgresql(program_files, program_data)
    _recovery()


def _wait_service(name: str, state: int, timeout: float = 30.0) -> dict[str, Any]:
    import time
    import win32service  # type: ignore[import-not-found]

    manager = win32service.OpenSCManager(None, None, win32service.SC_MANAGER_CONNECT)
    service = win32service.OpenService(manager, name, win32service.SERVICE_QUERY_STATUS)
    try:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            value = win32service.QueryServiceStatusEx(service)
            if value["CurrentState"] == state:
                return value
            time.sleep(0.1)
        raise ProvisionError(f"service state deadline expired: {name}")
    finally:
        win32service.CloseServiceHandle(service)
        win32service.CloseServiceHandle(manager)


def qualify_postgresql(program_files: Path, program_data: Path) -> None:
    import win32service  # type: ignore[import-not-found]

    _wait_service(CONTRACT.postgresql_service, win32service.SERVICE_RUNNING)
    data = program_data / "PostgreSQL" / "Data"
    if tuple(filter(None, (data / "pg_hba.conf").read_text().splitlines())) != FINAL_HBA:
        raise ProvisionError("final HBA differs")
    if tuple(filter(None, (data / "pg_ident.conf").read_text().splitlines())) != IDENT:
        raise ProvisionError("final ident differs")
    postgres = program_files / "PostgreSQL" / "bin" / "postgres.exe"
    if (
        " 17.11"
        not in subprocess.run(
            [str(postgres), "--version"], check=True, capture_output=True, text=True
        ).stdout
    ):
        raise ProvisionError("private PostgreSQL version differs")
    pg_isready = postgres.with_name("pg_isready.exe")
    subprocess.run(
        [
            str(pg_isready),
            "-h",
            CONTRACT.postgresql_host,
            "-p",
            str(CONTRACT.postgresql_port),
            "-d",
            "freshness_gate",
        ],
        check=True,
        timeout=10,
    )


def qualify_mtls(program_files: Path, program_data: Path) -> None:
    import time
    import psycopg

    subprocess.run(["sc.exe", "start", CONTRACT.verifier_service], check=True, capture_output=True)
    markers = (
        program_data / "Runtime" / "backend-readiness.json",
        program_data / "Runtime" / "Verifier" / "verifier-readiness.json",
    )
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline and not all(path.is_file() for path in markers):
        time.sleep(0.1)
    values = [json.loads(path.read_text()) for path in markers]
    expected_roles = ("freshness_runtime", "freshness_crypto_verifier")
    if any(
        value.get("db_role") != role
        or value.get("cross_key_denied") is not True
        or value.get("cross_roles_denied") is not True
        for value, role in zip(values, expected_roles, strict=True)
    ):
        raise ProvisionError("service-token mTLS matrix marker differs")
    for role in (*expected_roles, "postgres"):
        try:
            psycopg.connect(
                host=CONTRACT.postgresql_host,
                port=CONTRACT.postgresql_port,
                dbname="freshness_gate",
                user=role,
                sslmode="verify-full",
                sslrootcert=str(program_data / "Security" / "ca.crt"),
                connect_timeout=5,
            )
        except psycopg.Error:
            pass
        else:
            raise ProvisionError(f"interactive no-cert connection entered {role}")
    subprocess.run(["sc.exe", "stop", CONTRACT.verifier_service], check=True, capture_output=True)


def qualify_backend(program_files: Path, program_data: Path) -> None:
    import time
    import win32service  # type: ignore[import-not-found]

    first = _wait_service(CONTRACT.backend_service, win32service.SERVICE_RUNNING)
    marker = json.loads((program_data / "Runtime" / "backend-readiness.json").read_text())
    if marker.get("pid") != first["ProcessId"] or marker.get("corehost_lock") is not True:
        raise ProvisionError("backend readiness identity differs")
    time.sleep(3)
    second = _wait_service(CONTRACT.backend_service, win32service.SERVICE_RUNNING)
    if second["ProcessId"] != first["ProcessId"]:
        raise ProvisionError("backend was not stable")


def qualify_logging(program_files: Path, program_data: Path) -> None:
    import win32service  # type: ignore[import-not-found]

    log = program_data / "Logs" / "backend.log"
    before = log.read_bytes()
    subprocess.run(["sc.exe", "stop", CONTRACT.backend_service], check=True, capture_output=True)
    _wait_service(CONTRACT.backend_service, win32service.SERVICE_STOPPED)
    subprocess.run(["sc.exe", "start", CONTRACT.backend_service], check=True, capture_output=True)
    _wait_service(CONTRACT.backend_service, win32service.SERVICE_RUNNING)
    after = log.read_bytes()
    if len(after) <= len(before) or not after.startswith(before):
        raise ProvisionError("persistent backend log did not survive restart")
    sids = _service_sids()
    grants = {sid: mask for sid, mask, _ in expected_aces("LOGS", sids[CONTRACT.backend_service])}
    _qualify_protected(program_data / "Logs", grants, inherit=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "install",
            "enroll",
            "rollback",
            "commit",
            "qualify",
            "qualify-dacl",
            "qualify-postgresql",
            "qualify-mtls",
            "qualify-backend",
            "qualify-logging",
        ),
    )
    parser.add_argument("--program-files", required=True, type=Path)
    parser.add_argument("--program-data", required=True, type=Path)
    args = parser.parse_args(argv)
    if os.name != "nt":
        print("Windows required", file=sys.stderr)
        return 1
    current_stage = "UNCLASSIFIED"

    def set_stage(stage: str) -> None:
        nonlocal current_stage
        current_stage = stage

    try:
        if args.command == "install":
            clear_safe_failure()
            install(args.program_files, args.program_data, set_stage)
        elif args.command == "enroll":
            enroll(args.program_files, args.program_data)
        elif args.command == "rollback":
            rollback(args.program_files, args.program_data)
        elif args.command == "commit":
            commit(args.program_files, args.program_data)
            clear_safe_failure()
        elif args.command == "qualify":
            qualify(args.program_files, args.program_data)
        else:
            {
                "qualify-dacl": qualify_dacl,
                "qualify-postgresql": qualify_postgresql,
                "qualify-mtls": qualify_mtls,
                "qualify-backend": qualify_backend,
                "qualify-logging": qualify_logging,
            }[args.command](args.program_files, args.program_data)
    except Exception as exc:
        # Preserve the first failure class without leaking provider exception
        # text, which may contain credentials or private ceremony material.
        if args.command == "install":
            write_safe_failure(args.command, current_stage, exc)
        print(
            f"provisioning failed: operation={args.command} "
            f"first_exception={_first_exception_type(exc)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
