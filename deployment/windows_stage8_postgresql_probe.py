"""Live, fail-closed Windows PostgreSQL substrate and service-SID-bound mTLS qualification.

This is an acceptance probe, not an installer.  It owns an isolated temporary
cluster and temporary SCM helper services and never opens or changes a system
PGDATA or a pre-installed PostgreSQL service.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable
from datetime import datetime, timedelta, timezone
import ipaddress

from deployment.host_identity import canonical_host_os
from deployment.platform_evidence import WINDOWS_STAGE8_ITEMS

SUBSTRATE, PRINCIPAL_AUTH = WINDOWS_STAGE8_ITEMS
MINIMUM_MAJOR = 16
COMMAND_TIMEOUT = 30
START_TIMEOUT = 30
SERVICE_TIMEOUT = 20
AUTHORITY_STATEMENT_TIMEOUT_MS = 60_000
AUTHORITY_LOCK_TIMEOUT_MS = 5_000
STDERR_LIMIT = 8192
# postmaster.pid records whole Unix seconds while Win32 FILETIME has sub-second
# precision; three seconds also covers the short spawn-to-pidfile-write interval.
POSTMASTER_START_TOLERANCE_SECONDS = 3.0
DATABASE = "stage8_freshness"
RUNTIME_SERVICE = "CryptoHunterBackend"
VERIFIER_SERVICE = "CryptoHunterFreshnessVerifier"
RUNTIME_ACCOUNT = rf"NT SERVICE\{RUNTIME_SERVICE}"
VERIFIER_ACCOUNT = rf"NT SERVICE\{VERIFIER_SERVICE}"
RUNTIME_ROLE = "freshness_runtime"
VERIFIER_ROLE = "freshness_crypto_verifier"
MAP_NAME = "stage8_cert"
RUNTIME_CERT_IDENTITY = RUNTIME_SERVICE
VERIFIER_CERT_IDENTITY = VERIFIER_SERVICE
# Win32 returns these stable, concrete file masks from GetAce() for the values
# passed to AddAccessAllowedAce().  Keep them independent of pywin32 so the
# exact ACL-shape contract is executable on every CI platform.
PRIVATE_KEY_FULL_MASK = 0x001F01FF
PRIVATE_KEY_READ_MASK = 0x00120089
PRIVATE_KEY_ACE_FLAGS = 0
PRIVATE_KEY_ALLOW_ACE_TYPE = 0
REQUIRED_BINARIES = ("initdb.exe", "postgres.exe", "pg_ctl.exe", "psql.exe")
FORBIDDEN_ONLINE_METHODS = frozenset({"trust", "password", "md5", "scram-sha-256", "sspi"})
DIAGNOSTIC_LABELS = (
    "POSTGRESQL_DISCOVERY",
    "POSTGRESQL_CLUSTER_INIT",
    "POSTGRESQL_START",
    "POSTGRESQL_DURABILITY",
    "POSTGRESQL_SCHEMA_PROVISION",
    "POSTGRESQL_SCHEMA_QUALIFICATION",
    "WINDOWS_TLS_PKI_PROVISION",
    "WINDOWS_TLS_KEY_DACL_QUALIFICATION",
    "WINDOWS_TLS_RUNTIME_CONNECT",
    "WINDOWS_TLS_VERIFIER_CONNECT",
    "WINDOWS_TLS_CROSS_ROLE_DENIAL",
    "WINDOWS_TLS_OUTSIDER_DENIAL",
    "WINDOWS_TLS_HBA_QUALIFICATION",
    "WINDOWS_TLS_IDENT_QUALIFICATION",
    "STAGE8_CLEANUP",
)

PHASES = (
    "POSTGRESQL_DISCOVERY",
    "POSTGRESQL_CLUSTER_INIT",
    "HELPER_SERVICE_PREPARE",
    "TLS_PKI_PROVISION",
    "TLS_KEY_DACL_QUALIFICATION",
    "POSTGRESQL_START",
    "POSTGRESQL_DURABILITY",
    "POSTGRESQL_SCHEMA_PROVISION",
    "POSTGRESQL_SCHEMA_QUALIFICATION",
    "TLS_RUNTIME_KEY_ACCESS",
    "TLS_VERIFIER_KEY_ACCESS",
    "HBA_BOOTSTRAP_SESSION_OPEN",
    "FINAL_IDENT_WRITE",
    "FINAL_HBA_WRITE",
    "FINAL_HBA_PARSER_QUALIFICATION",
    "FINAL_CLUSTER_RESTART",
    "MATRIX_RUNTIME",
    "MATRIX_VERIFIER",
    "MATRIX_RUNTIME_TO_VERIFIER_DENIAL",
    "MATRIX_VERIFIER_TO_RUNTIME_DENIAL",
    "MATRIX_WRONG_ROLE_DENIAL",
    "INTERACTIVE_OUTSIDER_DENIAL",
    "CLEANUP_SERVICES",
    "CLEANUP_POSTGRESQL",
    "CLEANUP_SCRATCH",
)

_status_file: Path | None = None
_status: dict[str, Any] = {}


def _write_status(**updates: Any) -> None:
    """Publish non-secret ownership and progress state for the supervising process."""
    _status.update(updates)
    if _status_file is not None:
        _atomic_json(_status_file, _status)


@contextmanager
def _phase(name: str):
    started = time.monotonic()
    _write_status(last_phase=name, phase_started=started)
    print(f"[STAGE8_PHASE] START {name}", flush=True)
    try:
        yield
    except BaseException:
        elapsed = time.monotonic() - started
        print(f"[STAGE8_PHASE] FAIL {name} elapsed={elapsed:.3f}s", flush=True)
        raise
    else:
        elapsed = time.monotonic() - started
        print(f"[STAGE8_PHASE] PASS {name} elapsed={elapsed:.3f}s", flush=True)


@contextmanager
def _principal_auth_phase(name: str, label: str):
    """Attribute every failure in an mTLS phase to principal authentication."""
    with _phase(name):
        try:
            yield
        except Stage8PostgreSQLProbeError as exc:
            if exc.item == PRINCIPAL_AUTH:
                raise
            raise Stage8PostgreSQLProbeError(label, PRINCIPAL_AUTH, str(exc)) from exc
        except Exception as exc:
            raise Stage8PostgreSQLProbeError(
                label, PRINCIPAL_AUTH, f"{type(exc).__name__}: {exc}"
            ) from exc


class Stage8PostgreSQLProbeError(RuntimeError):
    def __init__(self, label: str, item: str, detail: str) -> None:
        self.label, self.item, self.detail = label, item, detail
        super().__init__(f"[{label}] {detail}")


@dataclass(frozen=True)
class PostgreSQLBinaries:
    bindir: Path
    major: int

    def executable(self, name: str) -> Path:
        return self.bindir / name


@dataclass(frozen=True)
class PgCtlResult:
    """Bounded result from the special no-pipe pg_ctl process boundary."""

    returncode: int | None
    elapsed: float
    timed_out: bool
    stdout_tail: str
    stderr_tail: str


@dataclass(frozen=True)
class PostmasterPidFile:
    pid: int
    data: Path
    start_time: float


@dataclass(frozen=True)
class PostmasterWitness:
    pid: int
    process: Any
    start_time: float
    executable: Path


@dataclass(frozen=True)
class ProcessIdentitySnapshot:
    pid: int
    process: Any
    creation_time: float
    executable: Path


def _tail(path: Path, limit: int = STDERR_LIMIT) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - limit))
            return stream.read(limit).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _run_pg_ctl(
    pg_ctl: Path,
    data: Path,
    diagnostic_root: Path,
    operation: str,
    arguments: list[str],
    *,
    timeout: int = START_TIMEOUT,
) -> PgCtlResult:
    """Run pg_ctl with real files so a postgres child cannot retain a Python pipe."""
    token = f"{operation}-{time.time_ns()}"
    stdout_path = diagnostic_root / f"pg_ctl-{token}.stdout.log"
    stderr_path = diagnostic_root / f"pg_ctl-{token}.stderr.log"
    command = [str(pg_ctl), "-D", str(data), *arguments]
    started = time.monotonic()
    process: subprocess.Popen[bytes] | None = None
    timed_out = False
    returncode: int | None = None
    # Binary file handles are deliberately passed directly to Popen. In particular,
    # neither stream is PIPE and there is no communicate()/pipe-EOF dependency.
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.terminate()
            try:
                returncode = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                try:
                    returncode = process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    returncode = None
    return PgCtlResult(
        returncode=returncode,
        elapsed=time.monotonic() - started,
        timed_out=timed_out,
        stdout_tail=_tail(stdout_path),
        stderr_tail=_tail(stderr_path),
    )


def _run(command: list[str], *, timeout: int = COMMAND_TIMEOUT) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_DISCOVERY", SUBSTRATE, "bounded command timed out"
        ) from exc


def discover_postgresql(environment: dict[str, str] | None = None) -> PostgreSQLBinaries:
    values = os.environ if environment is None else environment
    candidates: list[Path] = []
    if values.get("PGBIN"):
        candidates.append(Path(values["PGBIN"]))
    pg_config = shutil.which("pg_config.exe") or shutil.which("pg_config")
    if pg_config:
        result = _run([pg_config, "--bindir"])
        if result.returncode == 0 and result.stdout.strip():
            candidates.append(Path(result.stdout.strip()))
    for candidate in candidates:
        missing = [name for name in REQUIRED_BINARIES if not (candidate / name).is_file()]
        if missing:
            continue
        version = _run([str(candidate / "postgres.exe"), "--version"])
        match = re.search(r"(\d+)(?:\.\d+)?", version.stdout + version.stderr)
        if version.returncode or not match:
            continue
        major = int(match.group(1))
        if major < MINIMUM_MAJOR:
            raise Stage8PostgreSQLProbeError(
                "POSTGRESQL_DISCOVERY", SUBSTRATE, f"PostgreSQL {major} is below required major 16"
            )
        return PostgreSQLBinaries(candidate.resolve(), major)
    raise Stage8PostgreSQLProbeError(
        "POSTGRESQL_DISCOVERY",
        SUBSTRATE,
        "required PostgreSQL binaries were not found via PGBIN or pg_config",
    )


def final_hba_lines() -> tuple[str, ...]:
    options = f"cert map={MAP_NAME}"
    return (
        f"hostssl {DATABASE} {VERIFIER_ROLE} 127.0.0.1/32 {options}",
        f"hostssl {DATABASE} {RUNTIME_ROLE} 127.0.0.1/32 {options}",
        f"host {DATABASE} all 127.0.0.1/32 reject",
        "host all all 127.0.0.1/32 reject",
        "host all all 0.0.0.0/0 reject",
        "host all all ::0/0 reject",
    )


def ident_lines() -> tuple[str, ...]:
    return (
        f"{MAP_NAME} {RUNTIME_CERT_IDENTITY} {RUNTIME_ROLE}",
        f"{MAP_NAME} {VERIFIER_CERT_IDENTITY} {VERIFIER_ROLE}",
    )


def qualify_hba_rows(rows: list[tuple[Any, ...]]) -> None:
    expected = [
        (
            "hostssl",
            DATABASE,
            VERIFIER_ROLE,
            "127.0.0.1",
            "255.255.255.255",
            "cert",
            [f"map={MAP_NAME}", "clientcert=verify-full"],
        ),
        (
            "hostssl",
            DATABASE,
            RUNTIME_ROLE,
            "127.0.0.1",
            "255.255.255.255",
            "cert",
            [f"map={MAP_NAME}", "clientcert=verify-full"],
        ),
        ("host", DATABASE, "all", "127.0.0.1", "255.255.255.255", "reject", None),
        ("host", "all", "all", "127.0.0.1", "255.255.255.255", "reject", None),
        ("host", "all", "all", "0.0.0.0", "0.0.0.0", "reject", None),
        ("host", "all", "all", "::", "::", "reject", None),
    ]
    if len(rows) != len(expected):
        raise Stage8PostgreSQLProbeError(
            "WINDOWS_TLS_HBA_QUALIFICATION", PRINCIPAL_AUTH, "unexpected effective HBA row count"
        )
    previous = -1
    for index, (row, wanted) in enumerate(zip(rows, expected, strict=True)):
        line, kind, databases, users, address, netmask, method, options, error = row
        (
            expected_kind,
            database,
            user,
            expected_address,
            expected_netmask,
            auth,
            expected_options,
        ) = wanted
        if error is not None or line <= previous or kind != expected_kind:
            raise Stage8PostgreSQLProbeError(
                "WINDOWS_TLS_HBA_QUALIFICATION",
                PRINCIPAL_AUTH,
                "invalid HBA parse result or ordering",
            )
        previous = line
        if (
            databases != [database]
            or users != [user]
            or address != expected_address
            or netmask != expected_netmask
            or method != auth
            or options != expected_options
        ):
            observed = (kind, databases, users, address, netmask, method, options)
            raise Stage8PostgreSQLProbeError(
                "WINDOWS_TLS_HBA_QUALIFICATION",
                PRINCIPAL_AUTH,
                f"HBA row mismatch index={index} expected={wanted!r} observed={observed!r}",
            )
        if auth == "cert" and (address != "127.0.0.1" or netmask != "255.255.255.255"):
            raise Stage8PostgreSQLProbeError(
                "WINDOWS_TLS_HBA_QUALIFICATION",
                PRINCIPAL_AUTH,
                "online HBA is not exact loopback certificate authentication",
            )
        if user in {RUNTIME_ROLE, VERIFIER_ROLE} and method in FORBIDDEN_ONLINE_METHODS:
            raise Stage8PostgreSQLProbeError(
                "WINDOWS_TLS_HBA_QUALIFICATION",
                PRINCIPAL_AUTH,
                "forbidden online authentication method",
            )


def _qualify_hba_via_connection(connection: Any) -> None:
    """Qualify the current HBA using PostgreSQL's parser on an existing session."""
    rows = connection.execute(
        "SELECT line_number, type, database, user_name, address, netmask, "
        "auth_method, options, error FROM pg_catalog.pg_hba_file_rules "
        "ORDER BY line_number"
    ).fetchall()
    qualify_hba_rows(rows)


def _install_and_qualify_final_auth(
    psycopg: Any, port: int, data: Path, pg_ctl: Path, root: Path
) -> None:
    """Lock down auth while retaining one bounded, pre-lockdown parser session."""
    connection: Any | None = None
    try:
        with _principal_auth_phase("HBA_BOOTSTRAP_SESSION_OPEN", "WINDOWS_TLS_HBA_QUALIFICATION"):
            connection = psycopg.connect(_admin_dsn(port, DATABASE), autocommit=True)
            identity = connection.execute("SELECT session_user, current_database()").fetchone()
            if identity != ("stage8_bootstrap", DATABASE):
                raise Stage8PostgreSQLProbeError(
                    "WINDOWS_TLS_HBA_QUALIFICATION",
                    PRINCIPAL_AUTH,
                    f"unexpected bootstrap session identity: {identity}",
                )
        with _principal_auth_phase("FINAL_IDENT_WRITE", "WINDOWS_TLS_IDENT_QUALIFICATION"):
            (data / "pg_ident.conf").write_text("\n".join(ident_lines()) + "\n", encoding="utf-8")
        with _principal_auth_phase("FINAL_HBA_WRITE", "WINDOWS_TLS_HBA_QUALIFICATION"):
            (data / "pg_hba.conf").write_text("\n".join(final_hba_lines()) + "\n", encoding="utf-8")
            reload_result = _run_pg_ctl(pg_ctl, data, root, "reload", ["reload"])
            if reload_result.timed_out or reload_result.returncode != 0:
                raise Stage8PostgreSQLProbeError(
                    "WINDOWS_TLS_HBA_QUALIFICATION",
                    PRINCIPAL_AUTH,
                    "final configuration reload failed",
                )
        with _principal_auth_phase(
            "FINAL_HBA_PARSER_QUALIFICATION", "WINDOWS_TLS_HBA_QUALIFICATION"
        ):
            _qualify_hba_via_connection(connection)
    finally:
        if connection is not None:
            connection.close()


def _write_private_key(path: Path, key: Any) -> None:
    from cryptography.hazmat.primitives import serialization

    path.write_bytes(
        key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.TraditionalOpenSSL,
            serialization.NoEncryption(),
        )
    )


def _provision_tls_pki(root: Path) -> dict[str, Path]:
    """Create and validate an ephemeral CA, server identity and two client identities."""
    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

    pki = root / "pki"
    (pki / "runtime").mkdir(parents=True)
    (pki / "verifier").mkdir()
    now = datetime.now(timezone.utc)
    ca_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    ca_name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "CryptoHunter Stage8 CA")])
    ca_cert = (
        x509.CertificateBuilder()
        .subject_name(ca_name)
        .issuer_name(ca_name)
        .public_key(ca_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - timedelta(minutes=5))
        .not_valid_after(now + timedelta(days=2))
        .add_extension(x509.BasicConstraints(ca=True, path_length=0), critical=True)
        .sign(ca_key, hashes.SHA256())
    )
    assets: dict[str, Path] = {"pki": pki, "ca_cert": pki / "ca.crt", "ca_key": pki / "ca.key"}
    _write_private_key(assets["ca_key"], ca_key)
    assets["ca_cert"].write_bytes(ca_cert.public_bytes(serialization.Encoding.PEM))

    issued: dict[str, tuple[Any, Any]] = {}
    specifications = (
        ("server", "127.0.0.1", ExtendedKeyUsageOID.SERVER_AUTH, pki),
        ("runtime", RUNTIME_CERT_IDENTITY, ExtendedKeyUsageOID.CLIENT_AUTH, pki / "runtime"),
        ("verifier", VERIFIER_CERT_IDENTITY, ExtendedKeyUsageOID.CLIENT_AUTH, pki / "verifier"),
    )
    for name, common_name, eku, directory in specifications:
        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        builder = (
            x509.CertificateBuilder()
            .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)]))
            .issuer_name(ca_name)
            .public_key(key.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(now - timedelta(minutes=5))
            .not_valid_after(now + timedelta(days=1))
            .add_extension(x509.ExtendedKeyUsage([eku]), critical=False)
        )
        if name == "server":
            builder = builder.add_extension(
                x509.SubjectAlternativeName([x509.IPAddress(ipaddress.ip_address("127.0.0.1"))]),
                critical=False,
            )
        certificate = builder.sign(ca_key, hashes.SHA256())
        key_path = directory / ("server.key" if name == "server" else "client.key")
        cert_path = directory / ("server.crt" if name == "server" else "client.crt")
        _write_private_key(key_path, key)
        cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
        assets[f"{name}_key"], assets[f"{name}_cert"] = key_path, cert_path
        issued[name] = (key, certificate)

    # Executable cryptographic qualification: signature, key pairing, SAN, EKU,
    # validity and distinct client certificates are all checked before use.
    fingerprints: set[bytes] = set()
    for name, (key, certificate) in issued.items():
        ca_key.public_key().verify(
            certificate.signature,
            certificate.tbs_certificate_bytes,
            padding.PKCS1v15(),
            certificate.signature_hash_algorithm,
        )
        if key.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        ) != certificate.public_key().public_bytes(
            serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
        ):
            raise RuntimeError(f"{name} private key does not match certificate")
        if not (certificate.not_valid_before_utc <= now <= certificate.not_valid_after_utc):
            raise RuntimeError(f"{name} certificate is outside its validity period")
        if name != "server":
            eku = certificate.extensions.get_extension_for_class(x509.ExtendedKeyUsage).value
            if ExtendedKeyUsageOID.CLIENT_AUTH not in eku:
                raise RuntimeError(f"{name} certificate lacks clientAuth EKU")
            fingerprints.add(certificate.fingerprint(hashes.SHA256()))
    san = issued["server"][1].extensions.get_extension_for_class(x509.SubjectAlternativeName).value
    if ipaddress.ip_address("127.0.0.1") not in san.get_values_for_type(x509.IPAddress):
        raise RuntimeError("server certificate lacks exact 127.0.0.1 IP SAN")
    if len(fingerprints) != 2:
        raise RuntimeError("client certificates are not distinct")
    return assets


def _protect_private_key(path: Path, account: str) -> None:
    """Install a protected exact allow-list DACL on one private key."""
    import ntsecuritycon
    import win32security

    system, _, _ = win32security.LookupAccountName(None, "SYSTEM")
    service, _, _ = win32security.LookupAccountName(None, account)
    dacl = win32security.ACL()
    if (
        ntsecuritycon.FILE_ALL_ACCESS != PRIVATE_KEY_FULL_MASK
        or ntsecuritycon.FILE_GENERIC_READ != PRIVATE_KEY_READ_MASK
    ):
        raise RuntimeError("pywin32 private-key rights differ from the reviewed mask contract")
    dacl.AddAccessAllowedAce(win32security.ACL_REVISION, PRIVATE_KEY_FULL_MASK, system)
    dacl.AddAccessAllowedAce(win32security.ACL_REVISION, PRIVATE_KEY_READ_MASK, service)
    win32security.SetNamedSecurityInfo(
        str(path),
        win32security.SE_FILE_OBJECT,
        win32security.DACL_SECURITY_INFORMATION | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
        None,
        None,
        dacl,
        None,
    )


def _protect_owner_private_key(path: Path) -> None:
    """Keep CA/server material usable by the acceptance/PostgreSQL owner only."""
    import ntsecuritycon
    import win32api
    import win32con
    import win32security

    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        owner = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
    finally:
        token.Close()
    system, _, _ = win32security.LookupAccountName(None, "SYSTEM")
    if ntsecuritycon.FILE_ALL_ACCESS != PRIVATE_KEY_FULL_MASK:
        raise RuntimeError("pywin32 private-key full rights differ from reviewed mask contract")
    dacl = win32security.ACL()
    for sid in {
        win32security.ConvertSidToStringSid(system): system,
        win32security.ConvertSidToStringSid(owner): owner,
    }.values():
        dacl.AddAccessAllowedAce(win32security.ACL_REVISION, PRIVATE_KEY_FULL_MASK, sid)
    win32security.SetNamedSecurityInfo(
        str(path),
        win32security.SE_FILE_OBJECT,
        win32security.DACL_SECURITY_INFORMATION | win32security.PROTECTED_DACL_SECURITY_INFORMATION,
        None,
        None,
        dacl,
        None,
    )


def _qualify_exact_private_key_aces(
    observed: list[tuple[str, int, int, int]],
    expected: list[tuple[str, int, int, int]],
) -> None:
    """Require the exact ACE count, order, SID, type, mask and flags."""
    if len(observed) != len(expected) or observed != expected:
        raise RuntimeError(
            f"private key DACL differs from exact contract: expected={expected!r}, "
            f"observed={observed!r}"
        )


def _account_sid(account: str, win32security: Any) -> str:
    sid, _, _ = win32security.LookupAccountName(None, account)
    return str(win32security.ConvertSidToStringSid(sid))


def _current_process_sid(win32security: Any) -> str:
    import win32api
    import win32con

    token = win32security.OpenProcessToken(win32api.GetCurrentProcess(), win32con.TOKEN_QUERY)
    try:
        sid = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
        return str(win32security.ConvertSidToStringSid(sid))
    finally:
        token.Close()


def _qualify_private_key_dacl(path: Path, expected: list[tuple[str, int]]) -> None:
    import win32security

    descriptor = win32security.GetNamedSecurityInfo(
        str(path), win32security.SE_FILE_OBJECT, win32security.DACL_SECURITY_INFORMATION
    )
    control = descriptor.GetSecurityDescriptorControl()[0]
    if not control & win32security.SE_DACL_PRESENT:
        raise RuntimeError(f"private key DACL is not present: {path.name}")
    if not control & win32security.SE_DACL_PROTECTED:
        raise RuntimeError(f"private key DACL is not protected: {path.name}")
    dacl = descriptor.GetSecurityDescriptorDacl()
    if dacl is None:
        raise RuntimeError(f"private key has a null DACL: {path.name}")
    ace_count = dacl.GetAceCount()
    if ace_count != len(expected):
        raise RuntimeError(
            f"private key DACL has unexpected ACE count: expected={len(expected)}, "
            f"observed={ace_count}"
        )
    observed: list[tuple[str, int, int, int]] = []
    for index in range(ace_count):
        header, mask, sid = dacl.GetAce(index)
        observed.append(
            (
                str(win32security.ConvertSidToStringSid(sid)),
                int(header[0]),
                int(mask),
                int(header[1]),
            )
        )
    exact = [
        (sid, PRIVATE_KEY_ALLOW_ACE_TYPE, mask, PRIVATE_KEY_ACE_FLAGS) for sid, mask in expected
    ]
    _qualify_exact_private_key_aces(observed, exact)


def _port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _postgresql_pid(data: Path) -> int | None:
    try:
        return int((data / "postmaster.pid").read_text(encoding="utf-8").splitlines()[0])
    except (OSError, ValueError, IndexError):
        return None


def _postmaster_pidfile(data: Path) -> PostmasterPidFile | None:
    try:
        lines = (data / "postmaster.pid").read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY", SUBSTRATE, f"could not read owned postmaster.pid: {exc}"
        ) from exc
    if len(lines) < 3:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY", SUBSTRATE, "owned postmaster.pid is truncated"
        )
    try:
        pid = int(lines[0])
        start_time = float(lines[2])
    except ValueError as exc:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY", SUBSTRATE, "owned postmaster.pid identity is malformed"
        ) from exc
    if pid <= 0 or not lines[1].strip() or start_time <= 0:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY", SUBSTRATE, "owned postmaster.pid identity is invalid"
        )
    pid_data = Path(lines[1])
    try:
        matches = os.path.normcase(str(pid_data.resolve())) == os.path.normcase(str(data.resolve()))
    except OSError as exc:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY", SUBSTRATE, f"could not canonicalize owned PGDATA: {exc}"
        ) from exc
    if not matches:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY",
            SUBSTRATE,
            f"postmaster.pid data directory does not match exact owned PGDATA: {pid_data}",
        )
    return PostmasterPidFile(pid, pid_data, start_time)


def _filetime_seconds(value: Any) -> float:
    timestamp = getattr(value, "timestamp", None)
    if callable(timestamp):
        return float(timestamp())
    return float(value)


def _query_process_image_path(process_handle: Any) -> Path:
    """Read a process image through minimal-rights QueryFullProcessImageNameW."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    query = kernel32.QueryFullProcessImageNameW
    query.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    )
    query.restype = wintypes.BOOL
    # The Win32 extended path limit is 32,767 UTF-16 code units.  Supplying
    # that bound avoids truncating a reviewed executable identity.
    capacity = 32_768
    buffer = ctypes.create_unicode_buffer(capacity)
    length = wintypes.DWORD(capacity)
    try:
        native_handle = wintypes.HANDLE(int(process_handle))
    except (TypeError, ValueError, OverflowError) as exc:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY", SUBSTRATE, "could not convert the Win32 process handle"
        ) from exc
    if not query(native_handle, 0, buffer, ctypes.byref(length)):
        error = ctypes.get_last_error()
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY",
            SUBSTRATE,
            f"QueryFullProcessImageNameW failed: winerror={error}",
        )
    if length.value <= 0 or length.value >= capacity:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY", SUBSTRATE, "invalid process image path length"
        )
    return Path(buffer.value)


def _process_identity_snapshot(pid: int) -> ProcessIdentitySnapshot | None:
    """Open and describe one live Win32 process object; caller owns the handle."""
    import pywintypes
    import win32api
    import win32con
    import win32event
    import win32process

    access = win32con.SYNCHRONIZE | win32con.PROCESS_QUERY_LIMITED_INFORMATION
    try:
        process = win32api.OpenProcess(access, False, pid)
    except pywintypes.error as exc:
        if exc.winerror == 87:
            return None
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY",
            SUBSTRATE,
            f"process identity query failed for pid={pid}: winerror={exc.winerror}",
        ) from exc
    try:
        state = win32event.WaitForSingleObject(process, 0)
        if state == win32event.WAIT_OBJECT_0:
            process.Close()
            return None
        if state != win32event.WAIT_TIMEOUT:
            raise RuntimeError(f"unexpected wait result {state}")
        image = _query_process_image_path(process)
        creation = _filetime_seconds(win32process.GetProcessTimes(process)["CreationTime"])
        return ProcessIdentitySnapshot(pid, process, creation, image)
    except Exception as exc:
        process.Close()
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY",
            SUBSTRATE,
            f"process identity query was ambiguous for pid={pid}: {exc}",
        ) from exc


def _postmaster_witness(data: Path, postgres: Path) -> PostmasterWitness | None:
    """Bind pidfile identity to one live Win32 process object without mutation."""
    identity = _postmaster_pidfile(data)
    if identity is None:
        return None
    snapshot = _process_identity_snapshot(identity.pid)
    if snapshot is None:
        return None
    try:
        image_matches = os.path.normcase(str(snapshot.executable.resolve())) == os.path.normcase(
            str(postgres.resolve())
        )
        time_matches = (
            abs(snapshot.creation_time - identity.start_time) <= POSTMASTER_START_TOLERANCE_SECONDS
        )
        if not image_matches or not time_matches:
            snapshot.process.Close()
            mismatch = "executable" if not image_matches else "creation time"
            raise Stage8PostgreSQLProbeError(
                "POSTGRESQL_PROCESS_IDENTITY",
                SUBSTRATE,
                f"live PID does not match owned postmaster {mismatch}: pid={identity.pid}",
            )
        return PostmasterWitness(
            identity.pid, snapshot.process, identity.start_time, snapshot.executable
        )
    except Stage8PostgreSQLProbeError:
        raise
    except Exception as exc:
        snapshot.process.Close()
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY",
            SUBSTRATE,
            f"postmaster identity query was ambiguous for pid={identity.pid}: {exc}",
        ) from exc


def _close_postmaster_witness(witness: PostmasterWitness) -> None:
    witness.process.Close()


def _require_exact_owned_data(data: Path, root: Path) -> None:
    try:
        matches = data.resolve() == (root / "cluster").resolve()
    except OSError as exc:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY",
            SUBSTRATE,
            f"could not canonicalize Stage-8 PGDATA: {exc}",
        ) from exc
    if not matches:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_IDENTITY",
            SUBSTRATE,
            "PGDATA is not the exact Stage-8-owned cluster",
        )


def _process_exists(pid: int | None) -> bool:
    """Query process liveness on Windows without signalling or mutating it."""
    if pid is None or pid <= 0:
        return False
    import pywintypes
    import win32api
    import win32con
    import win32event

    try:
        process = win32api.OpenProcess(win32con.SYNCHRONIZE, False, pid)
    except pywintypes.error as exc:
        # ERROR_INVALID_PARAMETER is returned for a PID with no process object.
        if exc.winerror == 87:
            return False
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_STATE",
            SUBSTRATE,
            f"non-destructive process-state query failed for pid={pid}: winerror={exc.winerror}",
        ) from exc
    try:
        state = win32event.WaitForSingleObject(process, 0)
        if state == win32event.WAIT_TIMEOUT:
            return True
        if state == win32event.WAIT_OBJECT_0:
            return False
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_PROCESS_STATE",
            SUBSTRATE,
            f"unexpected process-state result for pid={pid}: wait_result={state}",
        )
    finally:
        process.Close()


def _cluster_running(pg_ctl: Path, data: Path, root: Path) -> tuple[bool, int | None]:
    """Require exact pidfile, process identity and exact-PGDATA pg_ctl status."""
    _require_exact_owned_data(data, root)
    witness = _postmaster_witness(data, pg_ctl.with_name("postgres.exe"))
    if witness is None:
        return False, _postgresql_pid(data)
    try:
        status = _run_pg_ctl(pg_ctl, data, root, "status", ["status"], timeout=10)
        return not status.timed_out and status.returncode == 0, witness.pid
    finally:
        _close_postmaster_witness(witness)


def _pg_ctl_failure_detail(
    operation: str, result: PgCtlResult, data: Path, running: bool, pid: int | None, log: Path
) -> str:
    return (
        f"pg_ctl operation={operation} returncode={result.returncode} "
        f"timed_out={result.timed_out} elapsed={result.elapsed:.3f}s cluster={data} "
        f"postmaster_pid={pid} cluster_running={running}; "
        f"stdout_tail={result.stdout_tail[-STDERR_LIMIT:]!r}; "
        f"stderr_tail={result.stderr_tail[-STDERR_LIMIT:]!r}; "
        f"postgresql_log_tail={_tail(log)!r}"
    )


def _stop_owned_cluster(
    pg_ctl: Path,
    data: Path,
    root: Path,
    log: Path,
    *,
    mode: str = "fast",
    timeout: int = START_TIMEOUT,
) -> str | None:
    _require_exact_owned_data(data, root)
    witness = _postmaster_witness(data, pg_ctl.with_name("postgres.exe"))
    if witness is None:
        return None
    pid = witness.pid
    witness_error: str | None = None
    try:
        result = _run_pg_ctl(
            pg_ctl, data, root, "stop", ["-m", mode, "-w", "stop"], timeout=timeout
        )
        status = _run_pg_ctl(pg_ctl, data, root, "status", ["status"], timeout=10)
        running = not status.timed_out and status.returncode == 0
        try:
            _wait_process_exit(witness.process, timeout=timeout)
        except Exception as exc:
            witness_error = str(exc)
        status_stopped = not status.timed_out and status.returncode == 3
        if (
            result.timed_out
            or result.returncode != 0
            or not status_stopped
            or witness_error is not None
        ):
            detail = _pg_ctl_failure_detail("stop", result, data, running, pid, log)
            detail += (
                f"; status_returncode={status.returncode}; status_timed_out={status.timed_out}"
            )
            return detail if witness_error is None else f"{detail}; process_exit={witness_error}"
        return None
    finally:
        _close_postmaster_witness(witness)


def _stop_if_owned_cluster_running(
    pg_ctl: Path,
    data: Path,
    root: Path,
    log: Path,
    *,
    cluster_started: bool,
    mode: str = "fast",
    timeout: int = START_TIMEOUT,
) -> str | None:
    """Stop based on live cluster state, not solely on the possibly stale marker."""
    # The stale cache marker is deliberately not authority to signal anything.
    # _stop_owned_cluster obtains one exact identity witness before invoking pg_ctl.
    return _stop_owned_cluster(pg_ctl, data, root, log, mode=mode, timeout=timeout)


def _restart_owned_cluster(
    pg_ctl: Path,
    data: Path,
    root: Path,
    log: Path,
    port: int,
    *,
    on_stopped: Callable[[], None],
) -> int:
    """Stop the exact owned postmaster, then start and identify its replacement."""
    old_witness = _postmaster_witness(data, pg_ctl.with_name("postgres.exe"))
    if old_witness is None:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_START", SUBSTRATE, "no owned postmaster identity before final restart"
        )
    old_identity = (old_witness.pid, old_witness.start_time)
    _close_postmaster_witness(old_witness)

    stop_error = _stop_owned_cluster(pg_ctl, data, root, log, timeout=START_TIMEOUT)
    if stop_error is not None:
        raise Stage8PostgreSQLProbeError("POSTGRESQL_START", SUBSTRATE, stop_error)
    on_stopped()

    started = _run_pg_ctl(
        pg_ctl,
        data,
        root,
        "start",
        ["-l", str(log), "-o", f"-p {port}", "-w", "start"],
        timeout=START_TIMEOUT,
    )
    running, pid = _cluster_running(pg_ctl, data, root)
    if started.timed_out or started.returncode != 0 or not running or pid is None:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_START",
            SUBSTRATE,
            _pg_ctl_failure_detail("start", started, data, running, pid, log),
        )

    new_witness = _postmaster_witness(data, pg_ctl.with_name("postgres.exe"))
    if new_witness is None:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_START", SUBSTRATE, "no owned postmaster identity after final restart"
        )
    try:
        new_identity = (new_witness.pid, new_witness.start_time)
        if new_witness.pid != pid or new_identity == old_identity:
            raise Stage8PostgreSQLProbeError(
                "POSTGRESQL_START",
                SUBSTRATE,
                f"postmaster identity was not replaced: old={old_identity!r}; new={new_identity!r}",
            )
    finally:
        _close_postmaster_witness(new_witness)

    if tuple((data / "pg_hba.conf").read_text(encoding="utf-8").splitlines()) != final_hba_lines():
        raise Stage8PostgreSQLProbeError(
            "WINDOWS_TLS_HBA_QUALIFICATION", PRINCIPAL_AUTH, "final HBA changed across restart"
        )
    if tuple((data / "pg_ident.conf").read_text(encoding="utf-8").splitlines()) != ident_lines():
        raise Stage8PostgreSQLProbeError(
            "WINDOWS_TLS_IDENT_QUALIFICATION",
            PRINCIPAL_AUTH,
            "final ident changed across restart",
        )
    return pid


def _admin_dsn(port: int, database: str = "postgres") -> str:
    options = (
        f"-c statement_timeout={AUTHORITY_STATEMENT_TIMEOUT_MS} "
        f"-c lock_timeout={AUTHORITY_LOCK_TIMEOUT_MS}"
    )
    return (
        f"host=127.0.0.1 port={port} dbname={database} user=stage8_bootstrap "
        f"connect_timeout=5 options='{options}'"
    )


def _qualify_session_timeout_settings(rows: list[tuple[str, str, str]]) -> None:
    timeout_settings = {name: (int(setting), unit) for name, setting, unit in rows}
    expected = {
        "statement_timeout": (AUTHORITY_STATEMENT_TIMEOUT_MS, "ms"),
        "lock_timeout": (AUTHORITY_LOCK_TIMEOUT_MS, "ms"),
    }
    if timeout_settings != expected:
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_DURABILITY",
            SUBSTRATE,
            f"unsafe acceptance SQL timeout settings: {timeout_settings}",
        )


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _wait_invocation_message(
    path: Path,
    invocation_token: str,
    *,
    expected_pid: int | None = None,
    timeout: float = SERVICE_TIMEOUT,
) -> dict[str, Any]:
    """Wait for a message bound to this invocation, ignoring stale contents."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = _read_json(path)
        if value is not None and secrets.compare_digest(
            str(value.get("invocation_token", "")), invocation_token
        ):
            pid = value.get("pid")
            if type(pid) is int and (expected_pid is None or pid == expected_pid):
                return value
        time.sleep(0.05)
    raise TimeoutError(f"deadline expired waiting for fresh invocation message {path.name}")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")), encoding="utf-8")
    os.replace(temporary, path)


def _sc(*args: str, timeout: int = COMMAND_TIMEOUT) -> subprocess.CompletedProcess[str]:
    return _run(["sc.exe", *args], timeout=timeout)


def _service_command(service: str, request: Path) -> str:
    helper = Path(__file__).with_name("windows_stage8_postgresql_service.py").resolve()
    return f'"{sys.executable}" "{helper}" --service-name {service} --request "{request}"'


def _service_is_absent(result: subprocess.CompletedProcess[str]) -> bool:
    return result.returncode != 0 and "1060" in (result.stdout + result.stderr)


def _create_service(name: str, account: str, request: Path) -> None:
    existing = _sc("query", name)
    if not _service_is_absent(existing):
        raise RuntimeError(f"refusing to replace or adopt pre-existing service {name}")
    result = _sc(
        "create",
        name,
        "binPath=",
        _service_command(name, request),
        "start=",
        "demand",
        "obj=",
        account,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout)[-STDERR_LIMIT:]
        raise RuntimeError(f"SCM create failed for {name}: {detail}")


def _service_observation(name: str) -> tuple[str, int]:
    result = _sc("queryex", name)
    if result.returncode:
        raise RuntimeError(f"SCM query failed for owned service {name}")
    state_match = re.search(r"STATE\s*:\s*\d+\s+(\w+)", result.stdout, re.I)
    pid_match = re.search(r"PID\s*:\s*(\d+)", result.stdout, re.I)
    if not state_match or not pid_match:
        raise RuntimeError(f"SCM returned an unparseable state for {name}")
    return state_match.group(1).upper(), int(pid_match.group(1))


def _wait_service_state(name: str, expected: str, timeout: int = SERVICE_TIMEOUT) -> int:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state, pid = _service_observation(name)
        if state == expected:
            return pid
        time.sleep(0.05)
    raise TimeoutError(f"deadline expired waiting for service {name} state {expected}")


def _wait_service_absent(name: str, timeout: int = SERVICE_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = _sc("query", name)
        if _service_is_absent(result):
            return
        # 1072 is ERROR_SERVICE_MARKED_FOR_DELETE, not absence.
        time.sleep(0.05)
    raise TimeoutError(f"deadline expired waiting for deletion of owned service {name}")


def _token_identity(pid: int) -> tuple[str, str]:
    import win32api
    import win32con
    import win32security

    process = win32api.OpenProcess(win32con.PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    try:
        token = win32security.OpenProcessToken(process, win32con.TOKEN_QUERY)
        try:
            sid = win32security.GetTokenInformation(token, win32security.TokenUser)[0]
            name, domain, _ = win32security.LookupAccountSid(None, sid)
            return win32security.ConvertSidToStringSid(sid), f"{domain}\\{name}"
        finally:
            token.Close()
    finally:
        process.Close()


def _assert_expected_token(pid: int, expected: str) -> tuple[str, str]:
    import win32security

    sid, account = _token_identity(pid)
    expected_sid, _, _ = win32security.LookupAccountName(None, expected)
    if (
        sid != win32security.ConvertSidToStringSid(expected_sid)
        or account.casefold() != expected.casefold()
    ):
        raise RuntimeError(f"service token mismatch: pid={pid}, sid={sid}, account={account}")
    return sid, account


def _open_process_exit_witness(pid: int) -> Any:
    """Hold a handle to the exact helper process, avoiding PID-reuse ambiguity."""
    import win32api
    import win32con

    return win32api.OpenProcess(win32con.SYNCHRONIZE, False, pid)


def _wait_process_exit(process: Any, timeout: float = SERVICE_TIMEOUT) -> None:
    import win32event

    if win32event.WaitForSingleObject(process, int(timeout * 1000)) != win32event.WAIT_OBJECT_0:
        raise TimeoutError("deadline expired waiting for helper process exit")


def _close_process_exit_witness(process: Any) -> None:
    process.Close()


def _service_failure_label(service: str, *, identity: bool) -> str:
    if service == RUNTIME_SERVICE:
        return "WINDOWS_TLS_RUNTIME_KEY_ACCESS" if identity else "WINDOWS_TLS_RUNTIME_CONNECT"
    return "WINDOWS_TLS_VERIFIER_KEY_ACCESS" if identity else "WINDOWS_TLS_VERIFIER_CONNECT"


def _prepare_services(root: Path, owned_services: set[str]) -> dict[str, Path]:
    requests: dict[str, Path] = {}
    for service, account in (
        (RUNTIME_SERVICE, RUNTIME_ACCOUNT),
        (VERIFIER_SERVICE, VERIFIER_ACCOUNT),
    ):
        label = _service_failure_label(service, identity=True)
        request = root / f"{service}.current-request.json"
        try:
            _create_service(service, account, request)
        except Exception as exc:
            raise Stage8PostgreSQLProbeError(label, PRINCIPAL_AUTH, str(exc)) from exc
        owned_services.add(service)
        _write_status(owned_services=sorted(owned_services))
        requests[service] = request
        try:
            acl = _run(
                ["icacls.exe", str(root), "/grant", f"{account}:(OI)(CI)F"],
                timeout=COMMAND_TIMEOUT,
            )
        except Exception as exc:
            raise Stage8PostgreSQLProbeError(label, PRINCIPAL_AUTH, str(exc)) from exc
        if acl.returncode:
            raise Stage8PostgreSQLProbeError(
                label,
                PRINCIPAL_AUTH,
                "could not grant helper service access to probe-owned scratch: "
                f"{(acl.stderr or acl.stdout)[-512:]}",
            )
    return requests


def _invoke_service(
    root: Path,
    request: Path,
    service: str,
    account: str,
    port: int,
    role: str,
    database: str = DATABASE,
    *,
    failure_label: str | None = None,
    operation: str = "connect",
) -> dict[str, Any]:
    label = failure_label or _service_failure_label(service, identity=False)
    invocation_token = secrets.token_hex(32)
    invocation = root / "invocations" / invocation_token
    invocation.mkdir(parents=True)
    ready, go, result = (invocation / name for name in ("ready.json", "go.json", "result.json"))
    _atomic_json(
        request,
        {
            "invocation_token": invocation_token,
            "port": port,
            "database": database,
            "role": role,
            "operation": operation,
            "ready": str(ready),
            "go": str(go),
            "result": str(result),
        },
    )
    witness: Any | None = None
    try:
        started = _sc("start", service)
        if started.returncode:
            raise RuntimeError((started.stderr or started.stdout)[-STDERR_LIMIT:])
        ready_payload = _wait_invocation_message(ready, invocation_token)
        if ready_payload.get("ready") is not True:
            raise RuntimeError("fresh READY message did not assert readiness")
        scm_pid = _wait_service_state(service, "RUNNING")
        if ready_payload["pid"] != scm_pid:
            raise RuntimeError(
                f"READY PID does not match SCM PID: ready={ready_payload['pid']}, scm={scm_pid}"
            )
        witness = _open_process_exit_witness(scm_pid)
        # The independently observed process token is authority. GO is published
        # only after this check succeeds, so the child cannot connect beforehand.
        sid, observed = _assert_expected_token(scm_pid, account)
        _atomic_json(go, {"invocation_token": invocation_token, "pid": scm_pid, "go": True})
        payload = _wait_invocation_message(result, invocation_token, expected_pid=scm_pid)
        _wait_service_state(service, "STOPPED")
        _wait_process_exit(witness)
        payload.update({"service": service, "pid": scm_pid, "sid": sid, "account": observed})
        return payload
    except Exception as exc:
        raise Stage8PostgreSQLProbeError(label, PRINCIPAL_AUTH, str(exc)) from exc
    finally:
        if witness is not None:
            _close_process_exit_witness(witness)


def _cleanup_service(name: str) -> list[str]:
    errors: list[str] = []
    witness: Any | None = None
    try:
        state, pid = _service_observation(name)
        if pid > 0:
            witness = _open_process_exit_witness(pid)
        if state != "STOPPED":
            stopped = _sc("stop", name, timeout=10)
            if stopped.returncode:
                errors.append(
                    f"service {name} stop failed: {(stopped.stderr or stopped.stdout)[-512:]}"
                )
                return errors
            else:
                _wait_service_state(name, "STOPPED")
        if witness is not None:
            _wait_process_exit(witness)
        deleted = _sc("delete", name, timeout=10)
        if deleted.returncode:
            errors.append(
                f"service {name} delete failed: {(deleted.stderr or deleted.stdout)[-512:]}"
            )
        else:
            _wait_service_absent(name)
    except Exception as exc:
        errors.append(f"service {name}: {exc}")
    finally:
        if witness is not None:
            _close_process_exit_witness(witness)
    return errors


def _finish_cleanup(primary: BaseException | None, cleanup_errors: list[str]) -> None:
    if not cleanup_errors:
        return
    detail = "; ".join(cleanup_errors)
    if primary is not None:
        print(f"[STAGE8_CLEANUP] secondary failure: {detail}", file=sys.stderr, flush=True)
        return
    raise Stage8PostgreSQLProbeError("STAGE8_CLEANUP", SUBSTRATE, detail)


def _matrix_label(service: str, role: str, expected: bool) -> str:
    if expected and service == RUNTIME_SERVICE and role == RUNTIME_ROLE:
        return "WINDOWS_TLS_RUNTIME_CONNECT"
    if expected and service == VERIFIER_SERVICE and role == VERIFIER_ROLE:
        return "WINDOWS_TLS_VERIFIER_CONNECT"
    if (service, role) in {
        (RUNTIME_SERVICE, VERIFIER_ROLE),
        (VERIFIER_SERVICE, RUNTIME_ROLE),
    }:
        return "WINDOWS_TLS_CROSS_ROLE_DENIAL"
    return "WINDOWS_TLS_OUTSIDER_DENIAL"


def run_probe(scratch_parent: Path, *, status_file: Path | None = None) -> dict[str, str]:
    global _status_file, _status
    _status_file = status_file
    _status = {
        "last_phase": None,
        "cluster_started": False,
        "owned_services": [],
        "root": None,
    }
    if canonical_host_os() != "Windows" or os.name != "nt":
        raise Stage8PostgreSQLProbeError(
            "POSTGRESQL_DISCOVERY", SUBSTRATE, "live Stage-8 requires a real Windows host"
        )
    with _phase("POSTGRESQL_DISCOVERY"):
        binaries = discover_postgresql()
        _write_status(pgbin=str(binaries.bindir))
    primary: BaseException | None = None
    cleanup_errors: list[str] = []
    cluster_started = False
    root: Path | None = None
    owned_services: set[str] = set()
    try:
        root = Path(tempfile.mkdtemp(prefix="CryptoHunter-Stage8-", dir=scratch_parent))
        _write_status(root=str(root))
        data, log = root / "cluster", root / "postgresql.log"
        with _phase("POSTGRESQL_CLUSTER_INIT"):
            init = _run(
                [
                    str(binaries.executable("initdb.exe")),
                    "-D",
                    str(data),
                    "-U",
                    "stage8_bootstrap",
                    "--auth-host=trust",
                    "--auth-local=trust",
                ]
            )
            if init.returncode:
                raise Stage8PostgreSQLProbeError(
                    "POSTGRESQL_CLUSTER_INIT",
                    SUBSTRATE,
                    (init.stderr or init.stdout)[-STDERR_LIMIT:],
                )
        # SCM service creation is the authority that makes both NT SERVICE
        # identities resolvable.  The inherited scratch grant is deliberately
        # installed first; each private key is then replaced with a protected,
        # exact DACL and qualified before any service is started.
        with _principal_auth_phase("HELPER_SERVICE_PREPARE", "WINDOWS_TLS_PKI_PROVISION"):
            service_requests = _prepare_services(root, owned_services)
            _write_status(owned_services=sorted(owned_services))
        with _principal_auth_phase("TLS_PKI_PROVISION", "WINDOWS_TLS_PKI_PROVISION"):
            tls = _provision_tls_pki(root)
            _protect_private_key(tls["runtime_key"], RUNTIME_ACCOUNT)
            _protect_private_key(tls["verifier_key"], VERIFIER_ACCOUNT)
            _protect_owner_private_key(tls["server_key"])
            _protect_owner_private_key(tls["ca_key"])
        with _principal_auth_phase(
            "TLS_KEY_DACL_QUALIFICATION", "WINDOWS_TLS_KEY_DACL_QUALIFICATION"
        ):
            import win32security

            system_sid = _account_sid("SYSTEM", win32security)
            owner_sid = _current_process_sid(win32security)
            runtime_sid = _account_sid(RUNTIME_ACCOUNT, win32security)
            verifier_sid = _account_sid(VERIFIER_ACCOUNT, win32security)
            _qualify_private_key_dacl(
                tls["runtime_key"],
                [(system_sid, PRIVATE_KEY_FULL_MASK), (runtime_sid, PRIVATE_KEY_READ_MASK)],
            )
            _qualify_private_key_dacl(
                tls["verifier_key"],
                [(system_sid, PRIVATE_KEY_FULL_MASK), (verifier_sid, PRIVATE_KEY_READ_MASK)],
            )
            owner_contract = list(
                {
                    system_sid: (system_sid, PRIVATE_KEY_FULL_MASK),
                    owner_sid: (owner_sid, PRIVATE_KEY_FULL_MASK),
                }.values()
            )
            _qualify_private_key_dacl(tls["ca_key"], owner_contract)
            _qualify_private_key_dacl(tls["server_key"], owner_contract)
        port = _port()

        def pg_path(path: Path) -> str:
            return str(path).replace("\\", "/").replace("'", "''")

        with (data / "postgresql.conf").open("a", encoding="utf-8") as stream:
            stream.write(
                "\nlisten_addresses='127.0.0.1'\nfsync=on\nsynchronous_commit=on\n"
                "log_connections=on\nlog_disconnections=on\nlogging_collector=off\nssl=on\n"
                f"ssl_cert_file='{pg_path(tls['server_cert'])}'\n"
                f"ssl_key_file='{pg_path(tls['server_key'])}'\n"
                f"ssl_ca_file='{pg_path(tls['ca_cert'])}'\n"
            )
        with _phase("POSTGRESQL_START"):
            started = _run_pg_ctl(
                binaries.executable("pg_ctl.exe"),
                data,
                root,
                "start",
                [
                    "-l",
                    str(log),
                    "-o",
                    f"-p {port}",
                    "-w",
                    "start",
                ],
                timeout=START_TIMEOUT,
            )
            running, pid = _cluster_running(binaries.executable("pg_ctl.exe"), data, root)
            if started.timed_out or started.returncode != 0 or not running or pid is None:
                raise Stage8PostgreSQLProbeError(
                    "POSTGRESQL_START",
                    SUBSTRATE,
                    _pg_ctl_failure_detail("start", started, data, running, pid, log),
                )
            cluster_started = True
            _write_status(cluster_started=True, postgresql_pid=pid)
        import psycopg
        from bot_core.postgresql_freshness_authority import (
            PostgreSQLConnectionConfig,
            provision_postgresql_freshness_authority,
            qualify_postgresql_freshness_authority,
        )

        with _phase("POSTGRESQL_DURABILITY"):
            with psycopg.connect(_admin_dsn(port), autocommit=True) as connection:
                settings = connection.execute(
                    "SELECT current_setting('fsync'),current_setting('synchronous_commit'),current_setting('listen_addresses'),current_setting('port'),current_setting('server_version_num')"
                ).fetchone()
                if settings[:3] != ("on", "on", "127.0.0.1") or int(settings[4]) < 160000:
                    raise Stage8PostgreSQLProbeError(
                        "POSTGRESQL_DURABILITY", SUBSTRATE, f"unsafe settings: {settings}"
                    )
                _qualify_session_timeout_settings(
                    connection.execute(
                        "SELECT name, setting, unit FROM pg_catalog.pg_settings "
                        "WHERE name IN ('statement_timeout', 'lock_timeout')"
                    ).fetchall()
                )
                connection.execute(f'CREATE DATABASE "{DATABASE}"')
        authority = PostgreSQLConnectionConfig(_admin_dsn(port, DATABASE))
        with _phase("POSTGRESQL_SCHEMA_PROVISION"):
            try:
                provision_postgresql_freshness_authority(authority)
            except Exception as exc:
                raise Stage8PostgreSQLProbeError(
                    "POSTGRESQL_SCHEMA_PROVISION", SUBSTRATE, f"{type(exc).__name__}: {exc}"
                ) from exc
        with _phase("POSTGRESQL_SCHEMA_QUALIFICATION"):
            try:
                qualify_postgresql_freshness_authority(authority)
            except Exception as exc:
                raise Stage8PostgreSQLProbeError(
                    "POSTGRESQL_SCHEMA_QUALIFICATION", SUBSTRATE, f"{type(exc).__name__}: {exc}"
                ) from exc

        for access_phase, service, account, own, other in (
            ("TLS_RUNTIME_KEY_ACCESS", RUNTIME_SERVICE, RUNTIME_ACCOUNT, "runtime", "verifier"),
            ("TLS_VERIFIER_KEY_ACCESS", VERIFIER_SERVICE, VERIFIER_ACCOUNT, "verifier", "runtime"),
        ):
            with _principal_auth_phase(
                access_phase, _service_failure_label(service, identity=True)
            ):
                payload = _invoke_service(
                    root,
                    service_requests[service],
                    service,
                    account,
                    port,
                    RUNTIME_ROLE if service == RUNTIME_SERVICE else VERIFIER_ROLE,
                    failure_label=_service_failure_label(service, identity=True),
                    operation="key_probe",
                )
                keys = payload.get("keys", {})
                if (
                    keys.get(own, {}).get("accessible") is not True
                    or keys.get(other, {}).get("accessible") is not False
                ):
                    raise Stage8PostgreSQLProbeError(
                        _service_failure_label(service, identity=True),
                        PRINCIPAL_AUTH,
                        f"service-token private-key access mismatch: {keys}",
                    )

        _install_and_qualify_final_auth(
            psycopg, port, data, binaries.executable("pg_ctl.exe"), root
        )
        with _phase("FINAL_CLUSTER_RESTART"):
            def publish_stopped() -> None:
                nonlocal cluster_started
                cluster_started = False
                _write_status(cluster_started=False, postgresql_pid=None)

            pid = _restart_owned_cluster(
                binaries.executable("pg_ctl.exe"),
                data,
                root,
                log,
                port,
                on_stopped=publish_stopped,
            )
            cluster_started = True
            _write_status(cluster_started=True, postgresql_pid=pid)
        matrix = (
            (RUNTIME_SERVICE, RUNTIME_ACCOUNT, RUNTIME_ROLE, True, DATABASE),
            (VERIFIER_SERVICE, VERIFIER_ACCOUNT, VERIFIER_ROLE, True, DATABASE),
            (RUNTIME_SERVICE, RUNTIME_ACCOUNT, VERIFIER_ROLE, False, DATABASE),
            (VERIFIER_SERVICE, VERIFIER_ACCOUNT, RUNTIME_ROLE, False, DATABASE),
        )
        matrix_phases = (
            "MATRIX_RUNTIME",
            "MATRIX_VERIFIER",
            "MATRIX_RUNTIME_TO_VERIFIER_DENIAL",
            "MATRIX_VERIFIER_TO_RUNTIME_DENIAL",
        )
        for matrix_phase, (service, account, role, expected, database) in zip(
            matrix_phases, matrix, strict=True
        ):
            with _principal_auth_phase(matrix_phase, _matrix_label(service, role, expected)):
                payload = _invoke_service(
                    root,
                    service_requests[service],
                    service,
                    account,
                    port,
                    role,
                    database,
                    failure_label=_matrix_label(service, role, expected),
                )
                if bool(payload.get("ok")) != expected or (
                    expected and payload.get("session_user") != role
                ):
                    raise Stage8PostgreSQLProbeError(
                        _matrix_label(service, role, expected),
                        PRINCIPAL_AUTH,
                        f"connection matrix mismatch: service={service}, requested_role={role}, result={payload}",
                    )
        with _principal_auth_phase("MATRIX_WRONG_ROLE_DENIAL", "WINDOWS_TLS_OUTSIDER_DENIAL"):
            for service, account in (
                (RUNTIME_SERVICE, RUNTIME_ACCOUNT),
                (VERIFIER_SERVICE, VERIFIER_ACCOUNT),
            ):
                payload = _invoke_service(
                    root,
                    service_requests[service],
                    service,
                    account,
                    port,
                    "stage8_bootstrap",
                    DATABASE,
                    failure_label=_matrix_label(service, "stage8_bootstrap", False),
                )
                if payload.get("ok") is not False:
                    raise Stage8PostgreSQLProbeError(
                        _matrix_label(service, "stage8_bootstrap", False),
                        PRINCIPAL_AUTH,
                        f"client certificate reached bootstrap role: service={service}",
                    )
        with _principal_auth_phase("INTERACTIVE_OUTSIDER_DENIAL", "WINDOWS_TLS_OUTSIDER_DENIAL"):
            for role in (RUNTIME_ROLE, VERIFIER_ROLE, "stage8_bootstrap"):
                try:
                    psycopg.connect(
                        host="127.0.0.1",
                        port=port,
                        dbname=DATABASE,
                        user=role,
                        sslmode="verify-full",
                        sslrootcert=str(tls["ca_cert"]),
                        connect_timeout=5,
                    )
                except psycopg.Error:
                    pass
                else:
                    raise Stage8PostgreSQLProbeError(
                        "WINDOWS_TLS_OUTSIDER_DENIAL",
                        PRINCIPAL_AUTH,
                        f"interactive caller reached {role}",
                    )
        # Re-qualify schema state before final shutdown. The exact qualifier itself
        # already covers roles, memberships, owners, ACLs, functions and raw DML.
        # The server parser has already qualified the final HBA through the bounded
        # pre-lockdown session; retain an exact-text drift check after the matrix.
        if (
            tuple((data / "pg_hba.conf").read_text(encoding="utf-8").splitlines())
            != final_hba_lines()
        ):
            raise Stage8PostgreSQLProbeError(
                "WINDOWS_TLS_HBA_QUALIFICATION", PRINCIPAL_AUTH, "final HBA changed after reload"
            )
        if (
            tuple((data / "pg_ident.conf").read_text(encoding="utf-8").splitlines())
            != ident_lines()
        ):
            raise Stage8PostgreSQLProbeError(
                "WINDOWS_TLS_IDENT_QUALIFICATION",
                PRINCIPAL_AUTH,
                "final ident changed after reload",
            )
        return {
            SUBSTRATE: "PASS",
            PRINCIPAL_AUTH: "PASS",
            "details": f"PostgreSQL {binaries.major}; service-SID-bound distinct mTLS certificates; isolated port {port}",
        }
    except BaseException as exc:
        primary = exc
        raise
    finally:
        try:
            with _phase("CLEANUP_SERVICES"):
                for name in tuple(owned_services):
                    try:
                        cleanup_errors.extend(_cleanup_service(name))
                    except Exception as exc:
                        cleanup_errors.append(f"service cleanup {name}: {exc}")
                if cleanup_errors:
                    raise RuntimeError("; ".join(cleanup_errors))
        except RuntimeError:
            pass
        postgresql_cleanup_ok = True
        if root is not None:
            try:
                with _phase("CLEANUP_POSTGRESQL"):
                    data, log = root / "cluster", root / "postgresql.log"
                    stop_error = _stop_if_owned_cluster_running(
                        binaries.executable("pg_ctl.exe"),
                        data,
                        root,
                        log,
                        cluster_started=cluster_started,
                    )
                    if stop_error:
                        raise RuntimeError(stop_error)
                    _write_status(cluster_started=False, postgresql_pid=None)
            except Exception as exc:
                postgresql_cleanup_ok = False
                cleanup_errors.append(f"PostgreSQL cleanup: {exc}")
        else:
            with _phase("CLEANUP_POSTGRESQL"):
                pass
        if root is not None and postgresql_cleanup_ok:
            try:
                with _phase("CLEANUP_SCRATCH"):
                    shutil.rmtree(root)
                    _write_status(root=None)
            except OSError as exc:
                cleanup_errors.append(f"scratch cleanup: {exc}")
        else:
            with _phase("CLEANUP_SCRATCH"):
                pass
        _finish_cleanup(primary, cleanup_errors)


def emergency_cleanup(status: dict[str, Any], scratch_parent: Path) -> list[str]:
    """Bounded cleanup after the supervisor terminates a stuck Stage-8 worker."""
    errors: list[str] = []
    owned = status.get("owned_services", [])
    if isinstance(owned, list):
        for name in owned:
            if name not in {RUNTIME_SERVICE, VERIFIER_SERVICE}:
                errors.append(f"refused unexpected service ownership marker: {name}")
                continue
            errors.extend(_cleanup_service(name))
    root_value = status.get("root")
    root = Path(root_value) if isinstance(root_value, str) else None
    try:
        safe_root = (
            root is not None
            and root.name.startswith("CryptoHunter-Stage8-")
            and root.resolve().parent == scratch_parent.resolve()
        )
    except OSError:
        safe_root = False
    if root is not None and not safe_root:
        errors.append("refused scratch cleanup outside Stage-8-owned root")
        return errors
    postgresql_cleanup_ok = True
    if safe_root:
        pgbin = status.get("pgbin")
        if isinstance(pgbin, str):
            try:
                pg_ctl = Path(pgbin) / "pg_ctl.exe"
                data, log = root / "cluster", root / "postgresql.log"
                stop_error = _stop_if_owned_cluster_running(
                    pg_ctl,
                    data,
                    root,
                    log,
                    cluster_started=status.get("cluster_started") is True,
                    mode="immediate",
                    timeout=10,
                )
                if stop_error:
                    postgresql_cleanup_ok = False
                    errors.append(f"emergency PostgreSQL stop failed: {stop_error}")
            except Exception as exc:
                postgresql_cleanup_ok = False
                errors.append(f"emergency PostgreSQL stop: {exc}")
        elif status.get("cluster_started") is True or _postgresql_pid(root / "cluster") is not None:
            postgresql_cleanup_ok = False
            errors.append("emergency PostgreSQL stop lacks reviewed binary path")
    if safe_root and postgresql_cleanup_ok:
        try:
            shutil.rmtree(root)
        except OSError as exc:
            errors.append(f"emergency scratch cleanup: {exc}")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Internal reviewed Windows Stage-8 worker")
    parser.add_argument("--scratch-parent", type=Path, required=True)
    parser.add_argument("--status-file", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_probe(args.scratch_parent, status_file=args.status_file)
    except Stage8PostgreSQLProbeError as exc:
        _write_status(failed_item=exc.item, error=str(exc))
        print(json.dumps({"item": exc.item, "error": str(exc)}), file=sys.stderr, flush=True)
        return 1
    _write_status(
        completed=True,
        results={item: result[item] for item in WINDOWS_STAGE8_ITEMS},
    )
    print(json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
