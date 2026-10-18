"""Installed Linux identity boundary for the production lifecycle authority.

This module qualifies an independently installed service and acquires database
fences. It cannot issue a Root Proof, activate a key, or adapt a TEST_ONLY
deployment. The final production write composition remains contract-blocked.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import stat
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

from bot_core.postgresql_atomic_root_proof_qualification import (
    PostgreSQLAtomicRootProofPhysicalIdentity,
)
from bot_core.postgresql_production_root_lifecycle_qualification import (
    PostgreSQLProductionRootLifecyclePin,
    qualify_production_root_lifecycle_authority,
)
from bot_core.postgresql_production_root_lifecycle_schema import (
    PostgreSQLProductionRootLifecycleProvisioning,
)

INSTALLATION_FILE = Path("/etc/cryptohunter/root-proof-issuer/installation.json")
_PRINCIPALS = frozenset(
    {"issuer", "administrator", "cha", "requester", "claimant", "root_signer", "history_signer"}
)
_SIGNERS = frozenset({"root_signer", "history_signer"})
PRODUCTION_EMISSION = "BLOCKED"


class ProductionRootLifecycleError(RuntimeError):
    """Missing or changed installation evidence fails closed."""


class ProductionRootLifecycleIndeterminate(ProductionRootLifecycleError):
    """A prior operation may exist; only positive exact resolution is evidence."""


def _protected_bytes(path: Path) -> bytes:
    """Pin each root-owned path component; reject symlinks and writable ancestry."""
    descriptors: list[int] = []
    try:
        if not path.is_absolute() or ".." in path.parts:
            raise ProductionRootLifecycleError("PROTECTED_INSTALLATION_REQUIRED")
        directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        descriptors.append(directory)
        for component in path.parts[1:-1]:
            info = os.fstat(directory)
            if info.st_uid != 0 or info.st_mode & 0o022:
                raise ProductionRootLifecycleError("PROTECTED_INSTALLATION_REQUIRED")
            directory = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            descriptors.append(directory)
        parent = os.fstat(directory)
        if parent.st_uid != 0 or parent.st_mode & 0o022:
            raise ProductionRootLifecycleError("PROTECTED_INSTALLATION_REQUIRED")
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory
        )
        descriptors.append(descriptor)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_uid != 0
            or before.st_mode & 0o022
            or before.st_nlink != 1
            or not 0 < before.st_size <= 65536
        ):
            raise ProductionRootLifecycleError("PROTECTED_INSTALLATION_REQUIRED")
        raw = b""
        while len(raw) <= 65536:
            part = os.read(descriptor, min(8192, 65537 - len(raw)))
            if not part:
                break
            raw += part
        after = os.fstat(descriptor)
        stable_fields = (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_uid",
            "st_gid",
            "st_nlink",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
        )
        if (
            any(getattr(before, field) != getattr(after, field) for field in stable_fields)
            or len(raw) != before.st_size
        ):
            raise ProductionRootLifecycleError("INSTALLATION_CHANGED_DURING_READ")
        return raw
    except OSError:
        raise ProductionRootLifecycleError("PROTECTED_INSTALLATION_REQUIRED") from None
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate installation field")
        value[key] = item
    return value


@dataclass(frozen=True, slots=True)
class _InstalledBoundary:
    pin: PostgreSQLProductionRootLifecyclePin
    principals: tuple[tuple[str, str, int, int, str, int], ...]
    signer_sockets: tuple[tuple[str, str], ...]
    source_sha256: str

    def principal(self, role: str) -> tuple[str, int, int, str]:
        for selected, username, uid, gid, pg_role, _oid in self.principals:
            if selected == role:
                return username, uid, gid, pg_role
        raise ProductionRootLifecycleError("EXACT_SERVICE_PRINCIPAL_REQUIRED")


def _installed_boundary() -> _InstalledBoundary:
    if sys.platform != "linux":
        raise ProductionRootLifecycleError("LINUX_SERVICE_INSTALLATION_REQUIRED")
    import pwd

    raw = _protected_bytes(INSTALLATION_FILE)
    try:
        value = json.loads(raw, object_pairs_hook=_object)
        if type(value) is not dict or set(value) != {
            "schema_version",
            "profile",
            "provisioning",
            "identity",
            "catalog_sha256",
            "role_oids",
            "principals",
            "signer_sockets",
        }:
            raise ValueError("exact installation fields required")
        if value["schema_version"] != 1 or type(value["schema_version"]) is not int:
            raise ValueError("installation version required")
        if value["profile"] != "PRODUCTION_LOCAL":
            raise ValueError("production installation required")
        provisioning = PostgreSQLProductionRootLifecycleProvisioning(**value["provisioning"])
        identity_data = dict(value["identity"])
        # PostgreSQL's uint64 system identifier is serialized as decimal text.
        system_id = identity_data["system_identifier"]
        if type(system_id) is not str or str(int(system_id)) != system_id:
            raise ValueError("exact system identifier required")
        identity_data["system_identifier"] = int(system_id)
        identity = PostgreSQLAtomicRootProofPhysicalIdentity(**identity_data)
        pin = PostgreSQLProductionRootLifecyclePin(
            provisioning,
            identity,
            value["catalog_sha256"],
            tuple(tuple(pair) for pair in value["role_oids"]),
        )
        if set(value["principals"]) != _PRINCIPALS or set(value["signer_sockets"]) != _SIGNERS:
            raise ValueError("complete separate service principals required")
        principals = []
        for role, configured in sorted(value["principals"].items()):
            if set(configured) != {"unix_user", "uid", "gid", "postgresql_role", "postgresql_oid"}:
                raise ValueError("exact principal fields required")
            username, uid, gid, pg_role = (
                configured["unix_user"],
                configured["uid"],
                configured["gid"],
                configured["postgresql_role"],
            )
            if (
                type(username) is not str
                or type(pg_role) is not str
                or not pg_role
                or type(uid) is not int
                or uid <= 0
                or type(gid) is not int
                or gid <= 0
            ):
                raise ValueError("non-root exact service identities required")
            account = pwd.getpwnam(username)
            if (account.pw_uid, account.pw_gid) != (uid, gid):
                raise ValueError("installed system account changed")
            oid = configured["postgresql_oid"]
            if type(oid) is not int or oid <= 0:
                raise ValueError("exact installed PostgreSQL principal OID required")
            if set(os.getgrouplist(username, gid)) != {gid}:
                raise ValueError("dedicated service accounts cannot share supplementary groups")
            principals.append((role, username, uid, gid, pg_role, oid))
        for index in (1, 2, 3, 4, 5):
            if len({p[index] for p in principals}) != len(principals):
                raise ValueError("service identities must be disjoint")
        selected = {p[0]: p[4] for p in principals}
        if (
            selected["issuer"] != provisioning.issuer_role
            or selected["administrator"] != provisioning.administrator_role
            or provisioning.schema_owner_role in selected.values()
        ):
            raise ValueError("database and installed service roles differ")
        sockets = []
        for role, name in sorted(value["signer_sockets"].items()):
            path = Path(name)
            if type(name) is not str or not path.is_absolute() or ".." in path.parts:
                raise ValueError("fixed absolute signer socket required")
            sockets.append((role, name))
        if len({name for _, name in sockets}) != len(_SIGNERS):
            raise ValueError("signer endpoints must differ")
        return _InstalledBoundary(
            pin, tuple(principals), tuple(sockets), hashlib.sha256(raw).hexdigest()
        )
    except (ValueError, TypeError, KeyError, AttributeError):
        raise ProductionRootLifecycleError("EXACT_PRODUCTION_INSTALLATION_REQUIRED") from None


def _require_process(boundary: _InstalledBoundary, role: str) -> None:
    if sys.platform != "linux":
        raise ProductionRootLifecycleError("LINUX_SERVICE_INSTALLATION_REQUIRED")
    if role not in {"issuer", "administrator"}:
        raise ProductionRootLifecycleError("ISSUER_OR_ADMINISTRATOR_REQUIRED")
    _, uid, gid, _ = boundary.principal(role)
    if (os.getuid(), os.geteuid(), os.getgid(), os.getegid()) != (uid, uid, gid, gid):
        raise ProductionRootLifecycleError("EXACT_INSTALLED_PROCESS_IDENTITY_REQUIRED")
    # No shared supplementary service group may defeat directory separation.
    other_gids = {p[3] for p in boundary.principals if p[0] != role}
    if other_gids.intersection(os.getgroups()):
        raise ProductionRootLifecycleError("SERVICE_GROUPS_NOT_SEPARATED")


def _qualify_principals(conn: psycopg.Connection[Any], boundary: _InstalledBoundary) -> None:
    names = [p[4] for p in boundary.principals]
    rows = conn.execute(
        "SELECT oid,rolname,rolcanlogin,rolsuper,rolcreaterole,rolcreatedb,"
        "rolreplication,rolbypassrls,rolinherit,rolconfig FROM pg_catalog.pg_roles "
        "WHERE rolname=ANY(%s) ORDER BY rolname",
        (names,),
    ).fetchall()
    if len(rows) != len(names) or any(not row[2] or any(row[3:9]) or row[9] for row in rows):
        raise ProductionRootLifecycleError("MINIMAL_DIRECT_DATABASE_PRINCIPALS_REQUIRED")
    oids = [row[0] for row in rows]
    if {(row[1], row[0]) for row in rows} != {(p[4], p[5]) for p in boundary.principals}:
        raise ProductionRootLifecycleError("INSTALLED_DATABASE_PRINCIPAL_OID_CHANGED")
    if conn.execute(
        "SELECT 1 FROM pg_catalog.pg_auth_members WHERE roleid=ANY(%s) OR member=ANY(%s)",
        (oids, oids),
    ).fetchone():
        raise ProductionRootLifecycleError("DATABASE_ROLE_MEMBERSHIP_FORBIDDEN")
    # CHA, requester, claimant and both signers have no lifecycle schema port.
    for role, _, _, _, pg_role, _oid in boundary.principals:
        if role not in {"issuer", "administrator"} and conn.execute(
            "SELECT pg_catalog.has_schema_privilege(%s,%s,'USAGE') "
            "OR pg_catalog.has_schema_privilege(%s,%s,'CREATE')",
            (pg_role, boundary.pin.provisioning.schema, pg_role, boundary.pin.provisioning.schema),
        ).fetchone() != (False,):
            raise ProductionRootLifecycleError("UNAUTHORIZED_LIFECYCLE_SCHEMA_ACCESS")


class ProductionRootLifecycleBoundary:
    """Read-only qualification/fencing; no public issuance capability."""

    def __init__(self, boundary: _InstalledBoundary, role: str) -> None:
        self._boundary = boundary
        self._role = role

    def _current(self) -> _InstalledBoundary:
        observed = _installed_boundary()
        if observed != self._boundary:
            raise ProductionRootLifecycleError("INSTALLED_BOUNDARY_CHANGED")
        _require_process(observed, self._role)
        return observed

    def qualify(self, conn: psycopg.Connection[Any]) -> None:
        boundary = self._current()
        self._require_peer_authentication(conn, boundary)
        qualify_production_root_lifecycle_authority(conn, boundary.pin, expected_role=self._role)
        _qualify_principals(conn, boundary)

    def _require_peer_authentication(
        self, conn: psycopg.Connection[Any], boundary: _InstalledBoundary
    ) -> None:
        # PG16 reports the authentication method and original authenticated
        # identity through system_user; trust returns NULL. A Unix socket and a
        # requested database user alone establish neither fact.
        username = boundary.principal(self._role)[0]
        if conn.execute("SELECT system_user").fetchone() != ("peer:" + username,):
            raise ProductionRootLifecycleError("EXACT_INSTALLED_PEER_AUTHENTICATION_REQUIRED")

    def _administrative_operation(
        self, conn: psycopg.Connection[Any], function: str, arguments: tuple[Any, ...]
    ) -> dict[str, Any]:
        """Own one close/completion transaction; never retry an uncertain result."""
        if (
            self._role != "administrator"
            or type(conn) is not psycopg.Connection
            or not conn.autocommit
            or conn.info.transaction_status is not psycopg.pq.TransactionStatus.IDLE
        ):
            raise ProductionRootLifecycleError("IDLE_DIRECT_ADMINISTRATIVE_CONNECTION_REQUIRED")
        boundary = self._current()
        attempted = False
        try:
            with conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                self.qualify(conn)
                placeholders = sql.SQL(",").join(sql.Placeholder() for _ in arguments)
                attempted = True
                row = conn.execute(
                    sql.SQL("SELECT {}.{}({})").format(
                        sql.Identifier(boundary.pin.provisioning.schema),
                        sql.Identifier(function),
                        placeholders,
                    ),
                    arguments,
                ).fetchone()
                if row is None or len(row) != 1 or type(row[0]) is not dict:
                    raise ProductionRootLifecycleIndeterminate("EXACT_LIFECYCLE_RESULT_REQUIRED")
                result = row[0]
            # Leaving the context positively acknowledges its COMMIT. No custody
            # operation is invoked anywhere in this coordinator.
            self._current()
            return result
        except (
            psycopg.errors.CheckViolation,
            psycopg.errors.InvalidParameterValue,
            psycopg.errors.InsufficientPrivilege,
            psycopg.errors.UniqueViolation,
            psycopg.errors.SerializationFailure,
        ):
            raise ProductionRootLifecycleError("LIFECYCLE_OPERATION_REJECTED") from None
        except Exception:
            if attempted:
                raise ProductionRootLifecycleIndeterminate(
                    "LIFECYCLE_COMMIT_OUTCOME_UNKNOWN"
                ) from None
            raise ProductionRootLifecycleError("QUALIFIED_ADMINISTRATOR_UNAVAILABLE") from None

    def begin_close(self, conn: psycopg.Connection[Any], request: dict[str, Any]) -> dict[str, Any]:
        """Positively commit PENDING_CLOSED before a separate custody procedure."""
        from psycopg.types.json import Jsonb

        return self._administrative_operation(conn, "begin_close", (Jsonb(request),))

    def publish_administrator_completion(
        self, conn: psycopg.Connection[Any], operation_id: str, request_sha256: str, receipt: bytes
    ) -> dict[str, Any]:
        """Publish an admin-verified receipt; this method does not verify custody."""
        return self._administrative_operation(
            conn, "complete_close", (operation_id, request_sha256, receipt)
        )

    def resolve_close(
        self,
        conn: psycopg.Connection[Any],
        semantic_role: str,
        operation_id: str,
        request_sha256: str,
    ) -> dict[str, Any]:
        """Lock synchronized READ COMMITTED observation; absence stays unknown."""
        from bot_core.postgresql_production_root_lifecycle_qualification import (
            qualify_production_root_lifecycle_resolution,
        )

        if (
            self._role != "administrator"
            or type(conn) is not psycopg.Connection
            or not conn.autocommit
            or conn.info.transaction_status is not psycopg.pq.TransactionStatus.IDLE
        ):
            raise ProductionRootLifecycleError("IDLE_DIRECT_ADMINISTRATIVE_CONNECTION_REQUIRED")
        boundary = self._current()
        try:
            with conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                self._require_peer_authentication(conn, boundary)
                qualify_production_root_lifecycle_resolution(conn, boundary.pin)
                _qualify_principals(conn, boundary)
                row = conn.execute(
                    sql.SQL("SELECT {}.read_operation(%s,%s,%s)").format(
                        sql.Identifier(boundary.pin.provisioning.schema)
                    ),
                    (semantic_role, operation_id, request_sha256),
                ).fetchone()
                if row is None or len(row) != 1 or type(row[0]) is not dict:
                    raise ProductionRootLifecycleIndeterminate(
                        "CLOSE_NOT_FOUND_PRIOR_OUTCOME_UNKNOWN"
                    )
                result = row[0]
            self._current()
            return result
        except Exception:
            raise ProductionRootLifecycleIndeterminate(
                "POSITIVE_CLOSE_RESOLUTION_UNAVAILABLE"
            ) from None

    def qualify_installed_service(self) -> None:
        """Connect using fixed local peer authentication, then observe both signers."""
        if self._role != "issuer":
            raise ProductionRootLifecycleError("ISSUER_SERVICE_QUALIFICATION_REQUIRED")
        boundary = self._current()
        identity = boundary.pin.identity
        if not Path(identity.connection_host).is_absolute():
            raise ProductionRootLifecycleError("INSTALLED_LOCAL_PEER_DATABASE_REQUIRED")
        with psycopg.connect(
            host=identity.connection_host,
            port=identity.connection_port,
            dbname=identity.database_name,
            user=boundary.principal(self._role)[3],
            password="",
            passfile="/dev/null",
            options="",
            sslmode="disable",
            connect_timeout=5,
            application_name="cryptohunter-root-lifecycle-qualification",
            autocommit=True,
        ) as conn:
            with conn.transaction():
                conn.execute("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE")
                self.qualify(conn)
        for role in sorted(_SIGNERS):
            self.qualify_signer_peer(role)

    def qualify_signer_peer(self, role: str) -> tuple[int, int, int]:
        """Observe a separate installed Unix process, never accept a caller socket."""
        if sys.platform != "linux":
            raise ProductionRootLifecycleError("LINUX_SERVICE_INSTALLATION_REQUIRED")
        boundary = self._current()
        if role not in _SIGNERS:
            raise ProductionRootLifecycleError("EXACT_SIGNER_ROLE_REQUIRED")
        _, uid, gid, _ = boundary.principal(role)
        path = Path(dict(boundary.signer_sockets)[role])
        try:
            for ancestor in path.parents:
                info = ancestor.lstat()
                if not stat.S_ISDIR(info.st_mode) or info.st_uid != 0 or info.st_mode & 0o022:
                    raise ProductionRootLifecycleError("PROTECTED_SIGNER_SOCKET_REQUIRED")
            info = path.lstat()
            if (
                not stat.S_ISSOCK(info.st_mode)
                or info.st_uid != uid
                or info.st_gid != boundary.principal("issuer")[2]
                or stat.S_IMODE(info.st_mode) != 0o660
            ):
                raise ProductionRootLifecycleError("PROTECTED_SIGNER_SOCKET_REQUIRED")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as peer:
                peer.settimeout(3)
                peer.connect(str(path))
                pid, peer_uid, peer_gid = struct.unpack(
                    "3i",
                    peer.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")),
                )
            if (peer_uid, peer_gid) != (uid, gid) or pid <= 0 or pid == os.getpid():
                raise ProductionRootLifecycleError("EXACT_SEPARATE_SIGNER_PROCESS_REQUIRED")
            self._current()
            return pid, peer_uid, peer_gid
        except OSError:
            raise ProductionRootLifecycleError("INSTALLED_SIGNER_UNAVAILABLE") from None

    def assert_active_fences(
        self, conn: psycopg.Connection[Any], fences: list[dict[str, Any]]
    ) -> None:
        """Hold exact SQL fences in the caller's existing final transaction.

        Fences are expectations only. This port establishes no attempt or
        entitlement authorization and must precede requester/claimant locks.
        """
        from psycopg.types.json import Jsonb

        from bot_core.postgresql_atomic_root_proof_transaction import (
            capture_atomic_root_proof_transaction_owner,
            validate_atomic_root_proof_transaction_owner,
        )

        if self._role != "issuer":
            raise ProductionRootLifecycleError("ISSUER_FENCING_PORT_REQUIRED")
        boundary = self._current()
        token = capture_atomic_root_proof_transaction_owner(
            conn, expected_login=boundary.pin.provisioning.issuer_role
        )
        self.qualify(conn)
        conn.execute(
            sql.SQL("SELECT {}.assert_active_fences(%s)").format(
                sql.Identifier(boundary.pin.provisioning.schema)
            ),
            (Jsonb(fences),),
        ).fetchone()
        validate_atomic_root_proof_transaction_owner(conn, token)
        self._current()

    def commit(self, *_args: object, **_kwargs: object) -> None:
        raise ProductionRootLifecycleError(
            "PRODUCTION_BIND_BLOCKED_FROZEN_TRANSACTION_CONFLICT_AND_UNQUALIFIED_ATTEMPT_INGRESS"
        )


def open_installed_production_root_lifecycle() -> ProductionRootLifecycleBoundary:
    """No configuration, path, credential, profile or principal overrides."""
    if sys.platform != "linux":
        raise ProductionRootLifecycleError("LINUX_SERVICE_INSTALLATION_REQUIRED")
    boundary = _installed_boundary()
    for role in ("issuer", "administrator"):
        if os.geteuid() == boundary.principal(role)[1]:
            _require_process(boundary, role)
            return ProductionRootLifecycleBoundary(boundary, role)
    raise ProductionRootLifecycleError("EXACT_INSTALLED_PROCESS_IDENTITY_REQUIRED")
