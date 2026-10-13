"""DB-first, test-qualified administration of genuine local signing custody.

The database close commits before the filesystem lock is acquired. Completion
publication holds that lock, then locks the already closed database fence. A
lost COMMIT response requires an authoritative, lock-synchronized resolution.
This administrator never provisions or activates signing eligibility.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
import weakref
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, cast

import psycopg
from psycopg import sql

from bot_core import local_signing_custody as custody
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.local_signing_lifecycle_evidence import (
    DurableSigningLifecycleCompletion,
    durable_transition_local_signing_credential_lifecycle,
    verified_durable_signing_lifecycle_completion,
)
from bot_core.postgresql_atomic_root_proof_qualification import (
    PostgreSQLAtomicRootProofPhysicalIdentity,
)
from bot_core.postgresql_custody_lifecycle_qualification import (
    CustodyLifecycleQualificationError,
    PostgreSQLCustodyLifecycleTestDeployment,
    qualify_custody_lifecycle_authority,
    validate_custody_lifecycle_deployment,
)
from bot_core.postgresql_custody_lifecycle_schema import expected_custody_fence_key
from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    CredentialSemanticRole,
    ProviderRole,
    SecurityProfileIdentity,
)


PROTOCOL_VERSION = 2
ATOMIC_ISSUER_READY = "NOT_READY"
PRODUCTION_LIFECYCLE_ISOLATION = "BLOCKED"
_ROLES = {ProviderRole.ROOT_PROOF_SIGNING, ProviderRole.HISTORY_ATTESTATION_SIGNING}
_MAC_DOMAIN = b"CryptoHunter/CustodyLifecycleCompletion/V2\0"
_REQUEST_KEYS = {
    "protocol_version",
    "operation_id",
    "fence_key",
    "semantic_role",
    "credential_identity",
    "public_key_hex",
    "key_version",
    "lifecycle_generation",
    "predecessor_state",
    "target_state",
    "custody_scope",
    "database_identity",
}


class CustodyLifecycleError(RuntimeError):
    """Lifecycle administration failed closed."""


class CustodyLifecycleConflict(CustodyLifecycleError):
    """Operation identity, predecessor or evidence conflicts."""


class CustodyLifecycleUnavailable(CustodyLifecycleError):
    """No positive authority observation is available."""


class CustodyLifecycleIndeterminate(CustodyLifecycleError):
    """A COMMIT may have completed; resolve before custody mutation."""


def _scope(directory: Path, security: SecurityProfileIdentity) -> dict[str, str]:
    return {
        "directory": str(directory),
        "security_profile": security.profile.value,
        "trust_domain": security.trust_domain,
    }


def _database_identity(identity: PostgreSQLAtomicRootProofPhysicalIdentity) -> dict[str, Any]:
    value = asdict(identity)
    # Preserve PostgreSQL's uint64 identifier outside the canonical JSON exact
    # integer range as an exact decimal string.
    value["system_identifier"] = str(identity.system_identifier)
    return value


def _directory(directory: Path) -> Path:
    root = cast(Path, custody._validate_directory(directory, create=False))
    if os.name == "posix" and root.resolve(strict=True) != root:
        raise custody.LocalSigningCustodyError("custody directory ancestry is unsafe")
    return root


def prepare_custody_lifecycle_binding(
    deployment_identity: PostgreSQLAtomicRootProofPhysicalIdentity,
    directory: Path,
    *,
    security: SecurityProfileIdentity,
    role: ProviderRole,
    fence_key: str,
) -> dict[str, Any]:
    """Observe authentic custody under its lock for explicit V2 enrollment.

    Enrollment binds an existing isolated V1 fence; it grants no ACTIVE state.
    The bootstrap migration remains an explicit trusted administration step.
    """
    if type(deployment_identity) is not PostgreSQLAtomicRootProofPhysicalIdentity:
        raise TypeError("exact physical authority identity required")
    if type(role) is not ProviderRole or role not in _ROLES:
        raise TypeError("root/history signing role required")
    if type(fence_key) is not str or not fence_key.strip():
        raise TypeError("exact existing fence identity required")
    root = _directory(directory)
    trusted_security = custody._snapshot_security(security)
    with custody._custody_lock(root, exclusive=False):
        metadata = custody._read_record(custody._record_path(root, role), trusted_security, role)
        binding = _binding_from_metadata(
            deployment_identity, root, trusted_security, role, fence_key, metadata
        )
        if fence_key != expected_custody_fence_key(
            binding, predecessor_state=metadata.lifecycle_state.value
        ):
            raise CustodyLifecycleConflict("fence does not belong to exact custody credential")
        return binding


def _binding_from_metadata(
    deployment_identity: PostgreSQLAtomicRootProofPhysicalIdentity,
    root: Path,
    security: SecurityProfileIdentity,
    role: ProviderRole,
    fence_key: str,
    metadata: custody.LocalSigningMetadata,
) -> dict[str, Any]:
    return {
        "fence_key": fence_key,
        "semantic_role": role.value,
        "credential_identity": asdict(metadata.credential_identity()),
        "public_key_hex": metadata.public_key.hex(),
        "key_version": metadata.key_version,
        "lifecycle_generation": metadata.lifecycle_generation,
        "custody_scope": _scope(root, security),
        "database_identity": _database_identity(deployment_identity),
    }


@dataclass(frozen=True, slots=True)
class CustodyLifecycleRequest:
    """Immutable exact request; facts alone never authorize a mutation."""

    directory: Path
    security: SecurityProfileIdentity
    role: ProviderRole
    descriptor: bytes

    def __post_init__(self) -> None:
        if type(self.descriptor) is not bytes or type(self.role) is not ProviderRole:
            raise TypeError("exact lifecycle request required")
        root = _directory(self.directory)
        security = custody._snapshot_security(self.security)
        try:
            operation = json.loads(self.descriptor)
            if (
                type(operation) is not dict
                or set(operation) != _REQUEST_KEYS
                or canonical_json_bytes(operation) != self.descriptor
                or type(operation["protocol_version"]) is not int
                or operation["protocol_version"] != PROTOCOL_VERSION
                or self.role not in _ROLES
                or operation["semantic_role"] != self.role.value
                or operation["custody_scope"] != _scope(root, security)
                or type(operation["operation_id"]) is not str
                or len(operation["operation_id"]) > 128
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", operation["operation_id"]) is None
                or type(operation["fence_key"]) is not str
                or not operation["fence_key"].strip()
                or type(operation["key_version"]) is not int
                or operation["key_version"] < 1
                or type(operation["lifecycle_generation"]) is not int
                or not 1 <= operation["lifecycle_generation"] < 2**63 - 1
                or (operation["predecessor_state"], operation["target_state"])
                not in {
                    ("ACTIVE", "VERIFY_ONLY"),
                    ("ACTIVE", "REVOKED"),
                    ("VERIFY_ONLY", "REVOKED"),
                }
                or len(bytes.fromhex(operation["public_key_hex"])) != 32
                or bytes.fromhex(operation["public_key_hex"]).hex() != operation["public_key_hex"]
            ):
                raise ValueError("malformed lifecycle request")
            identity = self.credential_identity
            if identity.semantic_role != custody._SEMANTIC_ROLE[self.role]:
                raise ValueError("credential semantic role mismatch")
            if identity.key_material_identity != custody.public_key_material_identity(
                bytes.fromhex(operation["public_key_hex"])
            ):
                raise ValueError("credential/public material mismatch")
            database_identity = dict(operation["database_identity"])
            system_identifier = database_identity["system_identifier"]
            if (
                type(system_identifier) is not str
                or str(int(system_identifier)) != system_identifier
            ):
                raise ValueError("exact decimal system identifier required")
            database_identity["system_identifier"] = int(system_identifier)
            PostgreSQLAtomicRootProofPhysicalIdentity(**database_identity)
        except (ValueError, TypeError, KeyError) as exc:
            raise CustodyLifecycleConflict("invalid exact lifecycle descriptor") from exc
        object.__setattr__(self, "directory", root)
        object.__setattr__(self, "security", security)

    @property
    def operation(self) -> dict[str, Any]:
        value: dict[str, Any] = json.loads(self.descriptor)
        return value

    @property
    def operation_id(self) -> str:
        return str(self.operation["operation_id"])

    @property
    def credential_identity(self) -> CredentialRoleIdentity:
        value = dict(self.operation["credential_identity"])
        value["semantic_role"] = CredentialSemanticRole(value["semantic_role"])
        return CredentialRoleIdentity(**value)


def prepare_custody_lifecycle_request(
    deployment: PostgreSQLCustodyLifecycleTestDeployment,
    directory: Path,
    *,
    security: SecurityProfileIdentity,
    role: ProviderRole,
    administrative_operation_id: str,
    target: custody.SigningKeyLifecycle,
    fence_key: str,
) -> CustodyLifecycleRequest:
    """Prepare once and retain descriptor bytes for exact retry after restart."""
    validate_custody_lifecycle_deployment(deployment)
    if type(target) is not custody.SigningKeyLifecycle:
        raise TypeError("exact lifecycle target required")
    root = _directory(directory)
    trusted_security = custody._snapshot_security(security)
    if type(role) is not ProviderRole or role not in _ROLES:
        raise TypeError("root/history signing role required")
    with custody._custody_lock(root, exclusive=False):
        metadata = custody._read_record(custody._record_path(root, role), trusted_security, role)
        # Successors retain the original enrolled fence identity. Its initial
        # snapshot hash is immutable even after custody generation advances.
        binding = _binding_from_metadata(
            deployment.atomic_deployment.identity, root, trusted_security, role, fence_key, metadata
        )
        operation = dict(
            binding,
            protocol_version=PROTOCOL_VERSION,
            operation_id=administrative_operation_id,
            predecessor_state=metadata.lifecycle_state.value,
            target_state=target.value,
        )
    return CustodyLifecycleRequest(root, trusted_security, role, canonical_json_bytes(operation))


@dataclass(frozen=True, slots=True, eq=False, weakref_slot=True)
class DurableCloseAuthorization:
    """Positive committed DB observation; validated by object membership."""

    descriptor: bytes


_AUTHORIZATIONS: weakref.WeakKeyDictionary[DurableCloseAuthorization, bytes] = (
    weakref.WeakKeyDictionary()
)
_AUTHORIZATION_LOCK = threading.Lock()


def validate_durable_close_authorization(
    close_authorization: object, operation_descriptor: bytes
) -> None:
    with _AUTHORIZATION_LOCK:
        if (
            type(close_authorization) is not DurableCloseAuthorization
            or _AUTHORIZATIONS.get(close_authorization) != operation_descriptor
            or close_authorization.descriptor != operation_descriptor
        ):
            raise CustodyLifecycleConflict("positive committed exact close authorization required")


def _persistence_cut(name: str, connection: psycopg.Connection[Any]) -> None:
    """Fault-injection seam; production performs no extra persistence action."""


class PostgreSQLCustodyLifecycleAdministrator:
    """Trusted administrator; do not share its MAC credential with issuer runtime."""

    def __init__(self, deployment: PostgreSQLCustodyLifecycleTestDeployment) -> None:
        validate_custody_lifecycle_deployment(deployment)
        self._deployment = deployment

    def _require_request(self, request: CustodyLifecycleRequest) -> None:
        if type(request) is not CustodyLifecycleRequest:
            raise TypeError("exact lifecycle request required")
        if request.operation["database_identity"] != _database_identity(
            self._deployment.atomic_deployment.identity
        ):
            raise CustodyLifecycleConflict("request belongs to another physical authority")

    def _call(self, name: str, args: tuple[Any, ...], *, write: bool) -> dict[str, Any] | None:
        commit_started = False
        try:
            with psycopg.connect(self._deployment.admin_connection.dsn) as conn:
                conn.execute("SET TRANSACTION ISOLATION LEVEL READ COMMITTED")
                qualify_custody_lifecycle_authority(conn, self._deployment, expected_role="admin")
                row = conn.execute(
                    sql.SQL("SELECT {}.{}({})").format(
                        sql.Identifier(self._deployment.atomic_deployment.provisioning.schema),
                        sql.Identifier(name),
                        sql.SQL(",").join(sql.Placeholder() for _ in args),
                    ),
                    args,
                ).fetchone()
                value = None if row is None else row[0]
                if value is not None and type(value) is not dict:
                    raise CustodyLifecycleConflict("malformed lifecycle authority response")
                _persistence_cut("after_" + name + "_write", conn)
                commit_started = True
                conn.commit()
                _persistence_cut("after_" + name + "_commit", conn)
                return value
        except psycopg.IntegrityError as exc:
            raise CustodyLifecycleConflict("exact lifecycle operation refused") from exc
        except CustodyLifecycleQualificationError as exc:
            raise CustodyLifecycleUnavailable("lifecycle qualification unavailable") from exc
        except psycopg.Error as exc:
            if commit_started and write:
                raise CustodyLifecycleIndeterminate("lifecycle COMMIT outcome unknown") from exc
            raise CustodyLifecycleUnavailable("lifecycle PostgreSQL authority unavailable") from exc

    def _record(
        self, request: CustodyLifecycleRequest, record: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        if record is not None and (
            canonical_json_bytes(record.get("operation")) != request.descriptor
            or record.get("phase") not in {"PENDING_CLOSED", "COMPLETED"}
            or (
                record["phase"] == "COMPLETED"
                and record.get("terminal_state") != request.operation["target_state"]
            )
        ):
            raise CustodyLifecycleConflict("authority returned a different operation")
        return record

    def _authorize(self, request: CustodyLifecycleRequest) -> DurableCloseAuthorization:
        authorization = DurableCloseAuthorization(request.descriptor)
        with _AUTHORIZATION_LOCK:
            _AUTHORIZATIONS[authorization] = request.descriptor
        return authorization

    def begin_close(self, request: CustodyLifecycleRequest) -> DurableCloseAuthorization:
        self._require_request(request)
        record = self._record(
            request,
            self._call(
                "begin_lifecycle_close", (psycopg.types.json.Jsonb(request.operation),), write=True
            ),
        )
        if record is None:
            raise CustodyLifecycleIndeterminate("no durable close confirmation")
        return self._authorize(request)

    def resolve_close(self, request: CustodyLifecycleRequest) -> DurableCloseAuthorization | None:
        """Lock the fence before reading, settling any still-live prior backend."""
        self._require_request(request)
        record = self._record(
            request,
            self._call(
                "read_lifecycle_operation",
                (psycopg.types.json.Jsonb(request.operation),),
                write=False,
            ),
        )
        return None if record is None else self._authorize(request)

    def mutate_custody(
        self,
        authorization: DurableCloseAuthorization,
        request: CustodyLifecycleRequest,
        *,
        fault_injector: Callable[[str], None] | None = None,
    ) -> DurableSigningLifecycleCompletion:
        self._require_request(request)
        validate_durable_close_authorization(authorization, request.descriptor)
        # Recheck PostgreSQL even for an old in-process capability. Unavailability
        # or drift must not begin an irreversible filesystem operation.
        if self.resolve_close(request) is None:
            raise CustodyLifecycleIndeterminate("durable close not present")
        return durable_transition_local_signing_credential_lifecycle(
            request.directory,
            security=request.security,
            role=request.role,
            credential_identity=request.credential_identity,
            expected_lifecycle_generation=request.operation["lifecycle_generation"],
            target=custody.SigningKeyLifecycle(request.operation["target_state"]),
            lifecycle_operation_id=request.operation_id,
            operation_descriptor=request.descriptor,
            close_authorization=authorization,
            fault_injector=fault_injector,
        )

    def publish_completion(
        self,
        request: CustodyLifecycleRequest,
        completion: DurableSigningLifecycleCompletion,
    ) -> dict[str, Any]:
        self._require_request(request)
        with verified_durable_signing_lifecycle_completion(
            completion, operation_descriptor=request.descriptor
        ) as metadata:
            evidence = canonical_json_bytes(
                {
                    "operation": request.operation,
                    "observed": {
                        "credential_identity": asdict(metadata.credential_identity()),
                        "public_key_hex": metadata.public_key.hex(),
                        "key_version": metadata.key_version,
                        "lifecycle_state": metadata.lifecycle_state.value,
                        "lifecycle_generation": metadata.lifecycle_generation,
                        "custody_scope": _scope(request.directory, request.security),
                    },
                    "local_evidence_sha256": completion.completion_fingerprint_sha256,
                }
            )
            tag = hmac.digest(self._deployment.evidence_key, _MAC_DOMAIN + evidence, hashlib.sha256)
            record = self._record(
                request,
                self._call(
                    "complete_lifecycle_close", (request.operation_id, evidence, tag), write=True
                ),
            )
            if record is None or record["phase"] != "COMPLETED":
                raise CustodyLifecycleIndeterminate("completion not durably confirmed")
            return record

    def reconcile_completion(self, request: CustodyLifecycleRequest) -> dict[str, Any] | None:
        self._require_request(request)
        record = self._record(
            request,
            self._call(
                "read_lifecycle_operation",
                (psycopg.types.json.Jsonb(request.operation),),
                write=False,
            ),
        )
        return record if record is not None and record["phase"] == "COMPLETED" else None

    def execute(self, request: CustodyLifecycleRequest) -> dict[str, Any]:
        """Exact retry never repeats a mutation before observing its real state."""
        terminal = self.reconcile_completion(request)
        if terminal is not None:
            return terminal
        authorization = self.resolve_close(request)
        if authorization is None:
            try:
                authorization = self.begin_close(request)
            except CustodyLifecycleIndeterminate:
                authorization = self.resolve_close(request)
                if authorization is None:
                    raise CustodyLifecycleIndeterminate("unknown close has no positive resolution")
        completion = self.mutate_custody(authorization, request)
        try:
            return self.publish_completion(request, completion)
        except CustodyLifecycleIndeterminate:
            terminal = self.reconcile_completion(request)
            if terminal is None:
                raise
            return terminal
