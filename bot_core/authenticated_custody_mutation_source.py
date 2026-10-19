"""Qualification-only public verification of CustodyMutationExecutionV1.

Matching a signature to an explicit independently supplied policy authenticates
bytes to that policy only. It does not qualify an installed executor, a live
PostgreSQL close, execution, or prior durability. This module has no private-key,
signing, filesystem, SQL, mutation, publication, or production capability port.
There is deliberately no production policy loader or durable replay ledger.
The loader's expected operation and next log position are external obligations;
reusing an unchanged policy does not consume an event or prevent its replay.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import re
from typing import Any, Callable

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from bot_core.independent_custody_completion_qualification import (
    COMPLETION_SEMANTIC_ROLE,
    MAX_JSON_DEPTH,
    _bytes,
    _database,
    _digest,
    _fields,
    _integer,
    _json,
    _text,
)
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.local_signing_lifecycle_evidence import _decode_operation
from bot_core.root_proof_issuer_substrate import (
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
    public_key_material_identity,
)

SOURCE_DOMAIN = b"CryptoHunter/CustodyMutationExecution/V1\0"
SOURCE_SCHEMA = "cryptohunter.custody-mutation-execution"
SOURCE_ENVELOPE_SCHEMA = "cryptohunter.signed-custody-mutation-execution"
SOURCE_POLICY_SCHEMA = "cryptohunter.explicit-custody-source-verification-policy"
ROOT_SOURCE_ROLE = "ROOT_CUSTODY_EXECUTION_V1"
HISTORY_SOURCE_ROLE = "HISTORY_CUSTODY_EXECUTION_V1"
MAX_ENVELOPE_BYTES = 4 * 1024 * 1024
MAX_PAYLOAD_BYTES = 3 * 1024 * 1024
MAX_POLICY_BYTES = 1024 * 1024
MAX_DESCRIPTOR_BYTES = 262144
MAX_JOURNAL_BYTES = 1048576
MAX_METADATA_BYTES = 65536
MAX_CLOSE_REQUEST_BYTES = 65536
_SOURCE_ROLES = frozenset({ROOT_SOURCE_ROLE, HISTORY_SOURCE_ROLE})
_ROLE_BINDING = {
    ROOT_SOURCE_ROLE: ProviderRole.ROOT_PROOF_SIGNING,
    HISTORY_SOURCE_ROLE: ProviderRole.HISTORY_ATTESTATION_SIGNING,
}
_PHYSICAL_DATABASE_FIELDS = {
    "system_identifier",
    "database_oid",
    "database_name",
    "cluster_name",
    "connection_host",
    "connection_port",
    "server_address",
    "server_port",
    "instance_id",
}
_DESCRIPTOR_FIELDS = {
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
_BLOCKERS = (
    "INSTALLED_SOURCE_PROVENANCE_NOT_QUALIFIED",
    "INDEPENDENT_TRUST_ANCHOR_LOADER_NOT_QUALIFIED",
    "LIVE_COMMITTED_PRODUCTION_CLOSE_NOT_VERIFIED",
    "CONTROLLED_CUSTODY_EXECUTION_NOT_ATTESTED",
    "PRIOR_DURABILITY_NOT_ATTESTED",
    "DURABLE_REPLAY_CHECKPOINT_NOT_IMPLEMENTED",
    "INDEPENDENT_COMPLETION_AUTHORITY_NOT_QUALIFIED",
    "COMPLETION_PUBLICATION_NOT_AUTHORIZED",
    "CONTRACT_ADDENDUM_INTEGRATION_PENDING",
)


class SourceMutationVerificationError(ValueError):
    """Missing, malformed, conflicting or changed source inputs fail closed."""


def _version(value: object) -> None:
    if type(value) is not int or value != 1:
        raise ValueError("exact source protocol version required")


def _same(actual: object, expected: object) -> None:
    # Avoid Python's bool/int equality and preserve exact JSON value types.
    if canonical_json_bytes(actual) != canonical_json_bytes(expected):
        raise ValueError("exact source binding differs")


def _decimal(value: object) -> str:
    result: str = _text(value, limit=20)
    if re.fullmatch(r"[1-9][0-9]*", result) is None or int(result) > 2**64 - 1:
        raise ValueError("positive canonical PostgreSQL transaction id required")
    return result


def _installation(value: object) -> dict[str, Any]:
    result: dict[str, Any] = _fields(
        value,
        {
            "installation_id",
            "environment_id",
            "profile",
            "trust_domain",
            "installation_sha256",
            "trust_pin_id",
            "trust_pin_version",
            "trust_pin_sha256",
        },
    )
    for name in ("installation_id", "environment_id", "trust_domain", "trust_pin_id"):
        _text(result[name])
    for name in ("installation_sha256", "trust_pin_sha256"):
        _digest(result[name])
    _integer(result["trust_pin_version"])
    if result["profile"] != "PRODUCTION_LOCAL":
        raise ValueError("production-local source protocol only")
    return result


def _authority(value: object, *, roles: frozenset[str]) -> dict[str, Any]:
    result: dict[str, Any] = _fields(
        value,
        {
            "authority_id",
            "service_id",
            "host_id",
            "os_uid",
            "os_gid",
            "endpoint",
            "provider_namespace",
            "credential_identity",
            "semantic_role",
            "public_key_b64",
            "public_material_identity",
            "key_version",
            "lifecycle_generation",
            "lifecycle_state",
            "trust_epoch",
        },
    )
    for name in (
        "authority_id",
        "service_id",
        "host_id",
        "endpoint",
        "provider_namespace",
        "credential_identity",
        "semantic_role",
    ):
        _text(result[name])
    for name in ("os_uid", "os_gid"):
        _integer(result[name], maximum=2**32 - 2)
    for name in ("key_version", "lifecycle_generation", "trust_epoch"):
        _integer(result[name])
    if result["semantic_role"] not in roles or result["lifecycle_state"] != "ACTIVE":
        raise ValueError("exact ACTIVE role-scoped source credential required")
    public = _bytes(result["public_key_b64"], limit=32)
    if len(public) != 32 or result["public_material_identity"] != public_key_material_identity(
        public
    ):
        raise ValueError("source credential material identity differs")
    return result


def _database_record(value: object) -> dict[str, Any]:
    result: dict[str, Any] = _fields(
        value,
        _PHYSICAL_DATABASE_FIELDS
        | {"authority_identity", "schema_name", "schema_version", "catalog_sha256"},
    )
    _database({name: result[name] for name in _PHYSICAL_DATABASE_FIELDS})
    _text(result["authority_identity"])
    _text(result["schema_name"])
    _integer(result["schema_version"])
    _digest(result["catalog_sha256"])
    return result


def _descriptor(
    raw: bytes,
    installation: dict[str, Any],
    executor: dict[str, Any],
    database: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = _fields(_json(raw, limit=MAX_DESCRIPTOR_BYTES), _DESCRIPTOR_FIELDS)
    if type(result["protocol_version"]) is not int or result["protocol_version"] != 2:
        raise ValueError("original V2 descriptor required")
    for name in ("operation_id", "fence_key"):
        _text(result[name], limit=128 if name == "operation_id" else 512)
    if result["semantic_role"] != _ROLE_BINDING[executor["semantic_role"]].value:
        raise ValueError("cross-role custody source substitution")
    _integer(result["key_version"])
    _integer(result["lifecycle_generation"])
    if (
        type(result["predecessor_state"]) is not str
        or type(result["target_state"]) is not str
        or (result["predecessor_state"], result["target_state"])
        not in {("ACTIVE", "VERIFY_ONLY"), ("ACTIVE", "REVOKED"), ("VERIFY_ONLY", "REVOKED")}
    ):
        raise ValueError("closed custody lifecycle transition required")
    identity = _fields(
        result["credential_identity"],
        {
            "semantic_role",
            "credential_identity",
            "provider_namespace",
            "key_handle_or_version",
            "custody_lifecycle_namespace",
            "key_material_identity",
        },
    )
    for item in identity.values():
        _text(item)
    public_hex = result["public_key_hex"]
    if type(public_hex) is not str or re.fullmatch(r"[0-9a-f]{64}", public_hex) is None:
        raise ValueError("exact descriptor Ed25519 public material required")
    if identity["key_material_identity"] != public_key_material_identity(bytes.fromhex(public_hex)):
        raise ValueError("descriptor material identity differs")
    scope = _fields(result["custody_scope"], {"directory", "security_profile", "trust_domain"})
    directory = _text(scope["directory"], limit=4096)
    path = Path(directory)
    if not path.is_absolute() or str(path) != directory or ".." in path.parts:
        raise ValueError("exact absolute custody scope required")
    _same(scope["security_profile"], installation["profile"])
    _same(scope["trust_domain"], installation["trust_domain"])
    _same(result["database_identity"], {name: database[name] for name in _PHYSICAL_DATABASE_FIELDS})
    _same(database["instance_id"], installation["installation_id"])
    for name, item in (
        ("credential_identity", identity["credential_identity"]),
        ("provider_namespace", identity["provider_namespace"]),
        ("public_material_identity", identity["key_material_identity"]),
    ):
        if executor[name] == item:
            raise ValueError("source credential cannot alias custody signer")
    return result


def _close(value: object, descriptor: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = _fields(
        value,
        {"operation_id", "request_format", "request_b64", "request_sha256", "close_xid", "phase"},
    )
    if (
        result["request_format"] != "PostgreSQL16/jsonb-text/UTF8"
        or result["phase"] != "PENDING_CLOSED"
    ):
        raise ValueError("exact production close request format/phase required")
    _decimal(result["close_xid"])
    _same(result["operation_id"], descriptor["operation_id"])
    raw = _bytes(result["request_b64"], limit=MAX_CLOSE_REQUEST_BYTES)
    if hashlib.sha256(raw).hexdigest() != _digest(result["request_sha256"]):
        raise ValueError("original authoritative close bytes differ")
    # PostgreSQL retains jsonb::text, not frozen canonical JSON or caller wire bytes.
    request = _json(raw, limit=MAX_CLOSE_REQUEST_BYTES, canonical=False)
    _same(
        request,
        {
            "operation_id": descriptor["operation_id"],
            "semantic_role": descriptor["semantic_role"],
            "identity": descriptor["credential_identity"]["credential_identity"],
            "generation": descriptor["lifecycle_generation"],
            "predecessor_state": descriptor["predecessor_state"],
            "target_state": descriptor["target_state"],
        },
    )
    return result


def _close_observation(
    value: object, database: dict[str, Any], close: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = _fields(
        value, {"reader_role", "reader_role_oid", "observation_xid", "qualified_catalog_sha256"}
    )
    _text(result["reader_role"])
    _integer(result["reader_role_oid"], maximum=2**32 - 1)
    _decimal(result["observation_xid"])
    _digest(result["qualified_catalog_sha256"])
    if (
        result["observation_xid"] == close["close_xid"]
        or result["qualified_catalog_sha256"] != database["catalog_sha256"]
    ):
        raise ValueError("separate transaction and exact qualified catalog claim required")
    return result


def _log_position(value: object, executor: dict[str, Any], execution_id: object) -> dict[str, Any]:
    _text(execution_id, limit=128)
    result: dict[str, Any] = _fields(
        value, {"namespace", "sequence", "previous_entry_sha256", "entry_id"}
    )
    _text(result["namespace"])
    _text(result["entry_id"], limit=128)
    _integer(result["sequence"])
    _digest(result["previous_entry_sha256"])
    if (
        result["namespace"] != executor["provider_namespace"]
        or result["entry_id"] != execution_id
        or (result["sequence"] == 1) != (result["previous_entry_sha256"] == "0" * 64)
    ):
        raise ValueError("exact source namespace/entry/chain position required")
    return result


def _originals(value: object) -> dict[str, bytes]:
    limits = {
        "operation_descriptor": MAX_DESCRIPTOR_BYTES,
        "complete_journal": MAX_JOURNAL_BYTES,
        "before_metadata": MAX_METADATA_BYTES,
        "after_metadata": MAX_METADATA_BYTES,
    }
    document: dict[str, Any] = _fields(
        value, {name + suffix for name in limits for suffix in ("_b64", "_sha256")}
    )
    result: dict[str, bytes] = {}
    for name, limit in limits.items():
        raw = _bytes(document[name + "_b64"], limit=limit)
        if hashlib.sha256(raw).hexdigest() != _digest(document[name + "_sha256"]):
            raise ValueError("original source bytes differ from digest")
        _json(raw, limit=limit)
        result[name] = raw
    return result


def _payload(raw: bytes) -> dict[str, Any]:
    result: dict[str, Any] = _fields(
        _json(raw, limit=MAX_PAYLOAD_BYTES),
        {
            "schema",
            "version",
            "domain",
            "installation",
            "executor",
            "database",
            "custody",
            "close",
            "close_observation",
            "originals",
            "execution_id",
            "source_log_position",
            "execution_result",
            "durability_result",
        },
    )
    _version(result["version"])
    if (
        result["schema"] != SOURCE_SCHEMA
        or result["domain"] != SOURCE_DOMAIN[:-1].decode("ascii")
        or result["execution_result"] != "EXACT_LIFECYCLE_TRANSITION_COMPLETED"
        or result["durability_result"] != "METADATA_AND_COMPLETE_SYNCED"
    ):
        raise ValueError("exact source schema/domain/result claims required")
    installation = _installation(result["installation"])
    executor = _authority(result["executor"], roles=_SOURCE_ROLES)
    database = _database_record(result["database"])
    originals = _originals(result["originals"])
    descriptor = _descriptor(originals["operation_descriptor"], installation, executor, database)
    # Strict bounded parsing above precedes the existing V2 record decoder.
    journal = _json(originals["complete_journal"], limit=MAX_JOURNAL_BYTES)
    security = SecurityProfileIdentity(
        SecurityProfile.PRODUCTION_LOCAL, installation["trust_domain"]
    )
    role = _ROLE_BINDING[executor["semantic_role"]]
    record = _decode_operation(originals["complete_journal"], security, role)
    if record.state != "COMPLETE" or record.descriptor != originals["operation_descriptor"]:
        raise ValueError("exact original COMPLETE journal required")
    expected_descriptor = {
        "protocol_version": 2,
        "operation_id": record.operation_id,
        "fence_key": descriptor["fence_key"],
        "semantic_role": role.value,
        "credential_identity": asdict(record.before.credential_identity()),
        "public_key_hex": record.before.public_key.hex(),
        "key_version": record.before.key_version,
        "lifecycle_generation": record.before.lifecycle_generation,
        "predecessor_state": record.before.lifecycle_state.value,
        "target_state": record.after.lifecycle_state.value,
        "custody_scope": descriptor["custody_scope"],
        "database_identity": {name: database[name] for name in _PHYSICAL_DATABASE_FIELDS},
    }
    _same(descriptor, expected_descriptor)
    for name in ("before_metadata", "after_metadata"):
        if canonical_json_bytes(journal[name]) != originals[name]:
            raise ValueError("exact original embedded journal metadata required")
    custody = _fields(
        result["custody"],
        {
            "semantic_role",
            "authority_identity",
            "credential_identity",
            "key_handle_or_version",
            "custody_lifecycle_namespace",
            "public_key_b64",
            "public_material_identity",
            "key_version",
            "predecessor_generation",
            "successor_generation",
            "predecessor_state",
            "target_state",
        },
    )
    _same(
        custody,
        {
            "semantic_role": role.value,
            "authority_identity": record.before.provider_namespace,
            "credential_identity": record.before.credential_id,
            "key_handle_or_version": record.before.credential_identity().key_handle_or_version,
            "custody_lifecycle_namespace": record.before.lifecycle_namespace,
            "public_key_b64": journal["before_metadata"]["public_key_b64"],
            "public_material_identity": record.before.key_material_identity,
            "key_version": record.before.key_version,
            "predecessor_generation": record.before.lifecycle_generation,
            "successor_generation": record.after.lifecycle_generation,
            "predecessor_state": record.before.lifecycle_state.value,
            "target_state": record.after.lifecycle_state.value,
        },
    )
    close = _close(result["close"], descriptor)
    _close_observation(result["close_observation"], database, close)
    _log_position(result["source_log_position"], executor, result["execution_id"])
    return result


def _validated_event(raw: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        envelope: dict[str, Any] = _fields(
            _json(raw, limit=MAX_ENVELOPE_BYTES),
            {"schema", "version", "payload_b64", "credential_id", "key_version", "signature_b64"},
        )
        _version(envelope["version"])
        if envelope["schema"] != SOURCE_ENVELOPE_SCHEMA:
            raise ValueError("exact source envelope required")
        _text(envelope["credential_id"])
        _integer(envelope["key_version"])
        payload = _payload(_bytes(envelope["payload_b64"], limit=MAX_PAYLOAD_BYTES))
        signature = _bytes(envelope["signature_b64"], limit=64)
        if len(signature) != 64:
            raise ValueError("exact Ed25519 signature required")
        _same(envelope["credential_id"], payload["executor"]["credential_identity"])
        _same(envelope["key_version"], payload["executor"]["key_version"])
        return envelope, payload
    except Exception as exc:
        raise SourceMutationVerificationError("INVALID_CUSTODY_MUTATION_EXECUTION_V1") from exc


def _validated_policy(raw: bytes) -> dict[str, Any]:
    try:
        result: dict[str, Any] = _fields(
            _json(raw, limit=MAX_POLICY_BYTES),
            {
                "schema",
                "version",
                "installation",
                "executor",
                "database",
                "completion_authority",
                "operation_descriptor_b64",
                "close",
                "close_observation",
                "execution_id",
                "source_log_position",
            },
        )
        _version(result["version"])
        if result["schema"] != SOURCE_POLICY_SCHEMA:
            raise ValueError("exact explicit source policy required")
        installation = _installation(result["installation"])
        executor = _authority(result["executor"], roles=_SOURCE_ROLES)
        database = _database_record(result["database"])
        descriptor = _descriptor(
            _bytes(result["operation_descriptor_b64"], limit=MAX_DESCRIPTOR_BYTES),
            installation,
            executor,
            database,
        )
        completion = _authority(
            result["completion_authority"], roles=frozenset({COMPLETION_SEMANTIC_ROLE})
        )
        for name in (
            "authority_id",
            "service_id",
            "os_uid",
            "os_gid",
            "endpoint",
            "provider_namespace",
            "credential_identity",
            "public_material_identity",
        ):
            if completion[name] == executor[name]:
                raise ValueError("completion authority cannot alias source")
        for name, item in (
            ("provider_namespace", descriptor["credential_identity"]["provider_namespace"]),
            ("credential_identity", descriptor["credential_identity"]["credential_identity"]),
            (
                "public_material_identity",
                descriptor["credential_identity"]["key_material_identity"],
            ),
        ):
            if completion[name] == item:
                raise ValueError("completion authority cannot alias custody signer")
        close = _close(result["close"], descriptor)
        _close_observation(result["close_observation"], database, close)
        _log_position(result["source_log_position"], executor, result["execution_id"])
        return result
    except Exception as exc:
        raise SourceMutationVerificationError("INVALID_EXPLICIT_CUSTODY_SOURCE_POLICY") from exc


@dataclass(frozen=True, slots=True)
class CustodyMutationExecutionV1:
    """Closed original envelope; parsing alone does not authenticate its claims."""

    canonical_bytes: bytes

    def __post_init__(self) -> None:
        _validated_event(self.canonical_bytes)

    @property
    def document(self) -> dict[str, Any]:
        return _validated_event(self.canonical_bytes)[0]

    @property
    def payload(self) -> dict[str, Any]:
        return _validated_event(self.canonical_bytes)[1]

    @property
    def fingerprint_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class ExplicitSourceVerificationPolicy:
    """Original independent policy snapshot; not installed-provenance evidence."""

    canonical_bytes: bytes

    def __post_init__(self) -> None:
        _validated_policy(self.canonical_bytes)

    @property
    def document(self) -> dict[str, Any]:
        return _validated_policy(self.canonical_bytes)


def parse_custody_mutation_execution(raw: bytes) -> CustodyMutationExecutionV1:
    """Decode bounded canonical source originals without signing or I/O."""
    return CustodyMutationExecutionV1(raw)


def parse_source_verification_policy(raw: bytes) -> ExplicitSourceVerificationPolicy:
    """Decode explicit policy bytes; this does not qualify their installation."""
    return ExplicitSourceVerificationPolicy(raw)


@dataclass(frozen=True, slots=True)
class CustodyMutationSourceVerification:
    """A signature match conveys no execution or production authorization."""

    event: CustodyMutationExecutionV1
    policy_fingerprint_sha256: str
    status: str = field(default="NOT_QUALIFIED", init=False)
    production_emission: str = field(default="BLOCKED", init=False)
    signature: str = field(default="MATCHED_EXPLICIT_POLICY", init=False)
    source_authorization: str = field(default="NOT_QUALIFIED", init=False)
    execution: str = field(default="NOT_ATTESTED", init=False)
    durability: str = field(default="NOT_ATTESTED", init=False)
    blockers: tuple[str, ...] = field(default=_BLOCKERS, init=False)


def _load_policy(
    loader: Callable[[], ExplicitSourceVerificationPolicy | None],
) -> ExplicitSourceVerificationPolicy:
    try:
        policy = loader()
        if type(policy) is not ExplicitSourceVerificationPolicy:
            raise ValueError("independent exact policy snapshot required")
        _validated_policy(policy.canonical_bytes)
        return policy
    except Exception as exc:
        raise SourceMutationVerificationError("INDEPENDENT_SOURCE_POLICY_UNAVAILABLE") from exc


def verify_custody_mutation_execution(
    event: CustodyMutationExecutionV1,
    *,
    policy_loader: Callable[[], ExplicitSourceVerificationPolicy | None],
) -> CustodyMutationSourceVerification:
    """Match original signed bytes to a stable, independently loaded policy.

    The callback receives no envelope, key or caller-selected path. Its protected
    provenance remains unqualified here. Missing policy, a changed/revoked key,
    an advanced expected operation/sequence, or any changed snapshot fails
    closed. No event is consumed: the owner must durably advance its independent
    checkpoint, serialize concurrent decisions and maintain rollback protection.
    """
    before = _load_policy(policy_loader)
    before_bytes = before.canonical_bytes
    try:
        if type(event) is not CustodyMutationExecutionV1:
            raise ValueError("exact source event required")
        event_bytes = event.canonical_bytes
        envelope, payload = _validated_event(event_bytes)
        policy = _validated_policy(before_bytes)
        for name in (
            "installation",
            "executor",
            "database",
            "close",
            "close_observation",
            "execution_id",
            "source_log_position",
        ):
            _same(payload[name], policy[name])
        _same(payload["originals"]["operation_descriptor_b64"], policy["operation_descriptor_b64"])
        # The key comes from the independent snapshot, never from the envelope.
        public = _bytes(policy["executor"]["public_key_b64"], limit=32)
        Ed25519PublicKey.from_public_bytes(public).verify(
            _bytes(envelope["signature_b64"], limit=64),
            SOURCE_DOMAIN + _bytes(envelope["payload_b64"], limit=MAX_PAYLOAD_BYTES),
        )
        after = _load_policy(policy_loader)
        if after.canonical_bytes != before_bytes:
            raise ValueError("independent source policy changed during verification")
        return CustodyMutationSourceVerification(
            CustodyMutationExecutionV1(event_bytes), hashlib.sha256(before_bytes).hexdigest()
        )
    except Exception as exc:
        raise SourceMutationVerificationError("CUSTODY_SOURCE_VERIFICATION_DENIED") from exc


__all__ = [
    "HISTORY_SOURCE_ROLE",
    "MAX_CLOSE_REQUEST_BYTES",
    "MAX_DESCRIPTOR_BYTES",
    "MAX_ENVELOPE_BYTES",
    "MAX_JOURNAL_BYTES",
    "MAX_JSON_DEPTH",
    "MAX_METADATA_BYTES",
    "MAX_PAYLOAD_BYTES",
    "MAX_POLICY_BYTES",
    "ROOT_SOURCE_ROLE",
    "SOURCE_DOMAIN",
    "SOURCE_ENVELOPE_SCHEMA",
    "SOURCE_POLICY_SCHEMA",
    "SOURCE_SCHEMA",
    "CustodyMutationExecutionV1",
    "CustodyMutationSourceVerification",
    "ExplicitSourceVerificationPolicy",
    "SourceMutationVerificationError",
    "parse_custody_mutation_execution",
    "parse_source_verification_policy",
    "verify_custody_mutation_execution",
]
