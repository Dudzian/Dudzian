"""Qualification-only claims about a custody completion; never authority evidence.

Canonical bytes and an exact existing V2 journal can be compared independently
of Python receipt membership. Neither authenticates the writer, execution,
previous fsync, a completion principal, or a committed production close. This
module has no signing, mutation, provisioning, publication, or acceptance port.
Every assessment remains NOT_QUALIFIED and production emission remains BLOCKED.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterator, cast

from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.local_signing_lifecycle_evidence import (
    _decode_operation,
    observe_durable_signing_lifecycle_completion,
)
from bot_core.postgresql_atomic_root_proof_qualification import (
    PostgreSQLAtomicRootProofPhysicalIdentity,
)
from bot_core.root_proof_issuer_substrate import (
    ProviderRole,
    SecurityProfile,
    SecurityProfileIdentity,
    public_key_material_identity,
)

CLAIM_SCHEMA = "cryptohunter.independent-custody-completion-qualification-claim"
CLAIM_DOMAIN = "CryptoHunter/IndependentCustodyCompletion/Qualification/V1"
COMPLETION_SEMANTIC_ROLE = "CUSTODY_COMPLETION_ATTESTATION_V1"
MAX_CLAIM_BYTES = 2 * 1024 * 1024
MAX_JSON_DEPTH = 16
_BLOCKERS = (
    "COMPLETION_AUTHORITY_IDENTITY_NOT_INDEPENDENTLY_QUALIFIED",
    "COMPLETION_CREDENTIAL_LIFECYCLE_NOT_QUALIFIED",
    "EXTERNAL_TRUST_PIN_NOT_VERIFIED",
    "LIVE_PRODUCTION_CLOSE_NOT_VERIFIED",
    "CUSTODY_MUTATION_SOURCE_NOT_AUTHENTICATED",
    "CUSTODY_EXECUTION_NOT_ATTESTED",
    "PRIOR_DURABILITY_NOT_ATTESTED",
    "COMPLETION_PUBLICATION_NOT_AUTHORIZED",
    "CONTRACT_RECONCILIATION_PENDING_APPROVAL",
)
_ROLES = {ProviderRole.ROOT_PROOF_SIGNING, ProviderRole.HISTORY_ATTESTATION_SIGNING}


class CompletionQualificationError(ValueError):
    """Malformed, conflicting or changed qualification claims fail closed."""


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, item in pairs:
        if key in result:
            raise ValueError("duplicate qualification field")
        result[key] = item
    return result


def _forbidden_number(_value: str) -> Any:
    raise ValueError("floating point/non-finite qualification field")


def _bounded_depth(raw: bytes) -> None:
    # Count structural delimiters outside strings before recursive JSON parsing.
    # The JSON decoder remains responsible for syntax and delimiter matching.
    depth = 0
    quoted = False
    escaped = False
    for character in raw:
        if quoted:
            if escaped:
                escaped = False
            elif character == 92:
                escaped = True
            elif character == 34:
                quoted = False
        elif character == 34:
            quoted = True
        elif character in (91, 123):
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise ValueError("qualification JSON nesting exceeds limit")
        elif character in (93, 125):
            depth -= 1
            if depth < 0:
                raise ValueError("invalid qualification JSON nesting")


def _json(raw: bytes, *, limit: int, canonical: bool = True) -> dict[str, Any]:
    if type(raw) is not bytes or not 0 < len(raw) <= limit:
        raise ValueError("bounded original bytes required")
    _bounded_depth(raw)
    value = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=_object,
        parse_constant=_forbidden_number,
        parse_float=_forbidden_number,
    )
    if type(value) is not dict or (canonical and canonical_json_bytes(value) != raw):
        raise ValueError("exact qualification JSON object required")
    return cast(dict[str, Any], value)


def _fields(value: object, keys: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError("missing or unknown qualification field")
    return cast(dict[str, Any], value)


def _text(value: object, *, limit: int = 512) -> str:
    if type(value) is not str or not value.strip() or "\0" in value or len(value) > limit:
        raise ValueError("bounded exact qualification text required")
    return value


def _integer(value: object, *, maximum: int = 9_007_199_254_740_991) -> int:
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError("positive exact qualification integer required")
    return value


def _digest(value: object) -> str:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("exact SHA256 required")
    return value


def _bytes(value: object, *, limit: int) -> bytes:
    if type(value) is not str or not 0 < len(value) <= ((limit + 2) // 3) * 4:
        raise ValueError("bounded canonical base64 required")
    raw = base64.b64decode(value, validate=True)
    if len(raw) > limit or base64.b64encode(raw).decode("ascii") != value:
        raise ValueError("bounded canonical base64 required")
    return raw


def _database(value: object) -> dict[str, Any]:
    result = _fields(
        value,
        {
            "system_identifier",
            "database_oid",
            "database_name",
            "cluster_name",
            "connection_host",
            "connection_port",
            "server_address",
            "server_port",
            "instance_id",
        },
    )
    system = _text(result["system_identifier"], limit=20)
    if re.fullmatch(r"[1-9][0-9]*", system) is None or int(system) > 2**64 - 1:
        raise ValueError("exact decimal PostgreSQL system identifier required")
    _integer(result["database_oid"], maximum=2**32 - 1)
    _integer(result["connection_port"], maximum=65535)
    for name in ("database_name", "connection_host", "instance_id"):
        _text(result[name])
    if (
        type(result["cluster_name"]) is not str
        or len(result["cluster_name"]) > 512
        or "\0" in result["cluster_name"]
    ):
        raise ValueError("bounded cluster name required")
    if result["server_address"] is not None:
        _text(result["server_address"])
    if result["server_port"] is not None:
        _integer(result["server_port"], maximum=65535)
    coordinates = dict(result)
    coordinates["system_identifier"] = int(system)
    PostgreSQLAtomicRootProofPhysicalIdentity(**coordinates)
    return result


def _authority(value: object) -> dict[str, Any]:
    result = _fields(
        value,
        {
            "authority_id",
            "host_id",
            "os_uid",
            "os_gid",
            "credential_identity",
            "semantic_role",
            "key_version",
            "lifecycle_generation",
            "lifecycle_state",
            "public_key_b64",
            "public_material_identity",
        },
    )
    for name in ("authority_id", "host_id", "credential_identity"):
        _text(result[name])
    for name in ("os_uid", "os_gid"):
        _integer(result[name], maximum=2**32 - 2)
    _integer(result["key_version"])
    _integer(result["lifecycle_generation"])
    legal = {"ACTIVE": {1}, "VERIFY_ONLY": {2}, "REVOKED": {2, 3}}
    if (
        result["semantic_role"] != COMPLETION_SEMANTIC_ROLE
        or type(result["lifecycle_state"]) is not str
        or result["lifecycle_generation"] not in legal.get(result["lifecycle_state"], set())
    ):
        raise ValueError("exact claimed completion credential role/lifecycle required")
    public = _bytes(result["public_key_b64"], limit=32)
    if len(public) != 32 or result["public_material_identity"] != public_key_material_identity(
        public
    ):
        raise ValueError("claimed completion public identity differs")
    return result


def _validated(raw: bytes) -> dict[str, Any]:
    try:
        value = _json(raw, limit=MAX_CLAIM_BYTES)
        _fields(
            value,
            {
                "schema",
                "schema_version",
                "domain",
                "profile",
                "installation_id",
                "environment_id",
                "trust_domain",
                "installation_sha256",
                "external_trust_pin_sha256",
                "completion_authority_claim",
                "database_identity_claim",
                "production_close_request_b64",
                "production_close_request_sha256",
                "operation_descriptor_b64",
                "operation_descriptor_sha256",
                "complete_journal_b64",
                "complete_journal_sha256",
            },
        )
        if (
            value["schema"] != CLAIM_SCHEMA
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 1
            or value["domain"] != CLAIM_DOMAIN
            or value["profile"] != "PRODUCTION_LOCAL"
        ):
            raise ValueError("exact qualification domain/profile/version required")
        for name in ("installation_id", "environment_id", "trust_domain"):
            _text(value[name])
        for name in ("installation_sha256", "external_trust_pin_sha256"):
            _digest(value[name])
        authority = _authority(value["completion_authority_claim"])
        database = _database(value["database_identity_claim"])
        if value["installation_id"] != database["instance_id"]:
            raise ValueError("claimed installation differs from physical database instance")
        originals = {}
        for name, limit in (
            ("production_close_request", 65536),
            ("operation_descriptor", 262144),
            ("complete_journal", 1048576),
        ):
            original = _bytes(value[name + "_b64"], limit=limit)
            if hashlib.sha256(original).hexdigest() != _digest(value[name + "_sha256"]):
                raise ValueError("original qualification bytes differ from digest")
            originals[name] = original
        descriptor = _json(originals["operation_descriptor"], limit=262144)
        journal = _json(originals["complete_journal"], limit=1048576)
        before = _fields(
            journal,
            {
                "schema",
                "schema_version",
                "operation_id",
                "operation_descriptor_b64",
                "before_metadata",
                "after_metadata",
                "completion_state",
            },
        )["before_metadata"]
        if type(before) is not dict:
            raise ValueError("exact journal predecessor required")
        role = ProviderRole(before["provider_role"])
        if role not in _ROLES:
            raise ValueError("root/history completion only")
        security = SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, value["trust_domain"])
        record = _decode_operation(originals["complete_journal"], security, role)
        if record.state != "COMPLETE" or record.descriptor != originals["operation_descriptor"]:
            raise ValueError("exact original V2 COMPLETE descriptor required")
        _fields(
            descriptor,
            {
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
            },
        )
        scope = _fields(
            descriptor["custody_scope"], {"directory", "security_profile", "trust_domain"}
        )
        directory = _text(scope["directory"], limit=4096)
        path = Path(directory)
        if not path.is_absolute() or str(path) != directory or ".." in path.parts:
            raise ValueError("exact absolute custody scope required")
        expected = {
            "protocol_version": 2,
            "operation_id": record.operation_id,
            "semantic_role": role.value,
            "credential_identity": asdict(record.before.credential_identity()),
            "public_key_hex": record.before.public_key.hex(),
            "key_version": record.before.key_version,
            "lifecycle_generation": record.before.lifecycle_generation,
            "predecessor_state": record.before.lifecycle_state.value,
            "target_state": record.after.lifecycle_state.value,
            "custody_scope": {
                "directory": directory,
                "security_profile": "PRODUCTION_LOCAL",
                "trust_domain": value["trust_domain"],
            },
            "database_identity": database,
        }
        for name, item in expected.items():
            if canonical_json_bytes(descriptor[name]) != canonical_json_bytes(item):
                raise ValueError("journal/descriptor/physical identity binding differs")
        _text(descriptor["fence_key"])
        # PostgreSQL hashes convert_to(p_request::text, 'UTF8'), not the
        # canonical JSON profile. Preserve these untrusted original bytes.
        close = _json(originals["production_close_request"], limit=65536, canonical=False)
        expected_close = {
            "operation_id": record.operation_id,
            "semantic_role": role.value,
            "identity": record.before.credential_id,
            "generation": record.before.lifecycle_generation,
            "predecessor_state": record.before.lifecycle_state.value,
            "target_state": record.after.lifecycle_state.value,
        }
        if canonical_json_bytes(close) != canonical_json_bytes(expected_close):
            raise ValueError("production close differs from exact journal operation")
        if authority["public_material_identity"] == record.before.key_material_identity:
            raise ValueError("completion credential cannot alias the signer credential")
        return value
    except Exception as exc:
        raise CompletionQualificationError("INVALID_COMPLETION_QUALIFICATION_CLAIM") from exc


@dataclass(frozen=True, slots=True)
class CompletionQualificationClaim:
    """Validated untrusted claims; constructing one conveys no authority."""

    canonical_bytes: bytes

    def __post_init__(self) -> None:
        _validated(self.canonical_bytes)

    @property
    def document(self) -> dict[str, Any]:
        """Return a fresh snapshot; all identity/pin fields remain untrusted."""
        return _validated(self.canonical_bytes)

    @property
    def fingerprint_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


def parse_completion_qualification_claim(raw: bytes) -> CompletionQualificationClaim:
    """Strictly parse closed V1 claims without reading files or credentials."""
    return CompletionQualificationClaim(raw)


@dataclass(frozen=True, slots=True)
class CompletionQualificationObservation:
    """Non-authorizing assessment, including after exact local journal reads."""

    claim: CompletionQualificationClaim
    status: str = field(default="NOT_QUALIFIED", init=False)
    production_emission: str = field(default="BLOCKED", init=False)
    blockers: tuple[str, ...] = field(default=_BLOCKERS, init=False)


def qualify_completion_claim(
    claim: CompletionQualificationClaim,
) -> CompletionQualificationObservation:
    """Assess only structural binding; no caller claim can supply missing proof."""
    if type(claim) is not CompletionQualificationClaim:
        raise CompletionQualificationError("EXACT_QUALIFICATION_CLAIM_REQUIRED")
    _validated(claim.canonical_bytes)
    return CompletionQualificationObservation(claim)


@contextmanager
def observe_completion_qualification(
    claim: CompletionQualificationClaim,
) -> Iterator[CompletionQualificationObservation]:
    """Compare exact existing V2 bytes under its shared lock and exit rechecks.

    The caller-selected custody scope remains untrusted. This is an observation
    of files, not authentication of their source or of a production close.
    Missing evidence, INTENT, conflict and path changes fail closed; no recovery,
    fsync, secret access, reconnect or retry occurs here.
    """
    assessment = qualify_completion_claim(claim)
    value = claim.document
    descriptor = _bytes(value["operation_descriptor_b64"], limit=262144)
    operation = _json(descriptor, limit=262144)
    with observe_durable_signing_lifecycle_completion(
        Path(operation["custody_scope"]["directory"]),
        security=SecurityProfileIdentity(SecurityProfile.PRODUCTION_LOCAL, value["trust_domain"]),
        role=ProviderRole(operation["semantic_role"]),
        operation_descriptor=descriptor,
    ) as observation:
        if observation.evidence_bytes != _bytes(value["complete_journal_b64"], limit=1048576):
            raise CompletionQualificationError("EXACT_EXISTING_JOURNAL_CLAIM_REQUIRED")
        yield assessment


__all__ = [
    "CLAIM_DOMAIN",
    "CLAIM_SCHEMA",
    "COMPLETION_SEMANTIC_ROLE",
    "MAX_CLAIM_BYTES",
    "CompletionQualificationClaim",
    "CompletionQualificationError",
    "CompletionQualificationObservation",
    "observe_completion_qualification",
    "parse_completion_qualification_claim",
    "qualify_completion_claim",
]
