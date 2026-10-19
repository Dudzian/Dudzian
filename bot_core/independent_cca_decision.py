"""Closed qualification-only IndependentCustodyCompletionV1 verification.

An exact signature match is never installed authority, execution, durability,
publication or issuer capability. No production policy loader, signing key,
SQL connection, filesystem write or runtime CCA is provided here. The caller's
independent loader must retain policy/checkpoint provenance and serialize its
lifecycle; repeated verification does not consume a sequence or prevent rollback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from typing import Any, Callable, ContextManager, Protocol, cast

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

import bot_core.authenticated_custody_mutation_source as source
from bot_core.independent_custody_completion_qualification import (
    COMPLETION_SEMANTIC_ROLE,
    _bytes,
    _digest,
    _fields,
    _integer,
    _json,
    _text,
)
from bot_core.licensing.canonical import canonical_json_bytes
from bot_core.root_proof_issuer_substrate import ProviderRole

CCA_DOMAIN = b"CryptoHunter/IndependentCustodyCompletion/V1\0"
CCA_SCHEMA = "cryptohunter.independent-custody-completion"
CCA_ENVELOPE_SCHEMA = "cryptohunter.signed-independent-custody-completion"
CCA_POLICY_SCHEMA = "cryptohunter.explicit-independent-cca-verification-policy"
MAX_CCA_ENVELOPE_BYTES = 14 * 1024 * 1024
MAX_CCA_PAYLOAD_BYTES = 10 * 1024 * 1024
MAX_CCA_POLICY_BYTES = 2 * 1024 * 1024
_PAYLOAD_FIELDS = frozenset(
    {
        "schema",
        "version",
        "domain",
        "installation",
        "completion_authority",
        "database",
        "custody",
        "close",
        "close_observation",
        "originals",
        "source_event_envelope_b64",
        "source_event_envelope_sha256",
        "decision_id",
        "decision_log_position",
        "source_authentication_result",
        "execution_result",
        "durability_result",
        "completion_result",
    }
)
_DECISION_FIELDS = frozenset({"decision_id", "decision_log_position"})
_PRINCIPAL_ROLES = frozenset(
    {
        source.ROOT_SOURCE_ROLE,
        source.HISTORY_SOURCE_ROLE,
        COMPLETION_SEMANTIC_ROLE,
        ProviderRole.ROOT_PROOF_SIGNING.value,
        ProviderRole.HISTORY_ATTESTATION_SIGNING.value,
    }
)
_BLOCKERS = (
    "INDEPENDENT_CCA_INSTALLATION_AND_POLICY_PROVENANCE_NOT_QUALIFIED",
    "SOURCE_EVENT_RETAINED_PROVENANCE_NOT_QUALIFIED",
    "CONTROLLED_CUSTODY_EXECUTION_NOT_ATTESTED",
    "SOURCE_AND_CCA_DURABILITY_NOT_ATTESTED",
    "INDEPENDENT_COMMITTED_POSTGRESQL_CLOSE_NOT_VERIFIED",
    "HISTORICAL_CREDENTIAL_ELIGIBILITY_NOT_QUALIFIED",
    "SOURCE_CUSTODY_LEASE_NOT_QUALIFIED",
    "PRODUCTION_CCA_LOG_AND_LOCK_ORDER_NOT_QUALIFIED",
    "INDEPENDENT_ROLLBACK_DOMAIN_AND_TRUSTED_FLOOR_NOT_QUALIFIED",
    "COMPLETION_PUBLICATION_NOT_AUTHORIZED",
    "CONTRACT_ADDENDUM_INTEGRATION_PENDING",
)


class IndependentCommittedCloseReader(Protocol):
    """Future installed read port; no implementation or SQL grants exist here.

    The implementation must qualify its own direct-login/peer/physical/catalog
    identity, synchronize the exact committed close in a separate transaction,
    and yield canonical closed close/observation evidence preserving PG bytes.
    Missing, changed or uncertain outcome must raise, never yield approval flags.
    """

    def read_exact_committed_close(
        self, *, original_source_envelope: bytes
    ) -> ContextManager[bytes]: ...


class RetainedSourceEventReader(Protocol):
    """Future authenticated exact lookup; a signature alone is not provenance."""

    def lookup_original_event(
        self, *, installation_id: str, operation_id: str, execution_id: str
    ) -> bytes: ...


class SourceCustodyLease(Protocol):
    """Future source-owned lease held from final re-read through durable CCA ack."""

    def hold_exact_after_state(
        self, *, original_source_envelope: bytes
    ) -> ContextManager[None]: ...


class CredentialEligibilityLease(Protocol):
    """Future source/CCA eligibility interlock; no caller-supplied booleans.

    Rotation/revoke must use the same authority serialization until its retained
    event/decision is durable. Historical eligibility requires a separate reviewed
    authenticated history protocol, which this ACTIVE-only codec does not supply.
    """

    def hold_exact_active_credential(
        self, *, original_pinned_authority: bytes
    ) -> ContextManager[None]: ...


class IndependentCCAVerificationError(ValueError):
    """Malformed originals, unavailable policy and changed bindings fail closed."""


def _separate_authorities(completion: dict[str, Any], executor: dict[str, Any]) -> None:
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
            raise ValueError("CCA cannot alias a source authority")


def _own_observation(
    value: object, event: dict[str, Any], database: dict[str, Any], close: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = source._close_observation(value, database, close)
    for name in ("reader_role", "reader_role_oid", "observation_xid"):
        if result[name] == event["close_observation"][name]:
            raise ValueError("CCA requires its own independently pinned close reader")
    return result


def _payload(raw: bytes) -> dict[str, Any]:
    result: dict[str, Any] = _fields(
        _json(raw, limit=MAX_CCA_PAYLOAD_BYTES),
        set(_PAYLOAD_FIELDS),
    )
    source._version(result["version"])
    if (
        result["schema"] != CCA_SCHEMA
        or result["domain"] != CCA_DOMAIN[:-1].decode("ascii")
        or result["source_authentication_result"] != "QUALIFIED_INSTALLED_SOURCE"
        or result["execution_result"] != "EXACT_LIFECYCLE_TRANSITION_COMPLETED"
        or result["durability_result"] != "SOURCE_AND_CCA_RECORDS_SYNCED"
        or result["completion_result"] != "AUTHENTICATED_CUSTODY_COMPLETION_RETAINED"
    ):
        raise ValueError("exact CCA schema/domain/result assertions required")
    original = _bytes(result["source_event_envelope_b64"], limit=source.MAX_ENVELOPE_BYTES)
    if hashlib.sha256(original).hexdigest() != _digest(result["source_event_envelope_sha256"]):
        raise ValueError("original retained source envelope digest differs")
    event = source.parse_custody_mutation_execution(original).payload
    # Source parsing performs the existing exact V2 descriptor/journal/metadata,
    # physical DB, generation and PostgreSQL-original request mapping first.
    for name in ("installation", "database", "custody", "close", "originals"):
        source._same(result[name], event[name])
    completion = source._authority(
        result["completion_authority"], roles=frozenset({COMPLETION_SEMANTIC_ROLE})
    )
    _separate_authorities(completion, event["executor"])
    for name, custody_name in (
        ("provider_namespace", "authority_identity"),
        ("credential_identity", "credential_identity"),
        ("public_material_identity", "public_material_identity"),
    ):
        if completion[name] == event["custody"][custody_name]:
            raise ValueError("CCA cannot alias a custody signing credential")
    _own_observation(result["close_observation"], event, result["database"], result["close"])
    source._log_position(result["decision_log_position"], completion, result["decision_id"])
    return result


def validate_cca_immutable_binding(raw: bytes) -> dict[str, Any]:
    """Validate the full unsigned binding before a TEST log can retain intent.

    The binding is exact canonical payload bytes with only decision ID/position
    absent. Internally supplied validation-only first-position fields exercise
    the same closed payload validator; they are never signed, retained or
    returned. No signature match, policy provenance or production qualification
    is established. The returned document is a fresh mutable snapshot.
    """
    try:
        result: dict[str, Any] = _fields(
            _json(raw, limit=MAX_CCA_PAYLOAD_BYTES), set(_PAYLOAD_FIELDS - _DECISION_FIELDS)
        )
        validation_id = "qualification-binding-validation"
        _payload(
            canonical_json_bytes(
                {
                    **result,
                    "decision_id": validation_id,
                    "decision_log_position": {
                        "namespace": result["completion_authority"]["provider_namespace"],
                        "sequence": 1,
                        "previous_entry_sha256": "0" * 64,
                        "entry_id": validation_id,
                    },
                }
            )
        )
        return result
    except Exception as exc:
        raise IndependentCCAVerificationError("INVALID_IMMUTABLE_CCA_BINDING") from exc


def _immutable_binding_from_payload(payload: dict[str, Any]) -> bytes:
    """Extract from a just-validated payload; this grants no trust or capability."""
    return cast(
        bytes,
        canonical_json_bytes(
            {name: value for name, value in payload.items() if name not in _DECISION_FIELDS}
        ),
    )


def _validated_decision(raw: bytes) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        envelope = _fields(
            _json(raw, limit=MAX_CCA_ENVELOPE_BYTES),
            {"schema", "version", "payload_b64", "credential_id", "key_version", "signature_b64"},
        )
        source._version(envelope["version"])
        if envelope["schema"] != CCA_ENVELOPE_SCHEMA:
            raise ValueError("exact completion envelope required")
        _text(envelope["credential_id"])
        _integer(envelope["key_version"])
        signature = _bytes(envelope["signature_b64"], limit=64)
        if len(signature) != 64:
            raise ValueError("exact Ed25519 signature required")
        payload = _payload(_bytes(envelope["payload_b64"], limit=MAX_CCA_PAYLOAD_BYTES))
        source._same(
            envelope["credential_id"], payload["completion_authority"]["credential_identity"]
        )
        source._same(envelope["key_version"], payload["completion_authority"]["key_version"])
        return envelope, payload
    except Exception as exc:
        raise IndependentCCAVerificationError("INVALID_INDEPENDENT_CCA_DECISION_V1") from exc


def _principal_inventory(value: object, policy: dict[str, Any]) -> None:
    if type(value) is not list or len(value) != len(_PRINCIPAL_ROLES):
        raise ValueError("all five independent principal records required")
    records: dict[str, dict[str, Any]] = {}
    for item in value:
        record = _fields(
            item,
            {
                "semantic_role",
                "os_uid",
                "os_gid",
                "supplementary_gids",
                "provider_namespace",
                "credential_identity",
                "public_material_identity",
            },
        )
        for name in (
            "semantic_role",
            "provider_namespace",
            "credential_identity",
            "public_material_identity",
        ):
            _text(record[name])
        material = record["public_material_identity"]
        if not material.startswith("sha256:"):
            raise ValueError("exact principal public material identity required")
        _digest(material[7:])
        role = record["semantic_role"]
        if role not in _PRINCIPAL_ROLES or role in records:
            raise ValueError("exact unique principal semantic roles required")
        for name in ("os_uid", "os_gid"):
            _integer(record[name], maximum=2**32 - 2)
        groups = record["supplementary_gids"]
        if type(groups) is not list or len(groups) > 32:
            raise ValueError("bounded exact supplementary group inventory required")
        for group in groups:
            _integer(group, maximum=2**32 - 2)
        if groups != sorted(set(groups)):
            raise ValueError("canonical unique supplementary group inventory required")
        records[role] = record
    for name in (
        "os_uid",
        "os_gid",
        "provider_namespace",
        "credential_identity",
        "public_material_identity",
    ):
        if len({record[name] for record in records.values()}) != len(records):
            raise ValueError("principals cannot share identity, namespace or key material")
    for role, record in records.items():
        own_groups = {record["os_gid"], *record["supplementary_gids"]}
        for other_role, other in records.items():
            if role != other_role and own_groups.intersection(
                {other["os_gid"], *other["supplementary_gids"]}
            ):
                raise ValueError("authority supplementary groups cannot overlap")
    for authority in (policy["executor"], policy["completion_authority"]):
        for name in records[authority["semantic_role"]]:
            if name != "supplementary_gids":
                source._same(records[authority["semantic_role"]][name], authority[name])
    descriptor = _json(
        _bytes(policy["operation_descriptor_b64"], limit=source.MAX_DESCRIPTOR_BYTES),
        limit=source.MAX_DESCRIPTOR_BYTES,
    )
    signer = records[descriptor["semantic_role"]]
    for name, expected in (
        ("provider_namespace", descriptor["credential_identity"]["provider_namespace"]),
        ("credential_identity", descriptor["credential_identity"]["credential_identity"]),
        ("public_material_identity", descriptor["credential_identity"]["key_material_identity"]),
    ):
        source._same(signer[name], expected)


def _validated_policy(raw: bytes) -> tuple[dict[str, Any], source.ExplicitSourceVerificationPolicy]:
    try:
        result = _fields(
            _json(raw, limit=MAX_CCA_POLICY_BYTES),
            {
                "schema",
                "version",
                "source_verification_policy_b64",
                "source_event_envelope_sha256",
                "completion_authority",
                "close_observation",
                "decision_id",
                "decision_log_position",
                "principal_inventory",
            },
        )
        source._version(result["version"])
        if result["schema"] != CCA_POLICY_SCHEMA:
            raise ValueError("exact explicit CCA verification policy required")
        source_policy = source.parse_source_verification_policy(
            _bytes(result["source_verification_policy_b64"], limit=source.MAX_POLICY_BYTES)
        )
        pinned = source_policy.document
        _digest(result["source_event_envelope_sha256"])
        source._same(result["completion_authority"], pinned["completion_authority"])
        _own_observation(result["close_observation"], pinned, pinned["database"], pinned["close"])
        source._log_position(
            result["decision_log_position"], pinned["completion_authority"], result["decision_id"]
        )
        _principal_inventory(result["principal_inventory"], pinned)
        return result, source_policy
    except Exception as exc:
        raise IndependentCCAVerificationError("INVALID_EXPLICIT_CCA_VERIFICATION_POLICY") from exc


@dataclass(frozen=True, slots=True)
class IndependentCCADecisionV1:
    """Original closed envelope; successful parsing conveys no authority."""

    canonical_bytes: bytes

    def __post_init__(self) -> None:
        _validated_decision(self.canonical_bytes)

    @property
    def document(self) -> dict[str, Any]:
        return _validated_decision(self.canonical_bytes)[0]

    @property
    def payload(self) -> dict[str, Any]:
        return _validated_decision(self.canonical_bytes)[1]

    @property
    def payload_bytes(self) -> bytes:
        return cast(bytes, _bytes(self.document["payload_b64"], limit=MAX_CCA_PAYLOAD_BYTES))

    @property
    def immutable_binding_bytes(self) -> bytes:
        """Extract exact canonical payload binding under the shared closed schema."""
        return _immutable_binding_from_payload(self.payload)

    @property
    def fingerprint_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()


@dataclass(frozen=True, slots=True)
class ExplicitCCAVerificationPolicy:
    """Independent expected snapshot; construction does not qualify provenance."""

    canonical_bytes: bytes

    def __post_init__(self) -> None:
        _validated_policy(self.canonical_bytes)

    @property
    def document(self) -> dict[str, Any]:
        return _validated_policy(self.canonical_bytes)[0]


def parse_independent_cca_decision(raw: bytes) -> IndependentCCADecisionV1:
    """Parse bounded original evidence without authentication, signing or I/O."""
    return IndependentCCADecisionV1(raw)


def parse_cca_verification_policy(raw: bytes) -> ExplicitCCAVerificationPolicy:
    """Parse independently supplied expectations; never perform TOFU or re-pin."""
    return ExplicitCCAVerificationPolicy(raw)


@dataclass(frozen=True, slots=True)
class IndependentCCADecisionVerification:
    """Non-authorizing signature assessment, including for synthetic valid keys."""

    decision: IndependentCCADecisionV1
    policy_fingerprint_sha256: str
    status: str = field(default="NOT_QUALIFIED", init=False)
    production_emission: str = field(default="BLOCKED", init=False)
    signature: str = field(default="MATCHED_EXPLICIT_POLICY", init=False)
    source_signature: str = field(default="MATCHED_EXPLICIT_POLICY", init=False)
    source_authorization: str = field(default="NOT_QUALIFIED", init=False)
    committed_close: str = field(default="NOT_QUALIFIED", init=False)
    execution: str = field(default="NOT_ATTESTED", init=False)
    durability: str = field(default="NOT_ATTESTED", init=False)
    blockers: tuple[str, ...] = field(default=_BLOCKERS, init=False)


def _load_policy(
    loader: Callable[[], ExplicitCCAVerificationPolicy | None],
) -> ExplicitCCAVerificationPolicy:
    try:
        policy = loader()
        if type(policy) is not ExplicitCCAVerificationPolicy:
            raise ValueError("independent exact CCA policy snapshot required")
        _validated_policy(policy.canonical_bytes)
        return policy
    except Exception as exc:
        raise IndependentCCAVerificationError("INDEPENDENT_CCA_POLICY_UNAVAILABLE") from exc


def verify_independent_cca_decision(
    decision: IndependentCCADecisionV1,
    *,
    policy_loader: Callable[[], ExplicitCCAVerificationPolicy | None],
) -> IndependentCCADecisionVerification:
    """Match both original signatures against a stable independent policy.

    The loader receives no caller key, pin, envelope or path. Its protected
    provenance, retained source authentication, live close read, held source
    lease, historical eligibility and durable replay advancement are separate
    future obligations. ACTIVE-only matching cannot authorize production.
    """
    before = _load_policy(policy_loader)
    before_bytes = before.canonical_bytes
    try:
        if type(decision) is not IndependentCCADecisionV1:
            raise ValueError("exact original CCA decision required")
        original = decision.canonical_bytes
        envelope, payload = _validated_decision(original)
        policy, source_policy = _validated_policy(before_bytes)
        for name in (
            "completion_authority",
            "close_observation",
            "source_event_envelope_sha256",
            "decision_id",
            "decision_log_position",
        ):
            source._same(payload[name], policy[name])
        event = source.parse_custody_mutation_execution(
            _bytes(payload["source_event_envelope_b64"], limit=source.MAX_ENVELOPE_BYTES)
        )
        source.verify_custody_mutation_execution(event, policy_loader=lambda: source_policy)
        # Verify the original payload bytes. Canonical re-encoding is validation
        # only and is never substituted into the Ed25519 verification message.
        public = _bytes(policy["completion_authority"]["public_key_b64"], limit=32)
        Ed25519PublicKey.from_public_bytes(public).verify(
            _bytes(envelope["signature_b64"], limit=64),
            CCA_DOMAIN + _bytes(envelope["payload_b64"], limit=MAX_CCA_PAYLOAD_BYTES),
        )
        after = _load_policy(policy_loader)
        if after.canonical_bytes != before_bytes:
            raise ValueError("CCA credential, lifecycle, close or checkpoint changed")
        return IndependentCCADecisionVerification(
            IndependentCCADecisionV1(original), hashlib.sha256(before_bytes).hexdigest()
        )
    except Exception as exc:
        raise IndependentCCAVerificationError("INDEPENDENT_CCA_VERIFICATION_DENIED") from exc


__all__ = [
    "CCA_DOMAIN",
    "CCA_ENVELOPE_SCHEMA",
    "CCA_POLICY_SCHEMA",
    "CCA_SCHEMA",
    "MAX_CCA_ENVELOPE_BYTES",
    "MAX_CCA_PAYLOAD_BYTES",
    "MAX_CCA_POLICY_BYTES",
    "CredentialEligibilityLease",
    "ExplicitCCAVerificationPolicy",
    "IndependentCommittedCloseReader",
    "RetainedSourceEventReader",
    "SourceCustodyLease",
    "IndependentCCADecisionV1",
    "IndependentCCADecisionVerification",
    "IndependentCCAVerificationError",
    "parse_cca_verification_policy",
    "parse_independent_cca_decision",
    "validate_cca_immutable_binding",
    "verify_independent_cca_decision",
]
