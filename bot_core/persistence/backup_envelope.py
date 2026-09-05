"""Deterministic, integrity-only backup candidates for a verified StateStore snapshot."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from typing import Any, ClassVar, cast

from .fingerprints import canonical_json_sha256
from .records import PersistenceRecord, PersistenceRecordError, validate_persistence_record
from .state_store import (
    SQLiteStateStore,
    StateStoreError,
    StateStoreMetadata,
    StateStoreSnapshot,
)
from .transaction_descriptor import (
    StateStoreTransactionDescriptor,
    TransactionDescriptorError,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_RECORD_KINDS = frozenset(
    {
        "M0.3ProtectedMembership",
        "M0.3CurrentReference",
        "M0.3RetirementState",
        "LocalDurableStateEvidence",
        "LocalEvidenceRegistry",
        "LocalEvidenceReference",
        "LocalEvidenceCurrentDesignation",
        "AuthenticationProof",
        "CoreIssuedAuthenticationProofBinding",
        "PlatformBiometricAssertion",
        "CoreAcceptedPlatformBiometricAssertionBinding",
        "SecureStorePayload",
    }
)
_FORBIDDEN_PAYLOAD_FIELDS = frozenset(
    {
        "raw_pin",
        "api_secret",
        "password",
        "private_key",
        "bearer_token",
        "secure_store_payload",
        "biometric_material",
        "verifier",
    }
)
_PIN_VERIFIER_REPRESENTATION = "PinVerifierRecord accepted revisions"


class BackupEnvelopeError(ValueError):
    """The candidate violates the frozen BackupEnvelope integrity contract."""


def _require_positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BackupEnvelopeError(f"{name} must be a positive non-boolean integer")
    return value


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise BackupEnvelopeError(f"{name} must be lowercase SHA-256 hexadecimal")
    return value


def _contains_forbidden_field(
    value: object,
    *,
    path: tuple[str, ...] = (),
    legal_verifier_path: tuple[str, ...] | None = None,
) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            item_path = (*path, key)
            if key in _FORBIDDEN_PAYLOAD_FIELDS and item_path != legal_verifier_path:
                return True
            if _contains_forbidden_field(
                item, path=item_path, legal_verifier_path=legal_verifier_path
            ):
                return True
    elif isinstance(value, (list, tuple)):
        return any(
            _contains_forbidden_field(item, path=path, legal_verifier_path=legal_verifier_path)
            for item in value
        )
    return False


def _validate_backup_record(record: PersistenceRecord) -> None:
    validate_persistence_record(record)
    if record.representation_name in _FORBIDDEN_RECORD_KINDS:
        raise BackupEnvelopeError("backup contains a forbidden record kind")
    legal_verifier_path = (
        ("upstream_payload", "verifier")
        if record.representation_name == _PIN_VERIFIER_REPRESENTATION
        else None
    )
    if _contains_forbidden_field(record.payload, legal_verifier_path=legal_verifier_path):
        raise BackupEnvelopeError("backup contains a forbidden payload field")


def _parse_records(value: object, name: str) -> tuple[PersistenceRecord, ...]:
    if not isinstance(value, list):
        raise BackupEnvelopeError(f"{name} must be a JSON array")
    try:
        records = tuple(PersistenceRecord.from_mapping(item) for item in value)
        for record in records:
            _validate_backup_record(record)
    except (PersistenceRecordError, TypeError, ValueError) as exc:
        raise BackupEnvelopeError(f"{name} contains an invalid record") from exc
    if list(records) != sorted(
        records, key=lambda item: (item.representation_name, item.record_key)
    ):
        raise BackupEnvelopeError(f"{name} must already be canonically sorted")
    return records


@dataclass(frozen=True, slots=True)
class BackupIntegrityMetadata:
    """Exact nested integrity carrier; it contains no authority designation."""

    state_store_transaction_descriptors: tuple[StateStoreTransactionDescriptor, ...]

    def to_mapping(self) -> dict[str, object]:
        return {
            "state_store_transaction_descriptors": [
                descriptor.to_mapping() for descriptor in self.state_store_transaction_descriptors
            ]
        }


@dataclass(frozen=True, slots=True)
class BackupEnvelope:
    """Deeply immutable, deterministic backup candidate (integrity only)."""

    backup_envelope_schema_version: int
    state_store_schema_version: int
    account_id: str
    device_installation_id: str
    state_store_identity_fingerprint_sha256: str
    environment: str
    local_protected_freshness_generation: int
    state_fingerprint_sha256: str
    transaction_fingerprint_sha256: str
    history_tail_fingerprint_sha256: str
    canonical_durable_records: tuple[PersistenceRecord, ...]
    immutable_recovery_history: tuple[PersistenceRecord, ...]
    envelope_fingerprint_sha256: str
    integrity_metadata: BackupIntegrityMetadata

    _FIELDS: ClassVar[tuple[str, ...]] = (
        "backup_envelope_schema_version",
        "state_store_schema_version",
        "account_id",
        "device_installation_id",
        "state_store_identity_fingerprint_sha256",
        "environment",
        "local_protected_freshness_generation",
        "state_fingerprint_sha256",
        "transaction_fingerprint_sha256",
        "history_tail_fingerprint_sha256",
        "canonical_durable_records",
        "immutable_recovery_history",
        "envelope_fingerprint_sha256",
        "integrity_metadata",
    )

    def to_mapping(self) -> dict[str, object]:
        """Return the exact 14-field, fresh mutable transport projection."""

        return {
            "backup_envelope_schema_version": self.backup_envelope_schema_version,
            "state_store_schema_version": self.state_store_schema_version,
            "account_id": self.account_id,
            "device_installation_id": self.device_installation_id,
            "state_store_identity_fingerprint_sha256": (
                self.state_store_identity_fingerprint_sha256
            ),
            "environment": self.environment,
            "local_protected_freshness_generation": (self.local_protected_freshness_generation),
            "state_fingerprint_sha256": self.state_fingerprint_sha256,
            "transaction_fingerprint_sha256": self.transaction_fingerprint_sha256,
            "history_tail_fingerprint_sha256": self.history_tail_fingerprint_sha256,
            "canonical_durable_records": [
                record.to_mapping() for record in self.canonical_durable_records
            ],
            "immutable_recovery_history": [
                record.to_mapping() for record in self.immutable_recovery_history
            ],
            "envelope_fingerprint_sha256": self.envelope_fingerprint_sha256,
            "integrity_metadata": self.integrity_metadata.to_mapping(),
        }

    def to_fingerprint_mapping(self) -> dict[str, object]:
        projection = self.to_mapping()
        del projection["envelope_fingerprint_sha256"]
        return projection


def _parse_descriptors(value: object) -> tuple[StateStoreTransactionDescriptor, ...]:
    if not isinstance(value, list):
        raise BackupEnvelopeError("state_store_transaction_descriptors must be a JSON array")
    for raw_descriptor in value:
        if not isinstance(raw_descriptor, Mapping):
            continue
        for array_name in ("current_record_mutations", "immutable_history_appends"):
            raw_records = raw_descriptor.get(array_name)
            if not isinstance(raw_records, list):
                continue
            for raw_record in raw_records:
                if not isinstance(raw_record, Mapping):
                    continue
                legal_path = (
                    ("upstream_payload", "verifier")
                    if raw_record.get("representation_name") == _PIN_VERIFIER_REPRESENTATION
                    else None
                )
                if _contains_forbidden_field(
                    raw_record.get("payload"), legal_verifier_path=legal_path
                ):
                    raise BackupEnvelopeError("backup contains a forbidden payload field")
    try:
        descriptors = tuple(StateStoreTransactionDescriptor.from_mapping(item) for item in value)
        for descriptor in descriptors:
            for record in (
                *descriptor.current_record_mutations,
                *descriptor.immutable_history_appends,
            ):
                _validate_backup_record(record)
    except BackupEnvelopeError:
        raise
    except (PersistenceRecordError, TransactionDescriptorError, TypeError, ValueError) as exc:
        raise BackupEnvelopeError("invalid StateStore transaction descriptor") from exc
    return tuple(sorted(descriptors, key=lambda item: item.target_generation))


def _validate_integrity_metadata(value: object) -> BackupIntegrityMetadata:
    expected = {"state_store_transaction_descriptors"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise BackupEnvelopeError("integrity_metadata requires its exact field set")
    return BackupIntegrityMetadata(_parse_descriptors(value["state_store_transaction_descriptors"]))


def validate_backup_envelope(value: Mapping[str, object] | BackupEnvelope) -> BackupEnvelope:
    """Parse, canonicalize descriptor order, and validate candidate integrity."""

    raw: Mapping[str, object] = value.to_mapping() if isinstance(value, BackupEnvelope) else value
    if not isinstance(raw, Mapping) or set(raw) != set(BackupEnvelope._FIELDS):
        raise BackupEnvelopeError("BackupEnvelope requires its exact 14-field set")
    version = _require_positive_integer(
        raw["backup_envelope_schema_version"], "backup_envelope_schema_version"
    )
    if version != 1:
        raise BackupEnvelopeError("unsupported backup_envelope_schema_version")
    state_store_schema_version = _require_positive_integer(
        raw["state_store_schema_version"], "state_store_schema_version"
    )
    generation = _require_positive_integer(
        raw["local_protected_freshness_generation"],
        "local_protected_freshness_generation",
    )
    integrity = _validate_integrity_metadata(raw["integrity_metadata"])
    current = _parse_records(raw["canonical_durable_records"], "canonical_durable_records")
    history = _parse_records(raw["immutable_recovery_history"], "immutable_recovery_history")
    try:
        metadata = StateStoreMetadata(
            account_id=cast(str, raw["account_id"]),
            device_installation_id=cast(str, raw["device_installation_id"]),
            state_store_schema_version=state_store_schema_version,
            state_store_identity_fingerprint_sha256=_require_sha256(
                raw["state_store_identity_fingerprint_sha256"],
                "state_store_identity_fingerprint_sha256",
            ),
            environment=cast(str, raw["environment"]),
            protected_freshness_generation=generation,
            state_fingerprint_sha256=_require_sha256(
                raw["state_fingerprint_sha256"], "state_fingerprint_sha256"
            ),
            transaction_fingerprint_sha256=_require_sha256(
                raw["transaction_fingerprint_sha256"], "transaction_fingerprint_sha256"
            ),
            history_tail_fingerprint_sha256=_require_sha256(
                raw["history_tail_fingerprint_sha256"], "history_tail_fingerprint_sha256"
            ),
        )
        SQLiteStateStore.verify_snapshot(
            StateStoreSnapshot(
                metadata, current, history, integrity.state_store_transaction_descriptors
            )
        )
    except (StateStoreError, TypeError, ValueError) as exc:
        raise BackupEnvelopeError(
            "BackupEnvelope does not bind a verified StateStore snapshot"
        ) from exc
    fingerprint = _require_sha256(raw["envelope_fingerprint_sha256"], "envelope_fingerprint_sha256")
    candidate = BackupEnvelope(
        version,
        state_store_schema_version,
        metadata.account_id,
        metadata.device_installation_id,
        metadata.state_store_identity_fingerprint_sha256,
        metadata.environment,
        generation,
        metadata.state_fingerprint_sha256,
        metadata.transaction_fingerprint_sha256,
        metadata.history_tail_fingerprint_sha256,
        current,
        history,
        fingerprint,
        integrity,
    )
    try:
        expected_fingerprint = canonical_json_sha256(candidate.to_fingerprint_mapping())
    except (TypeError, ValueError) as exc:
        raise BackupEnvelopeError("BackupEnvelope is not canonical JSON") from exc
    if fingerprint != expected_fingerprint:
        raise BackupEnvelopeError("envelope fingerprint mismatch")
    return candidate


def create_backup_envelope(store: SQLiteStateStore) -> BackupEnvelope | None:
    """Create one candidate from exactly one coherent verified StateStore read."""

    snapshot = store.read_verified_snapshot()
    if snapshot is None:
        return None
    metadata = snapshot.metadata
    projection: dict[str, Any] = {
        "backup_envelope_schema_version": 1,
        "state_store_schema_version": metadata.state_store_schema_version,
        "account_id": metadata.account_id,
        "device_installation_id": metadata.device_installation_id,
        "state_store_identity_fingerprint_sha256": (
            metadata.state_store_identity_fingerprint_sha256
        ),
        "environment": metadata.environment,
        "local_protected_freshness_generation": metadata.protected_freshness_generation,
        "state_fingerprint_sha256": metadata.state_fingerprint_sha256,
        "transaction_fingerprint_sha256": metadata.transaction_fingerprint_sha256,
        "history_tail_fingerprint_sha256": metadata.history_tail_fingerprint_sha256,
        "canonical_durable_records": [record.to_mapping() for record in snapshot.current_records],
        "immutable_recovery_history": [
            record.to_mapping() for record in snapshot.immutable_history
        ],
        "integrity_metadata": {
            "state_store_transaction_descriptors": [
                descriptor.to_mapping()
                for descriptor in sorted(
                    snapshot.transaction_descriptors,
                    key=lambda item: item.target_generation,
                )
            ]
        },
    }
    projection["envelope_fingerprint_sha256"] = canonical_json_sha256(projection)
    return validate_backup_envelope(projection)
