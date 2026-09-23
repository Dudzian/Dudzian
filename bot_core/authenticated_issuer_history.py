"""Executable M0.5 authenticated issuer-history and local-checkpoint contract.

This is issuer substrate, not a semantic RootProofIssuer.  The in-memory history
is a reference authority used to make the append/verification rules executable;
production persistence must provide an equivalent durable CAS.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from threading import RLock
from types import MappingProxyType
from typing import Any, Mapping, Protocol
import uuid

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from bot_core.local_signing_custody import SigningKeyLifecycle
from bot_core.root_proof_issuer_substrate import (
    CredentialRoleIdentity,
    CredentialSemanticRole,
    HistoryAttestationSigningProvider,
    ProviderIdentity,
    ProviderRole,
    public_key_material_identity,
)


RECORD_DOMAIN = b"CryptoHunter/M0.5/IssuerAuthenticatedHistoryRecord/v1\0"
ATTESTATION_DOMAIN = b"CryptoHunter/M0.5/IssuerAuthenticatedHistoryHead/v2\0"
ACCEPTANCE_DOMAIN = b"CryptoHunter/M0.5/HistoricalHeadAcceptanceEvidence/v2\0"
NO_PREDECESSOR = "NO_PREDECESSOR"
GENESIS_SEQUENCE = 1


class HistoryContractError(RuntimeError):
    """Fail-closed history or checkpoint contract violation."""


class HistoricalRevokedSignatureVerificationUnavailable(HistoryContractError):
    """Independent retained acceptance evidence is absent or unavailable."""


class HistorySigningAuthorityUnavailable(HistoryContractError):
    """Trusted signer or lifecycle authority could not supply required evidence."""


class ReconciliationOutcome(str, Enum):
    EXACT_COMMITTED = "EXACT_COMMITTED"
    NOT_FOUND = "NOT_FOUND"
    STALE = "STALE"
    CONFLICT = "CONFLICT"
    CORRUPT = "CORRUPT"
    UNAVAILABLE = "UNAVAILABLE"
    INDETERMINATE = "INDETERMINATE"


def _text(value: object, name: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{name} must be an exact non-empty str")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise TypeError(f"{name} must be an exact int >= {minimum}")
    return value


def _freeze_json(value: object) -> object:
    """Validate the deliberately small, ambiguity-free JSON value domain."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is list:
        return tuple(_freeze_json(item) for item in value)
    if type(value) is dict:
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str or not key or key in frozen:
                raise TypeError("payload keys must be unique exact non-empty strings")
            frozen[key] = _freeze_json(item)
        return MappingProxyType(frozen)
    raise TypeError("payload contains a non-canonical JSON type")


def _plain(value: object) -> object:
    if type(value) is MappingProxyType:
        return {key: _plain(item) for key, item in value.items()}
    if type(value) is tuple:
        return [_plain(item) for item in value]
    return value


def canonical_json_bytes(value: Mapping[str, object]) -> bytes:
    if type(value) is not dict:
        raise TypeError("canonical JSON root must be an exact dict")
    frozen = _freeze_json(value)
    return json.dumps(
        _plain(frozen),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _digest(domain: bytes, value: Mapping[str, object]) -> str:
    return "sha256:" + hashlib.sha256(domain + canonical_json_bytes(value)).hexdigest()


def _credential_material(identity: CredentialRoleIdentity) -> dict[str, object]:
    """Canonical, complete and namespaced credential identity snapshot."""
    if type(identity) is not CredentialRoleIdentity:
        raise TypeError("signing credential identity must be exact")
    if identity.semantic_role is not CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING:
        raise HistoryContractError("history signer identity has the wrong semantic role")
    return {
        "semantic_role": identity.semantic_role.value,
        "credential_identity": identity.credential_identity,
        "provider_namespace": identity.provider_namespace,
        "key_handle_or_version": identity.key_handle_or_version,
        "custody_lifecycle_namespace": identity.custody_lifecycle_namespace,
        "key_material_identity": identity.key_material_identity,
    }


def _snapshot_credential(identity: CredentialRoleIdentity) -> CredentialRoleIdentity:
    material = _credential_material(identity)
    return CredentialRoleIdentity(
        CredentialSemanticRole(material["semantic_role"]),
        material["credential_identity"],
        material["provider_namespace"],
        material["key_handle_or_version"],
        material["custody_lifecycle_namespace"],
        material["key_material_identity"],
    )


@dataclass(frozen=True, slots=True)
class HistoryStreamIdentity:
    stream_id: str
    issuer_authority_identity: str
    security_profile: str
    environment: str
    trust_domain: str
    product_scope: str
    security_epoch: int

    def __post_init__(self) -> None:
        for name in (
            "stream_id",
            "issuer_authority_identity",
            "security_profile",
            "environment",
            "trust_domain",
            "product_scope",
        ):
            _text(getattr(self, name), name)
        _integer(self.security_epoch, "security_epoch", minimum=1)

    def material(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class HistoryEventIdentity:
    logical_operation_id: str
    issuance_attempt_id: str
    root_proof_id: str

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _text(getattr(self, name), name)

    def material(self) -> dict[str, object]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class HistoryRecord:
    stream: HistoryStreamIdentity
    sequence: int
    predecessor_authenticated_digest: str
    event_identity: HistoryEventIdentity
    canonical_event_payload: Mapping[str, object]
    event_digest: str
    authenticated_digest: str

    def __post_init__(self) -> None:
        if (
            type(self.stream) is not HistoryStreamIdentity
            or type(self.event_identity) is not HistoryEventIdentity
        ):
            raise TypeError("history identities must be exact contract objects")
        _integer(self.sequence, "sequence", minimum=GENESIS_SEQUENCE)
        _text(self.predecessor_authenticated_digest, "predecessor_authenticated_digest")
        if type(self.canonical_event_payload) not in (dict, MappingProxyType):
            raise TypeError("canonical_event_payload must be an exact dict snapshot")
        payload = _freeze_json(_plain(self.canonical_event_payload))
        object.__setattr__(self, "canonical_event_payload", payload)
        if self.event_digest != _digest(
            b"CryptoHunter/M0.5/IssuerHistoryEvent/v1\0", _plain(payload)
        ):
            raise HistoryContractError("event digest mismatch")
        if self.authenticated_digest != _digest(RECORD_DOMAIN, self.unsigned_material()):
            raise HistoryContractError("authenticated record digest mismatch")

    def unsigned_material(self) -> dict[str, object]:
        return {
            "stream": self.stream.material(),
            "sequence": self.sequence,
            "predecessor_authenticated_digest": self.predecessor_authenticated_digest,
            "event_identity": self.event_identity.material(),
            "canonical_event_payload": _plain(self.canonical_event_payload),
            "event_digest": self.event_digest,
        }


def build_record(
    stream: HistoryStreamIdentity,
    sequence: int,
    predecessor: str,
    event_identity: HistoryEventIdentity,
    payload: Mapping[str, object],
) -> HistoryRecord:
    if (
        type(stream) is not HistoryStreamIdentity
        or type(event_identity) is not HistoryEventIdentity
    ):
        raise TypeError("history identities must be exact contract objects")
    if type(payload) is not dict:
        raise TypeError("payload must be an exact canonical JSON dict")
    frozen = _freeze_json(payload)
    event_digest = _digest(b"CryptoHunter/M0.5/IssuerHistoryEvent/v1\0", _plain(frozen))
    material = {
        "stream": stream.material(),
        "sequence": sequence,
        "predecessor_authenticated_digest": predecessor,
        "event_identity": event_identity.material(),
        "canonical_event_payload": _plain(frozen),
        "event_digest": event_digest,
    }
    return HistoryRecord(
        stream,
        sequence,
        predecessor,
        event_identity,
        frozen,
        event_digest,
        _digest(RECORD_DOMAIN, material),
    )


@dataclass(frozen=True, slots=True)
class AttestedHistoryHead:
    stream: HistoryStreamIdentity
    sequence: int
    record_digest: str
    signing_credential_identity: CredentialRoleIdentity
    canonical_attestation_bytes: bytes
    signature: bytes

    def __post_init__(self) -> None:
        if type(self.stream) is not HistoryStreamIdentity:
            raise TypeError("stream must be an exact HistoryStreamIdentity")
        _integer(self.sequence, "sequence", minimum=GENESIS_SEQUENCE)
        _text(self.record_digest, "record_digest")
        _credential_material(self.signing_credential_identity)
        if type(self.canonical_attestation_bytes) is not bytes or type(self.signature) is not bytes:
            raise TypeError("attestation and signature must be exact bytes")


def attest_head(
    record: HistoryRecord, signer: HistoryAttestationSigningProvider
) -> AttestedHistoryHead:
    identity = signer.active_credential_identity()
    if (
        type(identity) is not CredentialRoleIdentity
        or identity.semantic_role is not CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING
    ):
        raise HistoryContractError("history head requires HISTORY_ATTESTATION_SIGNING role")
    _text(identity.key_handle_or_version, "key version")
    signer_identity = _snapshot_credential(identity)
    material = {
        "stream": record.stream.material(),
        "sequence": record.sequence,
        "record_digest": record.authenticated_digest,
        "signing_credential_identity": _credential_material(signer_identity),
    }
    canonical = ATTESTATION_DOMAIN + canonical_json_bytes(material)
    return AttestedHistoryHead(
        record.stream,
        record.sequence,
        record.authenticated_digest,
        signer_identity,
        canonical,
        signer.sign_history_head(canonical),
    )


def _trusted_history_credential(
    head: AttestedHistoryHead,
    authority: HistoryAttestationSigningProvider,
) -> tuple[CredentialRoleIdentity, bytes, SigningKeyLifecycle, int]:
    """Resolve one head signer and key exclusively from the trusted authority."""
    try:
        provider_identity = authority.identity
        credentials = authority.credential_identities()
    except Exception as exc:
        raise HistorySigningAuthorityUnavailable(
            "history signing authority evidence unavailable"
        ) from exc
    if (
        type(provider_identity) is not ProviderIdentity
        or provider_identity.role is not ProviderRole.HISTORY_ATTESTATION_SIGNING
        or provider_identity.security.profile.value != head.stream.security_profile
        or provider_identity.security.trust_domain != head.stream.trust_domain
        or type(credentials) is not tuple
    ):
        raise HistoryContractError("invalid history signing authority")
    matches = tuple(
        credential
        for credential in credentials
        if type(credential) is CredentialRoleIdentity
        and credential == head.signing_credential_identity
    )
    if len(matches) != 1:
        raise HistoryContractError("signer credential is not uniquely trusted")
    credential = matches[0]
    if (
        credential.semantic_role is not CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING
        or credential.provider_namespace != provider_identity.provider_namespace
        or credential.key_material_identity is None
    ):
        raise HistoryContractError("trusted signer identity/role/version mismatch")
    try:
        public_key = authority.public_key(credential.credential_identity)
        observed_identity = public_key_material_identity(public_key)
    except Exception as exc:
        raise HistorySigningAuthorityUnavailable("trusted signing public key unavailable") from exc
    if observed_identity != credential.key_material_identity:
        raise HistoryContractError("trusted public-key material identity mismatch")
    lifecycle_reader = getattr(authority, "lifecycle_state", None)
    generation_reader = getattr(authority, "lifecycle_generation", None)
    if not callable(lifecycle_reader) or not callable(generation_reader):
        raise HistorySigningAuthorityUnavailable("trusted signing lifecycle evidence unavailable")
    try:
        lifecycle = lifecycle_reader()
        generation = generation_reader()
    except Exception as exc:
        raise HistorySigningAuthorityUnavailable(
            "trusted signing lifecycle evidence unavailable"
        ) from exc
    if type(lifecycle) is not SigningKeyLifecycle or type(generation) is not int or generation < 1:
        raise HistoryContractError("invalid trusted signing lifecycle evidence")
    if lifecycle not in (
        SigningKeyLifecycle.ACTIVE,
        SigningKeyLifecycle.VERIFY_ONLY,
        SigningKeyLifecycle.REVOKED,
    ):
        raise HistoryContractError("unsupported trusted signing lifecycle state")
    return credential, public_key, lifecycle, generation


def verify_head(
    head: AttestedHistoryHead,
    authority: HistoryAttestationSigningProvider,
    *,
    expected_stream: HistoryStreamIdentity,
    historical_evidence_authority: LocalCheckpointProvider | None = None,
) -> None:
    if type(head) is not AttestedHistoryHead or type(expected_stream) is not HistoryStreamIdentity:
        raise TypeError("head and expected_stream must be exact contract objects")
    if head.stream != expected_stream:
        raise HistoryContractError("wrong stream/environment/trust/product/security epoch")
    _, public_key, lifecycle, generation = _trusted_history_credential(head, authority)
    material = {
        "stream": head.stream.material(),
        "sequence": head.sequence,
        "record_digest": head.record_digest,
        "signing_credential_identity": _credential_material(head.signing_credential_identity),
    }
    expected = ATTESTATION_DOMAIN + canonical_json_bytes(material)
    if head.canonical_attestation_bytes != expected:
        raise HistoryContractError("non-canonical head attestation")
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(head.signature, expected)
    except (ValueError, InvalidSignature) as exc:
        raise HistoryContractError("invalid history-head signature") from exc
    if lifecycle is SigningKeyLifecycle.REVOKED:
        if type(historical_evidence_authority) is not LocalCheckpointProvider:
            raise HistoricalRevokedSignatureVerificationUnavailable(
                "revoked historical signature lacks its trusted checkpoint authority"
            )
        historical_evidence_authority.verify_historical_acceptance(
            head,
            current_lifecycle_generation=generation,
        )


class ReferenceAuthenticatedHistory:
    """Executable CAS/idempotency oracle; it deliberately claims no durability."""

    def __init__(self, stream: HistoryStreamIdentity) -> None:
        self._stream = _snapshot_stream(stream)
        self._records: list[HistoryRecord] = []
        self._events: dict[HistoryEventIdentity, HistoryRecord] = {}
        self._lock = RLock()

    def current_record(self) -> HistoryRecord | None:
        with self._lock:
            return _snapshot_record(self._records[-1]) if self._records else None

    def append(
        self,
        *,
        expected_digest: str,
        event_identity: HistoryEventIdentity,
        payload: Mapping[str, object],
    ) -> tuple[HistoryRecord, bool]:
        with self._lock:
            self._verify_locked()
            admitted_event = _snapshot_event_identity(event_identity)
            prior = self._events.get(admitted_event)
            if prior is not None:
                candidate = build_record(
                    self._stream,
                    prior.sequence,
                    prior.predecessor_authenticated_digest,
                    admitted_event,
                    payload,
                )
                if candidate.event_digest != prior.event_digest:
                    raise HistoryContractError("conflicting idempotency replay")
                return _snapshot_record(prior), True
            head = self._records[-1] if self._records else None
            actual = NO_PREDECESSOR if head is None else head.authenticated_digest
            if expected_digest != actual:
                raise HistoryContractError("stale predecessor CAS; append race/fork rejected")
            record = build_record(
                self._stream,
                GENESIS_SEQUENCE if head is None else head.sequence + 1,
                actual,
                admitted_event,
                payload,
            )
            self._records.append(record)
            self._events[admitted_event] = record
            return _snapshot_record(record), False

    def verify(self) -> None:
        with self._lock:
            self._verify_locked()

    def _verify_locked(self) -> None:
        predecessor = NO_PREDECESSOR
        for expected_sequence, record in enumerate(self._records, GENESIS_SEQUENCE):
            if (
                record.stream != self._stream
                or record.sequence != expected_sequence
                or record.predecessor_authenticated_digest != predecessor
            ):
                raise HistoryContractError("gap, rewind, splice, fork, or invalid predecessor")
            HistoryRecord(**{name: getattr(record, name) for name in record.__dataclass_fields__})
            predecessor = record.authenticated_digest
        if len(self._events) != len(self._records) or any(
            self._events.get(record.event_identity) is not record for record in self._records
        ):
            raise HistoryContractError("history event index is inconsistent")

    def verify_attested_head(
        self,
        head: AttestedHistoryHead,
        authority: HistoryAttestationSigningProvider,
        historical_evidence_authority: LocalCheckpointProvider | None = None,
    ) -> VerifiedHistoryHead:
        """Verify the complete retained chain, exact head relation, and attestation."""
        with self._lock:
            snapshot = _snapshot_head(head)
            self._verify_locked()
            if not self._records:
                raise HistoryContractError("attested head has no retained history")
            record = self._records[-1]
            if (
                snapshot.stream != self._stream
                or snapshot.sequence != record.sequence
                or snapshot.record_digest != record.authenticated_digest
            ):
                raise HistoryContractError(
                    "attested head does not name the current verified history"
                )
            verify_head(
                snapshot,
                authority,
                expected_stream=self._stream,
                historical_evidence_authority=historical_evidence_authority,
            )
            return VerifiedHistoryHead(snapshot)

    def verify_attested_historical_head(
        self,
        head: AttestedHistoryHead,
        authority: HistoryAttestationSigningProvider,
        historical_evidence_authority: LocalCheckpointProvider,
    ) -> VerifiedHistoryHead:
        """Verify one exact retained record using checkpoint-owned acceptance."""
        if type(historical_evidence_authority) is not LocalCheckpointProvider:
            raise TypeError("historical evidence authority must be exact")
        with self._lock:
            snapshot = _snapshot_head(head)
            self._verify_locked()
            if snapshot.stream != self._stream or snapshot.sequence > len(self._records):
                raise HistoryContractError("historical head is outside retained history")
            record = self._records[snapshot.sequence - GENESIS_SEQUENCE]
            if snapshot.record_digest != record.authenticated_digest:
                raise HistoryContractError("historical head does not name its exact record")
            verify_head(
                snapshot,
                authority,
                expected_stream=self._stream,
                historical_evidence_authority=historical_evidence_authority,
            )
            return VerifiedHistoryHead(snapshot)

    def reconcile(
        self,
        *,
        head: AttestedHistoryHead | None,
        verification_authority: HistoryAttestationSigningProvider | None,
        checkpoint_authority: LocalCheckpointProvider,
    ) -> ReconciliationOutcome:
        """Verify authoritative history before classifying its checkpoint."""
        if type(checkpoint_authority) is not LocalCheckpointProvider:
            raise TypeError("checkpoint authority must be the exact reference provider")
        if not self._records:
            if head is not None:
                return ReconciliationOutcome.CORRUPT
            try:
                checkpoint = checkpoint_authority.current_checkpoint()
            except HistoryContractError:
                return ReconciliationOutcome.CORRUPT
            return (
                ReconciliationOutcome.NOT_FOUND
                if checkpoint is None
                else ReconciliationOutcome.CORRUPT
            )
        if head is None or verification_authority is None:
            return ReconciliationOutcome.CORRUPT
        try:
            verified = self.verify_attested_head(
                head,
                verification_authority,
                checkpoint_authority,
            )
        except HistoricalRevokedSignatureVerificationUnavailable:
            raise
        except HistorySigningAuthorityUnavailable:
            return ReconciliationOutcome.UNAVAILABLE
        except (HistoryContractError, TypeError):
            return ReconciliationOutcome.CORRUPT
        try:
            checkpoint = checkpoint_authority.current_checkpoint()
        except HistoryContractError:
            return ReconciliationOutcome.CORRUPT
        return _compare_verified_checkpoint(verified.head, checkpoint)


@dataclass(frozen=True, slots=True)
class LocalCheckpoint:
    checkpoint_id: str
    stream: HistoryStreamIdentity
    history_sequence: int
    authenticated_head_digest: str
    history_signing_credential_identity: CredentialRoleIdentity
    checkpoint_revision: int

    def __post_init__(self) -> None:
        if type(self.stream) is not HistoryStreamIdentity:
            raise TypeError("stream must be an exact HistoryStreamIdentity")
        _text(self.checkpoint_id, "checkpoint_id")
        _integer(self.history_sequence, "history_sequence", minimum=GENESIS_SEQUENCE)
        _integer(self.checkpoint_revision, "checkpoint_revision", minimum=1)
        _text(self.authenticated_head_digest, "authenticated_head_digest")
        _credential_material(self.history_signing_credential_identity)


@dataclass(frozen=True, slots=True)
class VerifiedHistoryHead:
    """Immutable reporting snapshot; never an authorization capability."""

    head: AttestedHistoryHead

    def __post_init__(self) -> None:
        if type(self.head) is not AttestedHistoryHead:
            raise TypeError("head must be an exact AttestedHistoryHead")


def _snapshot_head(head: AttestedHistoryHead) -> AttestedHistoryHead:
    if type(head) is not AttestedHistoryHead:
        raise TypeError("head must be an exact AttestedHistoryHead")
    return AttestedHistoryHead(
        _snapshot_stream(head.stream),
        head.sequence,
        head.record_digest,
        _snapshot_credential(head.signing_credential_identity),
        head.canonical_attestation_bytes,
        head.signature,
    )


def _snapshot_stream(stream: HistoryStreamIdentity) -> HistoryStreamIdentity:
    if type(stream) is not HistoryStreamIdentity:
        raise TypeError("stream must be an exact HistoryStreamIdentity")
    return HistoryStreamIdentity(**stream.material())


def _snapshot_event_identity(event: HistoryEventIdentity) -> HistoryEventIdentity:
    if type(event) is not HistoryEventIdentity:
        raise TypeError("event identity must be exact")
    return HistoryEventIdentity(**event.material())


def _snapshot_record(record: HistoryRecord) -> HistoryRecord:
    if type(record) is not HistoryRecord:
        raise TypeError("record must be exact")
    return HistoryRecord(
        _snapshot_stream(record.stream),
        record.sequence,
        record.predecessor_authenticated_digest,
        _snapshot_event_identity(record.event_identity),
        _plain(record.canonical_event_payload),
        record.event_digest,
        record.authenticated_digest,
    )


def _snapshot_checkpoint(checkpoint: LocalCheckpoint) -> LocalCheckpoint:
    if type(checkpoint) is not LocalCheckpoint:
        raise TypeError("checkpoint must be exact")
    return LocalCheckpoint(
        checkpoint.checkpoint_id,
        _snapshot_stream(checkpoint.stream),
        checkpoint.history_sequence,
        checkpoint.authenticated_head_digest,
        _snapshot_credential(checkpoint.history_signing_credential_identity),
        checkpoint.checkpoint_revision,
    )


@dataclass(frozen=True, slots=True)
class HistoricalHeadAcceptanceEvidence:
    """Detached report of an acceptance retained by a checkpoint authority."""

    checkpoint_authority_identity: str
    checkpoint_revision: int
    stream: HistoryStreamIdentity
    accepted_history_sequence: int
    accepted_authenticated_head_digest: str
    history_signing_credential_identity: CredentialRoleIdentity
    lifecycle_generation_at_acceptance: int
    lifecycle_state_at_acceptance: SigningKeyLifecycle
    predecessor_acceptance_digest: str
    acceptance_digest: str

    def __post_init__(self) -> None:
        if type(self.stream) is not HistoryStreamIdentity:
            raise TypeError("stream must be an exact HistoryStreamIdentity")
        for name in (
            "checkpoint_authority_identity",
            "accepted_authenticated_head_digest",
            "predecessor_acceptance_digest",
            "acceptance_digest",
        ):
            _text(getattr(self, name), name)
        _integer(self.checkpoint_revision, "checkpoint_revision", minimum=1)
        _integer(self.accepted_history_sequence, "accepted_history_sequence", minimum=1)
        _credential_material(self.history_signing_credential_identity)
        _integer(
            self.lifecycle_generation_at_acceptance,
            "lifecycle_generation_at_acceptance",
            minimum=1,
        )
        if type(self.lifecycle_state_at_acceptance) is not SigningKeyLifecycle:
            raise TypeError("lifecycle_state_at_acceptance must be exact")
        if self.acceptance_digest != _acceptance_digest(self):
            raise HistoryContractError("historical acceptance digest mismatch")


def _acceptance_material(evidence: HistoricalHeadAcceptanceEvidence) -> dict[str, object]:
    return {
        "checkpoint_authority_identity": evidence.checkpoint_authority_identity,
        "checkpoint_revision": evidence.checkpoint_revision,
        "stream": evidence.stream.material(),
        "accepted_history_sequence": evidence.accepted_history_sequence,
        "accepted_authenticated_head_digest": evidence.accepted_authenticated_head_digest,
        "history_signing_credential_identity": _credential_material(
            evidence.history_signing_credential_identity
        ),
        "lifecycle_generation_at_acceptance": evidence.lifecycle_generation_at_acceptance,
        "lifecycle_state_at_acceptance": evidence.lifecycle_state_at_acceptance.value,
        "predecessor_acceptance_digest": evidence.predecessor_acceptance_digest,
    }


def _acceptance_digest(evidence: HistoricalHeadAcceptanceEvidence) -> str:
    return _digest(ACCEPTANCE_DOMAIN, _acceptance_material(evidence))


def _snapshot_acceptance(
    evidence: HistoricalHeadAcceptanceEvidence,
) -> HistoricalHeadAcceptanceEvidence:
    if type(evidence) is not HistoricalHeadAcceptanceEvidence:
        raise TypeError("acceptance evidence must be exact")
    return HistoricalHeadAcceptanceEvidence(
        evidence.checkpoint_authority_identity,
        evidence.checkpoint_revision,
        _snapshot_stream(evidence.stream),
        evidence.accepted_history_sequence,
        evidence.accepted_authenticated_head_digest,
        _snapshot_credential(evidence.history_signing_credential_identity),
        evidence.lifecycle_generation_at_acceptance,
        evidence.lifecycle_state_at_acceptance,
        evidence.predecessor_acceptance_digest,
        evidence.acceptance_digest,
    )


class LocalCheckpointProvider:
    """Reference atomic CAS semantics, not independent rollback protection."""

    def __init__(self, checkpoint_id: str, stream: HistoryStreamIdentity) -> None:
        self._checkpoint_id = _text(checkpoint_id, "checkpoint_id")
        self._stream = _snapshot_stream(stream)
        self._authority_identity = f"local-checkpoint-authority:{uuid.uuid4().hex}"
        self._current = None
        self._acceptances: list[HistoricalHeadAcceptanceEvidence] = []
        self._lock = RLock()

    def current_checkpoint(self) -> LocalCheckpoint | None:
        with self._lock:
            self._verify_authority_state_locked()
            return _snapshot_checkpoint(self._current) if self._current is not None else None

    def historical_acceptance(
        self,
        head: AttestedHistoryHead,
    ) -> HistoricalHeadAcceptanceEvidence | None:
        """Return a detached exact-head report; the value itself grants no authority."""
        snapshot = _snapshot_head(head)
        with self._lock:
            self._verify_authority_state_locked()
            match = next(
                (item for item in self._acceptances if _acceptance_matches_head(item, snapshot)),
                None,
            )
            return _snapshot_acceptance(match) if match is not None else None

    def verify_historical_acceptance(
        self,
        head: AttestedHistoryHead,
        *,
        current_lifecycle_generation: int,
    ) -> None:
        """Authority lookup, not validation of a caller-supplied evidence value."""
        snapshot = _snapshot_head(head)
        _integer(current_lifecycle_generation, "current_lifecycle_generation", minimum=1)
        with self._lock:
            self._verify_authority_state_locked()
            match = next(
                (item for item in self._acceptances if _acceptance_matches_head(item, snapshot)),
                None,
            )
            if match is None:
                raise HistoricalRevokedSignatureVerificationUnavailable(
                    "no exact independently retained historical acceptance"
                )
            if (
                match.lifecycle_state_at_acceptance is not SigningKeyLifecycle.ACTIVE
                or match.lifecycle_generation_at_acceptance >= current_lifecycle_generation
            ):
                raise HistoryContractError("invalid lifecycle ordering in acceptance evidence")

    def _verify_authority_state_locked(self) -> None:
        """Validate the one checkpoint/evidence authority state definition."""
        predecessor = NO_PREDECESSOR
        previous_sequence = 0
        for revision, evidence in enumerate(self._acceptances, 1):
            if (
                type(evidence) is not HistoricalHeadAcceptanceEvidence
                or evidence.checkpoint_authority_identity != self._authority_identity
                or evidence.stream != self._stream
                or evidence.checkpoint_revision != revision
                or evidence.predecessor_acceptance_digest != predecessor
                or evidence.lifecycle_state_at_acceptance is not SigningKeyLifecycle.ACTIVE
                or type(evidence.lifecycle_generation_at_acceptance) is not int
                or evidence.lifecycle_generation_at_acceptance < 1
                or evidence.accepted_history_sequence <= previous_sequence
            ):
                raise HistoryContractError("retained historical acceptance chain corrupt")
            try:
                HistoricalHeadAcceptanceEvidence(
                    **{name: getattr(evidence, name) for name in evidence.__dataclass_fields__}
                )
            except (HistoryContractError, TypeError) as exc:
                raise HistoryContractError("retained historical acceptance chain corrupt") from exc
            predecessor = evidence.acceptance_digest
            previous_sequence = evidence.accepted_history_sequence

        if (self._current is None) != (not self._acceptances):
            raise HistoryContractError("checkpoint/evidence authority cardinality corrupt")
        if self._current is None:
            return
        if type(self._current) is not LocalCheckpoint:
            raise HistoryContractError("retained checkpoint type corrupt")
        try:
            LocalCheckpoint(
                **{
                    name: getattr(self._current, name)
                    for name in self._current.__dataclass_fields__
                }
            )
        except (HistoryContractError, TypeError) as exc:
            raise HistoryContractError("retained checkpoint corrupt") from exc
        if (
            self._current.checkpoint_id != self._checkpoint_id
            or self._current.stream != self._stream
            or len(self._acceptances) != self._current.checkpoint_revision
        ):
            raise HistoryContractError("checkpoint/evidence authority cardinality corrupt")
        last = self._acceptances[-1]
        if (
            last.checkpoint_revision,
            last.stream,
            last.accepted_history_sequence,
            last.accepted_authenticated_head_digest,
            last.history_signing_credential_identity,
        ) != (
            self._current.checkpoint_revision,
            self._current.stream,
            self._current.history_sequence,
            self._current.authenticated_head_digest,
            self._current.history_signing_credential_identity,
        ):
            raise HistoryContractError("current checkpoint/last acceptance binding corrupt")

    def advance(
        self,
        *,
        expected_revision: int,
        history: ReferenceAuthenticatedHistory,
        head: AttestedHistoryHead,
        verification_authority: HistoryAttestationSigningProvider,
    ) -> tuple[LocalCheckpoint, bool]:
        if type(history) is not ReferenceAuthenticatedHistory:
            raise TypeError("history must be the exact reference authority")
        verified = history.verify_attested_head(head, verification_authority)
        snapshot = verified.head
        try:
            lifecycle_generation = verification_authority.lifecycle_generation()
            lifecycle = verification_authority.lifecycle_state()
            confirmed_generation = verification_authority.lifecycle_generation()
        except Exception as exc:
            raise HistorySigningAuthorityUnavailable(
                "signing lifecycle unavailable at checkpoint acceptance"
            ) from exc
        if (
            lifecycle is not SigningKeyLifecycle.ACTIVE
            or type(lifecycle_generation) is not int
            or lifecycle_generation < 1
            or confirmed_generation != lifecycle_generation
        ):
            raise HistoryContractError(
                "checkpoint acceptance requires an ACTIVE history credential"
            )
        if snapshot.stream != self._stream:
            raise HistoryContractError("checkpoint cross-stream splice")
        with self._lock:
            self._verify_authority_state_locked()
            current = self._current
            revision = 0 if current is None else current.checkpoint_revision
            if expected_revision != revision:
                raise HistoryContractError("checkpoint CAS conflict")
            if current is not None:
                if snapshot.sequence < current.history_sequence:
                    raise HistoryContractError("checkpoint rewind / valid-prefix rollback")
                if snapshot.sequence == current.history_sequence:
                    exact = (
                        snapshot.record_digest,
                        snapshot.signing_credential_identity,
                    ) == (
                        current.authenticated_head_digest,
                        current.history_signing_credential_identity,
                    )
                    if not exact:
                        raise HistoryContractError("same-sequence split brain")
                    return _snapshot_checkpoint(current), True
            successor = LocalCheckpoint(
                self._checkpoint_id,
                self._stream,
                snapshot.sequence,
                snapshot.record_digest,
                _snapshot_credential(snapshot.signing_credential_identity),
                revision + 1,
            )
            predecessor = (
                NO_PREDECESSOR if not self._acceptances else self._acceptances[-1].acceptance_digest
            )
            acceptance_material = {
                "checkpoint_authority_identity": self._authority_identity,
                "checkpoint_revision": revision + 1,
                "stream": self._stream.material(),
                "accepted_history_sequence": snapshot.sequence,
                "accepted_authenticated_head_digest": snapshot.record_digest,
                "history_signing_credential_identity": _credential_material(
                    snapshot.signing_credential_identity
                ),
                "lifecycle_generation_at_acceptance": lifecycle_generation,
                "lifecycle_state_at_acceptance": lifecycle.value,
                "predecessor_acceptance_digest": predecessor,
            }
            acceptance = HistoricalHeadAcceptanceEvidence(
                self._authority_identity,
                revision + 1,
                self._stream,
                snapshot.sequence,
                snapshot.record_digest,
                _snapshot_credential(snapshot.signing_credential_identity),
                lifecycle_generation,
                lifecycle,
                predecessor,
                _digest(ACCEPTANCE_DOMAIN, acceptance_material),
            )
            self._current = successor
            self._acceptances.append(acceptance)
            return _snapshot_checkpoint(successor), False


def _acceptance_matches_head(
    evidence: HistoricalHeadAcceptanceEvidence,
    head: AttestedHistoryHead,
) -> bool:
    return (
        evidence.stream == head.stream
        and evidence.accepted_history_sequence == head.sequence
        and evidence.accepted_authenticated_head_digest == head.record_digest
        and evidence.history_signing_credential_identity == head.signing_credential_identity
    )


def _compare_verified_checkpoint(
    history_head: AttestedHistoryHead,
    checkpoint: LocalCheckpoint | None,
) -> ReconciliationOutcome:
    """Compare only after the owning history authority verified its head."""
    if checkpoint is not None and checkpoint.stream != history_head.stream:
        return ReconciliationOutcome.CONFLICT
    if checkpoint is None or checkpoint.history_sequence < history_head.sequence:
        return ReconciliationOutcome.STALE
    if checkpoint.history_sequence > history_head.sequence:
        return ReconciliationOutcome.CORRUPT
    if checkpoint.authenticated_head_digest != history_head.record_digest:
        return ReconciliationOutcome.CONFLICT
    if checkpoint.history_signing_credential_identity != history_head.signing_credential_identity:
        return ReconciliationOutcome.CONFLICT
    return ReconciliationOutcome.EXACT_COMMITTED


def reconcile_checkpoint(
    *,
    history: ReferenceAuthenticatedHistory,
    history_head: AttestedHistoryHead | None,
    verification_authority: HistoryAttestationSigningProvider | None,
    checkpoint_authority: LocalCheckpointProvider,
) -> ReconciliationOutcome:
    """Authority operation; a raw head is never sufficient evidence."""
    if type(history) is not ReferenceAuthenticatedHistory:
        raise TypeError("history must be the exact reference authority")
    return history.reconcile(
        head=history_head,
        verification_authority=verification_authority,
        checkpoint_authority=checkpoint_authority,
    )


AUTHENTICATED_ISSUER_HISTORY_LOCAL_CHECKPOINT_EXECUTABLE_SEMANTIC_CONTRACT_FROZEN = True
ROOT_PROOF_ISSUER_IMPLEMENTED = False
PRODUCTION_LOCAL_RUNTIME_AVAILABLE = False
