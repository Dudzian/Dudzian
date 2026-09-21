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
from types import MappingProxyType
from typing import Any, Mapping, Protocol

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from bot_core.root_proof_issuer_substrate import (
    CredentialSemanticRole,
    HistoryAttestationSigningProvider,
)


RECORD_DOMAIN = b"CryptoHunter/M0.5/IssuerAuthenticatedHistoryRecord/v1\0"
ATTESTATION_DOMAIN = b"CryptoHunter/M0.5/IssuerAuthenticatedHistoryHead/v1\0"
NO_PREDECESSOR = "NO_PREDECESSOR"
GENESIS_SEQUENCE = 1


class HistoryContractError(RuntimeError):
    """Fail-closed history or checkpoint contract violation."""


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
    frozen = _freeze_json(dict(value))
    return json.dumps(
        _plain(frozen), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _digest(domain: bytes, value: Mapping[str, object]) -> str:
    return "sha256:" + hashlib.sha256(domain + canonical_json_bytes(value)).hexdigest()


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
        for name in ("stream_id", "issuer_authority_identity", "security_profile", "environment", "trust_domain", "product_scope"):
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
        if type(self.stream) is not HistoryStreamIdentity or type(self.event_identity) is not HistoryEventIdentity:
            raise TypeError("history identities must be exact contract objects")
        _integer(self.sequence, "sequence", minimum=GENESIS_SEQUENCE)
        _text(self.predecessor_authenticated_digest, "predecessor_authenticated_digest")
        payload = _freeze_json(dict(self.canonical_event_payload))
        object.__setattr__(self, "canonical_event_payload", payload)
        if self.event_digest != _digest(b"CryptoHunter/M0.5/IssuerHistoryEvent/v1\0", _plain(payload)):
            raise HistoryContractError("event digest mismatch")
        if self.authenticated_digest != _digest(RECORD_DOMAIN, self.unsigned_material()):
            raise HistoryContractError("authenticated record digest mismatch")

    def unsigned_material(self) -> dict[str, object]:
        return {"stream": self.stream.material(), "sequence": self.sequence,
                "predecessor_authenticated_digest": self.predecessor_authenticated_digest,
                "event_identity": self.event_identity.material(),
                "canonical_event_payload": _plain(self.canonical_event_payload),
                "event_digest": self.event_digest}


def build_record(stream: HistoryStreamIdentity, sequence: int, predecessor: str,
                 event_identity: HistoryEventIdentity, payload: Mapping[str, object]) -> HistoryRecord:
    frozen = _freeze_json(dict(payload))
    event_digest = _digest(b"CryptoHunter/M0.5/IssuerHistoryEvent/v1\0", _plain(frozen))
    material = {"stream": stream.material(), "sequence": sequence,
                "predecessor_authenticated_digest": predecessor,
                "event_identity": event_identity.material(), "canonical_event_payload": _plain(frozen),
                "event_digest": event_digest}
    return HistoryRecord(stream, sequence, predecessor, event_identity, frozen, event_digest,
                         _digest(RECORD_DOMAIN, material))


@dataclass(frozen=True, slots=True)
class AttestedHistoryHead:
    stream: HistoryStreamIdentity
    sequence: int
    record_digest: str
    signing_credential_id: str
    signing_key_version: str
    canonical_attestation_bytes: bytes
    signature: bytes

    def __post_init__(self) -> None:
        _integer(self.sequence, "sequence", minimum=GENESIS_SEQUENCE)
        for name in ("record_digest", "signing_credential_id", "signing_key_version"):
            _text(getattr(self, name), name)
        if type(self.canonical_attestation_bytes) is not bytes or type(self.signature) is not bytes:
            raise TypeError("attestation and signature must be exact bytes")


def attest_head(record: HistoryRecord, signer: HistoryAttestationSigningProvider) -> AttestedHistoryHead:
    identity = signer.active_credential_identity()
    if identity.semantic_role is not CredentialSemanticRole.HISTORY_ATTESTATION_SIGNING:
        raise HistoryContractError("history head requires HISTORY_ATTESTATION_SIGNING role")
    version = _text(identity.key_handle_or_version, "key version")
    material = {"stream": record.stream.material(), "sequence": record.sequence,
                "record_digest": record.authenticated_digest,
                "signing_role": "HISTORY_ATTESTATION_SIGNING",
                "signing_credential_id": identity.credential_identity,
                "signing_key_version": version}
    canonical = ATTESTATION_DOMAIN + canonical_json_bytes(material)
    return AttestedHistoryHead(record.stream, record.sequence, record.authenticated_digest,
                               identity.credential_identity, version, canonical,
                               signer.sign_history_head(canonical))


def verify_head(head: AttestedHistoryHead, public_key: bytes, *, expected_stream: HistoryStreamIdentity,
                expected_credential_id: str, expected_key_version: str) -> None:
    if head.stream != expected_stream:
        raise HistoryContractError("wrong stream/environment/trust/product/security epoch")
    if (head.signing_credential_id, head.signing_key_version) != (expected_credential_id, expected_key_version):
        raise HistoryContractError("signer credential/version mismatch")
    material = {"stream": head.stream.material(), "sequence": head.sequence,
                "record_digest": head.record_digest, "signing_role": "HISTORY_ATTESTATION_SIGNING",
                "signing_credential_id": head.signing_credential_id,
                "signing_key_version": head.signing_key_version}
    expected = ATTESTATION_DOMAIN + canonical_json_bytes(material)
    if head.canonical_attestation_bytes != expected:
        raise HistoryContractError("non-canonical head attestation")
    try:
        Ed25519PublicKey.from_public_bytes(public_key).verify(head.signature, expected)
    except (ValueError, InvalidSignature) as exc:
        raise HistoryContractError("invalid history-head signature") from exc


class ReferenceAuthenticatedHistory:
    """Executable CAS/idempotency oracle; it deliberately claims no durability."""
    def __init__(self, stream: HistoryStreamIdentity) -> None:
        self._stream = stream
        self._records: list[HistoryRecord] = []
        self._events: dict[HistoryEventIdentity, HistoryRecord] = {}

    def current_record(self) -> HistoryRecord | None:
        return self._records[-1] if self._records else None

    def append(self, *, expected_digest: str, event_identity: HistoryEventIdentity,
               payload: Mapping[str, object]) -> tuple[HistoryRecord, bool]:
        prior = self._events.get(event_identity)
        if prior is not None:
            candidate = build_record(self._stream, prior.sequence,
                                     prior.predecessor_authenticated_digest, event_identity, payload)
            if candidate.event_digest != prior.event_digest:
                raise HistoryContractError("conflicting idempotency replay")
            return prior, True
        head = self.current_record()
        actual = NO_PREDECESSOR if head is None else head.authenticated_digest
        if expected_digest != actual:
            raise HistoryContractError("stale predecessor CAS; append race/fork rejected")
        record = build_record(self._stream, GENESIS_SEQUENCE if head is None else head.sequence + 1,
                              actual, event_identity, payload)
        self._records.append(record)
        self._events[event_identity] = record
        return record, False

    def verify(self) -> None:
        predecessor = NO_PREDECESSOR
        for expected_sequence, record in enumerate(self._records, GENESIS_SEQUENCE):
            if record.stream != self._stream or record.sequence != expected_sequence or record.predecessor_authenticated_digest != predecessor:
                raise HistoryContractError("gap, rewind, splice, fork, or invalid predecessor")
            HistoryRecord(**{name: getattr(record, name) for name in record.__dataclass_fields__})
            predecessor = record.authenticated_digest


@dataclass(frozen=True, slots=True)
class LocalCheckpoint:
    checkpoint_id: str
    stream: HistoryStreamIdentity
    history_sequence: int
    authenticated_head_digest: str
    history_signing_credential_id: str
    history_signing_key_version: str
    checkpoint_revision: int

    def __post_init__(self) -> None:
        _text(self.checkpoint_id, "checkpoint_id")
        _integer(self.history_sequence, "history_sequence", minimum=GENESIS_SEQUENCE)
        _integer(self.checkpoint_revision, "checkpoint_revision", minimum=1)
        for name in ("authenticated_head_digest", "history_signing_credential_id", "history_signing_key_version"):
            _text(getattr(self, name), name)


class LocalCheckpointProvider:
    """Reference atomic CAS semantics, not independent rollback protection."""
    def __init__(self, checkpoint_id: str, stream: HistoryStreamIdentity) -> None:
        self._checkpoint_id, self._stream, self._current = _text(checkpoint_id, "checkpoint_id"), stream, None

    def current_checkpoint(self) -> LocalCheckpoint | None:
        return self._current

    def advance(self, *, expected_revision: int, head: AttestedHistoryHead) -> tuple[LocalCheckpoint, bool]:
        if head.stream != self._stream:
            raise HistoryContractError("checkpoint cross-stream splice")
        current = self._current
        revision = 0 if current is None else current.checkpoint_revision
        if expected_revision != revision:
            raise HistoryContractError("checkpoint CAS conflict")
        if current is not None:
            if head.sequence < current.history_sequence:
                raise HistoryContractError("checkpoint rewind / valid-prefix rollback")
            if head.sequence == current.history_sequence:
                exact = (head.record_digest, head.signing_credential_id, head.signing_key_version) == (current.authenticated_head_digest, current.history_signing_credential_id, current.history_signing_key_version)
                if not exact:
                    raise HistoryContractError("same-sequence split brain")
                return current, True
        successor = LocalCheckpoint(self._checkpoint_id, self._stream, head.sequence, head.record_digest,
                                    head.signing_credential_id, head.signing_key_version, revision + 1)
        self._current = successor
        return successor, False


def reconcile_checkpoint(history_head: AttestedHistoryHead | None,
                         checkpoint: LocalCheckpoint | None) -> ReconciliationOutcome:
    if history_head is None and checkpoint is None:
        return ReconciliationOutcome.NOT_FOUND
    if history_head is None:
        return ReconciliationOutcome.CORRUPT
    if checkpoint is None or checkpoint.history_sequence < history_head.sequence:
        return ReconciliationOutcome.STALE
    if checkpoint.history_sequence > history_head.sequence:
        return ReconciliationOutcome.CORRUPT
    if checkpoint.authenticated_head_digest != history_head.record_digest:
        return ReconciliationOutcome.CONFLICT
    return ReconciliationOutcome.EXACT_COMMITTED


AUTHENTICATED_ISSUER_HISTORY_LOCAL_CHECKPOINT_EXECUTABLE_SEMANTIC_CONTRACT_FROZEN = True
ROOT_PROOF_ISSUER_IMPLEMENTED = False
PRODUCTION_LOCAL_RUNTIME_AVAILABLE = False
