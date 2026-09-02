"""M0.11 secret-reference handoff with reconciliation-before-retry semantics."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import Any, Protocol

from .lifecycle_records import (
    LifecycleIntegrityError,
    fingerprint_without,
    persistence_record,
    validate_chain,
)
from .records import PersistenceRecord
from .secret_handoff_contract import (
    SecretHandoffContractError,
    secret_metadata_fingerprint_value,
    secret_operation_fingerprint_value,
    validate_raw_secret_handoff_record,
)

HANDOFF_TRANSITIONS = {
    "PREPARED": frozenset({"COMMITTED", "UNKNOWN_RECONCILIATION"}),
    "COMMITTED": frozenset({"CLEANUP_PENDING"}),
    "CLEANUP_PENDING": frozenset(),
    "UNKNOWN_RECONCILIATION": frozenset(),
}
_SHA = __import__("re").compile(r"^[0-9a-f]{64}$")


class SecretHandoffError(LifecycleIntegrityError):
    pass


@dataclass(frozen=True, slots=True)
class SecretHandoffRecord:
    handoff_id: str
    scope: tuple[str, str]
    operation: str
    old_reference: str | None
    new_reference: str | None
    metadata_fingerprint_sha256: str
    operation_fingerprint_sha256: str
    reconciliation_metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.handoff_id, str)
            or not self.handoff_id
            or not isinstance(self.operation, str)
            or not self.operation
        ):
            raise SecretHandoffError("handoff identity and operation must be non-empty")
        if (
            not isinstance(self.scope, tuple)
            or len(self.scope) != 2
            or not all(isinstance(x, str) and x for x in self.scope)
        ):
            raise SecretHandoffError("scope must contain account and device identifiers")
        if not all(
            x is None or isinstance(x, str) for x in (self.old_reference, self.new_reference)
        ):
            raise SecretHandoffError("references must be opaque strings or null")
        if not isinstance(self.reconciliation_metadata, Mapping):
            raise SecretHandoffError("reconciliation_metadata must be an object")
        frozen = _freeze_json(self.reconciliation_metadata)
        object.__setattr__(self, "reconciliation_metadata", frozen)
        for name in ("metadata_fingerprint_sha256", "operation_fingerprint_sha256"):
            if not isinstance(getattr(self, name), str) or not _SHA.fullmatch(getattr(self, name)):
                raise SecretHandoffError(f"{name} must be lowercase SHA-256")
        if self.metadata_fingerprint_sha256 != secret_metadata_fingerprint(frozen):
            raise SecretHandoffError("metadata fingerprint does not bind reconciliation metadata")
        if self.operation_fingerprint_sha256 != secret_operation_fingerprint(
            scope=self.scope,
            operation=self.operation,
            old_reference=self.old_reference,
            new_reference=self.new_reference,
            metadata_fingerprint_sha256=self.metadata_fingerprint_sha256,
        ):
            raise SecretHandoffError("operation fingerprint does not bind exact operation")
        try:
            validate_raw_secret_handoff_record(self.to_mapping())
        except SecretHandoffContractError as exc:
            raise SecretHandoffError(str(exc)) from exc

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SecretHandoffRecord":
        if set(value) != {f.name for f in fields(cls)}:
            raise SecretHandoffError("SecretHandoffRecord requires its exact field set")
        data = dict(value)
        if not isinstance(data["scope"], list):
            raise SecretHandoffError("JSON scope must be an array")
        data["scope"] = tuple(data["scope"])
        return cls(**data)

    def to_mapping(self) -> dict[str, Any]:
        return {
            "handoff_id": self.handoff_id,
            "scope": list(self.scope),
            "operation": self.operation,
            "old_reference": self.old_reference,
            "new_reference": self.new_reference,
            "metadata_fingerprint_sha256": self.metadata_fingerprint_sha256,
            "operation_fingerprint_sha256": self.operation_fingerprint_sha256,
            "reconciliation_metadata": _thaw_json(self.reconciliation_metadata),
        }


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        secret_metadata_fingerprint_value({"value": value})
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise SecretHandoffError("reconciliation metadata keys must be strings")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    raise SecretHandoffError("reconciliation metadata must be an exact JSON value")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def secret_metadata_fingerprint(metadata: Mapping[str, Any]) -> str:
    return secret_metadata_fingerprint_value(_thaw_json(metadata))


def secret_operation_fingerprint(
    *,
    scope: tuple[str, str],
    operation: str,
    old_reference: str | None,
    new_reference: str | None,
    metadata_fingerprint_sha256: str,
) -> str:
    return secret_operation_fingerprint_value(
        scope=scope,
        operation=operation,
        old_reference=old_reference,
        new_reference=new_reference,
        metadata_fingerprint_sha256=metadata_fingerprint_sha256,
    )


def handoff_transition(**fields_: Any) -> dict[str, Any]:
    value = dict(fields_)
    value["transition_fingerprint_sha256"] = fingerprint_without(
        value, "transition_fingerprint_sha256"
    )
    return value


def handoff_current(**fields_: Any) -> dict[str, Any]:
    value = dict(fields_)
    value["designation_fingerprint_sha256"] = fingerprint_without(
        value, "designation_fingerprint_sha256"
    )
    return value


def handoff_transition_carrier(value: Mapping[str, Any]):
    return persistence_record(
        "SecretHandoff transition/history revisions",
        f"handoff-transition:{value['handoff_id']}:{value['transition_revision']}",
        value,
    )


def handoff_current_carrier(value: Mapping[str, Any]):
    return persistence_record(
        "SecretHandoff current state/designation",
        f"handoff-current:{value['handoff_id']}",
        value,
    )


def handoff_descriptor_carrier(record: SecretHandoffRecord) -> PersistenceRecord:
    """Create the immutable descriptor through the ordinary Stage-1 path."""

    return persistence_record(
        "SecretHandoff immutable descriptor",
        f"handoff-descriptor:{record.handoff_id}",
        record.to_mapping(),
    )


def validate_handoff_descriptor_identity(
    existing: PersistenceRecord, candidate: PersistenceRecord
) -> None:
    """Fail closed when one durable handoff identity is assigned different facts."""

    if existing.record_key != candidate.record_key or existing != candidate:
        raise SecretHandoffError("immutable handoff descriptor conflict")


def validate_secret_handoff_lifecycle(
    record: SecretHandoffRecord,
    history: Sequence[Mapping[str, Any]],
    current: Mapping[str, Any] | None,
) -> None:
    try:
        validate_chain(
            history,
            current,
            identity_field="handoff_id",
            allowed=HANDOFF_TRANSITIONS,
            transition_hash_field="transition_fingerprint_sha256",
            current_hash_field="designation_fingerprint_sha256",
        )
        for item in history:
            if (
                item["operation_fingerprint_sha256"] != record.operation_fingerprint_sha256
                or item["metadata_fingerprint_sha256"] != record.metadata_fingerprint_sha256
            ):
                raise SecretHandoffError("transition is not bound to descriptor")
        if (
            current is not None
            and current["operation_fingerprint_sha256"] != record.operation_fingerprint_sha256
        ):
            raise SecretHandoffError("current designation is not bound to descriptor")
    except LifecycleIntegrityError as exc:
        raise SecretHandoffError(str(exc)) from exc


class ExternalOutcome(Enum):
    COMMITTED = "COMMITTED"
    UNRESOLVED = "UNRESOLVED"
    NOT_STARTED = "NOT_STARTED"


class SecretExternalResourcePort(Protocol):
    def begin(self, descriptor: SecretHandoffRecord) -> ExternalOutcome: ...
    def reconcile(self, descriptor: SecretHandoffRecord) -> ExternalOutcome: ...
    def cleanup(self, descriptor: SecretHandoffRecord) -> None: ...


class SecretHandoffCoordinator:
    """Process serialization plus a mandatory reconcile fence before any restart."""

    def __init__(self, port: SecretExternalResourcePort) -> None:
        self._port, self._lock, self._attempted = port, RLock(), set()

    def resume(
        self,
        record: SecretHandoffRecord,
        history: Sequence[Mapping[str, Any]],
        current: Mapping[str, Any] | None,
        *,
        first_dispatch: bool = False,
    ) -> str:
        with self._lock:
            validate_secret_handoff_lifecycle(record, history, current)
            if current is None:
                raise SecretHandoffError("descriptor alone has no lifecycle authority")
            state = str(current["state"])
            if state in {"CLEANUP_PENDING", "UNKNOWN_RECONCILIATION"}:
                return state
            if state == "COMMITTED":
                self._port.cleanup(record)
                return "CLEANUP_PENDING"
            # PREPARED after restart is never evidence that mutation was not sent.
            if first_dispatch and record.handoff_id not in self._attempted:
                self._attempted.add(record.handoff_id)
                outcome = self._port.begin(record)
            else:
                outcome = self._port.reconcile(record)
            if outcome is ExternalOutcome.COMMITTED:
                return "COMMITTED"
            if outcome is ExternalOutcome.UNRESOLVED:
                return "UNKNOWN_RECONCILIATION"
            if first_dispatch:
                raise SecretHandoffError("external operation did not start")
            raise SecretHandoffError("restart cannot blind retry a PREPARED handoff")
