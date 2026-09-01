"""Small shared integrity helpers for the two frozen M0.11 lifecycles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .fingerprints import canonical_json_sha256
from .record_registry import PERSISTENCE_RECORD_REGISTRY
from .records import PersistenceRecord, validate_persistence_record


class LifecycleIntegrityError(RuntimeError):
    """A durable lifecycle is malformed or internally inconsistent."""


def fingerprint_without(value: Mapping[str, Any], excluded: str) -> str:
    return canonical_json_sha256({key: item for key, item in value.items() if key != excluded})


def validate_chain(
    history: Sequence[Mapping[str, Any]],
    current: Mapping[str, Any] | None,
    *,
    identity_field: str,
    allowed: Mapping[str, frozenset[str]],
    transition_hash_field: str,
    current_hash_field: str,
) -> None:
    """Validate frozen revision, transition, fingerprint and designation semantics."""

    if current is None:
        if history:
            raise LifecycleIntegrityError("dangling transition history")
        return
    if not history:
        raise LifecycleIntegrityError("current designation has no transition history")
    expected_identity = current[identity_field]
    for index, transition in enumerate(history, 1):
        revision = transition.get("transition_revision")
        if isinstance(revision, bool) or revision != index:
            raise LifecycleIntegrityError("transition revisions must be positive and contiguous")
        if transition.get(identity_field) != expected_identity:
            raise LifecycleIntegrityError("transition identity mismatch")
        expected_previous = None if index == 1 else history[index - 2]["state"]
        if transition.get("previous_state") != expected_previous:
            raise LifecycleIntegrityError("previous_state does not match durable history")
        state = transition.get("state")
        if index == 1:
            if state != "PREPARED":
                raise LifecycleIntegrityError("revision 1 must designate PREPARED")
        elif state not in allowed[str(expected_previous)]:
            raise LifecycleIntegrityError("illegal lifecycle transition")
        if transition.get(transition_hash_field) != fingerprint_without(
            transition, transition_hash_field
        ):
            raise LifecycleIntegrityError("transition fingerprint mismatch")
    tail = history[-1]
    if current.get("current_transition_revision") != len(history):
        raise LifecycleIntegrityError("current revision does not designate history tail")
    if current.get("state") != tail.get("state"):
        raise LifecycleIntegrityError("current state does not designate history tail")
    if current.get(current_hash_field) != fingerprint_without(current, current_hash_field):
        raise LifecycleIntegrityError("current designation fingerprint mismatch")


def append_idempotently(
    history: Sequence[Mapping[str, Any]], candidate: Mapping[str, Any]
) -> tuple[Mapping[str, Any], ...]:
    revision = candidate.get("transition_revision")
    if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
        raise LifecycleIntegrityError("transition revision must be a positive integer")
    if revision <= len(history):
        if history[revision - 1] == candidate:
            return tuple(history)
        raise LifecycleIntegrityError("conflicting transition revision")
    if revision != len(history) + 1:
        raise LifecycleIntegrityError("transition revision gap")
    return (*history, candidate)


def persistence_record(
    representation_name: str, record_key: str, payload: Mapping[str, Any]
) -> PersistenceRecord:
    """Build and Stage-1 validate a lifecycle carrier using the frozen registry."""

    entry = PERSISTENCE_RECORD_REGISTRY[representation_name]
    record = PersistenceRecord(
        representation_name=representation_name,
        representation_category=str(entry["representation_category"]),
        semantic_owner_milestone=str(entry["semantic_owner_milestone"]),
        semantic_artifact=str(entry["semantic_artifact"]),
        semantic_json_pointer=str(entry["semantic_json_pointer"]),
        semantic_contract_fingerprint_sha256=str(entry["semantic_contract_fingerprint_sha256"]),
        record_key=record_key,
        payload=dict(payload),
        payload_fingerprint_sha256=canonical_json_sha256(payload),
    )
    validate_persistence_record(record)
    return record
