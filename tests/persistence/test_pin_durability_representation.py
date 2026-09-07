"""M0.11 representation closure for same-revision PIN authentication states."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bot_core.persistence.backup_envelope import create_backup_envelope, validate_backup_envelope
from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.persistence.lifecycle_records import persistence_record
from bot_core.persistence.records import PersistenceRecordError, validate_persistence_record
from bot_core.persistence.state_store import SQLiteStateStore
from tests.persistence.test_state_store_records import _metadata


ACCOUNT = "acct_018f0000-0000-7000-8000-000000000001"
DEVICE = "dev_018f0000-0000-7000-8000-000000000002"
OPERATOR = "op_018f0000-0000-7000-8000-000000000003"


def _pin(failed_attempts: int, lockout_until_utc: str | None = None) -> dict[str, object]:
    values: dict[str, object] = {
        "account_id": ACCOUNT,
        "operator_id": OPERATOR,
        "device_installation_id": DEVICE,
        "algorithm_id": "M010-DETERMINISTIC-REFERENCE-NOT-PRODUCTION-KDF",
        "parameter_policy_version": 1,
        "salt_reference": "secure-store://opaque/pin-salt",
        "verifier": "c" * 64,
        "pin_revision": 1,
        "failed_attempts": failed_attempts,
        "lockout_until_utc": lockout_until_utc,
        "security_generation": 1,
    }
    return {**values, "content_fingerprint_sha256": canonical_json_sha256(values)}


def _history(pin: dict[str, object]):
    payload = {
        "fact_kind": "PinVerifierRecord accepted revisions",
        "upstream_payload": pin,
        "upstream_payload_fingerprint_sha256": canonical_json_sha256(pin),
    }
    key = (
        "immutable:PinVerifierRecord accepted revisions:"
        f"{OPERATOR}:1:1:{pin['content_fingerprint_sha256']}"
    )
    return persistence_record("PinVerifierRecord accepted revisions", key, payload)


def _current(pin: dict[str, object]):
    scope = f"{ACCOUNT}:{OPERATOR}:{DEVICE}"
    payload: dict[str, object] = {
        "scope_key": scope,
        "current_reference": pin["content_fingerprint_sha256"],
        "current_revision": pin["pin_revision"],
        "current_generation": pin["security_generation"],
    }
    payload["content_fingerprint_sha256"] = canonical_json_sha256(payload)
    return persistence_record("PinVerifierRecord current designation", f"current:{scope}", payload)


def _commit(
    store: SQLiteStateStore,
    generation: int,
    pin: dict[str, object],
    *,
    append_history: bool = True,
) -> None:
    expected = None if generation == 1 else generation - 1
    history = (_history(pin),) if append_history else ()
    target = _metadata(
        generation,
        account_id=ACCOUNT,
        device_installation_id=DEVICE,
        state_store_schema_version=2,
    )
    prepared = store.derive_prepared_metadata(
        target,
        current_records=(_current(pin),),
        immutable_history=history,
        expected_current_generation=expected,
    )
    store.commit_prepared_state(
        prepared,
        current_records=(_current(pin),),
        immutable_history=history,
        expected_current_generation=expected,
    )


def test_distinct_same_revision_pin_states_have_distinct_immutable_keys() -> None:
    initial = _history(_pin(0))
    failed_once = _history(_pin(1))

    assert initial.record_key != failed_once.record_key


def test_same_revision_lockout_lifecycle_preserves_history_and_one_current(
    tmp_path: Path,
) -> None:
    states = (
        _pin(0),
        _pin(1),
        _pin(2),
        _pin(3, "2026-08-10T10:05:00Z"),
        _pin(0),
    )
    credential_fields = (
        "account_id",
        "operator_id",
        "device_installation_id",
        "algorithm_id",
        "parameter_policy_version",
        "salt_reference",
        "verifier",
        "pin_revision",
        "security_generation",
    )
    assert all(
        tuple(state[field] for field in credential_fields)
        == tuple(states[0][field] for field in credential_fields)
        for state in states
    )
    assert len({state["content_fingerprint_sha256"] for state in states}) == 4

    with SQLiteStateStore(tmp_path / "pin.sqlite3") as store:
        for generation, state in enumerate(states, 1):
            # P4 is mathematically identical to P0: designate the existing fact
            # instead of inventing an event ID or duplicating immutable content.
            _commit(store, generation, state, append_history=generation != 5)

        snapshot = store.read_verified_snapshot()
        assert snapshot is not None
        history = tuple(
            item
            for item in snapshot.immutable_history
            if item.representation_name == "PinVerifierRecord accepted revisions"
        )
        current = tuple(
            item
            for item in snapshot.current_records
            if item.representation_name == "PinVerifierRecord current designation"
            and item.payload["scope_key"] == f"{ACCOUNT}:{OPERATOR}:{DEVICE}"
        )
        assert len(history) == len({item.record_key for item in history}) == 4
        assert {item.payload["upstream_payload_fingerprint_sha256"] for item in history} == {
            canonical_json_sha256(state) for state in states
        }
        assert len(current) == 1
        assert current[0].payload["current_reference"] == states[-1]["content_fingerprint_sha256"]
        assert current[0].payload["current_revision"] == 1
        assert current[0].payload["current_generation"] == 1
        assert "raw_pin" not in repr(snapshot)
        assert all(
            "verifier" not in item.payload
            or item.representation_name == "PinVerifierRecord accepted revisions"
            for item in snapshot.immutable_history
        )
        backup = create_backup_envelope(store)
        assert backup is not None
        assert validate_backup_envelope(backup) == backup


def test_pin_history_key_payload_inconsistency_fails_closed() -> None:
    valid = _history(_pin(1))
    corrupted = replace(valid, record_key=valid.record_key[:-1] + "0")

    with pytest.raises(PersistenceRecordError, match="record_key derivation mismatch"):
        validate_persistence_record(corrupted)
