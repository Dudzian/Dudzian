from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import Any

import pytest

from bot_core.persistence.durable_observation import DurableStateObservation
from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.persistence.local_durable_evidence import (
    LocalDurableEvidenceRegistry,
    LocalDurableStateEvidence,
)
from bot_core.persistence.state_store import SQLiteStateStore, StateStoreError
from tests.persistence.test_state_store_records import _account, _commit, _metadata, _runtime

EVIDENCE_FIELDS = {
    "account_id",
    "device_installation_id",
    "state_store_identity_fingerprint_sha256",
    "generation",
    "state_fingerprint_sha256",
    "transaction_fingerprint_sha256",
    "durability_state",
    "evidence_revision",
    "evidence_fingerprint_sha256",
}


def _generation_one(store: SQLiteStateStore) -> None:
    _commit(store, _metadata(), current=(_account(),), history=(_runtime(),))


def _scope(evidence: LocalDurableStateEvidence) -> tuple[str, str, str]:
    return (
        evidence.account_id,
        evidence.device_installation_id,
        evidence.state_store_identity_fingerprint_sha256,
    )


def _manual_evidence(**changes: Any) -> LocalDurableStateEvidence:
    projection: dict[str, object] = {
        "account_id": "acct_01890f4c-7b9a-7cc1-8a2b-123456789abc",
        "device_installation_id": "dev_01890f4c-7b9a-7cc1-8a2b-123456789abc",
        "state_store_identity_fingerprint_sha256": "1" * 64,
        "generation": 1,
        "state_fingerprint_sha256": "2" * 64,
        "transaction_fingerprint_sha256": "3" * 64,
        "durability_state": "DURABLE_COMMITTED",
        "evidence_revision": 1,
    }
    projection.update(changes)
    return LocalDurableStateEvidence(  # type: ignore[arg-type]
        **projection,
        evidence_fingerprint_sha256=canonical_json_sha256(projection),
    )


def test_exact_evidence_payload_membership_derivation_and_fingerprint(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _generation_one(store)
        snapshot = store.read_verified_snapshot()
        registry = LocalDurableEvidenceRegistry()
        ref = registry.publish_verified_state(store)

    assert snapshot is not None and ref is not None
    evidence = registry._accepted[ref]
    mapping = evidence.to_mapping()
    projection = {
        key: value for key, value in mapping.items() if key != "evidence_fingerprint_sha256"
    }
    assert {field.name for field in fields(evidence)} == set(mapping) == EVIDENCE_FIELDS
    assert evidence.account_id == snapshot.metadata.account_id
    assert evidence.device_installation_id == snapshot.metadata.device_installation_id
    assert (
        evidence.state_store_identity_fingerprint_sha256
        == snapshot.metadata.state_store_identity_fingerprint_sha256
    )
    assert evidence.generation == snapshot.metadata.protected_freshness_generation == 1
    assert evidence.state_fingerprint_sha256 == snapshot.metadata.state_fingerprint_sha256
    assert (
        evidence.transaction_fingerprint_sha256 == snapshot.metadata.transaction_fingerprint_sha256
    )
    assert evidence.durability_state == "DURABLE_COMMITTED"
    assert evidence.evidence_revision == 1
    assert evidence.evidence_fingerprint_sha256 == canonical_json_sha256(projection)
    mapping["generation"] = 99
    assert evidence.generation == 1
    assert registry.verify_current(_scope(evidence), ref)


@pytest.mark.parametrize("field", sorted(EVIDENCE_FIELDS))
def test_every_projection_field_is_fingerprint_bound(field: str) -> None:
    evidence = _manual_evidence()
    mapping = evidence.to_mapping()
    projection = {
        key: value for key, value in mapping.items() if key != "evidence_fingerprint_sha256"
    }
    if field == "evidence_fingerprint_sha256":
        assert field not in projection
        return
    projection[field] = "changed" if field not in {"generation", "evidence_revision"} else 2
    assert canonical_json_sha256(projection) != evidence.evidence_fingerprint_sha256


@pytest.mark.parametrize("field", ["generation", "evidence_revision"])
@pytest.mark.parametrize("value", [True, False, 0, -1, "1"])
def test_positive_integer_type_gates(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="positive non-boolean integer"):
        _manual_evidence(**{field: value})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("account_id", "dev_01890f4c-7b9a-7cc1-8a2b-123456789abc"),
        ("device_installation_id", "acct_01890f4c-7b9a-7cc1-8a2b-123456789abc"),
        ("state_fingerprint_sha256", "A" * 64),
        ("transaction_fingerprint_sha256", "bad"),
        ("durability_state", "CURRENT"),
    ],
)
def test_intrinsic_scope_sha_and_durability_gates(field: str, value: object) -> None:
    with pytest.raises(ValueError):
        _manual_evidence(**{field: value})


def test_bad_stored_evidence_fingerprint_is_rejected() -> None:
    evidence = _manual_evidence()
    with pytest.raises(ValueError, match="does not match"):
        replace(evidence, evidence_fingerprint_sha256="a" * 64)


def test_self_hashed_evidence_and_caller_ref_have_no_membership() -> None:
    evidence = _manual_evidence()
    registry = LocalDurableEvidenceRegistry()
    assert not registry.verify_current(_scope(evidence), "caller-opaque-ref")
    assert registry._revision == 0 and registry._accepted == {} and registry._current == {}


def test_manual_observation_cannot_enroll(tmp_path: Path) -> None:
    fake = DurableStateObservation(
        account_id="acct_01890f4c-7b9a-7cc1-8a2b-123456789abc",
        device_installation_id="dev_01890f4c-7b9a-7cc1-8a2b-123456789abc",
        state_store_schema_version=1,
        state_store_identity_fingerprint_sha256="1" * 64,
        environment="LIVE",
        protected_freshness_generation=1,
        state_fingerprint_sha256="2" * 64,
        transaction_fingerprint_sha256="3" * 64,
        history_tail_fingerprint_sha256="4" * 64,
        durable_confirmed=True,
        authoritative_history_integrity=True,
        current_commit=True,
    )
    registry = LocalDurableEvidenceRegistry()
    with pytest.raises(TypeError, match="SQLiteStateStore"):
        registry.publish_verified_state(fake)  # type: ignore[arg-type]
    assert registry._revision == 0 and registry._accepted == {} and registry._current == {}


def test_empty_store_does_not_mutate_registry(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "empty.sqlite3") as store:
        registry = LocalDurableEvidenceRegistry()
        assert registry.publish_verified_state(store) is None
    assert registry._revision == 0 and registry._accepted == {} and registry._current == {}


@pytest.mark.parametrize(
    "statement",
    [
        "UPDATE state_store_metadata SET state_fingerprint_sha256='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'",
        "DELETE FROM state_store_transaction_descriptors",
        "UPDATE state_store_current_records SET record_json='{}'",
    ],
)
def test_corruption_does_not_mutate_registry(tmp_path: Path, statement: str) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _generation_one(store)
        store._connection.execute(statement)
        registry = LocalDurableEvidenceRegistry()
        with pytest.raises(StateStoreError):
            registry.publish_verified_state(store)
    assert registry._revision == 0 and registry._accepted == {} and registry._current == {}


def test_success_wrong_scope_ref_and_opaque_reference(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _generation_one(store)
        registry = LocalDurableEvidenceRegistry()
        ref = registry.publish_verified_state(store)
    assert isinstance(ref, str)
    evidence = registry._accepted[ref]
    scope = _scope(evidence)
    assert all(
        value not in ref
        for value in (
            *scope,
            evidence.state_fingerprint_sha256,
            evidence.transaction_fingerprint_sha256,
        )
    )
    assert registry.verify_current(scope, ref)
    assert not registry.verify_current(("wrong", scope[1], scope[2]), ref)
    assert not registry.verify_current((scope[0], "wrong", scope[2]), ref)
    assert not registry.verify_current((scope[0], scope[1], "wrong"), ref)
    assert not registry.verify_current(scope, "caller")
    assert not registry.verify_current(scope, None)
    assert not registry.verify_current(scope, 1)


def test_same_generation_republish_is_new_current_revision(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _generation_one(store)
        before = store.read_verified_snapshot()
        registry = LocalDurableEvidenceRegistry()
        ref1 = registry.publish_verified_state(store)
        ref2 = registry.publish_verified_state(store)
        after = store.read_verified_snapshot()
    assert ref1 is not None and ref2 is not None and ref1 != ref2
    evidence1, evidence2 = registry._accepted[ref1], registry._accepted[ref2]
    assert evidence1.generation == evidence2.generation == 1
    assert (evidence1.evidence_revision, evidence2.evidence_revision) == (1, 2)
    assert not registry.verify_current(_scope(evidence1), ref1)
    assert registry.verify_current(_scope(evidence2), ref2)
    assert before == after


def test_g1_to_g2_keeps_old_evidence_immutable_but_not_current(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _generation_one(store)
        registry = LocalDurableEvidenceRegistry()
        ref1 = registry.publish_verified_state(store)
        assert ref1 is not None
        evidence1 = registry._accepted[ref1]
        _commit(
            store,
            _metadata(2),
            history=(_runtime("run_01890f4c-7b9a-7cc1-8a2b-123456789abd"),),
            expected=1,
        )
        ref2 = registry.publish_verified_state(store)
    assert ref2 is not None and ref1 != ref2
    evidence2 = registry._accepted[ref2]
    assert (evidence1.generation, evidence2.generation) == (1, 2)
    assert (evidence1.evidence_revision, evidence2.evidence_revision) == (1, 2)
    assert not registry.verify_current(_scope(evidence1), ref1)
    assert registry.verify_current(_scope(evidence2), ref2)
    with pytest.raises((FrozenInstanceError, AttributeError)):
        evidence1.generation = 9  # type: ignore[misc]


def test_registry_restart_is_empty_and_old_ref_is_rejected(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _generation_one(store)
        first = LocalDurableEvidenceRegistry()
        ref1 = first.publish_verified_state(store)
        assert ref1 is not None
        scope = _scope(first._accepted[ref1])
        second = LocalDurableEvidenceRegistry()
        assert not second.verify_current(scope, ref1)
        ref2 = second.publish_verified_state(store)
    assert ref2 is not None and ref2 != ref1
    assert second.verify_current(scope, ref2)
    assert not second.verify_current(scope, ref1)


def test_multiple_exact_scopes_remain_independently_current(tmp_path: Path) -> None:
    registry = LocalDurableEvidenceRegistry()
    with (
        SQLiteStateStore(tmp_path / "a.sqlite3") as first,
        SQLiteStateStore(tmp_path / "b.sqlite3") as second,
    ):
        _generation_one(first)
        _commit(
            second,
            _metadata(
                account_id="acct_01890f4c-7b9a-7cc1-8a2b-123456789abd",
                state_store_identity_fingerprint_sha256="b" * 64,
            ),
            current=(_account("acct_01890f4c-7b9a-7cc1-8a2b-123456789abd"),),
            history=(_runtime(),),
        )
        ref1 = registry.publish_verified_state(first)
        ref2 = registry.publish_verified_state(second)
    assert ref1 is not None and ref2 is not None
    scope1, scope2 = _scope(registry._accepted[ref1]), _scope(registry._accepted[ref2])
    assert registry.verify_current(scope1, ref1)
    assert registry.verify_current(scope2, ref2)
    assert not registry.verify_current(scope1, ref2)
    assert not registry.verify_current(scope2, ref1)


def test_publication_and_verification_leave_store_and_schema_unchanged(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.sqlite3") as store:
        _generation_one(store)
        before = store.read_verified_snapshot()
        registry = LocalDurableEvidenceRegistry()
        ref = registry.publish_verified_state(store)
        assert ref is not None
        registry.publish_verified_state(store)
        registry.verify_current(_scope(registry._accepted[ref]), ref)
        after = store.read_verified_snapshot()
        tables = {
            row[0]
            for row in store._connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
    assert before == after
    assert tables == {
        "state_store_metadata",
        "state_store_current_records",
        "state_store_immutable_history",
        "state_store_transaction_descriptors",
    }


def test_no_public_enrollment_or_authority_expansion_api() -> None:
    forbidden = {
        "from_mapping",
        "accept",
        "authorize",
        "grant",
        "promote",
        "set_current",
        "designate_current",
        "restore",
        "enable_live",
        "register",
        "enroll",
        "insert",
    }
    assert forbidden.isdisjoint(dir(LocalDurableStateEvidence))
    assert forbidden.isdisjoint(dir(LocalDurableEvidenceRegistry))
