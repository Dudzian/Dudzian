from __future__ import annotations

from dataclasses import replace
from typing import cast

import pytest

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.protected_freshness_handoff import ProtectedFreshnessHandoffCoordinator
from bot_core.persistence.records import validate_persistence_record
from bot_core.persistence.runtime_session_history import (
    RUNTIME_SESSION_REPRESENTATION,
    RuntimeSessionHistoryError,
    RuntimeSessionHistoryPublisher,
    runtime_session_carrier,
)
from bot_core.persistence.state_store import SQLiteStateStore
from bot_core.runtime.runtime_session import RuntimeSession
from tests.persistence.test_protected_freshness_handoff import Boundary, record
from tests.persistence.test_state_store_records import DEVICE_ID, SESSION_1, _account, _metadata

SESSION_2 = "run_01890f47-8f2a-7abc-8def-1234567890ac"


class ObservingEvidenceRegistry(LocalDurableEvidenceRegistry):
    def __init__(self) -> None:
        super().__init__()
        self.published_refs: list[str | None] = []

    def publish_verified_state(self, store: SQLiteStateStore) -> str | None:
        reference = cast(str | None, super().publish_verified_state(store))
        self.published_refs.append(reference)
        return reference


def _initialized(tmp_path, evidence=None):  # type: ignore[no-untyped-def]
    store = SQLiteStateStore(tmp_path / "runtime-session.db")
    boundary = Boundary(record("UNINITIALIZED"))
    evidence = evidence or LocalDurableEvidenceRegistry()
    protected = ProtectedFreshnessHandoffCoordinator(store, evidence, boundary)
    initial = protected.advance_protected_state(_metadata(), current_records=(_account(),))
    return store, boundary, evidence, protected, initial


def test_exact_canonical_carrier_shape_and_stage_one_validation() -> None:
    session = RuntimeSession(SESSION_1, DEVICE_ID)
    carrier = runtime_session_carrier(session)
    upstream = {"runtime_session_id": SESSION_1, "device_installation_id": DEVICE_ID}
    assert carrier.representation_name == RUNTIME_SESSION_REPRESENTATION
    assert carrier.record_key == SESSION_1
    assert carrier.payload == {
        "fact_kind": "RuntimeSession",
        "upstream_payload": upstream,
        "upstream_payload_fingerprint_sha256": canonical_json_sha256(upstream),
    }
    validate_persistence_record(carrier)


def test_publication_is_one_protected_g_plus_one_history_only_mutation(tmp_path) -> None:  # type: ignore[no-untyped-def]
    evidence = ObservingEvidenceRegistry()
    store, boundary, evidence, protected, initial = _initialized(tmp_path, evidence)
    try:
        p1b_ref = evidence.publish_verified_state(store)
        assert len(evidence.published_refs) == 2
        result = RuntimeSessionHistoryPublisher(store, protected).publish_current_session(
            RuntimeSession(SESSION_1, DEVICE_ID)
        )
        final = store.read_verified_snapshot()
        assert final is not None
        assert result.source_generation == initial.protected_freshness_generation
        assert result.final_metadata == final.metadata
        assert (
            final.metadata.protected_freshness_generation
            == initial.protected_freshness_generation + 1
        )
        assert final.current_records == (_account(),)
        assert [item.record_key for item in final.immutable_history] == [SESSION_1]
        assert boundary.calls.count("prepare") == boundary.calls.count("finalize") == 2
        assert len(evidence.published_refs) == 3
        ref = evidence.published_refs[-1]
        assert ref != p1b_ref
        current = evidence.resolve_current(
            (
                final.metadata.account_id,
                final.metadata.device_installation_id,
                final.metadata.state_store_identity_fingerprint_sha256,
            ),
            ref,
        )
        assert (
            current is not None
            and current.generation == final.metadata.protected_freshness_generation
        )
        assert current.state_fingerprint_sha256 == final.metadata.state_fingerprint_sha256
        assert (
            current.transaction_fingerprint_sha256 == final.metadata.transaction_fingerprint_sha256
        )
        assert (
            current.account_id,
            current.device_installation_id,
            current.state_store_identity_fingerprint_sha256,
        ) == (
            final.metadata.account_id,
            final.metadata.device_installation_id,
            final.metadata.state_store_identity_fingerprint_sha256,
        )
        assert boundary.value["lifecycle"] == "COMMITTED"
    finally:
        store.close()


def test_different_old_history_is_preserved(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store, _, _, protected, _ = _initialized(tmp_path)
    publisher = RuntimeSessionHistoryPublisher(store, protected)
    try:
        publisher.publish_current_session(RuntimeSession(SESSION_1, DEVICE_ID))
        old = store.read_verified_snapshot()
        publisher.publish_current_session(RuntimeSession(SESSION_2, DEVICE_ID))
        final = store.read_verified_snapshot()
        assert old is not None and final is not None
        expected = {
            item.record_key: item
            for item in (
                *old.immutable_history,
                runtime_session_carrier(RuntimeSession(SESSION_2, DEVICE_ID)),
            )
        }
        assert {item.record_key: item for item in final.immutable_history} == expected
    finally:
        store.close()


def test_same_id_collision_fails_before_prepare(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store, boundary, _, protected, _ = _initialized(tmp_path)
    publisher = RuntimeSessionHistoryPublisher(store, protected)
    try:
        publisher.publish_current_session(RuntimeSession(SESSION_1, DEVICE_ID))
        before = store.read_verified_snapshot()
        prepares = boundary.calls.count("prepare")
        with pytest.raises(RuntimeSessionHistoryError, match="collision"):
            publisher.publish_current_session(RuntimeSession(SESSION_1, DEVICE_ID))
        assert store.read_verified_snapshot() == before
        assert boundary.calls.count("prepare") == prepares
    finally:
        store.close()


def test_device_mismatch_fails_before_prepare(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store, boundary, _, protected, _ = _initialized(tmp_path)
    try:
        prepares = boundary.calls.count("prepare")
        with pytest.raises(RuntimeSessionHistoryError, match="device binding"):
            RuntimeSessionHistoryPublisher(store, protected).publish_current_session(
                RuntimeSession(SESSION_1, "dev_01890f47-8f2a-7abc-8def-1234567890ff")
            )
        assert boundary.calls.count("prepare") == prepares
    finally:
        store.close()


def test_fresh_source_fence_rejects_scope_drift(tmp_path) -> None:  # type: ignore[no-untyped-def]
    store, _, _, protected, _ = _initialized(tmp_path)
    original = store.read_verified_snapshot

    def changed():
        snapshot = original()
        assert snapshot is not None
        return replace(
            snapshot,
            metadata=replace(
                snapshot.metadata, device_installation_id="dev_01890f47-8f2a-7abc-8def-1234567890ff"
            ),
        )

    store.read_verified_snapshot = changed  # type: ignore[method-assign]
    try:
        with pytest.raises(RuntimeSessionHistoryError, match="device binding"):
            RuntimeSessionHistoryPublisher(store, protected).publish_current_session(
                RuntimeSession(SESSION_1, DEVICE_ID)
            )
    finally:
        store.close()
