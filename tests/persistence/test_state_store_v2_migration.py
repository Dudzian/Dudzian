from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import cast

import pytest

from bot_core.persistence.fingerprints import (
    canonical_json_sha256,
    history_tail_fingerprint_sha256,
    state_fingerprint_sha256,
    transaction_fingerprint_sha256,
)
from bot_core.persistence.lifecycle_records import persistence_record
from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.records import (
    PersistenceRecordError,
    validate_persistence_record_for_schema,
)
from bot_core.persistence.migration_execution_engine import MigrationExecutionCoordinator
from bot_core.persistence.migration_execution import (
    MigrationExecutionDeclaration,
    operation_plan_fingerprint,
)
from bot_core.persistence.migration_execution_contract import (
    migration_definition_fingerprint,
    thaw_json,
)
from bot_core.persistence.migration_protocol import (
    PRODUCTION_MIGRATION_REGISTRY,
    DurableMigrationLifecycleCoordinator,
    MigrationError,
)
from bot_core.persistence.protected_freshness_handoff import (
    ProtectedFreshnessHandoffCoordinator,
)
from bot_core.persistence.state_store import (
    SQLiteStateStore,
    StateStoreError,
    StateStoreSnapshot,
)
from bot_core.persistence.transaction_descriptor import TransactionDescriptorError
from bot_core.persistence.state_store_v2_migration import (
    MIGRATION_ID,
    SQLITE_PHYSICAL_FINGERPRINT_SHA256,
    STATE_STORE_V2_MIGRATION_AUTHORITY,
    STATE_STORE_V2_MIGRATION_DEFINITION,
    STATE_STORE_V2_MIGRATION_OPERATIONS,
    STATE_STORE_V2_MIGRATION_PLAN,
)
from tests.persistence.test_protected_freshness_handoff import Boundary, record
from tests.persistence.test_state_store_records import ACCOUNT_ID, DEVICE_ID, _account, _metadata

OPERATOR_ID = "op_01890f4c-7b9a-7cc1-8a2b-123456789abc"


def _pin_payload() -> dict[str, object]:
    value: dict[str, object] = {
        "account_id": ACCOUNT_ID,
        "operator_id": OPERATOR_ID,
        "device_installation_id": DEVICE_ID,
        "algorithm_id": "M010-DETERMINISTIC-REFERENCE-NOT-PRODUCTION-KDF",
        "parameter_policy_version": 1,
        "salt_reference": "secure-store://opaque/pin-salt",
        "verifier": "c" * 64,
        "pin_revision": 1,
        "failed_attempts": 0,
        "lockout_until_utc": None,
        "security_generation": 1,
    }
    value["content_fingerprint_sha256"] = canonical_json_sha256(value)
    return value


def _legacy_pin_records():
    pin = _pin_payload()
    history_payload = {
        "fact_kind": "PinVerifierRecord accepted revisions",
        "upstream_payload": pin,
        "upstream_payload_fingerprint_sha256": canonical_json_sha256(pin),
    }
    scope = f"{ACCOUNT_ID}:{OPERATOR_ID}:{DEVICE_ID}"
    current_payload: dict[str, object] = {
        "scope_key": scope,
        "current_reference": pin["content_fingerprint_sha256"],
        "current_revision": 1,
        "current_generation": 1,
    }
    current_payload["content_fingerprint_sha256"] = canonical_json_sha256(current_payload)
    current = persistence_record(
        "PinVerifierRecord current designation",
        f"current:{scope}",
        current_payload,
    )
    history = persistence_record(
        "PinVerifierRecord accepted revisions",
        f"immutable:PinVerifierRecord accepted revisions:{OPERATOR_ID}:1:1:{pin['content_fingerprint_sha256']}",
        history_payload,
    )
    return (
        replace(current, record_key=f"current:{scope}:{pin['content_fingerprint_sha256']}:1:1"),
        replace(
            history, record_key=f"immutable:PinVerifierRecord accepted revisions:{OPERATOR_ID}:1:1"
        ),
    )


def _protected(store: SQLiteStateStore, boundary: Boundary):
    return ProtectedFreshnessHandoffCoordinator(store, LocalDurableEvidenceRegistry(), boundary)


def _applying_v1(path: Path):
    store = SQLiteStateStore(path)
    boundary = Boundary(record("UNINITIALIZED"))
    protected = _protected(store, boundary)
    current_pin, history_pin = _legacy_pin_records()
    protected.advance_protected_state(
        _metadata(state_store_schema_version=1),
        current_records=(_account(), current_pin),
        immutable_history=(history_pin,),
    )
    lifecycle = DurableMigrationLifecycleCoordinator(
        store, PRODUCTION_MIGRATION_REGISTRY, protected
    )
    lifecycle.prepare(MIGRATION_ID)
    lifecycle.begin_applying(MIGRATION_ID)
    snapshot = store.read_verified_snapshot()
    assert snapshot is not None and snapshot.metadata.state_store_schema_version == 1
    return store, boundary, snapshot


def _coordinator(store: SQLiteStateStore, boundary: Boundary):
    return MigrationExecutionCoordinator(
        store, PRODUCTION_MIGRATION_REGISTRY, _protected(store, boundary)
    )


def test_production_v1_pin_migration_and_reopen_recovery(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    store, boundary, before = _applying_v1(path)
    physical = store.sqlite_schema_fingerprint()
    descriptor_json = tuple(
        row[0]
        for row in store._connection.execute(
            "SELECT descriptor_json FROM state_store_transaction_descriptors ORDER BY target_generation"
        )
    )
    old_current, old_history = _legacy_pin_records()
    result = _coordinator(store, boundary).execute(MIGRATION_ID)
    after = store.read_verified_snapshot()
    assert after is not None
    assert after.metadata.state_store_schema_version == 2
    assert (
        after.metadata.protected_freshness_generation
        == before.metadata.protected_freshness_generation + 1
    )
    for field in (
        "account_id",
        "device_installation_id",
        "environment",
        "state_store_identity_fingerprint_sha256",
    ):
        assert getattr(after.metadata, field) == getattr(before.metadata, field)
    pin_history = tuple(
        r
        for r in after.immutable_history
        if r.representation_name == "PinVerifierRecord accepted revisions"
    )
    pin_current = tuple(
        r
        for r in after.current_records
        if r.representation_name == "PinVerifierRecord current designation"
    )
    assert len(pin_history) == len(pin_current) == 1
    assert pin_history[0].payload == old_history.payload
    assert (
        pin_history[0].record_key
        == f"{old_history.record_key}:{_pin_payload()['content_fingerprint_sha256']}"
    )
    assert pin_current[0].payload == old_current.payload
    assert pin_current[0].record_key == f"current:{ACCOUNT_ID}:{OPERATOR_ID}:{DEVICE_ID}"
    assert all(r.record_key != old_history.record_key for r in after.immutable_history)
    assert all(r.record_key != old_current.record_key for r in after.current_records)
    assert (
        tuple(
            row[0]
            for row in store._connection.execute(
                "SELECT descriptor_json FROM state_store_transaction_descriptors WHERE target_generation < ? ORDER BY target_generation",
                (after.metadata.protected_freshness_generation,),
            )
        )
        == descriptor_json
    )
    descriptor = after.transaction_descriptors[-1]
    declarations = tuple(
        r
        for r in after.immutable_history
        if r.representation_name == "Migration execution declaration"
    )
    assert descriptor.state_store_schema_version == 2
    assert descriptor.current_record_mutations == ()
    assert descriptor.immutable_history_appends == declarations
    assert descriptor.post_history_tail_fingerprint_sha256 == history_tail_fingerprint_sha256(
        after.immutable_history
    )
    assert descriptor.post_state_fingerprint_sha256 == state_fingerprint_sha256(
        account_id=after.metadata.account_id,
        device_installation_id=after.metadata.device_installation_id,
        state_store_schema_version=2,
        state_store_identity_fingerprint_sha256=after.metadata.state_store_identity_fingerprint_sha256,
        environment=after.metadata.environment,
        protected_freshness_generation=after.metadata.protected_freshness_generation,
        current_records=after.current_records,
        history_tail_fingerprint_sha256=after.metadata.history_tail_fingerprint_sha256,
    )
    assert physical == store.sqlite_schema_fingerprint() == SQLITE_PHYSICAL_FINGERPRINT_SHA256
    generation = after.metadata.protected_freshness_generation
    store.close()
    with SQLiteStateStore(path) as reopened:
        recovered = _coordinator(reopened, boundary).execute(MIGRATION_ID)
        assert recovered == result
        final = reopened.read_verified_snapshot()
        assert final is not None and final.metadata.protected_freshness_generation == generation
        assert (
            sum(
                r.representation_name == "Migration execution declaration"
                for r in final.immutable_history
            )
            == 1
        )


def test_trial_derivation_is_non_durable_and_matches_commit(tmp_path: Path) -> None:
    store, boundary, before = _applying_v1(tmp_path / "trial.db")
    definition = PRODUCTION_MIGRATION_REGISTRY.definition_for(MIGRATION_ID)
    plan = PRODUCTION_MIGRATION_REGISTRY.authorized_plan_for(definition, before)
    declaration = MigrationExecutionCoordinator._declaration(definition, plan, before)
    seed = replace(
        before.metadata,
        state_store_schema_version=2,
        protected_freshness_generation=declaration.target_generation,
    )
    candidate = store._derive_migration_execution_metadata(seed, declaration)
    assert not store._connection.in_transaction
    assert store.read_verified_snapshot() == before
    assert not any(
        r.representation_name == "Migration execution declaration" for r in before.immutable_history
    )
    _coordinator(store, boundary).execute(MIGRATION_ID)
    assert store.read_metadata() == candidate
    store.close()


def test_real_second_operation_failure_rolls_back_and_protected_recovery_aborts(
    tmp_path: Path,
) -> None:
    store, boundary, before = _applying_v1(tmp_path / "rollback.db")
    original = store._derive_migration_execution_metadata
    updates: list[str] = []

    def derive_then_deny_current(seed, declaration):
        candidate = original(seed, declaration)

        def authorizer(action, table, column, database, trigger):
            if action == sqlite3.SQLITE_UPDATE:
                updates.append(table)
                if table == "state_store_current_records":
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        store._connection.set_authorizer(authorizer)
        return candidate

    store._derive_migration_execution_metadata = derive_then_deny_current  # type: ignore[method-assign]
    with pytest.raises(MigrationError, match="execution failed"):
        _coordinator(store, boundary).execute(MIGRATION_ID)
    store._connection.set_authorizer(None)
    assert updates[0] == "state_store_immutable_history"
    assert "state_store_current_records" in updates
    assert store.read_verified_snapshot() == before
    assert boundary.value["lifecycle"] == "PREPARED"
    recovered = _protected(store, boundary).recover_protected_state(
        (
            before.metadata.account_id,
            before.metadata.device_installation_id,
            before.metadata.state_store_identity_fingerprint_sha256,
        )
    )
    assert recovered == before.metadata
    assert boundary.value["lifecycle"] == "COMMITTED"
    store.close()


def test_production_constants_are_sealed_to_canonical_document() -> None:
    canonical = json.loads(
        Path(
            "docs/architecture/cryptohunter_product_architecture/persistence_versioning_migrations_backup_and_recovery.json"
        ).read_text()
    )["state_store_representation_lineage_v2"]["sealed_production_migrations"][0]
    assert (
        STATE_STORE_V2_MIGRATION_DEFINITION.fingerprint()
        == canonical["definition"]["migration_definition_fingerprint_sha256"]
    )
    assert [
        operation.to_mapping() for operation in STATE_STORE_V2_MIGRATION_OPERATIONS
    ] == canonical["operations"]
    assert (
        STATE_STORE_V2_MIGRATION_PLAN.operation_plan_fingerprint_sha256
        == canonical["operation_plan_fingerprint_sha256"]
    )
    STATE_STORE_V2_MIGRATION_AUTHORITY.assert_definition(STATE_STORE_V2_MIGRATION_DEFINITION)
    STATE_STORE_V2_MIGRATION_AUTHORITY.assert_plan(STATE_STORE_V2_MIGRATION_PLAN)


def test_pin_record_contract_is_selected_only_by_schema_version() -> None:
    old_current, old_history = _legacy_pin_records()
    pin = _pin_payload()
    new_current = replace(old_current, record_key=f"current:{ACCOUNT_ID}:{OPERATOR_ID}:{DEVICE_ID}")
    new_history = replace(
        old_history,
        record_key=f"{old_history.record_key}:{pin['content_fingerprint_sha256']}",
    )
    for record in (old_current, old_history):
        validate_persistence_record_for_schema(record, state_store_schema_version=1)
        with pytest.raises(PersistenceRecordError, match="record_key derivation"):
            validate_persistence_record_for_schema(record, state_store_schema_version=2)
    for record in (new_current, new_history):
        validate_persistence_record_for_schema(record, state_store_schema_version=2)
        with pytest.raises(PersistenceRecordError, match="record_key derivation"):
            validate_persistence_record_for_schema(record, state_store_schema_version=1)
    for version in (0, True, 1.0, 3, 999):
        with pytest.raises(PersistenceRecordError, match="unsupported"):
            validate_persistence_record_for_schema(
                old_current,
                state_store_schema_version=version,  # type: ignore[arg-type]
            )


def test_descriptor_embedded_pin_uses_descriptor_owned_version(tmp_path: Path) -> None:
    store, _, before = _applying_v1(tmp_path / "descriptor.db")
    old_current, _ = _legacy_pin_records()
    new_current = replace(old_current, record_key=f"current:{ACCOUNT_ID}:{OPERATOR_ID}:{DEVICE_ID}")
    descriptor = before.transaction_descriptors[-1]
    with pytest.raises(TransactionDescriptorError, match="invalid record"):
        replace(descriptor, current_record_mutations=(new_current,))
    with pytest.raises(TransactionDescriptorError, match="invalid record"):
        replace(descriptor, state_store_schema_version=2, current_record_mutations=(old_current,))
    store.close()


def test_top_level_metadata_version_rejects_opposite_pin_representation(
    tmp_path: Path,
) -> None:
    v1, boundary, _ = _applying_v1(tmp_path / "mixed-v1.db")
    operation = STATE_STORE_V2_MIGRATION_OPERATIONS[0]
    v1._connection.execute(operation.statement, operation.parameters)
    with pytest.raises(StateStoreError, match="PersistenceRecord"):
        v1.read_verified_snapshot()
    v1.close()

    v2, boundary, _ = _applying_v1(tmp_path / "mixed-v2.db")
    migrated = _coordinator(v2, boundary).execute(MIGRATION_ID)
    assert migrated.target_schema_version == 2
    _, old_history = _legacy_pin_records()
    new_key = f"{old_history.record_key}:{_pin_payload()['content_fingerprint_sha256']}"
    encoded = v2._connection.execute(
        "SELECT record_json FROM state_store_immutable_history WHERE record_key=?", (new_key,)
    ).fetchone()[0]
    value = json.loads(encoded)
    value["record_key"] = old_history.record_key
    v2._connection.execute(
        "UPDATE state_store_immutable_history SET record_key=?, record_json=? WHERE record_key=?",
        (old_history.record_key, json.dumps(value, sort_keys=True, separators=(",", ":")), new_key),
    )
    with pytest.raises(StateStoreError, match="PersistenceRecord"):
        v2.read_verified_snapshot()
    v2.close()


def _rewritten_edge(
    snapshot: StateStoreSnapshot,
    declaration: MigrationExecutionDeclaration,
    *,
    extra_append=None,
) -> StateStoreSnapshot:
    old_carrier = snapshot.transaction_descriptors[-1].immutable_history_appends[0]
    carrier = declaration.carrier()
    history = tuple(
        carrier if record == old_carrier else record for record in snapshot.immutable_history
    )
    appends = (carrier,) if extra_append is None else (carrier, extra_append)
    if extra_append is not None:
        history = (*history, extra_append)
    history_tail = history_tail_fingerprint_sha256(history)
    metadata = snapshot.metadata
    state = state_fingerprint_sha256(
        account_id=metadata.account_id,
        device_installation_id=metadata.device_installation_id,
        state_store_schema_version=metadata.state_store_schema_version,
        state_store_identity_fingerprint_sha256=metadata.state_store_identity_fingerprint_sha256,
        environment=metadata.environment,
        protected_freshness_generation=metadata.protected_freshness_generation,
        current_records=snapshot.current_records,
        history_tail_fingerprint_sha256=history_tail,
    )
    descriptor = replace(
        snapshot.transaction_descriptors[-1],
        post_state_fingerprint_sha256=state,
        post_history_tail_fingerprint_sha256=history_tail,
        immutable_history_appends=appends,
        transaction_fingerprint_sha256="0" * 64,
    )
    descriptor = replace(
        descriptor,
        transaction_fingerprint_sha256=transaction_fingerprint_sha256(
            descriptor.to_fingerprint_mapping()
        ),
    )
    metadata = replace(
        metadata,
        state_fingerprint_sha256=state,
        history_tail_fingerprint_sha256=history_tail,
        transaction_fingerprint_sha256=descriptor.transaction_fingerprint_sha256,
    )
    return StateStoreSnapshot(
        metadata,
        snapshot.current_records,
        tuple(sorted(history, key=lambda item: (item.representation_name, item.record_key))),
        (*snapshot.transaction_descriptors[:-1], descriptor),
    )


def _migrated_snapshot(tmp_path: Path) -> StateStoreSnapshot:
    store, boundary, _ = _applying_v1(tmp_path / "forged.db")
    _coordinator(store, boundary).execute(MIGRATION_ID)
    snapshot = store.read_verified_snapshot()
    assert snapshot is not None
    store.close()
    return cast(StateStoreSnapshot, snapshot)


def _declaration(snapshot: StateStoreSnapshot) -> MigrationExecutionDeclaration:
    carrier = snapshot.transaction_descriptors[-1].immutable_history_appends[0]
    return MigrationExecutionDeclaration.from_mapping(thaw_json(carrier.payload))


def test_self_consistent_fake_migration_schema_edge_is_rejected(tmp_path: Path) -> None:
    snapshot = _migrated_snapshot(tmp_path)
    declaration = _declaration(snapshot)
    fake_id = "fake-migration"
    forged = replace(
        declaration,
        migration_id=fake_id,
        migration_definition_fingerprint_sha256=migration_definition_fingerprint(
            migration_id=fake_id,
            source_schema_version=1,
            target_schema_version=2,
            ordered_path=declaration.ordered_path,
            rollback_policy=declaration.rollback_policy,
        ),
    )
    with pytest.raises(StateStoreError, match="unauthorized descriptor schema-version edge"):
        SQLiteStateStore.verify_snapshot(_rewritten_edge(snapshot, forged))


def test_self_consistent_unsealed_operation_plan_is_rejected(tmp_path: Path) -> None:
    snapshot = _migrated_snapshot(tmp_path)
    declaration = _declaration(snapshot)
    changed = replace(declaration.operations[0], operation_id="unsealed-rekey-operation")
    operations = (changed, *declaration.operations[1:])
    forged = replace(
        declaration,
        operations=operations,
        operation_plan_fingerprint_sha256=operation_plan_fingerprint(operations),
    )
    with pytest.raises(StateStoreError, match="unauthorized descriptor schema-version edge"):
        SQLiteStateStore.verify_snapshot(_rewritten_edge(snapshot, forged))


def test_self_consistent_unsealed_ordered_path_is_rejected(tmp_path: Path) -> None:
    snapshot = _migrated_snapshot(tmp_path)
    declaration = _declaration(snapshot)
    path = ("unsealed-path",)
    forged = replace(
        declaration,
        ordered_path=path,
        migration_definition_fingerprint_sha256=migration_definition_fingerprint(
            migration_id=declaration.migration_id,
            source_schema_version=1,
            target_schema_version=2,
            ordered_path=path,
            rollback_policy=declaration.rollback_policy,
        ),
    )
    with pytest.raises(StateStoreError, match="unauthorized descriptor schema-version edge"):
        SQLiteStateStore.verify_snapshot(_rewritten_edge(snapshot, forged))


def test_sealed_declaration_must_bind_runtime_descriptor_scope(tmp_path: Path) -> None:
    snapshot = _migrated_snapshot(tmp_path)
    forged = replace(_declaration(snapshot), environment="LIVE")
    with pytest.raises(StateStoreError, match="unauthorized descriptor schema-version edge"):
        SQLiteStateStore.verify_snapshot(_rewritten_edge(snapshot, forged))


def test_schema_edge_rejects_additional_immutable_append(tmp_path: Path) -> None:
    snapshot = _migrated_snapshot(tmp_path)
    extra = next(
        record
        for record in snapshot.immutable_history
        if record.representation_name == "PinVerifierRecord accepted revisions"
    )
    with pytest.raises(StateStoreError, match="unauthorized descriptor schema-version edge"):
        SQLiteStateStore.verify_snapshot(
            _rewritten_edge(snapshot, _declaration(snapshot), extra_append=extra)
        )
