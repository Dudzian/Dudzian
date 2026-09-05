from __future__ import annotations

from pathlib import Path

import pytest

from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.migration_execution import MigrationExecutionAuthority
from bot_core.persistence.migration_protocol import (
    DurableMigrationLifecycleCoordinator,
    MigrationDefinition,
    MigrationError,
    MigrationRegistry,
    migration_current_carrier,
    migration_transition_carrier,
)
from bot_core.persistence.protected_freshness_handoff import (
    ProtectedFreshnessHandoffCoordinator,
)
from bot_core.persistence.secret_handoff import (
    DurableSecretHandoffLifecycleCoordinator,
    ExternalOutcome,
    SecretHandoffError,
)
from bot_core.persistence.state_store import SQLiteStateStore
from tests.persistence.test_protected_freshness_handoff import Boundary, record
from tests.persistence.test_migration_protocol import lifecycle, plan
from tests.persistence.test_secret_handoff import descriptor
from tests.persistence.test_state_store_records import _account, _metadata


def _migration_registry() -> MigrationRegistry:
    definition = MigrationDefinition("migration-1", 1, 2, ("test-step",))
    authority = MigrationExecutionAuthority(
        definition.migration_id,
        definition.source_schema_version,
        definition.target_schema_version,
        definition.ordered_path,
        definition.rollback_policy,
        definition.fingerprint(),
        "a" * 64,
        "a" * 64,
        "a" * 64,
    )
    return MigrationRegistry(((definition, authority, lambda snapshot: snapshot),))


def _initialize(store: SQLiteStateStore, boundary: Boundary, *, schema: int = 1) -> None:
    ProtectedFreshnessHandoffCoordinator(
        store, LocalDurableEvidenceRegistry(), boundary
    ).advance_protected_state(
        _metadata(state_store_schema_version=schema), current_records=(_account(),)
    )


def _protected(store: SQLiteStateStore, boundary: Boundary):
    return ProtectedFreshnessHandoffCoordinator(store, LocalDurableEvidenceRegistry(), boundary)


def _descriptor(**changes):
    return descriptor(scope=(_metadata().account_id, _metadata().device_installation_id), **changes)


def _initialize_with_migration(
    store: SQLiteStateStore, boundary: Boundary, states: tuple[str, ...], *, schema: int
) -> None:
    history, current = lifecycle(states)
    ProtectedFreshnessHandoffCoordinator(
        store, LocalDurableEvidenceRegistry(), boundary
    ).advance_protected_state(
        _metadata(state_store_schema_version=schema),
        current_records=(_account(), migration_current_carrier(current)),
        immutable_history=tuple(migration_transition_carrier(item) for item in history),
    )


def test_migration_genesis_duplicate_transition_and_observation(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        _initialize(store, boundary)
        coordinator = DurableMigrationLifecycleCoordinator(
            store, _migration_registry(), _protected(store, boundary)
        )
        prepared = coordinator.prepare("migration-1")
        assert prepared.current is not None
        assert prepared.current["state"] == "PREPARED"
        assert len(prepared.history) == 1
        assert all(
            item.representation_name != "MigrationRecord" for item in store.read_immutable_history()
        )
        assert prepared.history[0]["protected_freshness_generation"] == 1
        generation = store.read_metadata().protected_freshness_generation
        assert coordinator.prepare("migration-1") == prepared
        assert store.read_metadata().protected_freshness_generation == generation
        applying = coordinator.begin_applying("migration-1")
        assert applying.current is not None and applying.current["state"] == "APPLYING"
        assert applying.history[-1]["protected_freshness_generation"] == generation
        with pytest.raises(MigrationError, match="materially proven"):
            coordinator.record_durable_migrated("migration-1", plan())


def test_migration_restart_uses_registry_and_durable_records(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(path) as store:
        _initialize(store, boundary)
        DurableMigrationLifecycleCoordinator(
            store, _migration_registry(), _protected(store, boundary)
        ).prepare("migration-1")
    with SQLiteStateStore(path) as reopened:
        coordinator = DurableMigrationLifecycleCoordinator(
            reopened, _migration_registry(), _protected(reopened, boundary)
        )
        assert coordinator.begin_applying("migration-1").current["state"] == "APPLYING"
        with pytest.raises(MigrationError):
            DurableMigrationLifecycleCoordinator(
                reopened, MigrationRegistry(), _protected(reopened, boundary)
            ).discover("migration-1")


def test_secret_genesis_is_atomic_duplicate_and_restart_safe(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    handoff = _descriptor()
    with SQLiteStateStore(path) as store:
        _initialize(store, boundary)
        coordinator = DurableSecretHandoffLifecycleCoordinator(store, _protected(store, boundary))
        prepared = coordinator.prepare(handoff)
        assert prepared.descriptor == handoff
        assert prepared.current is not None and prepared.current["state"] == "PREPARED"
        assert (
            sum(
                item.representation_name == "SecretHandoff immutable descriptor"
                for item in store.read_immutable_history()
            )
            == 1
        )
        generation = store.read_metadata().protected_freshness_generation
        assert coordinator.prepare(handoff) == prepared
        assert store.read_metadata().protected_freshness_generation == generation
        assert (
            sum(
                item.representation_name == "SecretHandoff immutable descriptor"
                for item in store.read_immutable_history()
            )
            == 1
        )
    with SQLiteStateStore(path) as reopened:
        coordinator = DurableSecretHandoffLifecycleCoordinator(
            reopened, _protected(reopened, boundary)
        )
        committed = coordinator.record_external_outcome(
            handoff.handoff_id, ExternalOutcome.COMMITTED
        )
        assert committed.current is not None and committed.current["state"] == "COMMITTED"
        assert coordinator.mark_cleanup_pending(handoff.handoff_id).current["state"] == (
            "CLEANUP_PENDING"
        )


def test_secret_descriptor_conflict_and_illegal_outcome_fail_closed(
    tmp_path: Path,
) -> None:
    with SQLiteStateStore(tmp_path / "state.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        _initialize(store, boundary)
        coordinator = DurableSecretHandoffLifecycleCoordinator(store, _protected(store, boundary))
        coordinator.prepare(_descriptor())
        generation = store.read_metadata().protected_freshness_generation
        with pytest.raises(SecretHandoffError):
            coordinator.prepare(_descriptor(new_reference="ref:other"))
        with pytest.raises(SecretHandoffError):
            coordinator.record_external_outcome("handoff-1", ExternalOutcome.NOT_STARTED)
        assert store.read_metadata().protected_freshness_generation == generation


def test_post_commit_lost_ack_reopens_as_zero_mutation_duplicate(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(path) as store:
        _initialize(store, boundary)
        boundary.ack_lost = True
        coordinator = DurableMigrationLifecycleCoordinator(
            store, _migration_registry(), _protected(store, boundary)
        )
        with pytest.raises(MigrationError, match="protected migration CAS failed"):
            coordinator.prepare("migration-1")
        generation = store.read_metadata().protected_freshness_generation
    boundary.ack_lost = False
    with SQLiteStateStore(path) as reopened:
        coordinator = DurableMigrationLifecycleCoordinator(
            reopened, _migration_registry(), _protected(reopened, boundary)
        )
        assert coordinator.prepare("migration-1").current["state"] == "PREPARED"
        assert reopened.read_metadata().protected_freshness_generation == generation


def test_failure_before_commit_leaves_no_partial_lifecycle(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        _initialize(store, boundary)
        generation = store.read_metadata().protected_freshness_generation
        store._connection.execute("""
            CREATE TRIGGER fail_lifecycle_history
            BEFORE INSERT ON state_store_immutable_history
            BEGIN SELECT RAISE(ABORT, 'injected lifecycle failure'); END
            """)
        coordinator = DurableMigrationLifecycleCoordinator(
            store, _migration_registry(), _protected(store, boundary)
        )
        with pytest.raises(MigrationError, match="protected migration CAS failed"):
            coordinator.prepare("migration-1")
        assert store.read_metadata().protected_freshness_generation == generation
        assert coordinator.discover("migration-1").history == ()
        assert coordinator.discover("migration-1").current is None


def test_two_store_stale_same_transition_has_one_durable_winner(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(path) as first, SQLiteStateStore(path) as second:
        _initialize(first, boundary)
        a = DurableMigrationLifecycleCoordinator(
            first, _migration_registry(), _protected(first, boundary)
        )
        b = DurableMigrationLifecycleCoordinator(
            second, _migration_registry(), _protected(second, boundary)
        )
        a.prepare("migration-1")

        def let_second_win() -> None:
            boundary.before_prepare = None
            b.begin_applying("migration-1")

        boundary.before_prepare = let_second_win
        with pytest.raises(MigrationError, match="protected migration CAS failed"):
            a.begin_applying("migration-1")
        durable = a.discover("migration-1")
        assert durable.current is not None and durable.current["state"] == "APPLYING"
        assert [item["transition_revision"] for item in durable.history] == [1, 2]
        generation = first.read_metadata().protected_freshness_generation
        assert a.begin_applying("migration-1") == durable
        assert first.read_metadata().protected_freshness_generation == generation


def test_two_store_conflicting_transition_rejects_stale_loser(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(path) as first, SQLiteStateStore(path) as second:
        _initialize(first, boundary)
        a = DurableMigrationLifecycleCoordinator(
            first, _migration_registry(), _protected(first, boundary)
        )
        b = DurableMigrationLifecycleCoordinator(
            second, _migration_registry(), _protected(second, boundary)
        )
        a.prepare("migration-1")

        def let_applying_win() -> None:
            boundary.before_prepare = None
            b.begin_applying("migration-1")

        boundary.before_prepare = let_applying_win
        with pytest.raises(MigrationError, match="protected migration CAS failed"):
            a.fail("migration-1")
        durable = b.discover("migration-1")
        assert durable.current is not None and durable.current["state"] == "APPLYING"
        assert len(durable.history) == 2


@pytest.mark.parametrize("kind", ["migration", "handoff"])
def test_duplicate_recovers_local_durable_external_prepared(tmp_path: Path, kind: str) -> None:
    path = tmp_path / f"{kind}.db"
    boundary = Boundary(record("UNINITIALIZED"))
    with SQLiteStateStore(path) as store:
        _initialize(store, boundary)
        boundary.fail_finalize = True
        if kind == "migration":
            coordinator = DurableMigrationLifecycleCoordinator(
                store, _migration_registry(), _protected(store, boundary)
            )
            with pytest.raises(MigrationError, match="protected migration CAS failed"):
                coordinator.prepare("migration-1")
        else:
            coordinator = DurableSecretHandoffLifecycleCoordinator(
                store, _protected(store, boundary)
            )
            with pytest.raises(SecretHandoffError, match="protected handoff CAS failed"):
                coordinator.prepare(_descriptor())
        generation = store.read_metadata().protected_freshness_generation
        assert boundary.value["lifecycle"] == "PREPARED"
    boundary.fail_finalize = False
    with SQLiteStateStore(path) as reopened:
        if kind == "migration":
            recovered = DurableMigrationLifecycleCoordinator(
                reopened, _migration_registry(), _protected(reopened, boundary)
            ).prepare("migration-1")
        else:
            recovered = DurableSecretHandoffLifecycleCoordinator(
                reopened, _protected(reopened, boundary)
            ).prepare(_descriptor())
        assert len(recovered.history) == 1
        assert reopened.read_metadata().protected_freshness_generation == generation
        assert boundary.value["lifecycle"] == "COMMITTED"


def test_duplicate_fails_closed_when_external_current_mismatches(
    tmp_path: Path,
) -> None:
    with SQLiteStateStore(tmp_path / "state.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        _initialize(store, boundary)
        coordinator = DurableMigrationLifecycleCoordinator(
            store, _migration_registry(), _protected(store, boundary)
        )
        coordinator.prepare("migration-1")
        generation = store.read_metadata().protected_freshness_generation
        boundary.value = record(
            "COMMITTED",
            revision=boundary.value["authority_revision"] + 1,
            committed_generation=generation,
            committed_state_fingerprint_sha256="f" * 64,
        )
        with pytest.raises(MigrationError, match="duplicate recovery failed"):
            coordinator.prepare("migration-1")
        assert store.read_metadata().protected_freshness_generation == generation
        assert len(coordinator.discover("migration-1").history) == 1


def test_prepare_rejects_verified_target_schema_before_external_prepare(
    tmp_path: Path,
) -> None:
    with SQLiteStateStore(tmp_path / "state.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        _initialize(store, boundary, schema=2)
        generation = store.read_metadata().protected_freshness_generation
        calls = tuple(boundary.calls)
        coordinator = DurableMigrationLifecycleCoordinator(
            store, _migration_registry(), _protected(store, boundary)
        )
        with pytest.raises(MigrationError, match="PREPARED requires verified StateStore schema 1"):
            coordinator.prepare("migration-1")
        assert tuple(boundary.calls) == calls
        assert store.read_metadata().protected_freshness_generation == generation


def test_begin_applying_rejects_non_source_schema_before_external_prepare(
    tmp_path: Path,
) -> None:
    with SQLiteStateStore(tmp_path / "state.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        _initialize_with_migration(store, boundary, ("PREPARED",), schema=2)
        generation = store.read_metadata().protected_freshness_generation
        calls = tuple(boundary.calls)
        coordinator = DurableMigrationLifecycleCoordinator(
            store, _migration_registry(), _protected(store, boundary)
        )
        with pytest.raises(MigrationError, match="APPLYING requires verified StateStore schema 1"):
            coordinator.begin_applying("migration-1")
        assert tuple(boundary.calls) == calls
        assert store.read_metadata().protected_freshness_generation == generation


def test_complete_rejects_non_target_verified_schema(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "state.db") as store:
        boundary = Boundary(record("UNINITIALIZED"))
        _initialize_with_migration(
            store, boundary, ("PREPARED", "APPLYING", "DURABLE_MIGRATED"), schema=1
        )
        generation = store.read_metadata().protected_freshness_generation
        calls = tuple(boundary.calls)
        coordinator = DurableMigrationLifecycleCoordinator(
            store, _migration_registry(), _protected(store, boundary)
        )
        with pytest.raises(MigrationError, match="COMPLETED requires verified StateStore schema 2"):
            coordinator.complete("migration-1")
        assert tuple(boundary.calls) == calls
        assert store.read_metadata().protected_freshness_generation == generation
