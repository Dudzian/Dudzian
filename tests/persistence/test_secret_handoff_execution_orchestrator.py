from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import inspect
from pathlib import Path
from threading import Barrier, Event, Lock

import pytest

from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.protected_freshness_handoff import ProtectedFreshnessHandoffCoordinator
from bot_core.persistence.secret_handoff import (
    DurableSecretHandoffExecutionCoordinator,
    DurableSecretHandoffLifecycleCoordinator,
    ExternalOutcome,
    SecretHandoffError,
    SecretHandoffRecord,
)
from bot_core.persistence.state_store import SQLiteStateStore
from tests.persistence.test_protected_freshness_handoff import Boundary, record
from tests.persistence.test_secret_handoff import descriptor
from tests.persistence.test_state_store_records import _account, _metadata


class Port:
    """Thread-safe, stateful model of exact-once semantic external effects."""

    def __init__(
        self,
        begin: ExternalOutcome = ExternalOutcome.COMMITTED,
        reconcile: ExternalOutcome = ExternalOutcome.COMMITTED,
    ) -> None:
        self.begin_outcome = begin
        self.reconcile_outcome = reconcile
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.initial_effects: set[str] = set()
        self.cleanup_effects: set[str] = set()
        self.begin_effect_then_raise = False
        self.raise_reconcile = False
        self.cleanup_failure_before_effect = False
        self.cleanup_effect_then_raise = False
        self.before_action = None
        self._lock = Lock()

    def _record(self, action: str, value: SecretHandoffRecord) -> None:
        if self.before_action is not None:
            self.before_action(action)
        self.calls.append((action, value.to_mapping()))

    def begin(self, value: SecretHandoffRecord) -> ExternalOutcome:
        with self._lock:
            self._record("begin", value)
            if self.begin_effect_then_raise:
                self.initial_effects.add(value.handoff_id)
                raise RuntimeError("begin acknowledgement lost")
            if self.begin_outcome is ExternalOutcome.COMMITTED:
                self.initial_effects.add(value.handoff_id)
            return self.begin_outcome

    def reconcile(self, value: SecretHandoffRecord) -> ExternalOutcome:
        with self._lock:
            self._record("reconcile", value)
            if self.raise_reconcile:
                raise RuntimeError("reconcile unavailable")
            if value.handoff_id in self.initial_effects:
                return ExternalOutcome.COMMITTED
            return self.reconcile_outcome

    def cleanup(self, value: SecretHandoffRecord) -> None:
        with self._lock:
            self._record("cleanup", value)
            if self.cleanup_failure_before_effect:
                raise RuntimeError("cleanup failed before effect")
            self.cleanup_effects.add(value.handoff_id)
            if self.cleanup_effect_then_raise:
                raise RuntimeError("cleanup acknowledgement lost")

    def count(self, action: str) -> int:
        return sum(name == action for name, _ in self.calls)


def setup(path: Path, boundary: Boundary, port: Port):
    store = SQLiteStateStore(path)
    protected = ProtectedFreshnessHandoffCoordinator(
        store, LocalDurableEvidenceRegistry(), boundary
    )
    lifecycle = DurableSecretHandoffLifecycleCoordinator(store, protected)
    execution = DurableSecretHandoffExecutionCoordinator(store, lifecycle, protected, port)
    return store, lifecycle, execution


def initialize(store: SQLiteStateStore, boundary: Boundary) -> None:
    ProtectedFreshnessHandoffCoordinator(
        store, LocalDurableEvidenceRegistry(), boundary
    ).advance_protected_state(_metadata(), current_records=(_account(),))


def handoff(**changes):  # type: ignore[no-untyped-def]
    return descriptor(scope=(_metadata().account_id, _metadata().device_installation_id), **changes)


def state(value) -> str:  # type: ignore[no-untyped-def]
    assert value.current is not None
    return str(value.current["state"])


def secret_descriptors(snapshot):  # type: ignore[no-untyped-def]
    return tuple(
        item
        for item in snapshot.immutable_history
        if item.representation_name == "SecretHandoff immutable descriptor"
    )


def test_fresh_success_has_exact_structural_durable_evidence(tmp_path: Path) -> None:
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port()
    store, lifecycle, execution = setup(tmp_path / "state.db", boundary, port)
    with store:
        initialize(store, boundary)
        before = store.read_verified_snapshot()
        assert before is not None
        result = execution.start(handoff())
        after = store.read_verified_snapshot()
        assert after is not None and state(result) == "CLEANUP_PENDING"
        assert after.metadata.protected_freshness_generation == (
            before.metadata.protected_freshness_generation + 3
        )
        assert len(after.transaction_descriptors) == len(before.transaction_descriptors) + 3
        prepared, committed, cleanup = after.transaction_descriptors[-3:]
        assert [item["transition_revision"] for item in result.history] == [1, 2, 3]
        assert len(secret_descriptors(after)) == 1
        durable_descriptor = secret_descriptors(after)[0]
        assert durable_descriptor in prepared.immutable_history_appends
        assert durable_descriptor not in committed.immutable_history_appends
        assert durable_descriptor not in cleanup.immutable_history_appends
        for transition in (committed, cleanup):
            assert len(transition.immutable_history_appends) == 1
            assert transition.immutable_history_appends[0].representation_name == (
                "SecretHandoff transition/history revisions"
            )
            assert len(transition.current_record_mutations) == 1
            assert transition.current_record_mutations[0].representation_name == (
                "SecretHandoff current state/designation"
            )
        assert [name for name, _ in port.calls] == ["begin", "cleanup"]
        snapshot = store.read_verified_snapshot()
        assert execution.resume("handoff-1") == lifecycle.discover("handoff-1")
        assert store.read_verified_snapshot() == snapshot
        assert [name for name, _ in port.calls] == ["begin", "cleanup"]


def test_no_raw_secret_payload_is_durable(tmp_path: Path) -> None:
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port()
    store, _, execution = setup(tmp_path / "state.db", boundary, port)
    with store:
        initialize(store, boundary)
        execution.start(handoff())
        snapshot = store.read_verified_snapshot()
        assert snapshot is not None
        descriptor_carrier = secret_descriptors(snapshot)[0]
        assert set(descriptor_carrier.payload) == {
            "handoff_id",
            "scope",
            "operation",
            "old_reference",
            "new_reference",
            "metadata_fingerprint_sha256",
            "operation_fingerprint_sha256",
            "reconciliation_metadata",
        }
        assert descriptor_carrier.payload["old_reference"] == "ref:old"
        assert descriptor_carrier.payload["new_reference"] == "ref:new"
        secret_records = tuple(
            item
            for item in (*snapshot.current_records, *snapshot.immutable_history)
            if item.representation_name.startswith("SecretHandoff")
        )
        assert all("secret_payload" not in item.payload for item in secret_records)
        for transaction in snapshot.transaction_descriptors:
            for item in (
                *transaction.current_record_mutations,
                *transaction.immutable_history_appends,
            ):
                if item.representation_name.startswith("SecretHandoff"):
                    assert "secret_payload" not in item.payload


def test_unknown_terminal_same_process_and_true_reopen_are_zero_mutation(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port(begin=ExternalOutcome.UNRESOLVED)
    store, _, execution = setup(path, boundary, port)
    with store:
        initialize(store, boundary)
        result = execution.start(handoff())
        before = store.read_verified_snapshot()
        calls = tuple(port.calls)
        assert state(result) == "UNKNOWN_RECONCILIATION"
        assert execution.resume("handoff-1") == result
        assert store.read_verified_snapshot() == before and tuple(port.calls) == calls
    reopened, lifecycle, resumed = setup(path, boundary, port)
    with reopened:
        assert state(resumed.resume("handoff-1")) == "UNKNOWN_RECONCILIATION"
        assert reopened.read_verified_snapshot() == before
        assert lifecycle.discover("handoff-1").history == result.history
    assert tuple(port.calls) == calls


def test_cleanup_pending_true_reopen_is_zero_mutation(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port()
    store, _, execution = setup(path, boundary, port)
    with store:
        initialize(store, boundary)
        assert state(execution.start(handoff())) == "CLEANUP_PENDING"
        before = store.read_verified_snapshot()
        calls = tuple(port.calls)
    reopened, _, resumed = setup(path, boundary, port)
    with reopened:
        assert state(resumed.resume("handoff-1")) == "CLEANUP_PENDING"
        assert reopened.read_verified_snapshot() == before
    assert tuple(port.calls) == calls


@pytest.mark.parametrize("entry", ["start", "resume"])
def test_not_started_remains_prepared_and_never_blind_begins(tmp_path: Path, entry: str) -> None:
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port(begin=ExternalOutcome.NOT_STARTED, reconcile=ExternalOutcome.NOT_STARTED)
    store, lifecycle, execution = setup(tmp_path / "state.db", boundary, port)
    with store:
        initialize(store, boundary)
        with pytest.raises(SecretHandoffError, match="did not start"):
            execution.start(handoff())
        expected = "explicit recovery" if entry == "start" else "blind retry"
        with pytest.raises(SecretHandoffError, match=expected):
            getattr(execution, entry)(handoff() if entry == "start" else "handoff-1")
        assert state(lifecycle.discover("handoff-1")) == "PREPARED"
        assert port.count("begin") == 1


def test_prepared_resume_reconciles_only(tmp_path: Path) -> None:
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port(begin=ExternalOutcome.NOT_STARTED)
    store, _, execution = setup(tmp_path / "state.db", boundary, port)
    with store:
        initialize(store, boundary)
        with pytest.raises(SecretHandoffError):
            execution.start(handoff())
        assert state(execution.resume("handoff-1")) == "CLEANUP_PENDING"
    assert [name for name, _ in port.calls] == ["begin", "reconcile", "cleanup"]


def test_begin_effect_then_ack_loss_restarts_with_reconcile_only(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port()
    port.begin_effect_then_raise = True
    store, lifecycle, execution = setup(path, boundary, port)
    with store:
        initialize(store, boundary)
        with pytest.raises(RuntimeError, match="acknowledgement"):
            execution.start(handoff())
        assert state(lifecycle.discover("handoff-1")) == "PREPARED"
    port.begin_effect_then_raise = False
    reopened, lifecycle, resumed = setup(path, boundary, port)
    with reopened:
        result = resumed.resume("handoff-1")
        assert state(result) == "CLEANUP_PENDING"
        assert [item["state"] for item in lifecycle.discover("handoff-1").history].count(
            "COMMITTED"
        ) == 1
    assert port.count("begin") == 1
    assert port.count("reconcile") == 1
    assert port.initial_effects == {"handoff-1"}


def test_reconcile_exception_is_retryable_without_begin_replay(tmp_path: Path) -> None:
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port(begin=ExternalOutcome.NOT_STARTED)
    store, lifecycle, execution = setup(tmp_path / "state.db", boundary, port)
    with store:
        initialize(store, boundary)
        with pytest.raises(SecretHandoffError):
            execution.start(handoff())
        generation = store.read_metadata().protected_freshness_generation
        port.raise_reconcile = True
        with pytest.raises(RuntimeError, match="unavailable"):
            execution.resume("handoff-1")
        assert state(lifecycle.discover("handoff-1")) == "PREPARED"
        assert store.read_metadata().protected_freshness_generation == generation
        port.raise_reconcile = False
        assert state(execution.resume("handoff-1")) == "CLEANUP_PENDING"
    assert port.count("begin") == 1 and port.count("reconcile") == 2


def test_cleanup_failure_before_effect_retries_after_reopen(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port()
    port.cleanup_failure_before_effect = True
    store, lifecycle, execution = setup(path, boundary, port)
    with store:
        initialize(store, boundary)
        with pytest.raises(RuntimeError, match="before effect"):
            execution.start(handoff())
        assert port.cleanup_effects == set()
        assert state(lifecycle.discover("handoff-1")) == "COMMITTED"
    port.cleanup_failure_before_effect = False
    reopened, lifecycle, resumed = setup(path, boundary, port)
    with reopened:
        assert state(resumed.resume("handoff-1")) == "CLEANUP_PENDING"
        assert [item["state"] for item in lifecycle.discover("handoff-1").history].count(
            "CLEANUP_PENDING"
        ) == 1
    assert port.count("cleanup") == 2 and port.cleanup_effects == {"handoff-1"}


def test_cleanup_effect_then_ack_loss_redelivers_exactly(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port()
    port.cleanup_effect_then_raise = True
    store, lifecycle, execution = setup(path, boundary, port)
    with store:
        initialize(store, boundary)
        with pytest.raises(RuntimeError, match="acknowledgement"):
            execution.start(handoff())
        assert state(lifecycle.discover("handoff-1")) == "COMMITTED"
        assert port.cleanup_effects == {"handoff-1"}
    port.cleanup_effect_then_raise = False
    reopened, lifecycle, resumed = setup(path, boundary, port)
    with reopened:
        result = resumed.resume("handoff-1")
        assert state(result) == "CLEANUP_PENDING"
        assert "UNKNOWN_RECONCILIATION" not in [item["state"] for item in result.history]
        assert [item["state"] for item in result.history].count("CLEANUP_PENDING") == 1
    deliveries = [value for name, value in port.calls if name == "cleanup"]
    assert deliveries == [handoff().to_mapping(), handoff().to_mapping()]
    assert port.cleanup_effects == {"handoff-1"}


class FailCleanupMarkOnce:
    def __init__(self, lifecycle: DurableSecretHandoffLifecycleCoordinator) -> None:
        self.lifecycle = lifecycle
        self.failed = False

    def __getattr__(self, name: str):
        return getattr(self.lifecycle, name)

    def mark_cleanup_pending(self, handoff_id: str):
        if not self.failed:
            self.failed = True
            raise RuntimeError("crash before cleanup transition")
        return self.lifecycle.mark_cleanup_pending(handoff_id)


def test_cleanup_effect_then_crash_before_local_transition(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port()
    store, lifecycle, _ = setup(path, boundary, port)
    with store:
        initialize(store, boundary)
        protected = ProtectedFreshnessHandoffCoordinator(
            store, LocalDurableEvidenceRegistry(), boundary
        )
        execution = DurableSecretHandoffExecutionCoordinator(
            store,
            FailCleanupMarkOnce(lifecycle),
            protected,
            port,  # type: ignore[arg-type]
        )
        with pytest.raises(RuntimeError, match="crash before"):
            execution.start(handoff())
        assert state(lifecycle.discover("handoff-1")) == "COMMITTED"
        assert port.cleanup_effects == {"handoff-1"}
    reopened, lifecycle, resumed = setup(path, boundary, port)
    with reopened:
        result = resumed.resume("handoff-1")
        assert state(result) == "CLEANUP_PENDING"
        assert [item["state"] for item in result.history].count("CLEANUP_PENDING") == 1
    assert port.count("cleanup") == 2 and port.cleanup_effects == {"handoff-1"}


def _leave_protected_window(
    lifecycle: DurableSecretHandoffLifecycleCoordinator,
    boundary: Boundary,
    durable_state: str,
    ack_lost: bool,
) -> None:
    def fail(operation):  # type: ignore[no-untyped-def]
        boundary.ack_lost = ack_lost
        boundary.fail_finalize = not ack_lost
        with pytest.raises(SecretHandoffError, match="protected handoff CAS failed"):
            operation()
        boundary.ack_lost = False
        boundary.fail_finalize = False

    if durable_state == "PREPARED":
        fail(lambda: lifecycle.prepare(handoff()))
        return
    lifecycle.prepare(handoff())
    if durable_state == "COMMITTED":
        fail(lambda: lifecycle.record_external_outcome("handoff-1", ExternalOutcome.COMMITTED))
        return
    lifecycle.record_external_outcome("handoff-1", ExternalOutcome.COMMITTED)
    fail(lambda: lifecycle.mark_cleanup_pending("handoff-1"))


@pytest.mark.parametrize("durable_state", ["PREPARED", "COMMITTED", "CLEANUP_PENDING"])
@pytest.mark.parametrize("ack_lost", [False, True], ids=["external_prepared", "finalize_ack_lost"])
def test_protected_m03_restart_windows_recover_before_next_action(
    tmp_path: Path, durable_state: str, ack_lost: bool
) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port()
    store, lifecycle, _ = setup(path, boundary, port)
    with store:
        initialize(store, boundary)
        _leave_protected_window(lifecycle, boundary, durable_state, ack_lost)
        durable = lifecycle.discover("handoff-1")
        generation = store.read_metadata().protected_freshness_generation
        assert state(durable) == durable_state
        assert (
            len(durable.history)
            == {"PREPARED": 1, "COMMITTED": 2, "CLEANUP_PENDING": 3}[durable_state]
        )
        assert boundary.value["lifecycle"] == ("COMMITTED" if ack_lost else "PREPARED")
    port.before_action = lambda _action: (
        pytest.fail("port action before protected recovery")
        if (boundary.value["lifecycle"] != "COMMITTED")
        else None
    )
    reopened, lifecycle, resumed = setup(path, boundary, port)
    with reopened:
        result = resumed.resume("handoff-1")
        assert state(result) == "CLEANUP_PENDING"
        expected_advance = {"PREPARED": 2, "COMMITTED": 1, "CLEANUP_PENDING": 0}[durable_state]
        assert (
            reopened.read_metadata().protected_freshness_generation == generation + expected_advance
        )
        states = [item["state"] for item in lifecycle.discover("handoff-1").history]
        assert states == ["PREPARED", "COMMITTED", "CLEANUP_PENDING"]
    if durable_state == "PREPARED":
        assert port.count("begin") == 0 and port.count("reconcile") == 1
    elif durable_state == "COMMITTED":
        assert port.count("cleanup") == 1
    else:
        assert port.calls == []


def test_conflicting_descriptor_preserves_all_durable_invariants(tmp_path: Path) -> None:
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port(begin=ExternalOutcome.NOT_STARTED)
    store, lifecycle, execution = setup(tmp_path / "state.db", boundary, port)
    with store:
        initialize(store, boundary)
        with pytest.raises(SecretHandoffError):
            execution.start(handoff())
        before = store.read_verified_snapshot()
        durable = lifecycle.discover("handoff-1")
        calls = tuple(port.calls)
        with pytest.raises(SecretHandoffError, match="conflict"):
            execution.start(handoff(new_reference="ref:conflict"))
        assert store.read_verified_snapshot() == before
        assert lifecycle.discover("handoff-1") == durable
        assert tuple(port.calls) == calls


def test_production_orchestrator_api_and_authority_boundary() -> None:
    public = {
        name
        for name, member in DurableSecretHandoffExecutionCoordinator.__dict__.items()
        if callable(member) and not name.startswith("_")
    }
    assert public == {"start", "resume"}
    assert tuple(inspect.signature(DurableSecretHandoffExecutionCoordinator.start).parameters) == (
        "self",
        "descriptor",
    )
    assert tuple(inspect.signature(DurableSecretHandoffExecutionCoordinator.resume).parameters) == (
        "self",
        "handoff_id",
    )
    tree = ast.parse(inspect.getsource(DurableSecretHandoffExecutionCoordinator))
    forbidden = {
        "SecretHandoffCoordinator",
        "_attempted",
        "first_dispatch",
        "desired_state",
        "handoff_transition",
        "handoff_current",
        "handoff_transition_carrier",
        "handoff_current_carrier",
        "commit_prepared_state",
        "prepare",
        "finalize",
        "publish_verified_state",
    }
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert forbidden.isdisjoint(names | attributes)


def test_two_connection_fresh_start_exact_cardinality(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port()
    initial, _, _ = setup(path, boundary, port)
    with initial:
        initialize(initial, boundary)
        before = initial.read_verified_snapshot()
    assert before is not None
    gate = Barrier(2)

    def run(_):  # type: ignore[no-untyped-def]
        store, _, execution = setup(path, boundary, port)
        with store:
            gate.wait()
            try:
                return execution.start(handoff())
            except SecretHandoffError:
                return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, range(2)))
    reopened, lifecycle, execution = setup(path, boundary, port)
    with reopened:
        final = execution.resume("handoff-1")
        snapshot = reopened.read_verified_snapshot()
        assert snapshot is not None and state(final) == "CLEANUP_PENDING"
        assert [item["state"] for item in final.history] == [
            "PREPARED",
            "COMMITTED",
            "CLEANUP_PENDING",
        ]
        assert final.current["current_transition_revision"] == 3
        assert len(secret_descriptors(snapshot)) == 1
        assert snapshot.metadata.protected_freshness_generation == (
            before.metadata.protected_freshness_generation + 3
        )
        assert len(snapshot.transaction_descriptors) == len(before.transaction_descriptors) + 3
        assert port.count("begin") == 1 and sum(item is not None for item in results) >= 1
        calls = tuple(port.calls)
        assert execution.resume("handoff-1") == final
        assert reopened.read_verified_snapshot() == snapshot and tuple(port.calls) == calls


def test_duplicate_start_cannot_take_fresh_prepared_ownership(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port()
    initial, _, _ = setup(path, boundary, port)
    with initial:
        initialize(initial, boundary)

    prepared = Event()
    release_creator = Event()

    class PausingLifecycle:
        def __init__(self, lifecycle):  # type: ignore[no-untyped-def]
            self.lifecycle = lifecycle

        def prepare_with_disposition(self, value):  # type: ignore[no-untyped-def]
            result = self.lifecycle.prepare_with_disposition(value)
            prepared.set()
            assert release_creator.wait(timeout=10)
            return result

        def __getattr__(self, name):  # type: ignore[no-untyped-def]
            return getattr(self.lifecycle, name)

    def run_creator():  # type: ignore[no-untyped-def]
        creator_store = SQLiteStateStore(path)
        creator_protected = ProtectedFreshnessHandoffCoordinator(
            creator_store, LocalDurableEvidenceRegistry(), boundary
        )
        creator_lifecycle = DurableSecretHandoffLifecycleCoordinator(
            creator_store, creator_protected
        )
        creator = DurableSecretHandoffExecutionCoordinator(
            creator_store,
            PausingLifecycle(creator_lifecycle),  # type: ignore[arg-type]
            creator_protected,
            port,
        )
        with creator_store:
            return creator.start(handoff())

    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_creator)
        assert prepared.wait(timeout=10)
        duplicate_store, _, duplicate = setup(path, boundary, port)
        with duplicate_store:
            with pytest.raises(SecretHandoffError, match="explicit recovery is required"):
                duplicate.start(handoff())
        assert port.calls == []
        release_creator.set()
        result = future.result(timeout=10)

    assert state(result) == "CLEANUP_PENDING"
    assert [item["state"] for item in result.history] == [
        "PREPARED",
        "COMMITTED",
        "CLEANUP_PENDING",
    ]
    assert result.current["current_transition_revision"] == 3
    assert port.count("begin") == 1


def test_two_connection_prepared_resume_exact_cardinality(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port(begin=ExternalOutcome.NOT_STARTED)
    initial, _, execution = setup(path, boundary, port)
    with initial:
        initialize(initial, boundary)
        with pytest.raises(SecretHandoffError):
            execution.start(handoff())
    port.begin_outcome = ExternalOutcome.COMMITTED
    gate = Barrier(2)

    def run(_):  # type: ignore[no-untyped-def]
        store, _, resumed = setup(path, boundary, port)
        with store:
            gate.wait()
            try:
                return resumed.resume("handoff-1")
            except SecretHandoffError:
                return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, range(2)))
    reopened, lifecycle, resumed = setup(path, boundary, port)
    with reopened:
        final = resumed.resume("handoff-1")
        snapshot = reopened.read_verified_snapshot()
        assert snapshot is not None
        assert [item["state"] for item in final.history] == [
            "PREPARED",
            "COMMITTED",
            "CLEANUP_PENDING",
        ]
        assert len(secret_descriptors(snapshot)) == 1
        assert port.count("begin") == 1 and port.cleanup_effects == {"handoff-1"}
        calls = tuple(port.calls)
        assert resumed.resume("handoff-1") == lifecycle.discover("handoff-1")
        assert reopened.read_verified_snapshot() == snapshot and tuple(port.calls) == calls


def test_stale_coordinator_uses_fresh_terminal_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    boundary = Boundary(record("UNINITIALIZED"))
    port = Port()
    first, _, execution = setup(path, boundary, port)
    second, _, stale = setup(path, boundary, port)
    with first, second:
        initialize(first, boundary)
        assert state(execution.start(handoff())) == "CLEANUP_PENDING"
        before = second.read_verified_snapshot()
        calls = tuple(port.calls)
        assert state(stale.resume("handoff-1")) == "CLEANUP_PENDING"
        assert second.read_verified_snapshot() == before
        assert tuple(port.calls) == calls
