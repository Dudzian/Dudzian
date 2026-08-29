from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.persistence.local_durable_evidence import LocalDurableEvidenceRegistry
from bot_core.persistence.protected_freshness_handoff import (
    ProtectedFreshnessHandoffCoordinator,
    ProtectedFreshnessHandoffError,
)
from bot_core.persistence.state_store import SQLiteStateStore
from tests.persistence.test_protected_freshness_handoff import Boundary, SCOPE, record
from tests.persistence.test_state_store_records import _commit, _metadata

SCOPE_B = (
    "acct_01890f4c-7b9a-7cc1-8a2b-123456789abd",
    "dev_01890f4c-7b9a-7cc1-8a2b-123456789abd",
    "9" * 64,
)


class RecoveryBoundary(Boundary):
    def __init__(self, initial: dict[str, Any]) -> None:
        super().__init__(initial)
        self.before_abort = None
        self.before_finalize = None
        self.abort_ack_lost = False
        self.finalize_ack_lost = False
        self.after_abort_effect = None
        self.after_finalize_effect = None
        self.supplied_refs: list[object] = []

    def abort(self, current_ref, scope, *, evidence_ref, evidence_resolver):  # type: ignore[no-untyped-def]
        self.supplied_refs.append(evidence_ref)
        if self.before_abort:
            self.before_abort()
        super().abort(
            current_ref,
            scope,
            evidence_ref=evidence_ref,
            evidence_resolver=evidence_resolver,
        )
        if self.after_abort_effect:
            self.after_abort_effect()
        if self.abort_ack_lost:
            raise RuntimeError("abort ack lost")

    def finalize(self, current_ref, scope, *, evidence_ref, evidence_resolver):  # type: ignore[no-untyped-def]
        self.supplied_refs.append(evidence_ref)
        if self.before_finalize:
            self.before_finalize()
        super().finalize(
            current_ref,
            scope,
            evidence_ref=evidence_ref,
            evidence_resolver=evidence_resolver,
        )
        if self.after_finalize_effect:
            self.after_finalize_effect()
        if self.finalize_ack_lost:
            raise RuntimeError("finalize ack lost")


def prepared(local, *, committed=None, committed_state=None, **changes):  # type: ignore[no-untyped-def]
    fields = {
        "committed_generation": committed,
        "committed_state_fingerprint_sha256": committed_state,
        "prepared_generation": local.protected_freshness_generation,
        "prepared_state_fingerprint_sha256": local.state_fingerprint_sha256,
        "prepared_transaction_fingerprint_sha256": local.transaction_fingerprint_sha256,
    }
    fields.update(changes)
    return record("PREPARED", **fields)


def coordinator(store, boundary, registry=None):  # type: ignore[no-untyped-def]
    return ProtectedFreshnessHandoffCoordinator(
        store, registry or LocalDurableEvidenceRegistry(), boundary
    )


def no_business_mutation(monkeypatch, store):  # type: ignore[no-untyped-def]
    def denied(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("recovery attempted a local business transition")

    monkeypatch.setattr(store, "derive_prepared_metadata", denied)
    monkeypatch.setattr(store, "commit_prepared_state", denied)


def assert_ready(recovery):  # type: ignore[no-untyped-def]
    assert recovery._recovery_required is False
    assert recovery._recovery_scope is None


def assert_recovery(recovery, scope=SCOPE):  # type: ignore[no-untyped-def]
    assert recovery._recovery_required is True
    assert recovery._recovery_scope == scope


def test_r1_uninitialized_empty_is_stable_without_action(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "r1.db") as store:
        boundary = RecoveryBoundary(record("UNINITIALIZED"))
        recovery = coordinator(store, boundary)
        assert recovery.recover_protected_state(SCOPE) is None
        assert not {"prepare", "finalize", "abort"} & set(boundary.calls)
        assert_ready(recovery)


@pytest.mark.parametrize("generation", [1, 2])
def test_r2_r3_committed_exact_is_stable_without_evidence_or_action(
    tmp_path: Path, generation: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SQLiteStateStore(tmp_path / f"r{generation}.db") as store:
        local = _commit(store, _metadata())
        if generation == 2:
            local = _commit(store, _metadata(2), expected=1)
        registry = LocalDurableEvidenceRegistry()
        boundary = RecoveryBoundary(
            record(
                "COMMITTED",
                committed_generation=generation,
                committed_state_fingerprint_sha256=local.state_fingerprint_sha256,
            )
        )
        no_business_mutation(monkeypatch, store)
        result = coordinator(store, boundary, registry).recover_protected_state(SCOPE)
        assert result == local and registry.resolve_current(SCOPE, object()) is None
        assert not {"prepare", "finalize", "abort"} & set(boundary.calls)


def test_uninitialized_empty_becoming_g1_fails_stable_recovery_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "stable-empty-race.db"
    with SQLiteStateStore(path) as store, SQLiteStateStore(path) as writer:
        boundary = RecoveryBoundary(record("UNINITIALIZED"))
        original_read = store.read_verified_snapshot
        reads = 0

        def racing_read():  # type: ignore[no-untyped-def]
            nonlocal reads
            reads += 1
            if reads == 2:
                _commit(writer, _metadata())
            return original_read()

        monkeypatch.setattr(store, "read_verified_snapshot", racing_read)
        recovery = coordinator(store, boundary)
        with pytest.raises(ProtectedFreshnessHandoffError, match="changed during"):
            recovery.recover_protected_state(SCOPE)
        snapshot = store.read_verified_snapshot()
        assert snapshot is not None and snapshot.metadata.protected_freshness_generation == 1
        assert boundary.value["lifecycle"] == "UNINITIALIZED"
        assert not {"prepare", "finalize", "abort"} & set(boundary.calls)
        assert_recovery(recovery)


def test_committed_g1_becoming_local_g2_fails_stable_recovery_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "stable-committed-race.db"
    with SQLiteStateStore(path) as store:
        one = _commit(store, _metadata())
    with SQLiteStateStore(path) as store, SQLiteStateStore(path) as writer:
        boundary = RecoveryBoundary(
            record(
                "COMMITTED",
                committed_generation=1,
                committed_state_fingerprint_sha256=one.state_fingerprint_sha256,
            )
        )
        original_read = store.read_verified_snapshot
        reads = 0

        def racing_read():  # type: ignore[no-untyped-def]
            nonlocal reads
            reads += 1
            if reads == 2:
                _commit(writer, _metadata(2), expected=1)
            return original_read()

        monkeypatch.setattr(store, "read_verified_snapshot", racing_read)
        recovery = coordinator(store, boundary)
        with pytest.raises(ProtectedFreshnessHandoffError, match="changed during"):
            recovery.recover_protected_state(SCOPE)
        snapshot = store.read_verified_snapshot()
        assert snapshot is not None and snapshot.metadata.protected_freshness_generation == 2
        assert boundary.value["lifecycle"] == "COMMITTED"
        assert boundary.value["committed_generation"] == 1
        assert not {"prepare", "finalize", "abort"} & set(boundary.calls)
        assert_recovery(recovery)


def test_r4_normal_prepared_local_g_aborts_once_and_repeated_call_is_stable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SQLiteStateStore(tmp_path / "abort.db") as store:
        one = _commit(store, _metadata())
        candidate = store.derive_prepared_metadata(_metadata(2), expected_current_generation=1)
        boundary = RecoveryBoundary(
            prepared(candidate, committed=1, committed_state=one.state_fingerprint_sha256)
        )
        ref = boundary.ref
        no_business_mutation(monkeypatch, store)
        recovery = coordinator(store, boundary)
        assert recovery.recover_protected_state(SCOPE) == one
        assert boundary.ref is ref and boundary.value["lifecycle"] == "COMMITTED"
        assert boundary.calls.count("abort") == 1 and not recovery._recovery_required
        assert_ready(recovery)
        recovery.recover_protected_state(SCOPE)
        assert boundary.calls.count("abort") == 1


def test_r5_pending_local_candidate_finalizes_without_second_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SQLiteStateStore(tmp_path / "finalize.db") as store:
        one = _commit(store, _metadata())
        two = _commit(store, _metadata(2), expected=1)
        boundary = RecoveryBoundary(
            prepared(two, committed=1, committed_state=one.state_fingerprint_sha256)
        )
        ref = boundary.ref
        no_business_mutation(monkeypatch, store)
        recovery = coordinator(store, boundary)
        assert recovery.recover_protected_state(SCOPE) == two
        assert boundary.ref is ref and boundary.value["lifecycle"] == "COMMITTED"
        assert boundary.calls.count("finalize") == 1
        assert_ready(recovery)
        recovery.recover_protected_state(SCOPE)
        assert boundary.calls.count("finalize") == 1


def test_r6_genesis_pending_empty_retained_and_blocks_ordinary(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "genesis-empty.db") as store:
        boundary = RecoveryBoundary(
            record(
                "PREPARED",
                prepared_generation=1,
                prepared_state_fingerprint_sha256="2" * 64,
                prepared_transaction_fingerprint_sha256="3" * 64,
            )
        )
        recovery = coordinator(store, boundary)
        with pytest.raises(ProtectedFreshnessHandoffError, match="GENESIS_PENDING"):
            recovery.recover_protected_state(SCOPE)
        assert boundary.value["lifecycle"] == "PREPARED"
        assert not {"abort", "finalize", "prepare"} & set(boundary.calls)
        assert_recovery(recovery)
        with pytest.raises(ProtectedFreshnessHandoffError, match="requires recovery"):
            recovery.advance_protected_state(_metadata())


def test_genesis_failed_recovery_cannot_be_cleared_through_other_scope(
    tmp_path: Path,
) -> None:
    pending_a = record(
        "PREPARED",
        prepared_generation=1,
        prepared_state_fingerprint_sha256="2" * 64,
        prepared_transaction_fingerprint_sha256="3" * 64,
    )
    stable_b = record("UNINITIALIZED")
    stable_b.update(
        account_id=SCOPE_B[0],
        device_installation_id=SCOPE_B[1],
        state_store_identity_fingerprint_sha256=SCOPE_B[2],
    )
    stable_b["content_fingerprint_sha256"] = canonical_json_sha256(
        {key: value for key, value in stable_b.items() if key != "content_fingerprint_sha256"}
    )

    class MultiScopeBoundary(RecoveryBoundary):
        def __init__(self) -> None:
            super().__init__(pending_a)
            self.values = {SCOPE: pending_a, SCOPE_B: stable_b}
            self.resolved_scopes = []

        def resolve_current(self, scope):  # type: ignore[no-untyped-def]
            self.resolved_scopes.append(scope)
            value = self.values.get(scope)
            return None if value is None else self.ref, value

    with SQLiteStateStore(tmp_path / "cross-scope.db") as store:
        boundary = MultiScopeBoundary()
        recovery = coordinator(store, boundary)
        with pytest.raises(ProtectedFreshnessHandoffError, match="GENESIS_PENDING"):
            recovery.recover_protected_state(SCOPE)
        assert_recovery(recovery)
        with pytest.raises(ProtectedFreshnessHandoffError, match="scope does not match"):
            recovery.recover_protected_state(SCOPE_B)
        assert boundary.resolved_scopes == [SCOPE]
        assert boundary.values[SCOPE]["lifecycle"] == "PREPARED"
        assert boundary.values[SCOPE_B]["lifecycle"] == "UNINITIALIZED"
        assert_recovery(recovery)
        metadata_b = replace(
            _metadata(),
            account_id=SCOPE_B[0],
            device_installation_id=SCOPE_B[1],
            state_store_identity_fingerprint_sha256=SCOPE_B[2],
        )
        with pytest.raises(ProtectedFreshnessHandoffError, match="requires recovery"):
            recovery.advance_protected_state(metadata_b)
        assert store.read_verified_snapshot() is None


def test_ordinary_prepared_at_entry_binds_scope_and_blocks_other_scope(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "ordinary-pending.db") as store:
        boundary = RecoveryBoundary(
            record(
                "PREPARED",
                prepared_generation=1,
                prepared_state_fingerprint_sha256="2" * 64,
                prepared_transaction_fingerprint_sha256="3" * 64,
            )
        )
        recovery = coordinator(store, boundary)
        with pytest.raises(ProtectedFreshnessHandoffError, match="PREPARED requires recovery"):
            recovery.advance_protected_state(_metadata())
        assert_recovery(recovery)
        metadata_b = replace(
            _metadata(),
            account_id=SCOPE_B[0],
            device_installation_id=SCOPE_B[1],
            state_store_identity_fingerprint_sha256=SCOPE_B[2],
        )
        with pytest.raises(ProtectedFreshnessHandoffError, match="requires recovery"):
            recovery.advance_protected_state(metadata_b)
        assert_recovery(recovery)


def test_r7_genesis_exact_finalizes_and_r8_mismatch_denies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with SQLiteStateStore(tmp_path / "genesis.db") as store:
        one = _commit(store, _metadata())
        good = RecoveryBoundary(prepared(one))
        no_business_mutation(monkeypatch, store)
        assert coordinator(store, good).recover_protected_state(SCOPE) == one
        assert good.calls.count("finalize") == 1 and "abort" not in good.calls
    with SQLiteStateStore(tmp_path / "genesis-bad.db") as store:
        one = _commit(store, _metadata())
        bad = RecoveryBoundary(prepared(one, prepared_state_fingerprint_sha256="f" * 64))
        with pytest.raises(ProtectedFreshnessHandoffError):
            coordinator(store, bad).recover_protected_state(SCOPE)
        assert bad.value["lifecycle"] == "PREPARED" and "abort" not in bad.calls


def test_restart_discards_old_ref_and_rebuilds_evidence_for_local_commit(
    tmp_path: Path,
) -> None:
    with SQLiteStateStore(tmp_path / "restart.db") as store:
        one = _commit(store, _metadata())
        two = _commit(store, _metadata(2), expected=1)
        old_registry = LocalDurableEvidenceRegistry()
        old_ref = old_registry.publish_verified_state(store)
        new_registry = LocalDurableEvidenceRegistry()
        assert new_registry.resolve_current(SCOPE, old_ref) is None
        boundary = RecoveryBoundary(
            prepared(two, committed=1, committed_state=one.state_fingerprint_sha256)
        )
        coordinator(store, boundary, new_registry).recover_protected_state(SCOPE)
        new_ref = boundary.supplied_refs[0]
        assert new_ref is not old_ref
        assert new_registry.resolve_current(SCOPE, old_ref) is None
        assert new_registry.resolve_current(SCOPE, new_ref) is not None


@pytest.mark.parametrize(("external_generation", "local_generation"), [(2, 1), (1, 2)])
def test_external_ahead_and_store_ahead_without_pending_fail_closed(
    tmp_path: Path, external_generation: int, local_generation: int
) -> None:
    with SQLiteStateStore(tmp_path / f"ahead-{external_generation}.db") as store:
        one = _commit(store, _metadata())
        local = one
        if local_generation == 2:
            local = _commit(store, _metadata(2), expected=1)
        boundary = RecoveryBoundary(
            record(
                "COMMITTED",
                committed_generation=external_generation,
                committed_state_fingerprint_sha256=local.state_fingerprint_sha256,
            )
        )
        with pytest.raises(ProtectedFreshnessHandoffError):
            coordinator(store, boundary).recover_protected_state(SCOPE)
        assert not {"prepare", "finalize", "abort"} & set(boundary.calls)


@pytest.mark.parametrize("kind", ["gap", "state", "transaction", "baseline"])
def test_pending_mismatch_and_excessive_gap_fail_closed(tmp_path: Path, kind: str) -> None:
    with SQLiteStateStore(tmp_path / f"{kind}.db") as store:
        one = _commit(store, _metadata())
        two = _commit(store, _metadata(2), expected=1)
        local = two
        if kind == "gap":
            local = _commit(store, _metadata(3), expected=2)
        changes: dict[str, Any] = {}
        committed_state = one.state_fingerprint_sha256
        if kind == "state":
            changes["prepared_state_fingerprint_sha256"] = "f" * 64
        elif kind == "transaction":
            changes["prepared_transaction_fingerprint_sha256"] = "f" * 64
        elif kind == "baseline":
            committed_state = "f" * 64
        boundary = RecoveryBoundary(
            prepared(
                two,
                committed=1,
                committed_state=committed_state,
                **changes,
            )
        )
        with pytest.raises(ProtectedFreshnessHandoffError):
            coordinator(store, boundary).recover_protected_state(SCOPE)
        assert boundary.value["lifecycle"] == "PREPARED"
        assert not {"prepare", "finalize", "abort"} & set(boundary.calls)


def test_abort_same_generation_different_state_is_denied(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "abort-state.db") as store:
        one = _commit(store, _metadata())
        two = store.derive_prepared_metadata(_metadata(2), expected_current_generation=1)
        boundary = RecoveryBoundary(prepared(two, committed=1, committed_state="f" * 64))
        with pytest.raises(ProtectedFreshnessHandoffError):
            coordinator(store, boundary).recover_protected_state(SCOPE)
        assert "abort" not in boundary.calls and one.state_fingerprint_sha256 != "f" * 64


@pytest.mark.parametrize("action", ["abort", "finalize"])
def test_action_effect_then_ack_lost_is_success_without_blind_retry(
    tmp_path: Path, action: str
) -> None:
    with SQLiteStateStore(tmp_path / f"ack-{action}.db") as store:
        one = _commit(store, _metadata())
        two = store.derive_prepared_metadata(_metadata(2), expected_current_generation=1)
        local = one
        if action == "finalize":
            local = _commit(store, _metadata(2), expected=1)
            two = local
        boundary = RecoveryBoundary(
            prepared(two, committed=1, committed_state=one.state_fingerprint_sha256)
        )
        setattr(boundary, f"{action}_ack_lost", True)
        recovery = coordinator(store, boundary)
        assert recovery.recover_protected_state(SCOPE) == local
        assert boundary.calls.count(action) == 1 and not recovery._recovery_required
        assert_ready(recovery)


@pytest.mark.parametrize("action", ["abort", "finalize"])
def test_local_advance_after_external_effect_prevents_false_recovery_success(
    tmp_path: Path, action: str
) -> None:
    with SQLiteStateStore(tmp_path / f"post-effect-{action}.db") as store:
        one = _commit(store, _metadata())
        two = store.derive_prepared_metadata(_metadata(2), expected_current_generation=1)
        boundary = RecoveryBoundary(
            prepared(two, committed=1, committed_state=one.state_fingerprint_sha256)
        )
        if action == "abort":
            boundary.after_abort_effect = lambda: _commit(store, _metadata(2), expected=1)
        else:
            two = _commit(store, _metadata(2), expected=1)
            boundary.value = prepared(
                two, committed=1, committed_state=one.state_fingerprint_sha256
            )
            boundary.after_finalize_effect = lambda: _commit(store, _metadata(3), expected=2)
        recovery = coordinator(store, boundary)
        with pytest.raises(ProtectedFreshnessHandoffError, match="changed during"):
            recovery.recover_protected_state(SCOPE)
        assert boundary.value["lifecycle"] == "COMMITTED"
        expected_local = 2 if action == "abort" else 3
        assert (
            store.read_verified_snapshot().metadata.protected_freshness_generation  # type: ignore[union-attr]
            == expected_local
        )
        assert boundary.calls.count(action) == 1
        assert_recovery(recovery)


@pytest.mark.parametrize(("starting_revision", "terminal_revision"), [(3, 3), (3, 2)])
def test_abort_requires_strict_authority_revision_increase(
    tmp_path: Path, starting_revision: int, terminal_revision: int
) -> None:
    class NonIncreasingAbortBoundary(RecoveryBoundary):
        def abort(self, current_ref, scope, *, evidence_ref, evidence_resolver):  # type: ignore[no-untyped-def]
            super().abort(
                current_ref,
                scope,
                evidence_ref=evidence_ref,
                evidence_resolver=evidence_resolver,
            )
            committed = self.value
            self.value = record(
                "COMMITTED",
                revision=terminal_revision,
                committed_generation=committed["committed_generation"],
                committed_state_fingerprint_sha256=committed["committed_state_fingerprint_sha256"],
            )

    with SQLiteStateStore(tmp_path / f"revision-{terminal_revision}.db") as store:
        one = _commit(store, _metadata())
        two = store.derive_prepared_metadata(_metadata(2), expected_current_generation=1)
        boundary = NonIncreasingAbortBoundary(
            prepared(
                two,
                committed=1,
                committed_state=one.state_fingerprint_sha256,
                revision=starting_revision,
            )
        )
        recovery = coordinator(store, boundary)
        with pytest.raises(ProtectedFreshnessHandoffError, match="terminal state mismatch"):
            recovery.recover_protected_state(SCOPE)
        assert boundary.value["content_fingerprint_sha256"] == canonical_json_sha256(
            {
                key: value
                for key, value in boundary.value.items()
                if key != "content_fingerprint_sha256"
            }
        )
        assert boundary.calls.count("abort") == 1
        assert_recovery(recovery)


@pytest.mark.parametrize("action", ["abort", "finalize"])
def test_store_race_and_same_generation_republish_deny_action(tmp_path: Path, action: str) -> None:
    with SQLiteStateStore(tmp_path / f"race-{action}.db") as store:
        one = _commit(store, _metadata())
        two = store.derive_prepared_metadata(_metadata(2), expected_current_generation=1)
        if action == "finalize":
            two = _commit(store, _metadata(2), expected=1)
        boundary = RecoveryBoundary(
            prepared(two, committed=1, committed_state=one.state_fingerprint_sha256)
        )
        if action == "abort":
            boundary.before_abort = lambda: _commit(store, _metadata(2), expected=1)
        else:
            boundary.before_finalize = lambda: _commit(store, _metadata(3), expected=2)
        recovery = coordinator(store, boundary)
        with pytest.raises(ProtectedFreshnessHandoffError, match="unresolved"):
            recovery.recover_protected_state(SCOPE)
        assert boundary.value["lifecycle"] == "PREPARED"
        assert_recovery(recovery)
        assert boundary.calls.count(action) == 1


def test_same_generation_republish_makes_supplied_ref_stale(tmp_path: Path) -> None:
    with SQLiteStateStore(tmp_path / "republish.db") as store:
        one = _commit(store, _metadata())
        two = store.derive_prepared_metadata(_metadata(2), expected_current_generation=1)
        registry = LocalDurableEvidenceRegistry()
        boundary = RecoveryBoundary(
            prepared(two, committed=1, committed_state=one.state_fingerprint_sha256)
        )
        boundary.before_abort = lambda: registry.publish_verified_state(store)
        with pytest.raises(ProtectedFreshnessHandoffError, match="unresolved"):
            coordinator(store, boundary, registry).recover_protected_state(SCOPE)
        assert boundary.value["lifecycle"] == "PREPARED"


@pytest.mark.parametrize(
    "scope",
    [
        ("bad", SCOPE[1], SCOPE[2]),
        (SCOPE[0], "bad", SCOPE[2]),
        (SCOPE[0], SCOPE[1], "A" * 64),
    ],
)
def test_malformed_scope_fails_before_external_observation(tmp_path: Path, scope) -> None:  # type: ignore[no-untyped-def]
    with SQLiteStateStore(tmp_path / "scope.db") as store:
        boundary = RecoveryBoundary(record("UNINITIALIZED"))
        recovery = coordinator(store, boundary)
        with pytest.raises(ProtectedFreshnessHandoffError, match="scope is malformed"):
            recovery.recover_protected_state(scope)
        assert boundary.calls == []
        assert_ready(recovery)


def test_missing_malformed_and_corrupt_authority_or_store_fail_closed(tmp_path: Path) -> None:
    class Missing(RecoveryBoundary):
        def resolve_current(self, scope):  # type: ignore[no-untyped-def]
            self.calls.append("resolve")
            return None

    with SQLiteStateStore(tmp_path / "missing.db") as store:
        recovery = coordinator(store, Missing(record("UNINITIALIZED")))
        with pytest.raises(ProtectedFreshnessHandoffError, match="missing"):
            recovery.recover_protected_state(SCOPE)
        assert recovery._recovery_required
    with SQLiteStateStore(tmp_path / "malformed.db") as store:
        malformed = record("UNINITIALIZED")
        malformed["content_fingerprint_sha256"] = "f" * 64
        recovery = coordinator(store, RecoveryBoundary(malformed))
        with pytest.raises(ProtectedFreshnessHandoffError):
            recovery.recover_protected_state(SCOPE)
        assert recovery._recovery_required
    with SQLiteStateStore(tmp_path / "corrupt.db") as store:
        local = _commit(store, _metadata())
        store._connection.execute("DELETE FROM state_store_transaction_descriptors")
        boundary = RecoveryBoundary(
            record(
                "COMMITTED",
                committed_generation=1,
                committed_state_fingerprint_sha256=local.state_fingerprint_sha256,
            )
        )
        recovery = coordinator(store, boundary)
        with pytest.raises(Exception):
            recovery.recover_protected_state(SCOPE)
        assert recovery._recovery_required
        assert not {"abort", "finalize", "prepare"} & set(boundary.calls)


def test_recovery_surface_has_no_caller_latch_reset_and_never_prepares() -> None:
    assert "clear_recovery_required" not in vars(ProtectedFreshnessHandoffCoordinator)
    assert "set_ready" not in vars(ProtectedFreshnessHandoffCoordinator)
    assert "force_recovered" not in vars(ProtectedFreshnessHandoffCoordinator)
    assert "clear_recovery_scope" not in vars(ProtectedFreshnessHandoffCoordinator)
    assert "force_scope" not in vars(ProtectedFreshnessHandoffCoordinator)
