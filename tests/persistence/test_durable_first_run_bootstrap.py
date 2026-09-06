from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier

import pytest

from bot_core.persistence.first_run_bootstrap import (
    DurableFirstRunBootstrapCoordinator,
    DurableFirstRunBootstrapRegistry,
    _bootstrap_state_record,
    _consumed_history_record,
)
from bot_core.persistence.state_store import SQLiteStateStore, StateStoreMetadata
from bot_core.runtime.first_run_bootstrap import (
    AUTHORITY_SOURCE,
    FirstRunBootstrapClaim,
    FirstRunBootstrapError,
    ProvisioningMembershipBinding,
    claim_content_fingerprint,
    state_content_fingerprint,
    CoreCurrentBootstrapState,
    INITIAL_SECURITY_ESTABLISHMENT_ONLY,
    FirstRunBootstrapAuthority,
)

ACCOUNT = "acct_018f0000-0000-7000-8000-000000000001"
DEVICE = "dev_018f0000-0000-7000-8000-000000000002"
OPERATOR = "op_018f0000-0000-7000-8000-000000000003"


class Provisioning:
    def __init__(self, item: FirstRunBootstrapClaim) -> None:
        self.item = item
        self.binding = ProvisioningMembershipBinding(
            item.claim_fingerprint_sha256,
            claim_content_fingerprint(item),
            AUTHORITY_SOURCE,
            item.provisioning_context_fingerprint_sha256,
        )

    def resolve_membership(self, claim_reference: str) -> ProvisioningMembershipBinding:
        if claim_reference != self.item.claim_fingerprint_sha256:
            raise KeyError(claim_reference)
        return self.binding

    def resolve_accepted_claim(self, claim_reference: str) -> FirstRunBootstrapClaim:
        if claim_reference != self.item.claim_fingerprint_sha256:
            raise KeyError(claim_reference)
        return self.item


class RacingProvisioning(Provisioning):
    def __init__(self, item: FirstRunBootstrapClaim, barrier: Barrier) -> None:
        super().__init__(item)
        self.barrier = barrier

    def resolve_membership(self, claim_reference: str) -> ProvisioningMembershipBinding:
        self.barrier.wait(timeout=5)
        return super().resolve_membership(claim_reference)


class WrongTypeProvisioning:
    def __init__(self, item: FirstRunBootstrapClaim) -> None:
        self.item = item

    def resolve_membership(self, claim_reference: str) -> object:
        return {"claim_fingerprint_sha256": claim_reference}

    def resolve_accepted_claim(self, claim_reference: str) -> object:
        return {"claim_fingerprint_sha256": claim_reference}


def claim() -> FirstRunBootstrapClaim:
    item = FirstRunBootstrapClaim(
        ACCOUNT,
        DEVICE,
        OPERATOR,
        1,
        1,
        "2026-08-10T10:00:00Z",
        "2026-08-10T10:05:00Z",
        "a" * 64,
        "b" * 64,
        "0" * 64,
    )
    return replace(item, claim_fingerprint_sha256=claim_content_fingerprint(item))


def initial_state() -> CoreCurrentBootstrapState:
    state = CoreCurrentBootstrapState(
        "0" * 64,
        ACCOUNT,
        DEVICE,
        OPERATOR,
        "SETUP_REQUIRED",
        "PRE_INITIAL_SECURITY",
        "ABSENT",
        1,
        1,
        (),
        1,
    )
    return replace(state, state_fingerprint_sha256=state_content_fingerprint(state))


def initialize(
    store: SQLiteStateStore, item: FirstRunBootstrapClaim | None = None
) -> CoreCurrentBootstrapState:
    item = claim() if item is None else item
    initialize_metadata(store)
    reference = DurableFirstRunBootstrapCoordinator(
        store, Provisioning(item)
    ).materialize_initial_state(item.claim_fingerprint_sha256)
    state = DurableFirstRunBootstrapRegistry(store).resolve_accepted_state(reference)
    assert state == initial_state()
    return state


def initialize_metadata(store: SQLiteStateStore) -> StateStoreMetadata:
    metadata = StateStoreMetadata(
        ACCOUNT,
        DEVICE,
        1,
        "1" * 64,
        "PAPER",
        1,
        "2" * 64,
        "3" * 64,
        "4" * 64,
    )
    prepared = store.derive_prepared_metadata(metadata, expected_current_generation=None)
    store.commit_prepared_metadata(prepared, expected_current_generation=None)
    return prepared


def test_initial_materialization_rejects_self_inconsistent_claim(tmp_path: Path) -> None:
    fake = replace(claim(), claim_fingerprint_sha256="f" * 64)
    with SQLiteStateStore(tmp_path / "self-inconsistent.sqlite3") as store:
        before = initialize_metadata(store)
        with pytest.raises(FirstRunBootstrapError, match="BOOTSTRAP_AUTHORITY_DENIED"):
            DurableFirstRunBootstrapCoordinator(
                store, Provisioning(fake)
            ).materialize_initial_state(fake.claim_fingerprint_sha256)
        assert store.read_metadata() == before
        snapshot = store.read_verified_snapshot()
        assert snapshot is not None
        assert snapshot.current_records == ()
        assert snapshot.immutable_history == ()


def test_initial_materialization_requires_exact_provisioning_types(tmp_path: Path) -> None:
    item = claim()
    with SQLiteStateStore(tmp_path / "wrong-types.sqlite3") as store:
        before = initialize_metadata(store)
        with pytest.raises(FirstRunBootstrapError, match="BOOTSTRAP_AUTHORITY_DENIED"):
            DurableFirstRunBootstrapCoordinator(
                store,
                WrongTypeProvisioning(item),  # type: ignore[arg-type]
            ).materialize_initial_state(item.claim_fingerprint_sha256)
        assert store.read_metadata() == before


def test_tampered_result_lookup_does_not_leak_persistence_error(tmp_path: Path) -> None:
    item = claim()
    with SQLiteStateStore(tmp_path / "tampered-result.sqlite3") as store:
        initialize(store, item)
        result = DurableFirstRunBootstrapCoordinator(store, Provisioning(item)).consume(
            item, "2026-08-10T10:02:00Z"
        )
        tampered = replace(result, pre_state_fingerprint_sha256="f" * 64)
        authority = FirstRunBootstrapAuthority(
            Provisioning(item), DurableFirstRunBootstrapRegistry(store)
        )
        with pytest.raises(FirstRunBootstrapError):
            authority.revalidate(tampered, INITIAL_SECURITY_ESTABLISHMENT_ONLY)


def test_coordinator_metadata_failure_is_bootstrap_denial(tmp_path: Path) -> None:
    item = claim()
    with SQLiteStateStore(tmp_path / "malformed-metadata.sqlite3") as store:
        initialize_metadata(store)
        store._connection.execute(  # type: ignore[attr-defined]
            "UPDATE state_store_metadata SET environment='BROKEN' WHERE singleton_key=1"
        )
        with pytest.raises(FirstRunBootstrapError, match="CONTRACT_INCONSISTENT"):
            DurableFirstRunBootstrapCoordinator(
                store, Provisioning(item)
            ).materialize_initial_state(item.claim_fingerprint_sha256)
        assert store._connection.execute(  # type: ignore[attr-defined]
            "SELECT COUNT(*) FROM state_store_current_records"
        ).fetchone() == (0,)


def test_registry_raw_sqlite_read_failure_is_bootstrap_denial(tmp_path: Path) -> None:
    item = claim()
    store = SQLiteStateStore(tmp_path / "closed-store.sqlite3")
    initialize(store, item)
    registry = DurableFirstRunBootstrapRegistry(store)
    store.close()
    with pytest.raises(FirstRunBootstrapError, match="CONTRACT_INCONSISTENT"):
        registry.current_state_reference(ACCOUNT, DEVICE)


def publish_state_without_history(
    store: SQLiteStateStore, state: CoreCurrentBootstrapState
) -> None:
    metadata = store.read_metadata()
    assert metadata is not None
    target = replace(
        metadata,
        protected_freshness_generation=metadata.protected_freshness_generation + 1,
    )
    record = _bootstrap_state_record(state)
    prepared = store.derive_prepared_metadata(
        target,
        current_records=(record,),
        expected_current_generation=metadata.protected_freshness_generation,
    )
    store.commit_prepared_state(
        prepared,
        current_records=(record,),
        immutable_history=(),
        expected_current_generation=metadata.protected_freshness_generation,
    )


def publish_state_with_history(
    store: SQLiteStateStore,
    state: CoreCurrentBootstrapState,
) -> None:
    metadata = store.read_metadata()
    assert metadata is not None
    record = _bootstrap_state_record(state)
    history = _consumed_history_record(state.consumed_authorities[-1])
    target = replace(
        metadata,
        protected_freshness_generation=metadata.protected_freshness_generation + 1,
    )
    prepared = store.derive_prepared_metadata(
        target,
        current_records=(record,),
        immutable_history=(history,),
        expected_current_generation=metadata.protected_freshness_generation,
    )
    store.commit_prepared_state(
        prepared,
        current_records=(record,),
        immutable_history=(history,),
        expected_current_generation=metadata.protected_freshness_generation,
    )


def test_forged_revision_without_consumption_is_not_current(tmp_path: Path) -> None:
    item = claim()
    with SQLiteStateStore(tmp_path / "forged.sqlite3") as store:
        initialize(store, item)
        result = DurableFirstRunBootstrapCoordinator(store, Provisioning(item)).consume(
            item, "2026-08-10T10:02:00Z"
        )
        registry = DurableFirstRunBootstrapRegistry(store)
        post = registry.resolve_accepted_state(result.post_state_fingerprint_sha256)
        forged = replace(
            post,
            state_fingerprint_sha256="0" * 64,
            state_revision=post.state_revision + 1,
        )
        forged = replace(forged, state_fingerprint_sha256=state_content_fingerprint(forged))
        publish_state_without_history(store, forged)
        with pytest.raises(FirstRunBootstrapError, match="CONTRACT_INCONSISTENT"):
            DurableFirstRunBootstrapRegistry(store).current_state_reference(ACCOUNT, DEVICE)


def test_appended_consumption_without_durable_history_is_rejected(tmp_path: Path) -> None:
    item = claim()
    with SQLiteStateStore(tmp_path / "missing-history.sqlite3") as store:
        pre = initialize(store, item)
        _, consumed = FirstRunBootstrapAuthority(
            Provisioning(item), DurableFirstRunBootstrapRegistry(store)
        ).consume(
            item,
            pre.state_fingerprint_sha256,
            "2026-08-10T10:02:00Z",
            INITIAL_SECURITY_ESTABLISHMENT_ONLY,
        )
        publish_state_without_history(store, consumed)
        with pytest.raises(FirstRunBootstrapError, match="CONTRACT_INCONSISTENT"):
            DurableFirstRunBootstrapRegistry(store).resolve_accepted_state(
                consumed.state_fingerprint_sha256
            )


def test_stable_expected_fence_drift_is_rejected(tmp_path: Path) -> None:
    item = claim()
    with SQLiteStateStore(tmp_path / "drift.sqlite3") as store:
        pre = initialize(store, item)
        authority = FirstRunBootstrapAuthority(
            Provisioning(item), DurableFirstRunBootstrapRegistry(store)
        )
        _, post = authority.consume(
            item,
            pre.state_fingerprint_sha256,
            "2026-08-10T10:02:00Z",
            INITIAL_SECURITY_ESTABLISHMENT_ONLY,
        )
        drifted = replace(post, state_fingerprint_sha256="0" * 64, expected_revision=2)
        drifted = replace(drifted, state_fingerprint_sha256=state_content_fingerprint(drifted))
        publish_state_with_history(store, drifted)
        with pytest.raises(FirstRunBootstrapError, match="CONTRACT_INCONSISTENT"):
            DurableFirstRunBootstrapRegistry(store).current_state_reference(ACCOUNT, DEVICE)


def test_orphan_consumption_history_is_rejected(tmp_path: Path) -> None:
    item = claim()
    with SQLiteStateStore(tmp_path / "orphan.sqlite3") as store:
        pre = initialize(store, item)
        authority = FirstRunBootstrapAuthority(
            Provisioning(item), DurableFirstRunBootstrapRegistry(store)
        )
        result, _ = authority.consume(
            item,
            pre.state_fingerprint_sha256,
            "2026-08-10T10:02:00Z",
            INITIAL_SECURITY_ESTABLISHMENT_ONLY,
        )
        metadata = store.read_metadata()
        assert metadata is not None
        history = _consumed_history_record(result.consumed_authority)
        target = replace(
            metadata, protected_freshness_generation=metadata.protected_freshness_generation + 1
        )
        prepared = store.derive_prepared_metadata(
            target,
            immutable_history=(history,),
            expected_current_generation=metadata.protected_freshness_generation,
        )
        store.commit_prepared_state(
            prepared,
            current_records=(),
            immutable_history=(history,),
            expected_current_generation=metadata.protected_freshness_generation,
        )
        with pytest.raises(FirstRunBootstrapError, match="CONTRACT_INCONSISTENT"):
            DurableFirstRunBootstrapRegistry(store).current_state_reference(ACCOUNT, DEVICE)


def test_atomic_transition_reopen_and_replay_fence(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    item = claim()
    with SQLiteStateStore(path) as store:
        pre = initialize(store, item)
        with pytest.raises(FirstRunBootstrapError, match="BOOTSTRAP_AUTHORITY_DENIED"):
            DurableFirstRunBootstrapCoordinator(
                store, Provisioning(item)
            ).materialize_initial_state(item.claim_fingerprint_sha256)
        result = DurableFirstRunBootstrapCoordinator(store, Provisioning(item)).consume(
            item, "2026-08-10T10:02:00Z"
        )
        registry = DurableFirstRunBootstrapRegistry(store)
        post = registry.resolve_accepted_state(result.post_state_fingerprint_sha256)
        assert registry.resolve_accepted_state(pre.state_fingerprint_sha256) == pre
        assert registry.current_state_reference(ACCOUNT, DEVICE) == post.state_fingerprint_sha256
        assert post.state_revision == pre.state_revision + 1
        assert post.consumed_authorities == (result.consumed_authority,)
        assert result.consumed_authority.bootstrap_generation == pre.expected_generation
        assert result.consumed_authority.bootstrap_revision == pre.expected_revision
        assert len(store.read_immutable_history()) == 1
        assert post.startup_readiness == "SETUP_REQUIRED"
        assert post.initial_security_lifecycle == "PRE_INITIAL_SECURITY"
        assert post.first_operator_presence == "ABSENT"
        assert (
            FirstRunBootstrapAuthority(Provisioning(item), registry).revalidate(
                result, INITIAL_SECURITY_ESTABLISHMENT_ONLY
            )
            == post
        )

    with SQLiteStateStore(path) as reopened:
        registry = DurableFirstRunBootstrapRegistry(reopened)
        assert registry.resolve_accepted_state(pre.state_fingerprint_sha256) == pre
        post = registry.resolve_accepted_state(result.post_state_fingerprint_sha256)
        assert registry.current_state_reference(ACCOUNT, DEVICE) == post.state_fingerprint_sha256
        assert post.consumed_authorities == (result.consumed_authority,)
        assert (
            FirstRunBootstrapAuthority(Provisioning(item), registry).revalidate(
                result, INITIAL_SECURITY_ESTABLISHMENT_ONLY
            )
            == post
        )
        before = reopened.read_verified_snapshot()
        with pytest.raises(FirstRunBootstrapError):
            DurableFirstRunBootstrapCoordinator(reopened, Provisioning(item)).consume(
                item, "2026-08-10T10:02:00Z"
            )
        assert reopened.read_verified_snapshot() == before


def test_competing_stale_generation_cannot_publish_second_post(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    item = claim()
    with SQLiteStateStore(path) as initial:
        initialize(initial, item)
    barrier = Barrier(2)

    def attempt() -> str:
        with SQLiteStateStore(path) as store:
            try:
                DurableFirstRunBootstrapCoordinator(
                    store, RacingProvisioning(item, barrier)
                ).consume(item, "2026-08-10T10:02:00Z")
                return "committed"
            except FirstRunBootstrapError:
                return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _: attempt(), range(2)))
    assert sorted(outcomes) == ["committed", "rejected"]
    with SQLiteStateStore(path) as reopened:
        snapshot = reopened.read_verified_snapshot()
        assert snapshot is not None
        assert (
            len(
                [
                    r
                    for r in snapshot.current_records
                    if r.representation_name == "bootstrap consumed fence"
                ]
            )
            == 2
        )
        assert len(snapshot.immutable_history) == 1


def test_failure_inside_transaction_leaves_no_partial_transition(tmp_path: Path) -> None:
    path = tmp_path / "state.sqlite3"
    item = claim()
    store = SQLiteStateStore(path)
    pre = initialize(store, item)
    store._connection.execute(  # type: ignore[attr-defined]
        "CREATE TRIGGER fail_bootstrap_commit BEFORE UPDATE ON state_store_metadata "
        "BEGIN SELECT RAISE(ABORT, 'injected crash'); END"
    )
    with pytest.raises(FirstRunBootstrapError):
        DurableFirstRunBootstrapCoordinator(store, Provisioning(item)).consume(
            item, "2026-08-10T10:02:00Z"
        )
    store.close()
    with SQLiteStateStore(path) as reopened:
        registry = DurableFirstRunBootstrapRegistry(reopened)
        assert registry.current_state_reference(ACCOUNT, DEVICE) == pre.state_fingerprint_sha256
        assert len(reopened.read_current_records()) == 1
        assert reopened.read_immutable_history() == ()
