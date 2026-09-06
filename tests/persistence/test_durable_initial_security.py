"""Durable M0.10 establishment, terminal bootstrap, and restart fences."""

from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from bot_core.persistence.first_run_bootstrap import (
    DurableFirstRunBootstrapCoordinator,
    DurableFirstRunBootstrapRegistry,
)
from bot_core.persistence.initial_security import (
    DurableInitialSecurityCoordinator,
    DurableInitialSecurityError,
    DurableInitialSecurityRegistry,
    _designation,
    _history,
)
from bot_core.persistence.lifecycle_records import persistence_record
from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.persistence.runtime_session_history import runtime_session_carrier
from bot_core.persistence.state_store import SQLiteStateStore, StateStoreMetadata
from bot_core.runtime.first_run_bootstrap import (
    AUTHORITY_SOURCE,
    FirstRunBootstrapClaim,
    FirstRunBootstrapError,
    ProvisioningMembershipBinding,
    claim_content_fingerprint,
)
from bot_core.runtime.runtime_session import RuntimeSession
from bot_core.security.initial_security import (
    InitialSecurityAuthority,
    M03InitialSecurityBridge,
    PinVerifierMaterial,
    PinVerifierRecord,
    SessionSecurityState,
)

ACCOUNT = "acct_018f0000-0000-7000-8000-000000000001"
DEVICE = "dev_018f0000-0000-7000-8000-000000000002"
OPERATOR = "op_018f0000-0000-7000-8000-000000000003"
SESSION = "run_018f0000-0000-7000-8000-000000000004"
SESSION_B = "run_018f0000-0000-7000-8000-000000000005"
OPERATOR_B = "op_018f0000-0000-7000-8000-000000000006"


class Provisioning:
    def __init__(self, item: FirstRunBootstrapClaim) -> None:
        self.item = item
        self.binding = ProvisioningMembershipBinding(
            item.claim_fingerprint_sha256,
            claim_content_fingerprint(item),
            AUTHORITY_SOURCE,
            item.provisioning_context_fingerprint_sha256,
        )

    def resolve_accepted_claim(self, reference: str) -> FirstRunBootstrapClaim:
        if reference != self.item.claim_fingerprint_sha256:
            raise KeyError(reference)
        return self.item

    def resolve_membership(self, reference: str) -> ProvisioningMembershipBinding:
        if reference != self.item.claim_fingerprint_sha256:
            raise KeyError(reference)
        return self.binding


class PinFactory:
    def create(self, raw_pin: str) -> PinVerifierMaterial:
        assert raw_pin == "2468"
        return PinVerifierMaterial(
            "M010-DETERMINISTIC-REFERENCE-NOT-PRODUCTION-KDF",
            1,
            "secure-store://opaque/pin-salt",
            "c" * 64,
        )


class Sessions:
    def __init__(self, session: RuntimeSession | None = None) -> None:
        self.current_session = session or RuntimeSession(SESSION, DEVICE)

    def resolve_current(self, account_id: str, device_id: str) -> RuntimeSession:
        assert (account_id, device_id) == (ACCOUNT, DEVICE)
        return self.current_session


def _claim() -> FirstRunBootstrapClaim:
    value = FirstRunBootstrapClaim(
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
    return replace(value, claim_fingerprint_sha256=claim_content_fingerprint(value))


def _stack(path: Path, **faults: object):
    item = _claim()
    provisioning = Provisioning(item)
    store = SQLiteStateStore(path)
    metadata = StateStoreMetadata(
        ACCOUNT, DEVICE, 1, "1" * 64, "PAPER", 1, "2" * 64, "3" * 64, "4" * 64
    )
    prepared = store.derive_prepared_metadata(metadata, expected_current_generation=None)
    store.commit_prepared_metadata(prepared, expected_current_generation=None)
    bootstrap = DurableFirstRunBootstrapCoordinator(store, provisioning)
    bootstrap.materialize_initial_state(item.claim_fingerprint_sha256)
    transition = bootstrap.consume(item, "2026-08-10T10:02:00Z")
    snapshot = store.read_verified_snapshot()
    assert snapshot is not None
    runtime = runtime_session_carrier(RuntimeSession(SESSION, DEVICE))
    target = replace(
        snapshot.metadata,
        protected_freshness_generation=snapshot.metadata.protected_freshness_generation + 1,
    )
    durable = store.derive_prepared_metadata(
        target,
        immutable_history=(runtime,),
        expected_current_generation=snapshot.metadata.protected_freshness_generation,
    )
    store.commit_prepared_state(
        durable,
        current_records=(),
        immutable_history=(runtime,),
        expected_current_generation=snapshot.metadata.protected_freshness_generation,
    )
    bridge = M03InitialSecurityBridge(DurableFirstRunBootstrapRegistry(store), provisioning)
    sessions = Sessions()
    owner = InitialSecurityAuthority(bridge, PinFactory(), sessions)
    coordinator = DurableInitialSecurityCoordinator(
        store,
        provisioning,
        bridge,
        owner,
        **faults,  # type: ignore[arg-type]
    )
    return store, provisioning, transition, sessions, owner, coordinator


def test_success_is_complete_terminal_and_has_no_fabricated_biometric_history(
    tmp_path: Path,
) -> None:
    store, provisioning, transition, _, owner, coordinator = _stack(tmp_path / "success.db")
    result = coordinator.establish(transition, "2468")
    assert result.result == "INITIAL_SECURITY_ESTABLISHED"
    family = DurableInitialSecurityRegistry(store, provisioning).resolve_current(ACCOUNT, DEVICE)
    assert family.device.state == "TRUSTED"
    assert family.device.platform_enrollment_revision == 1
    assert family.terminal_bootstrap.initial_security_lifecycle == "INITIAL_SECURITY_COMPLETED"
    snapshot = store.read_verified_snapshot()
    assert snapshot is not None
    assert not any(
        record.representation_name == "platform enrollment revisions"
        for record in snapshot.immutable_history
    )
    assert owner.resolve_current_initial_security(ACCOUNT, DEVICE) == family.initial


def test_commit_precedes_semantic_publication_and_failure_before_commit_is_atomic(
    tmp_path: Path,
) -> None:
    def fail() -> None:
        raise DurableInitialSecurityError("INJECTED_BEFORE_COMMIT")

    store, _, transition, _, owner, coordinator = _stack(tmp_path / "before.db", before_commit=fail)
    generation = store.read_metadata().protected_freshness_generation  # type: ignore[union-attr]
    with pytest.raises(DurableInitialSecurityError, match="INJECTED_BEFORE_COMMIT"):
        coordinator.establish(transition, "2468")
    assert store.read_metadata().protected_freshness_generation == generation  # type: ignore[union-attr]
    assert not owner.snapshot.accepted_initial_states
    assert DurableFirstRunBootstrapRegistry(store).current_state_reference(ACCOUNT, DEVICE)


def test_post_commit_failure_reopens_rehydrates_and_replay_is_denied(tmp_path: Path) -> None:
    path = tmp_path / "after.db"

    def fail() -> None:
        raise DurableInitialSecurityError("INJECTED_AFTER_COMMIT")

    store, provisioning, transition, sessions, owner, coordinator = _stack(path, after_commit=fail)
    with pytest.raises(DurableInitialSecurityError, match="INJECTED_AFTER_COMMIT"):
        coordinator.establish(transition, "2468")
    assert not owner.snapshot.accepted_initial_states
    coordinator.rehydrate(ACCOUNT, DEVICE)
    assert owner.resolve_current_session(ACCOUNT, OPERATOR, DEVICE).runtime_session_id == SESSION
    store.close()
    with SQLiteStateStore(path) as reopened:
        fresh_sessions = Sessions(RuntimeSession(SESSION_B, DEVICE))
        bridge = M03InitialSecurityBridge(DurableFirstRunBootstrapRegistry(reopened), provisioning)
        fresh = InitialSecurityAuthority(bridge, PinFactory(), fresh_sessions)
        durable = DurableInitialSecurityCoordinator(reopened, provisioning, bridge, fresh)
        with pytest.raises(DurableInitialSecurityError, match="RUNTIME_SESSION_AUTHORITY_DENIED"):
            durable.rehydrate(ACCOUNT, DEVICE)
        assert not fresh.snapshot.accepted_sessions
        assert (
            DurableInitialSecurityRegistry(reopened, provisioning)
            .resolve_current(ACCOUNT, DEVICE)
            .session.runtime_session_id
            == SESSION
        )
        with pytest.raises(DurableInitialSecurityError, match="AUTHORIZATION_DENIED"):
            durable.establish(transition, "2468")
    assert sessions.current_session.runtime_session_id == SESSION


def test_terminal_bootstrap_is_not_usable_but_history_remains_auditable(tmp_path: Path) -> None:
    store, _, transition, _, _, coordinator = _stack(tmp_path / "terminal.db")
    coordinator.establish(transition, "2468")
    registry = DurableFirstRunBootstrapRegistry(store)
    with pytest.raises(FirstRunBootstrapError, match="BOOTSTRAP_AUTHORITY_DENIED"):
        registry.current_state_reference(ACCOUNT, DEVICE)
    assert registry.resolve_accepted_state(transition.post_state_fingerprint_sha256)


def test_two_independent_coordinators_have_exactly_one_winner(tmp_path: Path) -> None:
    path = tmp_path / "race.db"
    store, provisioning, transition, _, _, _ = _stack(path)
    source_generation = store.read_metadata().protected_freshness_generation  # type: ignore[union-attr]
    store.close()
    barrier = Barrier(2)

    def attempt() -> str:
        def synchronize() -> None:
            barrier.wait(timeout=5)

        with SQLiteStateStore(path) as contender:
            bridge = M03InitialSecurityBridge(
                DurableFirstRunBootstrapRegistry(contender), provisioning
            )
            owner = InitialSecurityAuthority(bridge, PinFactory(), Sessions())
            coordinator = DurableInitialSecurityCoordinator(
                contender,
                provisioning,
                bridge,
                owner,
                before_commit=synchronize,
            )
            try:
                coordinator.establish(transition, "2468")
            except DurableInitialSecurityError:
                return "denied"
            return "success"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = tuple(executor.map(lambda _: attempt(), range(2)))
    assert sorted(outcomes) == ["denied", "success"]
    with SQLiteStateStore(path) as verified:
        assert (
            verified.read_metadata().protected_freshness_generation  # type: ignore[union-attr]
            == source_generation + 1
        )
        family = DurableInitialSecurityRegistry(verified, provisioning).resolve_current(
            ACCOUNT, DEVICE
        )
        assert family.initial.state == "ESTABLISHED"


def test_recovery_recomputes_accepted_claim_content(tmp_path: Path) -> None:
    store, provisioning, transition, _, _, coordinator = _stack(tmp_path / "claim.db")
    coordinator.establish(transition, "2468")
    provisioning.item = replace(provisioning.item, issued_at_utc="2026-08-10T10:00:01Z")
    with pytest.raises(DurableInitialSecurityError, match="CONTRACT_INCONSISTENT"):
        DurableInitialSecurityRegistry(store, provisioning).resolve_current(ACCOUNT, DEVICE)


@pytest.mark.parametrize(
    "field",
    ["complete_claim_content_fingerprint_sha256", "provisioning_context_fingerprint_sha256"],
)
def test_recovery_requires_exact_provisioning_membership_relationship(
    tmp_path: Path, field: str
) -> None:
    store, provisioning, transition, _, _, coordinator = _stack(tmp_path / f"{field}.db")
    coordinator.establish(transition, "2468")
    provisioning.binding = replace(provisioning.binding, **{field: "f" * 64})
    with pytest.raises(DurableInitialSecurityError, match="CONTRACT_INCONSISTENT"):
        DurableInitialSecurityRegistry(store, provisioning).resolve_current(ACCOUNT, DEVICE)


def test_durable_family_has_exact_bootstrap_view_and_history_only_runtime(
    tmp_path: Path,
) -> None:
    store, provisioning, transition, sessions, _, coordinator = _stack(tmp_path / "facts.db")
    coordinator.establish(transition, "2468")
    family = DurableInitialSecurityRegistry(store, provisioning).resolve_current(ACCOUNT, DEVICE)
    assert (
        family.bootstrap_view.pre_state_fingerprint_sha256
        == transition.pre_state_fingerprint_sha256
    )
    assert (
        family.bootstrap_view.post_state_fingerprint_sha256
        == transition.post_state_fingerprint_sha256
    )
    assert family.runtime_session_history.record_key == SESSION
    assert not isinstance(family.runtime_session_history, RuntimeSession)
    assert sessions.current_session is not family.runtime_session_history


@pytest.mark.parametrize("mode", ["closed", "clone"])
def test_same_process_recovery_requires_exact_open_runtime_handle(
    tmp_path: Path, mode: str
) -> None:
    def fail() -> None:
        raise DurableInitialSecurityError("INJECTED_AFTER_COMMIT")

    store, _, transition, sessions, owner, coordinator = _stack(
        tmp_path / f"runtime-{mode}.db", after_commit=fail
    )
    with pytest.raises(DurableInitialSecurityError, match="INJECTED_AFTER_COMMIT"):
        coordinator.establish(transition, "2468")
    original = sessions.current_session
    if mode == "closed":
        original.close()
    else:
        sessions.current_session = RuntimeSession(original.runtime_session_id, DEVICE)
    with pytest.raises(DurableInitialSecurityError, match="RUNTIME_SESSION_AUTHORITY_DENIED"):
        coordinator.rehydrate(ACCOUNT, DEVICE)
    assert not owner.snapshot.accepted_sessions


def _replace_projection_store(
    path: Path,
    source: SQLiteStateStore,
    *,
    history_name: str,
    current_name: str,
    projection: PinVerifierRecord | SessionSecurityState,
) -> SQLiteStateStore:
    snapshot = source.read_verified_snapshot()
    assert snapshot is not None
    current = [
        item for item in snapshot.current_records if item.representation_name != current_name
    ]
    history = [
        item for item in snapshot.immutable_history if item.representation_name != history_name
    ]
    history.append(_history(history_name, projection))
    if isinstance(projection, PinVerifierRecord):
        current.append(
            _designation(
                current_name,
                f"{projection.account_id}:{projection.operator_id}:{projection.device_installation_id}",
                projection.content_fingerprint_sha256,
                projection.pin_revision,
                projection.security_generation,
            )
        )
    else:
        current.append(
            persistence_record(
                current_name,
                "direct:SessionSecurityState current generation/state:"
                f"{projection.account_id}:{projection.operator_id}:"
                f"{projection.device_installation_id}:{projection.runtime_session_id}:"
                f"{projection.session_generation}:{projection.security_generation}",
                projection.__dict__
                if hasattr(projection, "__dict__")
                else {name: getattr(projection, name) for name in projection.__dataclass_fields__},
            )
        )
    target = SQLiteStateStore(path)
    metadata = replace(snapshot.metadata, protected_freshness_generation=1)
    prepared = target.derive_prepared_metadata(
        metadata,
        current_records=current,
        immutable_history=history,
        expected_current_generation=None,
    )
    target.commit_prepared_state(
        prepared,
        current_records=current,
        immutable_history=history,
        expected_current_generation=None,
    )
    return target


@pytest.mark.parametrize(
    ("kind", "changes"),
    [
        ("pin", {"operator_id": OPERATOR_B}),
        ("session", {"security_generation": 2}),
        ("session", {"session_generation": 2}),
    ],
)
def test_registry_rejects_intrinsic_valid_cross_family_drift(
    tmp_path: Path, kind: str, changes: dict[str, object]
) -> None:
    store, provisioning, transition, _, _, coordinator = _stack(tmp_path / "source.db")
    coordinator.establish(transition, "2468")
    family = DurableInitialSecurityRegistry(store, provisioning).resolve_current(ACCOUNT, DEVICE)
    original = family.pin if kind == "pin" else family.session
    values = {
        name: getattr(original, name)
        for name in original.__dataclass_fields__
        if name != "content_fingerprint_sha256"
    }
    values.update(changes)
    values["content_fingerprint_sha256"] = canonical_json_sha256(values)
    projection = type(original)(**values)
    corrupted = _replace_projection_store(
        tmp_path / f"corrupt-{kind}-{tuple(changes)[0]}.db",
        store,
        history_name=(
            "PinVerifierRecord accepted revisions"
            if kind == "pin"
            else "SessionSecurityState revision history"
        ),
        current_name=(
            "PinVerifierRecord current designation"
            if kind == "pin"
            else "SessionSecurityState current generation/state"
        ),
        projection=projection,
    )
    with corrupted, pytest.raises(DurableInitialSecurityError, match="CONTRACT_INCONSISTENT"):
        DurableInitialSecurityRegistry(corrupted, provisioning).resolve_current(ACCOUNT, DEVICE)


def test_same_id_runtime_clone_is_rejected_before_commit_without_mutation(tmp_path: Path) -> None:
    store, _, transition, sessions, owner, coordinator = _stack(tmp_path / "pre-clone.db")
    original = sessions.current_session
    calls = 0

    def changing_current(account_id: str, device_id: str) -> RuntimeSession:
        nonlocal calls
        assert (account_id, device_id) == (ACCOUNT, DEVICE)
        calls += 1
        return original if calls == 1 else RuntimeSession(original.runtime_session_id, DEVICE)

    sessions.resolve_current = changing_current  # type: ignore[method-assign]
    generation = store.read_metadata().protected_freshness_generation  # type: ignore[union-attr]
    with pytest.raises(DurableInitialSecurityError, match="RUNTIME_SESSION_AUTHORITY_DENIED"):
        coordinator.establish(transition, "2468")
    assert store.read_metadata().protected_freshness_generation == generation  # type: ignore[union-attr]
    assert not owner.snapshot.accepted_initial_states
    assert DurableFirstRunBootstrapRegistry(store).current_state_reference(ACCOUNT, DEVICE)


def test_runtime_replacement_after_commit_preserves_durable_family_only(tmp_path: Path) -> None:
    sessions_ref: list[Sessions] = []

    def replace_runtime() -> None:
        sessions_ref[0].current_session = RuntimeSession(SESSION_B, DEVICE)

    store, provisioning, transition, sessions, owner, coordinator = _stack(
        tmp_path / "post-change.db", after_commit=replace_runtime
    )
    sessions_ref.append(sessions)
    with pytest.raises(DurableInitialSecurityError, match="CONTRACT_INCONSISTENT"):
        coordinator.establish(transition, "2468")
    assert not owner.snapshot.accepted_sessions
    family = DurableInitialSecurityRegistry(store, provisioning).resolve_current(ACCOUNT, DEVICE)
    assert family.session.runtime_session_id == SESSION
    assert family.terminal_bootstrap.initial_security_lifecycle == "INITIAL_SECURITY_COMPLETED"
    with pytest.raises(DurableInitialSecurityError, match="RUNTIME_SESSION_AUTHORITY_DENIED"):
        coordinator.rehydrate(ACCOUNT, DEVICE)


def _assert_no_p1d_mutation(
    store: SQLiteStateStore, owner: InitialSecurityAuthority, generation: int
) -> None:
    assert store.read_metadata().protected_freshness_generation == generation  # type: ignore[union-attr]
    snapshot = store.read_verified_snapshot()
    assert snapshot is not None
    assert not any(
        record.representation_name
        in DurableInitialSecurityRegistry._NAMES | {"InitialSecurityState current state"}
        for record in (*snapshot.current_records, *snapshot.immutable_history)
    )
    assert not owner.snapshot.accepted_initial_states
    assert DurableFirstRunBootstrapRegistry(store).current_state_reference(ACCOUNT, DEVICE)


@pytest.mark.parametrize("same_id", [False, True])
def test_runtime_replacement_inside_before_commit_is_rejected_without_mutation(
    tmp_path: Path, same_id: bool
) -> None:
    refs: list[Sessions] = []

    def replace_runtime() -> None:
        original = refs[0].current_session
        runtime_id = original.runtime_session_id if same_id else SESSION_B
        refs[0].current_session = RuntimeSession(runtime_id, DEVICE)

    store, _, transition, sessions, owner, coordinator = _stack(
        tmp_path / f"hook-runtime-{same_id}.db", before_commit=replace_runtime
    )
    refs.append(sessions)
    generation = store.read_metadata().protected_freshness_generation  # type: ignore[union-attr]
    with pytest.raises(DurableInitialSecurityError, match="RUNTIME_SESSION_AUTHORITY_DENIED"):
        coordinator.establish(transition, "2468")
    _assert_no_p1d_mutation(store, owner, generation)


def test_claim_change_inside_before_commit_is_revalidated_without_mutation(tmp_path: Path) -> None:
    refs: list[Provisioning] = []

    def change_claim() -> None:
        refs[0].item = replace(refs[0].item, expires_at_utc="2026-08-10T10:04:59Z")

    store, provisioning, transition, _, owner, coordinator = _stack(
        tmp_path / "hook-claim.db", before_commit=change_claim
    )
    refs.append(provisioning)
    generation = store.read_metadata().protected_freshness_generation  # type: ignore[union-attr]
    with pytest.raises(DurableInitialSecurityError, match="AUTHORIZATION_DENIED"):
        coordinator.establish(transition, "2468")
    _assert_no_p1d_mutation(store, owner, generation)


@pytest.mark.parametrize(
    "field",
    ["complete_claim_content_fingerprint_sha256", "provisioning_context_fingerprint_sha256"],
)
def test_membership_change_inside_before_commit_is_revalidated_without_mutation(
    tmp_path: Path, field: str
) -> None:
    refs: list[Provisioning] = []

    def change_membership() -> None:
        refs[0].binding = replace(refs[0].binding, **{field: "f" * 64})

    store, provisioning, transition, _, owner, coordinator = _stack(
        tmp_path / f"hook-membership-{field}.db", before_commit=change_membership
    )
    refs.append(provisioning)
    generation = store.read_metadata().protected_freshness_generation  # type: ignore[union-attr]
    with pytest.raises(DurableInitialSecurityError, match="AUTHORIZATION_DENIED"):
        coordinator.establish(transition, "2468")
    _assert_no_p1d_mutation(store, owner, generation)
