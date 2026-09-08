from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bot_core.persistence.first_run_bootstrap import (
    DurableFirstRunBootstrapCoordinator,
    DurableFirstRunBootstrapRegistry,
)
from bot_core.persistence.state_store import SQLiteStateStore, StateStoreMetadata
from bot_core.runtime.first_run_bootstrap import (
    AUTHORITY_SOURCE,
    FirstRunBootstrapClaim,
    ProvisioningMembershipBinding,
    claim_content_fingerprint,
)
from bot_core.runtime.runtime_session import RuntimeSession
from bot_core.security.initial_security import (
    InitialSecurityAuthority,
    InitialSecurityError,
    M03BootstrapAuthorityView,
    M03InitialSecurityBridge,
    PinVerifierMaterial,
)

ACCOUNT = "acct_018f0000-0000-7000-8000-000000000001"
DEVICE = "dev_018f0000-0000-7000-8000-000000000002"
OPERATOR = "op_018f0000-0000-7000-8000-000000000003"
SESSION = "run_018f0000-0000-7000-8000-000000000004"
RAW_PIN = "2468"


class Provisioning:
    def __init__(self, claim: FirstRunBootstrapClaim) -> None:
        self.claim = claim
        self.membership_missing = False
        self.binding = ProvisioningMembershipBinding(
            claim.claim_fingerprint_sha256,
            claim_content_fingerprint(claim),
            AUTHORITY_SOURCE,
            claim.provisioning_context_fingerprint_sha256,
        )

    def resolve_membership(self, reference: str) -> ProvisioningMembershipBinding:
        if getattr(self, "membership_missing", False):
            raise KeyError(reference)
        if reference != self.claim.claim_fingerprint_sha256:
            raise KeyError(reference)
        return self.binding

    def resolve_accepted_claim(self, reference: str) -> FirstRunBootstrapClaim:
        if reference != self.claim.claim_fingerprint_sha256:
            raise KeyError(reference)
        return self.claim


class PinFactory:
    def __init__(
        self,
        fail: bool = False,
        salt_reference: str = "secure-store://opaque/pin-salt",
        leak: bool = False,
    ) -> None:
        self.fail = fail
        self.salt_reference = salt_reference
        self.leak = leak

    def create(self, raw_pin: str) -> PinVerifierMaterial:
        if self.fail:
            raise RuntimeError(f"KDF unavailable:{raw_pin}" if self.leak else "KDF unavailable")
        assert raw_pin == RAW_PIN
        return PinVerifierMaterial("ARGON2ID", 1, self.salt_reference, "c" * 64)


class Sessions:
    def __init__(self) -> None:
        self.current_session = RuntimeSession(SESSION, DEVICE)

    def resolve_current(self, account_id: str, device_installation_id: str) -> RuntimeSession:
        assert account_id == ACCOUNT
        assert device_installation_id == DEVICE
        return self.current_session


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


def authority(
    path: Path, *, failing_pin: bool = False
) -> tuple[SQLiteStateStore, InitialSecurityAuthority, M03BootstrapAuthorityView]:
    item = claim()
    provisioning = Provisioning(item)
    store = SQLiteStateStore(path)
    metadata = StateStoreMetadata(
        ACCOUNT, DEVICE, 1, "1" * 64, "PAPER", 1, "2" * 64, "3" * 64, "4" * 64
    )
    prepared = store.derive_prepared_metadata(metadata, expected_current_generation=None)
    store.commit_prepared_metadata(prepared, expected_current_generation=None)
    coordinator = DurableFirstRunBootstrapCoordinator(store, provisioning)
    coordinator.materialize_initial_state(item.claim_fingerprint_sha256)
    transition = coordinator.consume(item, "2026-08-10T10:02:00Z")
    bridge = M03InitialSecurityBridge(DurableFirstRunBootstrapRegistry(store), provisioning)
    view = bridge.accept(transition)
    return store, InitialSecurityAuthority(bridge, PinFactory(failing_pin), Sessions()), view


def test_genuine_durable_bootstrap_atomically_establishes_initial_security(tmp_path: Path) -> None:
    store, owner, view = authority(tmp_path / "success.sqlite3")
    assert owner.snapshot.current_bootstrap[(ACCOUNT, DEVICE)] == view.claim_fingerprint_sha256
    with store:
        result = owner.establish_initial_security(view, RAW_PIN)
    state = owner.snapshot
    assert result.result == "INITIAL_SECURITY_ESTABLISHED"
    assert owner.resolve_current_identity(ACCOUNT, OPERATOR).state == "ACTIVE"
    assert owner.resolve_current_device(ACCOUNT, DEVICE).state == "TRUSTED"
    assert owner.resolve_current_session(ACCOUNT, OPERATOR, DEVICE).state == "UNLOCKED"
    assert owner.resolve_current_session(ACCOUNT, OPERATOR, DEVICE).runtime_session_id == SESSION
    assert owner.resolve_current_initial_security(ACCOUNT, DEVICE).state == "ESTABLISHED"
    assert owner.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE).failed_attempts == 0
    for accepted, current in (
        (state.accepted_identities, state.current_identities),
        (state.accepted_devices, state.current_devices),
        (state.accepted_pins, state.current_pins),
        (state.accepted_sessions, state.current_sessions),
        (state.accepted_initial_states, state.current_initial_states),
    ):
        assert tuple(current.values()) == tuple(accepted.keys())
    assert (ACCOUNT, DEVICE) not in state.current_bootstrap
    assert view.claim_fingerprint_sha256 in state.consumed_bootstrap_claims
    assert {
        owner.resolve_current_identity(ACCOUNT, OPERATOR).security_generation,
        owner.resolve_current_device(ACCOUNT, DEVICE).security_generation,
        owner.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE).security_generation,
        owner.resolve_current_session(ACCOUNT, OPERATOR, DEVICE).security_generation,
        owner.resolve_current_initial_security(ACCOUNT, DEVICE).security_generation,
    } == {1}
    assert (
        owner.resolve_current_session(ACCOUNT, OPERATOR, DEVICE).session_generation
        == owner.resolve_current_initial_security(ACCOUNT, DEVICE).session_generation
        == 1
    )
    assert RAW_PIN not in repr(result)
    assert RAW_PIN not in repr(state)
    assert not any(
        name in repr(state)
        for name in ("LiveAccessGrant", "ExecutionLease", "RiskDecision", "AuthenticationProof")
    )
    assert not state.accepted_operation_entitlements
    assert not state.current_operation_entitlements


def test_manual_and_tampered_views_are_denied(tmp_path: Path) -> None:
    store, owner, view = authority(tmp_path / "tamper.sqlite3")
    with store:
        manual = replace(view, claim_fingerprint_sha256="f" * 64)
        with pytest.raises(InitialSecurityError, match="AUTHORIZATION_DENIED"):
            owner.establish_initial_security(manual, RAW_PIN)
        for field in (
            "pre_state_fingerprint_sha256",
            "post_state_fingerprint_sha256",
            "consumed_authority_fingerprint_sha256",
            "consumed_claim_fingerprint_sha256",
            "consumed_challenge_fingerprint_sha256",
            "operator_id",
        ):
            value = (
                "op_018f0000-0000-7000-8000-000000000099" if field == "operator_id" else "e" * 64
            )
            with pytest.raises(InitialSecurityError, match="AUTHORIZATION_DENIED"):
                owner.establish_initial_security(
                    replace(view, **{field: value}),  # type: ignore[arg-type]
                    RAW_PIN,
                )
    assert not owner.snapshot.accepted_identities


def test_replay_is_denied_without_mutation(tmp_path: Path) -> None:
    store, owner, view = authority(tmp_path / "replay.sqlite3")
    with store:
        owner.establish_initial_security(view, RAW_PIN)
        before = owner.snapshot
        with pytest.raises(InitialSecurityError, match="BOOTSTRAP_REPLAY_DENIED"):
            owner.establish_initial_security(replace(view), RAW_PIN)
    assert owner.snapshot == before


def test_pin_dependency_failure_publishes_nothing_and_retains_no_pin(tmp_path: Path) -> None:
    store, owner, view = authority(tmp_path / "failure.sqlite3", failing_pin=True)
    with (
        store,
        pytest.raises(InitialSecurityError, match="PIN_VERIFIER_DEPENDENCY_FAILURE") as error,
    ):
        owner.establish_initial_security(view, RAW_PIN)
    assert (
        not owner.snapshot.accepted_identities
        and not owner.snapshot.accepted_devices
        and not owner.snapshot.accepted_pins
    )
    assert not owner.snapshot.accepted_sessions and not owner.snapshot.accepted_initial_states
    assert owner.snapshot.consumed_bootstrap_claims == frozenset()
    assert RAW_PIN not in str(error.value)
    assert RAW_PIN not in repr(owner.snapshot)


def test_second_owner_cannot_reuse_shared_bootstrap(tmp_path: Path) -> None:
    store, first, view = authority(tmp_path / "shared.sqlite3")
    second = InitialSecurityAuthority(first._bridge, PinFactory(), Sessions())  # noqa: SLF001
    with store:
        first.establish_initial_security(view, RAW_PIN)
        before = first.snapshot
        with pytest.raises(InitialSecurityError):
            second.establish_initial_security(view, RAW_PIN)
    assert second.snapshot == before
    assert view.claim_fingerprint_sha256 in before.consumed_bootstrap_claims
    assert (ACCOUNT, DEVICE) not in before.current_bootstrap


def test_invalid_pin_is_denied_before_any_publication(tmp_path: Path) -> None:
    store, owner, view = authority(tmp_path / "invalid-pin.sqlite3")
    with store, pytest.raises(InitialSecurityError, match="INVALID_PIN_INPUT"):
        owner.establish_initial_security(view, object())
    assert not owner.snapshot.accepted_identities
    assert owner.snapshot.current_bootstrap[(ACCOUNT, DEVICE)] == view.claim_fingerprint_sha256


def test_missing_upstream_membership_is_denied_after_durable_consume(tmp_path: Path) -> None:
    item = claim()
    provisioning = Provisioning(item)
    store = SQLiteStateStore(tmp_path / "membership.sqlite3")
    metadata = StateStoreMetadata(
        ACCOUNT, DEVICE, 1, "1" * 64, "PAPER", 1, "2" * 64, "3" * 64, "4" * 64
    )
    prepared = store.derive_prepared_metadata(metadata, expected_current_generation=None)
    store.commit_prepared_metadata(prepared, expected_current_generation=None)
    coordinator = DurableFirstRunBootstrapCoordinator(store, provisioning)
    coordinator.materialize_initial_state(item.claim_fingerprint_sha256)
    transition = coordinator.consume(item, "2026-08-10T10:02:00Z")
    provisioning.membership_missing = True
    with store, pytest.raises(InitialSecurityError, match="M03_BOOTSTRAP_AUTHORITY_DENIED"):
        M03InitialSecurityBridge(DurableFirstRunBootstrapRegistry(store), provisioning).accept(
            transition
        )


@pytest.mark.parametrize(
    "factory", [PinFactory(salt_reference="plaintext-salt"), PinFactory(fail=True, leak=True)]
)
def test_pin_boundary_failure_is_atomic_and_pin_safe(tmp_path: Path, factory: PinFactory) -> None:
    store, owner, view = authority(tmp_path / f"pin-{id(factory)}.sqlite3")
    owner._pin_factory = factory  # noqa: SLF001 -- deterministic dependency seam
    with store, pytest.raises(InitialSecurityError) as captured:
        owner.establish_initial_security(view, RAW_PIN)
    error = captured.value
    graph = repr(error) + str(error) + repr(error.__cause__) + repr(error.__context__)
    assert RAW_PIN not in graph
    assert not owner.snapshot.accepted_identities
    assert owner.snapshot.current_bootstrap[(ACCOUNT, DEVICE)] == view.claim_fingerprint_sha256
