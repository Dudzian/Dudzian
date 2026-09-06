# mypy: disable-error-code="arg-type"
from __future__ import annotations

from dataclasses import asdict, fields, replace
from typing import Any

import pytest

from bot_core.runtime.first_run_bootstrap import (
    AUTHORITY_SOURCE,
    INITIAL_SECURITY_ESTABLISHMENT_ONLY,
    BootstrapTransitionResult,
    ConsumedBootstrapAuthority,
    CoreCurrentBootstrapState,
    FirstRunBootstrapAuthority,
    FirstRunBootstrapClaim,
    FirstRunBootstrapError,
    ProvisioningMembershipBinding,
    claim_content_fingerprint,
    state_content_fingerprint,
)

ACCOUNT = "acct_018f0000-0000-7000-8000-000000000001"
DEVICE = "dev_018f0000-0000-7000-8000-000000000002"
OPERATOR = "op_018f0000-0000-7000-8000-000000000003"


class InMemoryProvisioningBoundary:
    def __init__(self) -> None:
        self.bindings: dict[str, Any] = {}
        self.claims: dict[str, Any] = {}

    def resolve_membership(self, claim_reference: str) -> Any:
        return self.bindings[claim_reference]

    def resolve_accepted_claim(self, claim_reference: str) -> Any:
        return self.claims[claim_reference]


class InMemoryBootstrapStateRegistry:
    def __init__(self) -> None:
        self.states: dict[str, Any] = {}
        self.current: dict[tuple[str, str], str] = {}

    def resolve_accepted_state(self, state_reference: str) -> Any:
        return self.states[state_reference]

    def current_state_reference(self, account_id: str, device_installation_id: str) -> str:
        return self.current[(account_id, device_installation_id)]


def claim(**changes: object) -> FirstRunBootstrapClaim:
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
    value = replace(value, claim_fingerprint_sha256=claim_content_fingerprint(value))
    return replace(value, **changes)


def state(**changes: object) -> CoreCurrentBootstrapState:
    value = CoreCurrentBootstrapState(
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
    value = replace(value, state_fingerprint_sha256=state_content_fingerprint(value))
    value = replace(value, **changes)
    if "state_fingerprint_sha256" not in changes:
        value = replace(value, state_fingerprint_sha256=state_content_fingerprint(value))
    return value


def authority() -> tuple[
    FirstRunBootstrapAuthority, InMemoryProvisioningBoundary, InMemoryBootstrapStateRegistry
]:
    provision = InMemoryProvisioningBoundary()
    registry = InMemoryBootstrapStateRegistry()
    item = claim()
    before = state()
    provision.bindings[item.claim_fingerprint_sha256] = ProvisioningMembershipBinding(
        item.claim_fingerprint_sha256,
        claim_content_fingerprint(item),
        AUTHORITY_SOURCE,
        item.provisioning_context_fingerprint_sha256,
    )
    provision.claims[item.claim_fingerprint_sha256] = item
    registry.states[before.state_fingerprint_sha256] = before
    registry.current[(ACCOUNT, DEVICE)] = before.state_fingerprint_sha256
    return FirstRunBootstrapAuthority(provision, registry), provision, registry


def consume(
    owner: FirstRunBootstrapAuthority,
    item: object | None = None,
    reference: object | None = None,
    now: object = "2026-08-10T10:02:00Z",
    purpose: object = INITIAL_SECURITY_ESTABLISHMENT_ONLY,
) -> tuple[BootstrapTransitionResult, CoreCurrentBootstrapState]:
    candidate = claim() if item is None else item
    if reference is None:
        reference = state().state_fingerprint_sha256
    return owner.consume(candidate, reference, now, purpose)


def publish_transition(
    owner: FirstRunBootstrapAuthority, registry: InMemoryBootstrapStateRegistry
) -> BootstrapTransitionResult:
    result, after = consume(owner)
    registry.states[after.state_fingerprint_sha256] = after
    registry.current[(ACCOUNT, DEVICE)] = after.state_fingerprint_sha256
    return result


def test_exact_pure_transition_and_revalidation() -> None:
    owner, _, registry = authority()
    before_states = dict(registry.states)
    before_current = dict(registry.current)
    result, after = consume(owner)
    assert result.outcome == "INITIAL_SECURITY_ESTABLISHMENT_AUTHORIZED"
    assert result.authority_purpose == INITIAL_SECURITY_ESTABLISHMENT_ONLY
    assert after.initial_security_lifecycle == "PRE_INITIAL_SECURITY"
    assert after.first_operator_presence == "ABSENT"
    assert after.state_revision == 2
    assert after.consumed_authorities == (result.consumed_authority,)
    assert registry.states == before_states
    assert registry.current == before_current
    registry.states[after.state_fingerprint_sha256] = after
    registry.current[(ACCOUNT, DEVICE)] = after.state_fingerprint_sha256
    assert owner.revalidate(result, INITIAL_SECURITY_ESTABLISHMENT_ONLY) == after


def test_exact_frozen_field_sets_and_slots() -> None:
    assert [item.name for item in fields(FirstRunBootstrapClaim)] == [
        "account_id",
        "device_installation_id",
        "intended_operator_id",
        "bootstrap_generation",
        "bootstrap_revision",
        "issued_at_utc",
        "expires_at_utc",
        "challenge_fingerprint_sha256",
        "provisioning_context_fingerprint_sha256",
        "claim_fingerprint_sha256",
    ]
    result, _ = consume(authority()[0])
    with pytest.raises((AttributeError, TypeError)):
        result.outcome = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        FirstRunBootstrapClaim(**{**asdict(claim()), "trusted": True})  # type: ignore[arg-type]


@pytest.mark.parametrize("purpose", ["LIVE", "ADMIN", "", None, True])
def test_all_other_purposes_fail_closed(purpose: object) -> None:
    with pytest.raises(FirstRunBootstrapError, match="BOOTSTRAP_SCOPE_DENIED"):
        consume(authority()[0], purpose=purpose)


@pytest.mark.parametrize(
    ("now", "reason"),
    [
        ("2026-08-10T09:59:59Z", "BOOTSTRAP_NOT_YET_VALID"),
        ("2026-08-10T10:05:01Z", "BOOTSTRAP_EXPIRED"),
        ("2026-08-10 10:00:00Z", "MALFORMED_NOW_UTC"),
    ],
)
def test_time_denials(now: object, reason: str) -> None:
    with pytest.raises(FirstRunBootstrapError, match=reason):
        consume(authority()[0], now=now)


@pytest.mark.parametrize("now", ["2026-08-10T10:00:00Z", "2026-08-10T10:05:00Z"])
def test_time_boundaries_are_inclusive(now: str) -> None:
    consume(authority()[0], now=now)


def test_unknown_provisioning_reference_fails_closed() -> None:
    owner, provision, _ = authority()
    provision.bindings.clear()
    with pytest.raises(FirstRunBootstrapError, match="BOOTSTRAP_AUTHORITY_DENIED"):
        consume(owner)


@pytest.mark.parametrize(
    "field",
    [
        "claim_fingerprint_sha256",
        "complete_claim_content_fingerprint_sha256",
        "provisioning_context_fingerprint_sha256",
    ],
)
def test_wrong_membership_fingerprints_fail_closed(field: str) -> None:
    owner, provision, _ = authority()
    item = claim()
    binding = provision.bindings[item.claim_fingerprint_sha256]
    assert isinstance(binding, ProvisioningMembershipBinding)
    provision.bindings[item.claim_fingerprint_sha256] = replace(binding, **{field: "c" * 64})
    with pytest.raises(FirstRunBootstrapError, match="BOOTSTRAP_AUTHORITY_DENIED"):
        consume(owner)


def test_wrong_authority_source_is_rejected_at_closed_carrier_boundary() -> None:
    with pytest.raises(FirstRunBootstrapError, match="BOOTSTRAP_AUTHORITY_DENIED"):
        ProvisioningMembershipBinding("a" * 64, "b" * 64, "caller", "c" * 64)


def test_accepted_membership_with_different_content_fails_closed() -> None:
    owner, provision, _ = authority()
    item = claim()
    provision.claims[item.claim_fingerprint_sha256] = replace(
        item, expires_at_utc="2026-08-10T10:06:00Z"
    )
    with pytest.raises(FirstRunBootstrapError, match="BOOTSTRAP_AUTHORITY_DENIED"):
        consume(owner)


@pytest.mark.parametrize(
    "fake", [ProvisioningMembershipBinding("a" * 64, "a" * 64, AUTHORITY_SOURCE, "a" * 64), state()]
)
def test_caller_created_authority_objects_are_not_references(fake: object) -> None:
    with pytest.raises(FirstRunBootstrapError):
        consume(authority()[0], reference=fake)


def test_self_hashed_unregistered_claim_fails_closed() -> None:
    fake = claim(challenge_fingerprint_sha256="c" * 64, claim_fingerprint_sha256="0" * 64)
    fake = replace(fake, claim_fingerprint_sha256=claim_content_fingerprint(fake))
    with pytest.raises(FirstRunBootstrapError, match="BOOTSTRAP_AUTHORITY_DENIED"):
        consume(authority()[0], item=fake)


@pytest.mark.parametrize(
    "mutation",
    [
        {"account_id": "bad"},
        {"bootstrap_generation": True},
        {"bootstrap_revision": True},
        {"claim_fingerprint_sha256": "A" * 64},
    ],
)
def test_malformed_claim_carriers_fail_closed(mutation: dict[str, object]) -> None:
    with pytest.raises(FirstRunBootstrapError):
        replace(claim(), **mutation)


def test_unknown_and_stale_core_references_fail_closed() -> None:
    owner, _, registry = authority()
    with pytest.raises(FirstRunBootstrapError, match="BOOTSTRAP_AUTHORITY_DENIED"):
        consume(owner, reference="f" * 64)
    registry.current[(ACCOUNT, DEVICE)] = "e" * 64
    with pytest.raises(FirstRunBootstrapError, match="STALE_CORE_BOOTSTRAP_STATE"):
        consume(owner)


def test_registry_key_internal_and_recomputed_mismatch_fail_closed() -> None:
    owner, _, registry = authority()
    reference = state().state_fingerprint_sha256
    registry.states[reference] = replace(state(), state_fingerprint_sha256="d" * 64)
    with pytest.raises(FirstRunBootstrapError, match="CONTRACT_INCONSISTENT"):
        consume(owner)
    broken = state()
    object.__setattr__(broken, "expected_revision", 2)
    registry.states[reference] = broken
    with pytest.raises(FirstRunBootstrapError, match="CONTRACT_INCONSISTENT"):
        consume(owner)


def test_completed_or_present_state_cannot_start_transition() -> None:
    completed = state(
        initial_security_lifecycle="INITIAL_SECURITY_COMPLETED", first_operator_presence="PRESENT"
    )
    owner, _, registry = authority()
    registry.states = {completed.state_fingerprint_sha256: completed}
    registry.current[(ACCOUNT, DEVICE)] = completed.state_fingerprint_sha256
    with pytest.raises(FirstRunBootstrapError, match="BOOTSTRAP_NOT_ELIGIBLE"):
        consume(owner, reference=completed.state_fingerprint_sha256)
    with pytest.raises(FirstRunBootstrapError, match="MALFORMED_CORE_BOOTSTRAP_STATE"):
        replace(state(), first_operator_presence="PRESENT")


def test_replay_and_duplicate_generation_fail_closed() -> None:
    item = claim()
    consumed = ConsumedBootstrapAuthority(
        ACCOUNT, DEVICE, 1, 1, item.claim_fingerprint_sha256, "a" * 64
    )
    replay = state(consumed_authorities=(consumed,))
    owner, _, registry = authority()
    registry.states = {replay.state_fingerprint_sha256: replay}
    registry.current[(ACCOUNT, DEVICE)] = replay.state_fingerprint_sha256
    with pytest.raises(FirstRunBootstrapError, match="BOOTSTRAP_REPLAY_DENIED"):
        consume(owner, reference=replay.state_fingerprint_sha256)


def test_malformed_duplicate_and_reordered_history_fail_closed() -> None:
    first = ConsumedBootstrapAuthority(ACCOUNT, DEVICE, 1, 1, "a" * 64, "b" * 64)
    second = ConsumedBootstrapAuthority(ACCOUNT, DEVICE, 2, 1, "c" * 64, "d" * 64)
    with pytest.raises(FirstRunBootstrapError, match="MALFORMED_CONSUMED_HISTORY"):
        state(consumed_authorities=(first, first))
    with pytest.raises(FirstRunBootstrapError, match="MALFORMED_CONSUMED_HISTORY"):
        state(consumed_authorities=(second, first))


@pytest.mark.parametrize(
    "change",
    [
        {"pre_state_fingerprint_sha256": "f" * 64},
        {"post_state_fingerprint_sha256": "e" * 64},
        {"authority_purpose": "LIVE"},
        {"outcome": "SUCCESS"},
        {
            "consumed_authority": ConsumedBootstrapAuthority(
                ACCOUNT, DEVICE, 2, 1, "a" * 64, "b" * 64
            )
        },
    ],
)
def test_tampered_result_fails_revalidation(change: dict[str, object]) -> None:
    owner, _, registry = authority()
    result = publish_transition(owner, registry)
    try:
        tampered = replace(result, **change)
    except FirstRunBootstrapError:
        return
    with pytest.raises(FirstRunBootstrapError):
        owner.revalidate(tampered, INITIAL_SECURITY_ESTABLISHMENT_ONLY)


def test_revalidation_rejects_noncurrent_post_and_changed_history() -> None:
    owner, _, registry = authority()
    result = publish_transition(owner, registry)
    registry.current[(ACCOUNT, DEVICE)] = result.pre_state_fingerprint_sha256
    with pytest.raises(FirstRunBootstrapError, match="BOOTSTRAP_SCOPE_DENIED"):
        owner.revalidate(result, INITIAL_SECURITY_ESTABLISHMENT_ONLY)


@pytest.mark.parametrize(
    ("consumed_generation", "consumed_revision"),
    [(2, 1), (1, 2)],
    ids=["wrong-generation", "wrong-revision"],
)
@pytest.mark.parametrize("iteration", range(20))
def test_revalidation_rejects_self_consistent_post_with_wrong_expected_fence(
    consumed_generation: int,
    consumed_revision: int,
    iteration: int,
) -> None:
    del iteration
    owner, _, registry = authority()
    before = state()
    wrong_consumed = ConsumedBootstrapAuthority(
        before.account_id,
        before.device_installation_id,
        consumed_generation,
        consumed_revision,
        claim().claim_fingerprint_sha256,
        claim().challenge_fingerprint_sha256,
    )
    after = replace(
        before,
        state_fingerprint_sha256="0" * 64,
        consumed_authorities=(wrong_consumed,),
        state_revision=before.state_revision + 1,
    )
    after = replace(after, state_fingerprint_sha256=state_content_fingerprint(after))
    result = BootstrapTransitionResult(
        "INITIAL_SECURITY_ESTABLISHMENT_AUTHORIZED",
        INITIAL_SECURITY_ESTABLISHMENT_ONLY,
        before.state_fingerprint_sha256,
        after.state_fingerprint_sha256,
        wrong_consumed,
    )
    registry.states[after.state_fingerprint_sha256] = after
    registry.current[(ACCOUNT, DEVICE)] = after.state_fingerprint_sha256

    with pytest.raises(FirstRunBootstrapError, match="BOOTSTRAP_SCOPE_DENIED"):
        owner.revalidate(result, INITIAL_SECURITY_ESTABLISHMENT_ONLY)


def test_no_authority_leakage_in_production_surface() -> None:
    import bot_core.runtime.first_run_bootstrap as module

    forbidden = {
        "OperatorIdentity",
        "DeviceTrust",
        "PinVerifierRecord",
        "AuthenticationProof",
        "LiveAccessGrant",
        "ExecutionLease",
        "RiskDecision",
        "ProductCapability",
    }
    assert forbidden.isdisjoint(vars(module))
    assert forbidden.isdisjoint({item.name for item in fields(BootstrapTransitionResult)})


def test_stress_valid_replay_stale_and_tamper_twenty_each() -> None:
    counts = {"valid": 0, "replay": 0, "stale": 0, "tamper": 0}
    for _ in range(20):
        owner, _, registry = authority()
        result, after = consume(owner)
        counts["valid"] += 1
        replay_owner, _, replay_registry = authority()
        replay_registry.states[after.state_fingerprint_sha256] = after
        replay_registry.current[(ACCOUNT, DEVICE)] = after.state_fingerprint_sha256
        with pytest.raises(FirstRunBootstrapError):
            consume(replay_owner, reference=after.state_fingerprint_sha256)
        counts["replay"] += 1
        registry.current[(ACCOUNT, DEVICE)] = after.state_fingerprint_sha256
        with pytest.raises(FirstRunBootstrapError, match="STALE_CORE_BOOTSTRAP_STATE"):
            consume(owner)
        counts["stale"] += 1
        registry.states[after.state_fingerprint_sha256] = after
        with pytest.raises(FirstRunBootstrapError):
            owner.revalidate(
                replace(result, post_state_fingerprint_sha256="f" * 64),
                INITIAL_SECURITY_ESTABLISHMENT_ONLY,
            )
        counts["tamper"] += 1
    assert counts == {"valid": 20, "replay": 20, "stale": 20, "tamper": 20}
