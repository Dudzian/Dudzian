from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path

import pytest

from bot_core.security.authentication import session_mutation_fingerprint
from bot_core.security.authorization import (
    AuthorizationAuthority,
    AuthorizationError,
    _seed_trusted_operation_entitlement,
)
from bot_core.security.session_transitions import SessionSecurityTransitionAuthority
from bot_core.security.session_unlock import SessionUnlockAuthority
from tests.security.test_authentication import (
    ACCOUNT,
    DEVICE,
    NOW,
    OPERATOR,
    RAW_PIN,
    Comparator,
    prepared,
    request,
)
from tests.security.test_authorization import entitlement
from tests.security.test_platform_biometric_authentication import (
    _seed_external_platform,
    assertion,
)


def _transition_request(operation: str, target: str, generation: int):
    provisional = request(operation)
    return replace(
        provisional,
        mutation_fingerprint_sha256=session_mutation_fingerprint(
            provisional, target, generation, generation + 1
        ),
    )


def _locked(tmp_path: Path, comparator: Comparator | None = None):  # type: ignore[no-untyped-def]
    security, authentication, comparison = prepared(tmp_path, comparator)
    authorization = AuthorizationAuthority(authentication)
    transitions = SessionSecurityTransitionAuthority(authorization)
    lock_request = _transition_request("LOCK_SESSION", "LOCKED", 1)
    pre_lock_proof = authentication.issue_authentication_proof(lock_request, RAW_PIN, NOW)
    _seed_trusted_operation_entitlement(authorization, entitlement(lock_request))
    assert transitions.transition_session(pre_lock_proof, lock_request, NOW) == "LOCKED"
    unlock_request = _transition_request("UNLOCK_SESSION", "UNLOCKED", 2)
    owner = SessionUnlockAuthority(authorization)
    return (
        security,
        authentication,
        authorization,
        owner,
        comparison,
        lock_request,
        pre_lock_proof,
        unlock_request,
    )


def _accepted_assertion(authentication, unlock_request, outcome="SUCCESS"):  # type: ignore[no-untyped-def]
    candidate = assertion(authentication, unlock_request, outcome)
    _seed_external_platform(authentication, candidate)
    return candidate


def test_genuine_lock_then_dedicated_unlock_preserves_history_and_fences_proof(
    tmp_path: Path,
) -> None:
    (
        security,
        authentication,
        authorization,
        owner,
        _,
        lock_request,
        pre_lock_proof,
        unlock_request,
    ) = _locked(tmp_path)
    assert tuple(inspect.signature(owner.unlock_session).parameters) == (
        "request",
        "now_utc",
        "raw_pin",
        "assertion",
    )
    locked = security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE)
    assert (locked.state, locked.session_generation) == ("LOCKED", 2)
    _seed_trusted_operation_entitlement(authorization, entitlement(unlock_request))
    candidate = _accepted_assertion(authentication, unlock_request)
    proofs_before = dict(authentication.snapshot.accepted_authentication_proofs)
    bindings_before = dict(authentication.snapshot.accepted_authentication_proof_bindings)

    assert owner.unlock_session(unlock_request, NOW, RAW_PIN, candidate) == "UNLOCKED"
    post = security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE)
    assert (post.state, post.session_generation) == ("UNLOCKED", 3)
    assert post.runtime_session_id == locked.runtime_session_id
    assert post.security_generation == locked.security_generation
    assert locked.content_fingerprint_sha256 in security.snapshot.accepted_sessions
    assert security.snapshot.current_sessions[(ACCOUNT, OPERATOR, DEVICE)] == (
        post.content_fingerprint_sha256
    )
    assert dict(authentication.snapshot.accepted_authentication_proofs) == proofs_before
    assert dict(authentication.snapshot.accepted_authentication_proof_bindings) == bindings_before
    with pytest.raises(AuthorizationError, match="PROOF_STALE"):
        authorization.authorize(pre_lock_proof, lock_request, NOW)

    assert owner.unlock_session(unlock_request, NOW, RAW_PIN, candidate) == "AUTHORIZATION_DENIED"
    assert security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE) == post


def test_wrong_pin_updates_failure_state_but_does_not_unlock(tmp_path: Path) -> None:
    security, authentication, authorization, owner, _, *_, unlock_request = _locked(tmp_path)
    _seed_trusted_operation_entitlement(authorization, entitlement(unlock_request))
    candidate = _accepted_assertion(authentication, unlock_request)
    locked = security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE)
    pin_before = security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE)

    assert owner.unlock_session(unlock_request, NOW, "wrong-pin", candidate) == (
        "AUTHENTICATION_FAILED"
    )
    assert security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE) == locked
    pin_after = security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE)
    assert pin_after.failed_attempts == pin_before.failed_attempts + 1
    assert pin_after.pin_revision == pin_before.pin_revision
    assert pin_after.security_generation == pin_before.security_generation
    assert pin_after.verifier == pin_before.verifier


def test_pin_lockout_and_dependency_failure_are_safe(tmp_path: Path) -> None:
    security, authentication, authorization, owner, _, *_, unlock_request = _locked(tmp_path)
    _seed_trusted_operation_entitlement(authorization, entitlement(unlock_request))
    candidate = _accepted_assertion(authentication, unlock_request)
    locked = security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE)
    assert owner.unlock_session(unlock_request, NOW, "wrong", candidate) == "AUTHENTICATION_FAILED"
    assert owner.unlock_session(unlock_request, NOW, "wrong", candidate) == "AUTHENTICATION_FAILED"
    assert owner.unlock_session(unlock_request, NOW, "wrong", candidate) == "AUTHENTICATION_FAILED"
    assert (
        authentication.verify_current_pin(ACCOUNT, OPERATOR, DEVICE, RAW_PIN, NOW) == "PIN_LOCKED"
    )
    assert security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE) == locked

    leaking = Comparator()
    security2, authentication2, authorization2, owner2, _, *_, request2 = _locked(
        tmp_path / "dependency", leaking
    )
    leaking.leak = True
    _seed_trusted_operation_entitlement(authorization2, entitlement(request2))
    candidate2 = _accepted_assertion(authentication2, request2)
    result = owner2.unlock_session(request2, NOW, RAW_PIN, candidate2)
    assert result == "PIN_VERIFIER_DEPENDENCY_FAILURE"
    assert RAW_PIN not in result
    assert security2.resolve_current_session(ACCOUNT, OPERATOR, DEVICE).state == "LOCKED"


def test_biometric_failure_keeps_session_locked_and_preserves_pin_reset(tmp_path: Path) -> None:
    security, authentication, authorization, owner, _, *_, unlock_request = _locked(tmp_path)
    _seed_trusted_operation_entitlement(authorization, entitlement(unlock_request))
    successful = _accepted_assertion(authentication, unlock_request)
    assert owner.unlock_session(unlock_request, NOW, "wrong", successful) == "AUTHENTICATION_FAILED"
    assert security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE).failed_attempts == 1
    failed = _accepted_assertion(authentication, unlock_request, "FAILED")
    locked = security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE)

    assert owner.unlock_session(unlock_request, NOW, RAW_PIN, failed) == "AUTHENTICATION_FAILED"
    assert security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE) == locked
    reset_pin = security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE)
    assert reset_pin.failed_attempts == 0
    assert reset_pin.lockout_until_utc is None


@pytest.mark.parametrize("case", ["mutation", "entitlement", "scope"])
def test_prefactor_denials_do_not_call_pin_comparator(tmp_path: Path, case: str) -> None:
    security, authentication, authorization, owner, comparator, *_, unlock_request = _locked(
        tmp_path
    )
    if case != "entitlement":
        _seed_trusted_operation_entitlement(authorization, entitlement(unlock_request))
    if case == "mutation":
        unlock_request = replace(
            unlock_request,
            mutation_fingerprint_sha256=session_mutation_fingerprint(
                unlock_request, "UNLOCKED", 3, 4
            ),
        )
    elif case == "scope":
        unlock_request = replace(unlock_request, scope_fingerprint_sha256="f" * 64)
    calls = comparator.calls
    locked = security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE)

    assert owner.unlock_session(unlock_request, NOW, RAW_PIN, object()) == "AUTHORIZATION_DENIED"
    assert comparator.calls == calls
    assert security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE) == locked
    assert authentication.snapshot.current_sessions[(ACCOUNT, OPERATOR, DEVICE)] == (
        locked.content_fingerprint_sha256
    )


def test_entitlement_generation_incoherence_precedes_factors(tmp_path: Path) -> None:
    security, _, authorization, owner, comparator, *_, unlock_request = _locked(tmp_path)
    _seed_trusted_operation_entitlement(authorization, entitlement(unlock_request, generation=2))
    calls = comparator.calls
    locked = security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE)

    assert owner.unlock_session(unlock_request, NOW, RAW_PIN, object()) == "CONTRACT_INCONSISTENT"
    assert comparator.calls == calls
    assert security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE) == locked


def test_logged_out_session_is_not_unlockable(tmp_path: Path) -> None:
    security, authentication, _ = prepared(tmp_path)
    authorization = AuthorizationAuthority(authentication)
    logout = _transition_request("LOGOUT_SESSION", "LOGGED_OUT", 1)
    proof = authentication.issue_authentication_proof(logout, RAW_PIN, NOW)
    _seed_trusted_operation_entitlement(authorization, entitlement(logout))
    transitions = SessionSecurityTransitionAuthority(authorization)
    assert transitions.transition_session(proof, logout, NOW) == "LOGGED_OUT"
    unlock = _transition_request("UNLOCK_SESSION", "UNLOCKED", 2)
    _seed_trusted_operation_entitlement(authorization, entitlement(unlock))
    owner = SessionUnlockAuthority(authorization)
    calls_before = authentication._comparator.calls  # noqa: SLF001

    assert owner.unlock_session(unlock, NOW, RAW_PIN, object()) == "AUTHORIZATION_DENIED"
    assert authentication._comparator.calls == calls_before  # noqa: SLF001
    current = security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE)
    assert (current.state, current.session_generation) == ("LOGGED_OUT", 2)
