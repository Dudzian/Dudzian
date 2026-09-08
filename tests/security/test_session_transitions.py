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
from tests.security.test_authentication import (
    ACCOUNT,
    DEVICE,
    NOW,
    OPERATOR,
    RAW_PIN,
    prepared,
    request,
)
from tests.security.test_authorization import entitlement


def _request(operation: str, target: str, current_generation: int = 1):
    provisional = request(operation)
    return replace(
        provisional,
        mutation_fingerprint_sha256=session_mutation_fingerprint(
            provisional, target, current_generation, current_generation + 1
        ),
    )


def _arranged(tmp_path: Path, operation: str, target: str):
    security, authentication, _ = prepared(tmp_path)
    authorization = AuthorizationAuthority(authentication)
    req = _request(operation, target)
    proof = authentication.issue_authentication_proof(req, RAW_PIN, NOW)
    return (
        security,
        authorization,
        SessionSecurityTransitionAuthority(authorization),
        req,
        proof,
    )


@pytest.mark.parametrize(
    ("operation", "expected_state"),
    (("LOCK_SESSION", "LOCKED"), ("LOGOUT_SESSION", "LOGGED_OUT")),
)
def test_genuine_transition_preserves_session_history_and_fences_old_proof(
    tmp_path: Path, operation: str, expected_state: str
) -> None:
    security, authorization, owner, req, proof = _arranged(tmp_path, operation, expected_state)
    assert tuple(inspect.signature(owner.transition_session).parameters) == (
        "proof",
        "request",
        "now_utc",
    )
    old = security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE)
    _seed_trusted_operation_entitlement(authorization, entitlement(req))

    assert owner.transition_session(proof, req, NOW) == expected_state
    post = security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE)
    assert (old.state, old.session_generation) == ("UNLOCKED", 1)
    assert (post.state, post.session_generation) == (expected_state, 2)
    assert post.runtime_session_id == old.runtime_session_id
    assert post.security_generation == old.security_generation
    assert old.content_fingerprint_sha256 in security.snapshot.accepted_sessions
    assert security.snapshot.current_sessions[(ACCOUNT, OPERATOR, DEVICE)] == (
        post.content_fingerprint_sha256
    )
    with pytest.raises(AuthorizationError, match="PROOF_STALE"):
        authorization.authorize(proof, req, NOW)


def test_replay_is_atomic_denial(tmp_path: Path) -> None:
    security, authorization, owner, req, proof = _arranged(tmp_path, "LOCK_SESSION", "LOCKED")
    _seed_trusted_operation_entitlement(authorization, entitlement(req))
    assert owner.transition_session(proof, req, NOW) == "LOCKED"
    transitioned = security.snapshot

    assert owner.transition_session(proof, req, NOW) == "AUTHORIZATION_DENIED"
    assert security.snapshot == transitioned


def test_wrong_generation_intent_and_missing_entitlement_are_atomic(tmp_path: Path) -> None:
    security, authorization, owner, req, proof = _arranged(tmp_path, "LOGOUT_SESSION", "LOGGED_OUT")
    before = security.snapshot
    assert owner.transition_session(proof, req, NOW) == "AUTHORIZATION_DENIED"
    assert security.snapshot == before

    _seed_trusted_operation_entitlement(authorization, entitlement(req))
    entitled = security.snapshot
    wrong = replace(
        req,
        mutation_fingerprint_sha256=session_mutation_fingerprint(req, "LOGGED_OUT", 2, 3),
    )
    assert owner.transition_session(proof, wrong, NOW) == "AUTHORIZATION_DENIED"
    assert security.snapshot == entitled


def test_unlock_is_not_an_ordinary_transition(tmp_path: Path) -> None:
    security, authorization, owner, _, proof = _arranged(tmp_path, "LOCK_SESSION", "LOCKED")
    unlock = _request("UNLOCK_SESSION", "UNLOCKED")
    _seed_trusted_operation_entitlement(authorization, entitlement(unlock))
    before = security.snapshot

    assert owner.transition_session(proof, unlock, NOW) == "AUTHORIZATION_DENIED"
    assert security.snapshot == before
