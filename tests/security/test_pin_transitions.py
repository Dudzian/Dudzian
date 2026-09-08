from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
from typing import cast

import pytest

from bot_core.security.authentication import AuthenticationError, pin_mutation_fingerprint
from bot_core.security.authorization import (
    AuthorizationAuthority,
    AuthorizationError,
    _seed_trusted_operation_entitlement,
)
from bot_core.security.initial_security import PinVerifierMaterial, PinVerifierRecord
from bot_core.security.pin_transitions import PinSecurityTransitionAuthority
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
from tests.security.test_platform_biometric_authentication import _seed_external_platform, assertion

NEW_PIN = "8642"


class Comparator:
    def compare(self, raw_pin: str, record: PinVerifierRecord) -> bool:
        return (raw_pin, record.verifier) in {(RAW_PIN, "c" * 64), (NEW_PIN, "d" * 64)}


class Factory:
    def __init__(
        self, *, fail: bool = False, invalid: bool = False, malformed: bool = False
    ) -> None:
        self.fail = fail
        self.invalid = invalid
        self.malformed = malformed

    def create(self, raw_pin: str) -> PinVerifierMaterial:
        if self.fail:
            raise RuntimeError(f"dependency exposed {raw_pin}")
        assert raw_pin == NEW_PIN
        if self.malformed:
            return PinVerifierMaterial(
                cast(str, object()),  # deliberate runtime-malformed dependency output
                1,
                "secure-store://opaque/replacement-pin-salt",
                "d" * 64,
            )
        return PinVerifierMaterial(
            "ARGON2ID",
            1,
            "invalid" if self.invalid else "secure-store://opaque/replacement-pin-salt",
            "d" * 64,
        )


def _request(operation: str, current_revision: int = 1):
    provisional = request(operation)
    return replace(
        provisional,
        mutation_fingerprint_sha256=pin_mutation_fingerprint(
            provisional, current_revision, current_revision + 1
        ),
    )


def _arranged(tmp_path: Path, operation: str, *, factory: Factory | None = None):
    security, authentication, _ = prepared(tmp_path, Comparator())
    authorization = AuthorizationAuthority(authentication)
    req = _request(operation)
    platform_assertion = assertion(authentication, req)
    _seed_external_platform(authentication, platform_assertion)
    proof = authentication.issue_authentication_proof(
        req, RAW_PIN, NOW, platform_assertion=platform_assertion
    )
    return (
        security,
        authentication,
        authorization,
        PinSecurityTransitionAuthority(authorization, factory or Factory()),
        req,
        proof,
    )


@pytest.mark.parametrize("operation", ["CHANGE_PIN", "RESET_PIN"])
def test_genuine_transition_preserves_history_and_fences_old_proof(
    tmp_path: Path, operation: str
) -> None:
    security, _, authorization, owner, req, proof = _arranged(tmp_path, operation)
    assert tuple(inspect.signature(owner.transition_pin).parameters) == (
        "proof",
        "request",
        "now_utc",
        "new_pin",
    )
    old = security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE)
    _seed_trusted_operation_entitlement(authorization, entitlement(req))

    assert owner.transition_pin(proof, req, NOW, NEW_PIN) == "PIN_CHANGED"
    post = security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE)
    assert (old.pin_revision, post.pin_revision) == (1, 2)
    assert post.security_generation == old.security_generation
    assert (post.failed_attempts, post.lockout_until_utc) == (0, None)
    assert old.content_fingerprint_sha256 in security.snapshot.accepted_pins
    assert security.snapshot.current_pins[(ACCOUNT, OPERATOR, DEVICE)] == (
        post.content_fingerprint_sha256
    )
    with pytest.raises(AuthorizationError, match="PROOF_STALE"):
        authorization.authorize(proof, req, NOW)


def test_replacement_material_authenticates_new_pin_and_rejects_old_pin(tmp_path: Path) -> None:
    security, authentication, authorization, owner, req, proof = _arranged(tmp_path, "CHANGE_PIN")
    _seed_trusted_operation_entitlement(authorization, entitlement(req))
    assert owner.transition_pin(proof, req, NOW, NEW_PIN) == "PIN_CHANGED"

    next_req = _request("RESET_PIN", 2)
    next_assertion = assertion(authentication, next_req)
    _seed_external_platform(authentication, next_assertion)
    new_proof = authentication.issue_authentication_proof(
        next_req, NEW_PIN, NOW, platform_assertion=next_assertion
    )
    assert new_proof.pin_revision == 2
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
        authentication.issue_authentication_proof(
            next_req, RAW_PIN, NOW, platform_assertion=next_assertion
        )
    assert security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE).failed_attempts == 1


def test_replay_wrong_revision_and_missing_entitlement_are_atomic(tmp_path: Path) -> None:
    security, _, authorization, owner, req, proof = _arranged(tmp_path, "CHANGE_PIN")
    before = security.snapshot
    assert owner.transition_pin(proof, req, NOW, NEW_PIN) == "AUTHORIZATION_DENIED"
    assert security.snapshot == before
    _seed_trusted_operation_entitlement(authorization, entitlement(req))
    entitled = security.snapshot
    wrong = replace(req, mutation_fingerprint_sha256=pin_mutation_fingerprint(req, 2, 3))
    assert owner.transition_pin(proof, wrong, NOW, NEW_PIN) == "AUTHORIZATION_DENIED"
    assert security.snapshot == entitled
    assert owner.transition_pin(proof, req, NOW, NEW_PIN) == "PIN_CHANGED"
    transitioned = security.snapshot
    assert owner.transition_pin(proof, req, NOW, NEW_PIN) == "AUTHORIZATION_DENIED"
    assert security.snapshot == transitioned


@pytest.mark.parametrize("new_pin", [object(), None, 1234])
def test_non_string_pin_is_atomic_denial(tmp_path: Path, new_pin: object) -> None:
    security, _, authorization, owner, req, proof = _arranged(tmp_path, "RESET_PIN")
    _seed_trusted_operation_entitlement(authorization, entitlement(req))
    before = security.snapshot
    assert owner.transition_pin(proof, req, NOW, new_pin) == "AUTHORIZATION_DENIED"
    assert security.snapshot == before


@pytest.mark.parametrize("factory", [Factory(fail=True), Factory(invalid=True)])
def test_factory_failures_are_pin_safe_and_do_not_publish(tmp_path: Path, factory: Factory) -> None:
    security, _, authorization, owner, req, proof = _arranged(
        tmp_path, "CHANGE_PIN", factory=factory
    )
    _seed_trusted_operation_entitlement(authorization, entitlement(req))
    before = security.snapshot
    result = owner.transition_pin(proof, req, NOW, NEW_PIN)
    assert result == "PIN_VERIFIER_DEPENDENCY_FAILURE"
    assert NEW_PIN not in result
    assert security.snapshot == before


def test_non_json_safe_material_is_controlled_and_does_not_publish(tmp_path: Path) -> None:
    security, _, authorization, owner, req, proof = _arranged(
        tmp_path, "CHANGE_PIN", factory=Factory(malformed=True)
    )
    _seed_trusted_operation_entitlement(authorization, entitlement(req))
    before = security.snapshot

    assert owner.transition_pin(proof, req, NOW, NEW_PIN) == "PIN_VERIFIER_DEPENDENCY_FAILURE"
    assert security.snapshot == before
