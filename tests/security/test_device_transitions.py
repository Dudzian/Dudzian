from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
from types import MappingProxyType

import pytest

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.security.authentication import device_mutation_fingerprint
from bot_core.security.authorization import (
    AuthorizationAuthority,
    AuthorizationError,
    _seed_trusted_operation_entitlement,
)
from bot_core.security.device_transitions import DeviceTrustTransitionAuthority
from bot_core.security.initial_security import DeviceTrustProjection
from tests.security.test_authentication import ACCOUNT, DEVICE, NOW, RAW_PIN, prepared, request
from tests.security.test_authorization import entitlement
from tests.security.test_platform_biometric_authentication import (
    _seed_external_platform,
    assertion,
)

TARGET_A = "dev_018f0000-0000-7000-8000-000000000021"
TARGET_B = "dev_018f0000-0000-7000-8000-000000000022"


def _request(operation: str, target: str, current_revision: int):
    provisional = request(operation)
    return replace(
        provisional,
        mutation_fingerprint_sha256=device_mutation_fingerprint(
            provisional, target, current_revision, current_revision + 1
        ),
    )


def _arranged(tmp_path: Path, operation: str, target: str, current_revision: int = 0):
    security, authentication, _ = prepared(tmp_path)
    authorization = AuthorizationAuthority(authentication)
    req = _request(operation, target, current_revision)
    platform_assertion = assertion(authentication, req)
    _seed_external_platform(authentication, platform_assertion)
    proof = authentication.issue_authentication_proof(
        req, RAW_PIN, NOW, platform_assertion=platform_assertion
    )
    return (
        security,
        authentication,
        authorization,
        DeviceTrustTransitionAuthority(authorization),
        req,
        proof,
    )


def _seed_device(authorization: AuthorizationAuthority, value: DeviceTrustProjection) -> None:
    """Test-only fixture for authority accepted by a preceding enrollment owner."""
    value = replace(
        value,
        content_fingerprint_sha256=canonical_json_sha256(
            {
                "account_id": value.account_id,
                "device_installation_id": value.device_installation_id,
                "state": value.state,
                "trust_revision": value.trust_revision,
                "security_generation": value.security_generation,
                "platform_enrollment_revision": value.platform_enrollment_revision,
            }
        ),
    )
    with authorization._state.lock:  # noqa: SLF001 - module-private trusted test fixture
        before = authorization._state.snapshot  # noqa: SLF001
        accepted = dict(before.accepted_devices)
        current = dict(before.current_devices)
        accepted[value.content_fingerprint_sha256] = value
        current[(value.account_id, value.device_installation_id)] = value.content_fingerprint_sha256
        authorization._state.snapshot = replace(  # noqa: SLF001
            before,
            accepted_devices=MappingProxyType(accepted),
            current_devices=MappingProxyType(current),
        )


def test_public_surface_and_genuine_absent_to_trusted(tmp_path: Path) -> None:
    security, _, authorization, owner, req, proof = _arranged(tmp_path, "TRUST_DEVICE", TARGET_A)
    assert tuple(inspect.signature(owner.transition_device).parameters) == (
        "proof",
        "request",
        "now_utc",
        "target_device_id",
    )
    _seed_trusted_operation_entitlement(authorization, entitlement(req))

    assert owner.transition_device(proof, req, NOW, TARGET_A) == "TRUSTED"
    post = security.resolve_current_device(ACCOUNT, TARGET_A)
    assert (post.state, post.trust_revision, post.security_generation) == ("TRUSTED", 1, 1)
    assert post.platform_enrollment_revision == 1
    assert post.content_fingerprint_sha256 in security.snapshot.accepted_devices


def test_replay_and_trusted_to_trusted_are_atomic_denials(tmp_path: Path) -> None:
    security, _, authorization, owner, req, proof = _arranged(tmp_path, "TRUST_DEVICE", TARGET_A)
    _seed_trusted_operation_entitlement(authorization, entitlement(req))
    assert owner.transition_device(proof, req, NOW, TARGET_A) == "TRUSTED"
    before = security.snapshot
    assert owner.transition_device(proof, req, NOW, TARGET_A) == "AUTHORIZATION_DENIED"
    assert security.snapshot == before

    next_req = _request("TRUST_DEVICE", TARGET_A, 1)
    next_assertion = assertion(owner._authentication, next_req)  # noqa: SLF001
    _seed_external_platform(owner._authentication, next_assertion)  # noqa: SLF001
    next_proof = owner._authentication.issue_authentication_proof(  # noqa: SLF001
        next_req, RAW_PIN, NOW, platform_assertion=next_assertion
    )
    _seed_trusted_operation_entitlement(authorization, entitlement(next_req))
    before_trusted_denial = security.snapshot
    assert owner.transition_device(next_proof, next_req, NOW, TARGET_A) == "AUTHORIZATION_DENIED"
    assert security.snapshot == before_trusted_denial


def test_enrolled_untrusted_to_trusted_preserves_history(tmp_path: Path) -> None:
    security, _, authorization, owner, req, proof = _arranged(tmp_path, "TRUST_DEVICE", TARGET_A, 3)
    enrolled = DeviceTrustProjection(ACCOUNT, TARGET_A, "ENROLLED_UNTRUSTED", 3, 1, 7, "")
    _seed_device(authorization, enrolled)
    old_fingerprint = security.snapshot.current_devices[(ACCOUNT, TARGET_A)]
    _seed_trusted_operation_entitlement(authorization, entitlement(req))

    assert owner.transition_device(proof, req, NOW, TARGET_A) == "TRUSTED"
    post = security.resolve_current_device(ACCOUNT, TARGET_A)
    assert (post.trust_revision, post.platform_enrollment_revision) == (4, 7)
    assert old_fingerprint in security.snapshot.accepted_devices
    assert security.snapshot.current_devices[(ACCOUNT, TARGET_A)] == post.content_fingerprint_sha256


def test_genuine_trusted_actor_to_revoked_fences_existing_proof(tmp_path: Path) -> None:
    security, _, authorization, owner, req, proof = _arranged(tmp_path, "REVOKE_DEVICE", DEVICE, 1)
    trusted_fingerprint = security.snapshot.current_devices[(ACCOUNT, DEVICE)]
    _seed_trusted_operation_entitlement(authorization, entitlement(req))

    assert owner.transition_device(proof, req, NOW, DEVICE) == "REVOKED"
    post = security.resolve_current_device(ACCOUNT, DEVICE)
    assert (post.state, post.trust_revision, post.security_generation) == ("REVOKED", 2, 1)
    assert trusted_fingerprint in security.snapshot.accepted_devices
    assert security.snapshot.current_devices[(ACCOUNT, DEVICE)] == post.content_fingerprint_sha256
    with pytest.raises(AuthorizationError, match="DEVICE_NOT_TRUSTED"):
        authorization.authorize(proof, req, NOW)


def test_target_binding_missing_entitlement_and_malformed_target_are_atomic(
    tmp_path: Path,
) -> None:
    security, _, authorization, owner, req, proof = _arranged(tmp_path, "TRUST_DEVICE", TARGET_A)
    before = security.snapshot
    assert owner.transition_device(proof, req, NOW, TARGET_A) == "AUTHORIZATION_DENIED"
    assert security.snapshot == before
    _seed_trusted_operation_entitlement(authorization, entitlement(req))
    entitled = security.snapshot
    assert owner.transition_device(proof, req, NOW, TARGET_B) == "AUTHORIZATION_DENIED"
    assert owner.transition_device(proof, req, NOW, "not-a-device") == "AUTHORIZATION_DENIED"
    assert security.snapshot == entitled
