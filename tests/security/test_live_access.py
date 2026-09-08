from __future__ import annotations

from dataclasses import replace
import inspect
from pathlib import Path
from types import MappingProxyType

import pytest

from bot_core.security.authentication import canonical_scope_fingerprint
from bot_core.security.authorization import (
    AuthorizationAuthority,
    _seed_trusted_operation_entitlement,
)
from bot_core.security.initial_security import LiveAccessGrantSecurityProjection
from bot_core.security.live_access import (
    LiveAccessGrantAuthority,
    live_access_grant_fingerprint,
    live_mutation_fingerprint,
)
from tests.security.test_authentication import (
    ACCOUNT,
    DEVICE,
    NOW,
    OPERATOR,
    RAW_PIN,
    prepared,
    request,
)
from tests.security.test_authorization import _install_current_change, entitlement
from tests.security.test_platform_biometric_authentication import (
    _seed_external_platform,
    assertion,
)

GRANT_A = "lgrant_018f0000-0000-7000-8000-000000000021"
GRANT_B = "lgrant_018f0000-0000-7000-8000-000000000022"
POLICY_SCOPE = "c" * 64


def _live_request(
    operation: str,
    grant_id: str,
    revision: int,
    *,
    policy_scope: str = POLICY_SCOPE,
    **changes: object,
):  # type: ignore[no-untyped-def]
    provisional = request(operation, "LIVE", **changes)
    provisional = replace(
        provisional, scope_fingerprint_sha256=canonical_scope_fingerprint(provisional)
    )
    target = {
        "GRANT_LIVE_ACCESS": "ACTIVE",
        "SUSPEND_LIVE_ACCESS": "SUSPENDED",
        "REVOKE_LIVE_ACCESS": "REVOKED",
    }[operation]
    return replace(
        provisional,
        mutation_fingerprint_sha256=live_mutation_fingerprint(
            provisional, grant_id, target, policy_scope, revision, revision + 1, 1
        ),
    )


def _arranged(tmp_path: Path):  # type: ignore[no-untyped-def]
    security, authentication, _ = prepared(tmp_path)
    authorization = AuthorizationAuthority(authentication)
    return security, authentication, authorization, LiveAccessGrantAuthority(authorization)


def _genuine(authentication, authorization, req):  # type: ignore[no-untyped-def]
    biometric = assertion(authentication, req)
    _seed_external_platform(authentication, biometric)
    proof = authentication.issue_authentication_proof(
        req, RAW_PIN, NOW, platform_assertion=biometric
    )
    _seed_trusted_operation_entitlement(authorization, entitlement(req))
    return proof


def _transition(authentication, authorization, owner, operation, grant_id, revision):  # type: ignore[no-untyped-def]
    req = _live_request(operation, grant_id, revision)
    proof = _genuine(authentication, authorization, req)
    return req, proof, owner.transition_live_grant(proof, req, NOW, grant_id, POLICY_SCOPE)


def test_exact_schema_shared_state_and_no_public_acceptance(tmp_path: Path) -> None:
    security, _, _, owner = _arranged(tmp_path)
    assert tuple(LiveAccessGrantSecurityProjection.__dataclass_fields__) == (
        "live_access_grant_id",
        "account_id",
        "operator_id",
        "device_installation_id",
        "policy_scope_fingerprint_sha256",
        "state",
        "grant_revision",
        "security_generation",
        "content_fingerprint_sha256",
    )
    assert tuple(inspect.signature(owner.transition_live_grant).parameters) == (
        "proof",
        "request",
        "now_utc",
        "grant_id",
        "policy_scope_fingerprint_sha256",
    )
    assert tuple(inspect.signature(owner.validate_live_grant).parameters) == ("untrusted",)
    assert not any(
        hasattr(owner, name)
        for name in ("accept", "register", "seed", "set_current", "mark_current", "commit")
    )
    assert isinstance(security.snapshot.accepted_live_access_grants, MappingProxyType)
    assert isinstance(security.snapshot.current_live_access_grants, MappingProxyType)


def test_genuine_grant_and_replay(tmp_path: Path) -> None:
    security, authentication, authorization, owner = _arranged(tmp_path)
    req, proof, (status, post) = _transition(
        authentication, authorization, owner, "GRANT_LIVE_ACCESS", GRANT_A, 0
    )
    assert status == "ACTIVE" and post is not None
    assert (
        post.live_access_grant_id,
        post.account_id,
        post.operator_id,
        post.device_installation_id,
        post.policy_scope_fingerprint_sha256,
        post.state,
        post.grant_revision,
        post.security_generation,
    ) == (GRANT_A, ACCOUNT, OPERATOR, DEVICE, POLICY_SCOPE, "ACTIVE", 1, proof.security_generation)
    assert security.snapshot.accepted_live_access_grants[post.content_fingerprint_sha256] == post
    assert security.snapshot.current_live_access_grants[(ACCOUNT, DEVICE, POLICY_SCOPE)] == (
        post.content_fingerprint_sha256
    )
    assert owner.validate_live_grant(post) == "GRANT_ACTIVE"
    after = security.snapshot
    assert owner.transition_live_grant(proof, req, NOW, GRANT_A, POLICY_SCOPE) == (
        "AUTHORIZATION_DENIED",
        None,
    )
    assert security.snapshot == after


def test_active_suspend_revoke_preserves_history_and_fences_old(tmp_path: Path) -> None:
    security, authentication, authorization, owner = _arranged(tmp_path)
    _, _, (_, active) = _transition(
        authentication, authorization, owner, "GRANT_LIVE_ACCESS", GRANT_A, 0
    )
    _, _, (status, suspended) = _transition(
        authentication, authorization, owner, "SUSPEND_LIVE_ACCESS", GRANT_A, 1
    )
    assert status == "SUSPENDED" and active is not None and suspended is not None
    assert suspended.grant_revision == 2
    assert owner.validate_live_grant(active) == "PROOF_STALE"
    assert owner.validate_live_grant(suspended) == "AUTHORIZATION_DENIED"
    _, _, (status, revoked) = _transition(
        authentication, authorization, owner, "REVOKE_LIVE_ACCESS", GRANT_A, 2
    )
    assert status == "REVOKED" and revoked is not None
    assert revoked.grant_revision == 3
    assert owner.validate_live_grant(suspended) == "PROOF_STALE"
    assert owner.validate_live_grant(revoked) == "AUTHORIZATION_DENIED"
    assert len(security.snapshot.accepted_live_access_grants) == 3


def test_direct_revoke_and_fresh_regrant_but_same_id_reuse_denied(tmp_path: Path) -> None:
    security, authentication, authorization, owner = _arranged(tmp_path)
    _transition(authentication, authorization, owner, "GRANT_LIVE_ACCESS", GRANT_A, 0)
    _, _, (status, revoked) = _transition(
        authentication, authorization, owner, "REVOKE_LIVE_ACCESS", GRANT_A, 1
    )
    assert status == "REVOKED" and revoked is not None and revoked.grant_revision == 2
    reuse_req = _live_request("GRANT_LIVE_ACCESS", GRANT_A, 0)
    reuse_proof = _genuine(authentication, authorization, reuse_req)
    assert owner.transition_live_grant(reuse_proof, reuse_req, NOW, GRANT_A, POLICY_SCOPE) == (
        "AUTHORIZATION_DENIED",
        None,
    )
    _, _, (status, fresh) = _transition(
        authentication, authorization, owner, "GRANT_LIVE_ACCESS", GRANT_B, 0
    )
    assert status == "ACTIVE" and fresh is not None and fresh.grant_revision == 1
    assert revoked.content_fingerprint_sha256 in security.snapshot.accepted_live_access_grants


@pytest.mark.parametrize("occupied_state", ["ACTIVE", "SUSPENDED"])
def test_second_identity_denied_while_scope_occupied(tmp_path: Path, occupied_state: str) -> None:
    security, authentication, authorization, owner = _arranged(tmp_path)
    _transition(authentication, authorization, owner, "GRANT_LIVE_ACCESS", GRANT_A, 0)
    if occupied_state == "SUSPENDED":
        _transition(authentication, authorization, owner, "SUSPEND_LIVE_ACCESS", GRANT_A, 1)
    req = _live_request("GRANT_LIVE_ACCESS", GRANT_B, 0)
    proof = _genuine(authentication, authorization, req)
    before = security.snapshot
    assert owner.transition_live_grant(proof, req, NOW, GRANT_B, POLICY_SCOPE) == (
        "AUTHORIZATION_DENIED",
        None,
    )
    assert security.snapshot == before


@pytest.mark.parametrize(
    ("request_change", "policy_scope"),
    [
        ({"device_installation_id": "dev_018f0000-0000-7000-8000-000000000099"}, POLICY_SCOPE),
        ({}, "d" * 64),
    ],
)
def test_grant_identity_cannot_be_reparented(
    tmp_path: Path, request_change: dict[str, object], policy_scope: str
) -> None:
    security, authentication, authorization, owner = _arranged(tmp_path)
    _transition(authentication, authorization, owner, "GRANT_LIVE_ACCESS", GRANT_A, 0)
    req = _live_request(
        "SUSPEND_LIVE_ACCESS", GRANT_A, 1, policy_scope=policy_scope, **request_change
    )
    before = security.snapshot
    assert owner.transition_live_grant(object(), req, NOW, GRANT_A, policy_scope) == (
        "AUTHORIZATION_DENIED",
        None,
    )
    assert security.snapshot == before


def test_wrong_mutation_and_missing_entitlement_do_not_publish(tmp_path: Path) -> None:
    security, authentication, authorization, owner = _arranged(tmp_path)
    req = replace(
        _live_request("GRANT_LIVE_ACCESS", GRANT_A, 0), mutation_fingerprint_sha256="d" * 64
    )
    biometric = assertion(authentication, req)
    _seed_external_platform(authentication, biometric)
    proof = authentication.issue_authentication_proof(
        req, RAW_PIN, NOW, platform_assertion=biometric
    )
    before = security.snapshot
    assert owner.transition_live_grant(proof, req, NOW, GRANT_A, POLICY_SCOPE)[0] == (
        "AUTHORIZATION_DENIED"
    )
    assert security.snapshot == before

    req = _live_request("GRANT_LIVE_ACCESS", GRANT_A, 0)
    biometric = assertion(authentication, req)
    _seed_external_platform(authentication, biometric)
    proof = authentication.issue_authentication_proof(
        req, RAW_PIN, NOW, platform_assertion=biometric
    )
    before = security.snapshot
    assert owner.transition_live_grant(proof, req, NOW, GRANT_A, POLICY_SCOPE)[0] == (
        "AUTHORIZATION_DENIED"
    )
    assert security.snapshot == before


def test_validation_intrinsic_membership_and_generation_fences(tmp_path: Path) -> None:
    _, authentication, authorization, owner = _arranged(tmp_path)
    nominal = LiveAccessGrantSecurityProjection(
        GRANT_A, ACCOUNT, OPERATOR, DEVICE, POLICY_SCOPE, "ACTIVE", 1, 1, ""
    )
    nominal = replace(nominal, content_fingerprint_sha256=live_access_grant_fingerprint(nominal))
    assert owner.validate_live_grant(nominal) == "PROOF_STALE"
    malformed = replace(nominal, grant_revision=True)
    malformed = replace(
        malformed, content_fingerprint_sha256=live_access_grant_fingerprint(malformed)
    )
    assert owner.validate_live_grant(malformed) == "AUTHORIZATION_DENIED"

    _, _, (_, active) = _transition(
        authentication, authorization, owner, "GRANT_LIVE_ACCESS", GRANT_A, 0
    )
    assert active is not None
    _install_current_change(
        authorization, "accepted_identities", "current_identities", security_generation=2
    )
    _install_current_change(
        authorization, "accepted_devices", "current_devices", security_generation=2
    )
    assert owner.validate_live_grant(active) == "PROOF_STALE"
