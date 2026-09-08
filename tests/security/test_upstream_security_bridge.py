from __future__ import annotations

from dataclasses import replace
from datetime import datetime
import inspect
from pathlib import Path

import pytest

from bot_core.security.authorization import (
    AuthorizationAuthority,
    _seed_trusted_operation_entitlement,
)
from bot_core.security.authentication import canonical_scope_fingerprint
from tests.security.test_authentication import NOW, RAW_PIN, prepared, request
from tests.security.test_authorization import entitlement
from tests.security.test_platform_biometric_authentication import (
    _seed_external_platform,
    assertion,
)


UPSTREAM_OPERATIONS = (
    "SETUP_PIN",
    "ROTATE_SECRET_REFERENCE",
    "REBIND_SECRET_REFERENCE",
    "ACTIVATE_CREDENTIAL_PROFILE",
    "DEACTIVATE_CREDENTIAL_PROFILE",
    "CHANGE_RISK_POLICY",
    "CHANGE_KILL_SWITCH",
    "CHANGE_PRODUCT_CAPABILITIES",
)


def upstream_arranged(tmp_path: Path, operation: str = "ROTATE_SECRET_REFERENCE"):
    security, authentication, _ = prepared(tmp_path)
    req = request(
        operation, scope_fingerprint_sha256="d" * 64, mutation_fingerprint_sha256="e" * 64
    )
    platform_assertion = assertion(authentication, req)
    _seed_external_platform(authentication, platform_assertion)
    proof = authentication.issue_authentication_proof(
        req, RAW_PIN, NOW, platform_assertion=platform_assertion
    )
    return security, AuthorizationAuthority(authentication), req, proof


def test_exact_public_signature() -> None:
    assert tuple(
        inspect.signature(AuthorizationAuthority.authorize_upstream_security_request).parameters
    ) == ("self", "proof", "request", "now_utc")


@pytest.mark.parametrize("operation", UPSTREAM_OPERATIONS)
def test_all_upstream_operations_authorize_read_only(tmp_path: Path, operation: str) -> None:
    _, owner, req, proof = upstream_arranged(tmp_path, operation)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    before = owner._state.snapshot  # noqa: SLF001 - read-only boundary observation

    assert owner.authorize_upstream_security_request(proof, req, NOW) == (
        "AUTHORIZED_SECURITY_REQUEST"
    )
    assert owner._state.snapshot == before  # noqa: SLF001 - read-only boundary observation


@pytest.mark.parametrize(
    ("req", "expected"),
    [
        (object(), "MALFORMED_UNTRUSTED_CONTEXT"),
        (request(scope_fingerprint_sha256="not-a-sha"), "MALFORMED_UNTRUSTED_CONTEXT"),
        (request("UNKNOWN_OPERATION"), "OPERATION_UNSUPPORTED"),
    ],
)
def test_preliminary_request_results_precede_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, req: object, expected: str
) -> None:
    _, authentication, _ = prepared(tmp_path)
    owner = AuthorizationAuthority(authentication)
    monkeypatch.setattr(owner, "authorize", lambda *_args: pytest.fail("authorize called"))
    assert owner.authorize_upstream_security_request(object(), req, object()) == expected


def test_core_owned_operation_is_fenced_before_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, authentication, _ = prepared(tmp_path)
    owner = AuthorizationAuthority(authentication)
    monkeypatch.setattr(owner, "authorize", lambda *_args: pytest.fail("authorize called"))
    assert owner.authorize_upstream_security_request(object(), request("CHANGE_PIN"), object()) == (
        "OPERATION_UNSUPPORTED"
    )


def test_non_proof_is_flattened(tmp_path: Path) -> None:
    _, owner, req, _ = upstream_arranged(tmp_path)
    assert owner.authorize_upstream_security_request(object(), req, NOW) == ("AUTHORIZATION_DENIED")


def test_invalid_now_is_flattened(tmp_path: Path) -> None:
    _, owner, req, proof = upstream_arranged(tmp_path)
    assert (
        owner.authorize_upstream_security_request(proof, req, datetime(2026, 8, 10))
        == "AUTHORIZATION_DENIED"
    )


def test_wrong_policy_environment_and_missing_entitlement_are_denied(tmp_path: Path) -> None:
    _, owner, req, proof = upstream_arranged(tmp_path)
    assert owner.authorize_upstream_security_request(proof, req, NOW) == "AUTHORIZATION_DENIED"
    assert (
        owner.authorize_upstream_security_request(proof, replace(req, environment="LIVE"), NOW)
        == "AUTHORIZATION_DENIED"
    )


def test_opaque_scope_and_mutation_are_authorized_and_repeatable(tmp_path: Path) -> None:
    _, owner, req, proof = upstream_arranged(tmp_path)
    assert req.scope_fingerprint_sha256 != canonical_scope_fingerprint(req)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    before = owner._state.snapshot  # noqa: SLF001 - read-only boundary observation

    first = owner.authorize_upstream_security_request(proof, req, NOW)
    second = owner.authorize_upstream_security_request(proof, req, NOW)

    assert first == second == "AUTHORIZED_SECURITY_REQUEST"
    assert owner._state.snapshot == before  # noqa: SLF001 - no replay consumption


@pytest.mark.parametrize(
    "change",
    [
        {"scope_fingerprint_sha256": "a" * 64},
        {"mutation_fingerprint_sha256": "b" * 64},
    ],
)
def test_exact_proof_request_binding_is_required(tmp_path: Path, change: dict[str, object]) -> None:
    _, owner, req, proof = upstream_arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    assert owner.authorize_upstream_security_request(proof, replace(req, **change), NOW) == (
        "AUTHORIZATION_DENIED"
    )


def test_expired_proof_is_denied(tmp_path: Path) -> None:
    _, owner, req, proof = upstream_arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    assert (
        owner.authorize_upstream_security_request(proof, req, NOW.replace(minute=NOW.minute + 2))
        == "AUTHORIZATION_DENIED"
    )
