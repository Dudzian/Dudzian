from __future__ import annotations

from dataclasses import asdict, replace
from datetime import timedelta, timezone
import inspect
from pathlib import Path
from types import MappingProxyType

import pytest

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.security.authorization import (
    AuthorizationAuthority,
    AuthorizationError,
    OperationEntitlementProjection,
    _seed_trusted_operation_entitlement,
    operation_entitlement_fingerprint,
)
from bot_core.security.authentication import (
    OPERATION_OWNERSHIP,
    OPERATION_POLICY_REGISTRY,
    authentication_proof_fingerprint,
    canonical_scope_fingerprint,
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
from tests.security.test_platform_biometric_authentication import (
    _seed_external_platform,
    assertion,
)


def entitlement(req, *, generation: int = 1, **changes: object):  # type: ignore[no-untyped-def]
    policy = OPERATION_POLICY_REGISTRY[req.operation]
    item = OperationEntitlementProjection(
        req.account_id,
        req.operator_id,
        req.operation,
        req.environment,
        policy.authorization_scope,
        1,
        generation,
        "",
    )
    item = replace(item, **changes)  # type: ignore[arg-type]
    return replace(item, content_fingerprint_sha256=operation_entitlement_fingerprint(item))


def arranged(tmp_path: Path):  # type: ignore[no-untyped-def]
    security, authentication, _ = prepared(tmp_path)
    req = request()
    proof = authentication.issue_authentication_proof(req, RAW_PIN, NOW)
    return security, authentication, AuthorizationAuthority(authentication), req, proof


def test_exact_schema_and_public_surface() -> None:
    assert tuple(OperationEntitlementProjection.__dataclass_fields__) == (
        "account_id",
        "operator_id",
        "operation",
        "environment",
        "authorization_scope",
        "entitlement_revision",
        "security_generation",
        "content_fingerprint_sha256",
    )
    assert tuple(inspect.signature(AuthorizationAuthority.authorize).parameters) == (
        "self",
        "proof",
        "request",
        "now_utc",
    )
    assert not any(
        hasattr(AuthorizationAuthority, name)
        for name in (
            "grant_entitlement",
            "accept_entitlement",
            "register_entitlement",
            "set_entitlement",
            "set_current_entitlement",
            "authorize_entitlement",
        )
    )


def test_genuine_proof_without_entitlement_is_denied(tmp_path: Path) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    with pytest.raises(AuthorizationError, match="AUTHORIZATION_DENIED"):
        owner.authorize(proof, req, NOW)


def test_exact_current_entitlement_authorizes_without_mutation(tmp_path: Path) -> None:
    security, authentication, owner, req, proof = arranged(tmp_path)
    before_family = (
        security.resolve_current_identity(ACCOUNT, OPERATOR),
        security.resolve_current_device(ACCOUNT, DEVICE),
        security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE),
        security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE),
    )
    before_proofs = authentication.snapshot.accepted_authentication_proofs
    _seed_trusted_operation_entitlement(owner, entitlement(req))

    assert owner.authorize(proof, req, NOW) == "AUTHORIZED"
    assert before_family == (
        security.resolve_current_identity(ACCOUNT, OPERATOR),
        security.resolve_current_device(ACCOUNT, DEVICE),
        security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE),
        security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE),
    )
    assert authentication.snapshot.accepted_authentication_proofs == before_proofs


def test_self_hashed_nominal_entitlement_cannot_self_enroll(tmp_path: Path) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    nominal = entitlement(req)
    assert operation_entitlement_fingerprint(nominal) == nominal.content_fingerprint_sha256
    with pytest.raises(TypeError):
        owner.authorize(proof, req, NOW, nominal)  # type: ignore[call-arg]
    with pytest.raises(AuthorizationError, match="AUTHORIZATION_DENIED"):
        owner.authorize(proof, req, NOW)


def test_self_hashed_nominal_proof_is_not_core_authority(tmp_path: Path) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    nominal = replace(proof, correlation_id="nominal", proof_fingerprint_sha256="")
    nominal = replace(nominal, proof_fingerprint_sha256=authentication_proof_fingerprint(nominal))
    with pytest.raises(AuthorizationError, match="AUTHENTICATION_REQUIRED"):
        owner.authorize(nominal, replace(req, correlation_id="nominal"), NOW)


def test_accepted_but_not_current_entitlement_is_denied(tmp_path: Path) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req), current=False)
    with pytest.raises(AuthorizationError, match="AUTHORIZATION_DENIED"):
        owner.authorize(proof, req, NOW)


@pytest.mark.parametrize(
    "changes",
    [
        {"account_id": "not-an-account"},
        {"operator_id": "not-an-operator"},
        {"operation": "UNKNOWN_OPERATION"},
        {"environment": "LIVE"},
        {"authorization_scope": "not_lock_session"},
        {"entitlement_revision": True},
        {"security_generation": True},
        {"content_fingerprint_sha256": "f" * 64},
    ],
)
def test_trusted_seed_rejects_noncanonical_entitlement_without_mutation(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    _, _, owner, req, _ = arranged(tmp_path)
    item = entitlement(req)
    item = replace(item, **changes)  # type: ignore[arg-type]
    if "content_fingerprint_sha256" not in changes:
        item = replace(item, content_fingerprint_sha256=operation_entitlement_fingerprint(item))
    before = owner._state.snapshot  # noqa: SLF001 - trusted-boundary observation
    with pytest.raises(AuthorizationError, match="CONTRACT_INCONSISTENT"):
        _seed_trusted_operation_entitlement(owner, item)
    assert owner._state.snapshot == before  # noqa: SLF001 - no admission mutation


def test_entitlement_generation_inconsistency_is_contract_error(tmp_path: Path) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req, generation=2))
    with pytest.raises(AuthorizationError, match="CONTRACT_INCONSISTENT"):
        owner.authorize(proof, req, NOW)


def test_missing_entitlement_precedes_current_family_generation_incoherence(
    tmp_path: Path,
) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    _install_current_change(
        owner,
        "accepted_devices",
        "current_devices",
        security_generation=2,
    )

    with pytest.raises(AuthorizationError) as denied:
        owner.authorize(proof, req, NOW)
    assert denied.value.reason == "AUTHORIZATION_DENIED"

    _seed_trusted_operation_entitlement(owner, entitlement(req))
    with pytest.raises(AuthorizationError) as inconsistent:
        owner.authorize(proof, req, NOW)
    assert inconsistent.value.reason == "CONTRACT_INCONSISTENT"


def test_dangling_current_entitlement_pointer_is_authorization_denied(tmp_path: Path) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    item = entitlement(req)
    _seed_trusted_operation_entitlement(owner, item)
    before = owner._state.snapshot  # noqa: SLF001 - corruption fixture
    owner._state.snapshot = replace(  # noqa: SLF001 - corruption fixture
        before,
        accepted_operation_entitlements=MappingProxyType({}),
    )
    with pytest.raises(AuthorizationError, match="AUTHORIZATION_DENIED"):
        owner.authorize(proof, req, NOW)


@pytest.mark.parametrize(
    "tampered_fingerprint",
    ["f" * 64, "not-a-canonical-fingerprint"],
)
def test_current_entitlement_with_bad_terminal_fingerprint_is_denied(
    tmp_path: Path, tampered_fingerprint: str
) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    item = entitlement(req)
    _seed_trusted_operation_entitlement(owner, item)
    before = owner._state.snapshot  # noqa: SLF001 - corruption fixture
    accepted = dict(before.accepted_operation_entitlements)
    accepted[item.content_fingerprint_sha256] = replace(
        item, content_fingerprint_sha256=tampered_fingerprint
    )
    owner._state.snapshot = replace(  # noqa: SLF001 - corruption fixture
        before,
        accepted_operation_entitlements=MappingProxyType(accepted),
    )
    with pytest.raises(AuthorizationError, match="AUTHORIZATION_DENIED"):
        owner.authorize(proof, req, NOW)


@pytest.mark.parametrize(
    "change",
    [
        {"mutation_fingerprint_sha256": "a" * 64},
        {"causation_id": "different-cause"},
        {"correlation_id": "different-correlation"},
        {"scope_fingerprint_sha256": "b" * 64},
        {"operation": "LOGOUT_SESSION"},
        {"environment": "PAPER"},
    ],
)
def test_proof_cannot_be_replayed_across_request_context(
    tmp_path: Path, change: dict[str, object]
) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    with pytest.raises(AuthorizationError, match="AUTHORIZATION_DENIED"):
        owner.authorize(proof, replace(req, **change), NOW)


def test_time_boundaries_and_expiry_are_controlled(tmp_path: Path) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    assert owner.authorize(proof, req, NOW) == "AUTHORIZED"
    assert owner.authorize(proof, req, NOW + timedelta(seconds=60)) == "AUTHORIZED"
    with pytest.raises(AuthorizationError, match="PROOF_EXPIRED"):
        owner.authorize(proof, req, NOW + timedelta(seconds=60, microseconds=1))
    with pytest.raises(AuthorizationError, match="MALFORMED_UNTRUSTED_CONTEXT"):
        owner.authorize(proof, req, NOW.replace(tzinfo=None))
    with pytest.raises(AuthorizationError, match="MALFORMED_UNTRUSTED_CONTEXT"):
        owner.authorize(proof, req, NOW.astimezone(timezone(timedelta(hours=1))))


def test_now_before_issued_requires_authentication(tmp_path: Path) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    with pytest.raises(AuthorizationError, match="AUTHENTICATION_REQUIRED"):
        owner.authorize(proof, req, NOW - timedelta(microseconds=1))


def _install_current_change(owner: AuthorizationAuthority, collection: str, current: str, **change):  # type: ignore[no-untyped-def]
    before = owner._state.snapshot  # noqa: SLF001 - legal state-change fixture
    accepted = dict(getattr(before, collection))
    designations = dict(getattr(before, current))
    old_fingerprint = next(iter(designations.values()))
    value = replace(
        accepted[old_fingerprint],
        **change,
        content_fingerprint_sha256="",
    )
    fingerprint = canonical_json_sha256(
        {key: field for key, field in asdict(value).items() if key != "content_fingerprint_sha256"}
    )
    value = replace(value, content_fingerprint_sha256=fingerprint)
    accepted[fingerprint] = value
    designations[next(iter(designations))] = fingerprint
    owner._state.snapshot = replace(  # noqa: SLF001 - legal state-change fixture
        before,
        **{
            collection: MappingProxyType(accepted),
            current: MappingProxyType(designations),
        },
    )


def test_revoked_identity_has_authorization_specific_reason(tmp_path: Path) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    _install_current_change(owner, "accepted_identities", "current_identities", state="REVOKED")
    with pytest.raises(AuthorizationError, match="IDENTITY_REVOKED"):
        owner.authorize(proof, req, NOW)


def test_locked_session_makes_proof_stale(tmp_path: Path) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    _install_current_change(owner, "accepted_sessions", "current_sessions", state="LOCKED")
    with pytest.raises(AuthorizationError, match="PROOF_STALE"):
        owner.authorize(proof, req, NOW)


def test_non_trusted_device_has_authorization_specific_reason(tmp_path: Path) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    _install_current_change(
        owner, "accepted_devices", "current_devices", state="ENROLLED_UNTRUSTED"
    )
    with pytest.raises(AuthorizationError, match="DEVICE_NOT_TRUSTED"):
        owner.authorize(proof, req, NOW)


@pytest.mark.parametrize(
    ("collection", "current", "change"),
    [
        ("accepted_identities", "current_identities", {"identity_revision": 2}),
        ("accepted_devices", "current_devices", {"trust_revision": 2}),
        ("accepted_devices", "current_devices", {"platform_enrollment_revision": 2}),
        ("accepted_pins", "current_pins", {"pin_revision": 2}),
        ("accepted_sessions", "current_sessions", {"session_generation": 2}),
    ],
)
def test_legal_current_revision_drift_makes_proof_stale(
    tmp_path: Path, collection: str, current: str, change: dict[str, int]
) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    _install_current_change(owner, collection, current, **change)
    with pytest.raises(AuthorizationError, match="PROOF_STALE"):
        owner.authorize(proof, req, NOW)


def test_expiry_precedes_revoked_current_identity(tmp_path: Path) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    _install_current_change(owner, "accepted_identities", "current_identities", state="REVOKED")
    with pytest.raises(AuthorizationError, match="PROOF_EXPIRED"):
        owner.authorize(proof, req, NOW + timedelta(seconds=61))


def test_malformed_proof_time_relation_is_controlled(tmp_path: Path) -> None:
    _, authentication, owner, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    malformed = replace(
        proof,
        expires_at_utc=proof.issued_at_utc,
        proof_fingerprint_sha256="",
    )
    malformed = replace(
        malformed,
        proof_fingerprint_sha256=authentication_proof_fingerprint(malformed),
    )
    before = owner._state.snapshot  # noqa: SLF001 - impossible-issuer corruption fixture
    proofs = dict(before.accepted_authentication_proofs)
    bindings = dict(before.accepted_authentication_proof_bindings)
    proofs[malformed.proof_fingerprint_sha256] = malformed
    bindings[malformed.proof_fingerprint_sha256] = authentication._binding(malformed)  # noqa: SLF001
    owner._state.snapshot = replace(  # noqa: SLF001 - impossible-issuer corruption fixture
        before,
        accepted_authentication_proofs=MappingProxyType(proofs),
        accepted_authentication_proof_bindings=MappingProxyType(bindings),
    )
    with pytest.raises(AuthorizationError, match="MALFORMED_UNTRUSTED_CONTEXT"):
        owner.authorize(malformed, req, NOW)


def _advance_generation(owner: AuthorizationAuthority, req) -> None:  # type: ignore[no-untyped-def]
    before = owner._state.snapshot  # noqa: SLF001 - coherent authority advance fixture
    replacements: dict[str, object] = {}
    for accepted_name, current_name in (
        ("accepted_identities", "current_identities"),
        ("accepted_devices", "current_devices"),
        ("accepted_pins", "current_pins"),
        ("accepted_sessions", "current_sessions"),
    ):
        accepted = dict(getattr(before, accepted_name))
        current = dict(getattr(before, current_name))
        fingerprint = next(iter(current.values()))
        old = accepted[fingerprint]
        advanced = replace(old, security_generation=2, content_fingerprint_sha256="")
        new_fingerprint = canonical_json_sha256(
            {
                key: value
                for key, value in asdict(advanced).items()
                if key != "content_fingerprint_sha256"
            }
        )
        advanced = replace(advanced, content_fingerprint_sha256=new_fingerprint)
        accepted[new_fingerprint] = advanced
        current[next(iter(current))] = new_fingerprint
        replacements[accepted_name] = MappingProxyType(accepted)
        replacements[current_name] = MappingProxyType(current)
    owner._state.snapshot = replace(before, **replacements)  # noqa: SLF001
    _seed_trusted_operation_entitlement(owner, entitlement(req, generation=2))


def test_old_proof_after_coherent_generation_advance_is_stale(tmp_path: Path) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    _advance_generation(owner, req)
    with pytest.raises(AuthorizationError, match="PROOF_STALE"):
        owner.authorize(proof, req, NOW)


@pytest.mark.parametrize("malformed", [object(), None, "request"])
def test_malformed_request_is_controlled(tmp_path: Path, malformed: object) -> None:
    _, _, owner, _, proof = arranged(tmp_path)
    with pytest.raises(AuthorizationError, match="MALFORMED_UNTRUSTED_CONTEXT"):
        owner.authorize(proof, malformed, NOW)


def test_invalid_proof_precedes_malformed_request(tmp_path: Path) -> None:
    _, _, owner, _, _ = arranged(tmp_path)
    with pytest.raises(AuthorizationError, match="AUTHENTICATION_REQUIRED"):
        owner.authorize(object(), object(), NOW)


def test_invalid_proof_precedes_unsupported_operation(tmp_path: Path) -> None:
    _, _, owner, req, _ = arranged(tmp_path)
    unsupported = replace(req, operation="UNKNOWN_OPERATION")
    with pytest.raises(AuthorizationError, match="AUTHENTICATION_REQUIRED"):
        owner.authorize(object(), unsupported, NOW)


def test_invalid_time_precedes_invalid_proof_and_request(tmp_path: Path) -> None:
    _, _, owner, _, _ = arranged(tmp_path)
    with pytest.raises(AuthorizationError, match="MALFORMED_UNTRUSTED_CONTEXT"):
        owner.authorize(object(), object(), NOW.replace(tzinfo=None))


def test_unknown_operation_is_controlled(tmp_path: Path) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    with pytest.raises(AuthorizationError, match="OPERATION_UNSUPPORTED"):
        owner.authorize(proof, replace(req, operation="UNKNOWN_OPERATION"), NOW)


@pytest.mark.parametrize(
    ("accepted_name", "current_name", "reason"),
    [
        ("accepted_identities", "current_identities", "IDENTITY_REVOKED"),
        ("accepted_devices", "current_devices", "DEVICE_NOT_TRUSTED"),
        ("accepted_pins", "current_pins", "PROOF_STALE"),
        ("accepted_sessions", "current_sessions", "PROOF_STALE"),
    ],
)
def test_dangling_current_family_uses_missing_authority_reason(
    tmp_path: Path, accepted_name: str, current_name: str, reason: str
) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    before = owner._state.snapshot  # noqa: SLF001 - dangling authority fixture
    assert getattr(before, current_name)
    owner._state.snapshot = replace(  # noqa: SLF001 - dangling authority fixture
        before,
        **{accepted_name: MappingProxyType({})},
    )
    with pytest.raises(AuthorizationError, match=reason):
        owner.authorize(proof, req, NOW)


def test_non_string_current_family_key_remains_contract_error(tmp_path: Path) -> None:
    _, _, owner, req, proof = arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    before = owner._state.snapshot  # noqa: SLF001 - structural corruption fixture
    current = dict(before.current_identities)
    current[(req.account_id, req.operator_id)] = 7  # type: ignore[assignment]
    owner._state.snapshot = replace(  # noqa: SLF001 - structural corruption fixture
        before,
        current_identities=MappingProxyType(current),
    )
    with pytest.raises(AuthorizationError, match="CONTRACT_INCONSISTENT"):
        owner.authorize(proof, req, NOW)


def test_operation_ownership_is_exact_closed_frozen_partition() -> None:
    owned = {
        "TRUST_DEVICE",
        "REVOKE_DEVICE",
        "CHANGE_PIN",
        "RESET_PIN",
        "LOCK_SESSION",
        "UNLOCK_SESSION",
        "LOGOUT_SESSION",
        "GRANT_LIVE_ACCESS",
        "SUSPEND_LIVE_ACCESS",
        "REVOKE_LIVE_ACCESS",
    }
    upstream = {
        "SETUP_PIN",
        "ROTATE_SECRET_REFERENCE",
        "REBIND_SECRET_REFERENCE",
        "ACTIVATE_CREDENTIAL_PROFILE",
        "DEACTIVATE_CREDENTIAL_PROFILE",
        "CHANGE_RISK_POLICY",
        "CHANGE_KILL_SWITCH",
        "CHANGE_PRODUCT_CAPABILITIES",
    }
    assert isinstance(OPERATION_OWNERSHIP, MappingProxyType)
    assert {
        operation
        for operation, owner in OPERATION_OWNERSHIP.items()
        if owner == "M0.10_OWNED_TRANSITION"
    } == owned
    assert {
        operation
        for operation, owner in OPERATION_OWNERSHIP.items()
        if owner == "AUTHORIZED_SECURITY_REQUEST_TO_UPSTREAM_OWNER"
    } == upstream
    assert not owned & upstream
    assert owned | upstream == set(OPERATION_POLICY_REGISTRY)


def _upstream_arranged(tmp_path: Path):  # type: ignore[no-untyped-def]
    security, authentication, _ = prepared(tmp_path)
    req = request("ROTATE_SECRET_REFERENCE", scope_fingerprint_sha256="d" * 64)
    assert req.scope_fingerprint_sha256 != canonical_scope_fingerprint(req)
    platform_assertion = assertion(authentication, req)
    _seed_external_platform(authentication, platform_assertion)
    proof = authentication.issue_authentication_proof(
        req, RAW_PIN, NOW, platform_assertion=platform_assertion
    )
    return security, authentication, AuthorizationAuthority(authentication), req, proof


def test_upstream_owned_opaque_scope_gets_genuine_proof_and_authorizes(
    tmp_path: Path,
) -> None:
    _, authentication, owner, req, proof = _upstream_arranged(tmp_path)
    assert proof.scope_fingerprint_sha256 == "d" * 64 == req.scope_fingerprint_sha256
    assert authentication.resolve_accepted_proof(proof) is proof
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    assert owner.authorize(proof, req, NOW) == "AUTHORIZED"


@pytest.mark.parametrize(
    "change",
    [
        {"scope_fingerprint_sha256": "e" * 64},
        {"mutation_fingerprint_sha256": "f" * 64},
    ],
)
def test_upstream_owned_handoff_remains_exactly_bound(
    tmp_path: Path, change: dict[str, str]
) -> None:
    _, _, owner, req, proof = _upstream_arranged(tmp_path)
    _seed_trusted_operation_entitlement(owner, entitlement(req))
    with pytest.raises(AuthorizationError, match="AUTHORIZATION_DENIED"):
        owner.authorize(proof, replace(req, **change), NOW)
