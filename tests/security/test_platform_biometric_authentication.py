from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
import inspect
from pathlib import Path
from types import MappingProxyType

import pytest

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.security.authentication import (
    AuthenticationAuthority,
    AuthenticationError,
    CoreAcceptedPlatformBiometricAssertionBinding,
    PlatformBiometricAssertion,
    complete_platform_biometric_assertion_fingerprint,
    core_expected_biometric_challenge,
    platform_biometric_assertion_fingerprint,
)
from tests.security.test_authentication import (
    ACCOUNT,
    ACCOUNT_B,
    DEVICE,
    DEVICE_B,
    NOW,
    OPERATOR,
    OPERATOR_B,
    prepared,
    request,
)


def assertion(owner: AuthenticationAuthority, req=None, outcome: str = "SUCCESS", **changes):
    req = req or request("UNLOCK_SESSION")
    data = {
        "account_id": req.account_id,
        "device_installation_id": req.device_installation_id,
        "platform_authenticator_source": "platform-fixture",
        "platform_enrollment_revision": 1,
        "challenge_fingerprint_sha256": owner.derive_platform_biometric_challenge(req),
        "outcome": outcome,
        "verified_at_utc": "2026-08-10T10:03:00Z",
        "expires_at_utc": "2026-08-10T10:04:00Z",
    }
    data.update(changes)
    candidate = PlatformBiometricAssertion(**data, assertion_fingerprint_sha256="")
    return replace(
        candidate,
        assertion_fingerprint_sha256=platform_biometric_assertion_fingerprint(candidate),
    )


def _seed_external_platform(owner: AuthenticationAuthority, candidate):
    """Test-only representation of membership accepted before the public Core call."""
    binding = CoreAcceptedPlatformBiometricAssertionBinding(
        candidate.assertion_fingerprint_sha256,
        complete_platform_biometric_assertion_fingerprint(candidate),
        "external_platform_authenticator",
        candidate.account_id,
        candidate.device_installation_id,
        candidate.platform_enrollment_revision,
        candidate.challenge_fingerprint_sha256,
    )
    before = owner.snapshot
    accepted = dict(before.accepted_platform_biometric_assertion_bindings)
    accepted[candidate.assertion_fingerprint_sha256] = binding
    owner._state.snapshot = replace(  # noqa: SLF001 - private trusted test fixture
        before,
        accepted_platform_biometric_assertion_bindings=MappingProxyType(accepted),
    )
    return binding


def _advance(owner, collection, current, scope, value, **changes):
    before = owner.snapshot
    changed = replace(value, **changes, content_fingerprint_sha256="")
    fingerprint = canonical_json_sha256(
        {k: v for k, v in asdict(changed).items() if k != "content_fingerprint_sha256"}
    )
    changed = replace(changed, content_fingerprint_sha256=fingerprint)
    accepted = dict(getattr(before, collection))
    accepted[fingerprint] = changed
    designation = dict(getattr(before, current))
    designation[scope] = fingerprint
    owner._state.snapshot = replace(  # noqa: SLF001 - adversarial epoch fixture
        before,
        **{collection: MappingProxyType(accepted), current: MappingProxyType(designation)},
    )


def _point_current_at(owner, collection, current, scope, value):
    before = owner.snapshot
    accepted = dict(getattr(before, collection))
    accepted[value.content_fingerprint_sha256] = value
    designation = dict(getattr(before, current))
    designation[scope] = value.content_fingerprint_sha256
    owner._state.snapshot = replace(  # noqa: SLF001 - adversarial authority fixture
        before,
        **{collection: MappingProxyType(accepted), current: MappingProxyType(designation)},
    )


def _recomputed(value, **changes):
    changed = replace(value, **changes, content_fingerprint_sha256="")
    fingerprint = canonical_json_sha256(
        {k: v for k, v in asdict(changed).items() if k != "content_fingerprint_sha256"}
    )
    return replace(changed, content_fingerprint_sha256=fingerprint)


def _point_current_at_recomputed_content(owner, collection, current, scope, value):
    fingerprint = canonical_json_sha256(
        {k: v for k, v in asdict(value).items() if k != "content_fingerprint_sha256"}
    )
    before = owner.snapshot
    accepted = dict(getattr(before, collection))
    accepted[fingerprint] = value
    designation = dict(getattr(before, current))
    designation[scope] = fingerprint
    owner._state.snapshot = replace(  # noqa: SLF001 - corrupt-terminal fixture
        before,
        **{collection: MappingProxyType(accepted), current: MappingProxyType(designation)},
    )


def test_exact_dto_shapes_and_complete_binding(tmp_path: Path) -> None:
    _, owner, _ = prepared(tmp_path)
    candidate = assertion(owner)
    binding = _seed_external_platform(owner, candidate)
    assert tuple(PlatformBiometricAssertion.__dataclass_fields__) == (
        "account_id",
        "device_installation_id",
        "platform_authenticator_source",
        "platform_enrollment_revision",
        "challenge_fingerprint_sha256",
        "outcome",
        "verified_at_utc",
        "expires_at_utc",
        "assertion_fingerprint_sha256",
    )
    assert tuple(CoreAcceptedPlatformBiometricAssertionBinding.__dataclass_fields__) == (
        "assertion_fingerprint_sha256",
        "complete_assertion_content_fingerprint_sha256",
        "authority_source",
        "account_id",
        "device_installation_id",
        "platform_enrollment_revision",
        "challenge_fingerprint_sha256",
    )
    assert binding.authority_source == "external_platform_authenticator"
    assert binding.complete_assertion_content_fingerprint_sha256 == canonical_json_sha256(
        asdict(candidate)
    )


def test_core_challenge_has_exact_frozen_input_order(tmp_path: Path) -> None:
    _, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    expected = canonical_json_sha256(
        [
            "M010-BIOMETRIC-CHALLENGE",
            req.account_id,
            req.operator_id,
            req.device_installation_id,
            req.environment,
            req.operation,
            req.scope_fingerprint_sha256,
            req.mutation_fingerprint_sha256,
            req.causation_id,
            req.correlation_id,
            1,
            1,
            1,
        ]
    )
    assert core_expected_biometric_challenge(req, 1, 1, 1) == expected
    assert owner.derive_platform_biometric_challenge(req) == expected


def test_genuine_membership_is_required_and_public_api_has_no_binding_input(tmp_path: Path) -> None:
    _, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = assertion(owner, req)
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
        owner.verify_platform_assertion(candidate, req, NOW)
    assert "binding" not in inspect.signature(owner.verify_platform_assertion).parameters
    _seed_external_platform(owner, candidate)
    assert owner.verify_platform_assertion(candidate, req, NOW) == "BIOMETRIC_ACCEPTED"


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        ("FAILED", "AUTHENTICATION_FAILED"),
        ("CANCELLED", "AUTHENTICATION_FAILED"),
        ("UNAVAILABLE", "FACTOR_UNAVAILABLE"),
    ],
)
def test_terminal_outcome_semantics(tmp_path: Path, outcome: str, reason: str) -> None:
    _, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = assertion(owner, req, outcome)
    _seed_external_platform(owner, candidate)
    with pytest.raises(AuthenticationError, match=reason):
        owner.verify_platform_assertion(candidate, req, NOW)


@pytest.mark.parametrize(
    "field", ["account_id", "device_installation_id", "challenge_fingerprint_sha256"]
)
def test_exact_assertion_scope_is_required(tmp_path: Path, field: str) -> None:
    _, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    changes = {
        field: "a" * 64
        if "fingerprint" in field
        else (
            "acct_018f0000-0000-7000-8000-000000000011"
            if field == "account_id"
            else "dev_018f0000-0000-7000-8000-000000000012"
        )
    }
    candidate = assertion(owner, req, **changes)
    _seed_external_platform(owner, candidate)
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
        owner.verify_platform_assertion(candidate, req, NOW)


def test_challenge_fences_request_and_current_epochs(tmp_path: Path) -> None:
    security, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = assertion(owner, req)
    _seed_external_platform(owner, candidate)
    for other in (
        request("TRUST_DEVICE"),
        replace(req, mutation_fingerprint_sha256="d" * 64),
        replace(req, correlation_id="different-correlation"),
    ):
        with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
            owner.verify_platform_assertion(candidate, other, NOW)
    session = security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE)
    _advance(
        owner,
        "accepted_sessions",
        "current_sessions",
        (ACCOUNT, OPERATOR, DEVICE),
        session,
        session_generation=2,
    )
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
        owner.verify_platform_assertion(candidate, req, NOW)


def test_security_and_enrollment_epoch_advances_fence_assertion(tmp_path: Path) -> None:
    security, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = assertion(owner, req)
    _seed_external_platform(owner, candidate)
    identity = security.resolve_current_identity(ACCOUNT, OPERATOR)
    _advance(
        owner,
        "accepted_identities",
        "current_identities",
        (ACCOUNT, OPERATOR),
        identity,
        security_generation=2,
        identity_revision=2,
    )
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
        owner.verify_platform_assertion(candidate, req, NOW)

    security2, owner2, _ = prepared(tmp_path / "other")
    candidate2 = assertion(owner2, req)
    _seed_external_platform(owner2, candidate2)
    device = security2.resolve_current_device(ACCOUNT, DEVICE)
    _advance(
        owner2,
        "accepted_devices",
        "current_devices",
        (ACCOUNT, DEVICE),
        device,
        platform_enrollment_revision=2,
        trust_revision=2,
    )
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
        owner2.verify_platform_assertion(candidate2, req, NOW)


@pytest.mark.parametrize("component", ["identity", "device", "session"])
def test_cross_scope_current_projection_reparenting_is_contract_inconsistent(
    tmp_path: Path, component: str
) -> None:
    security, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = assertion(owner, req)
    _seed_external_platform(owner, candidate)
    args: tuple[object, object, object, object]
    if component == "identity":
        value = _recomputed(
            security.resolve_current_identity(ACCOUNT, OPERATOR),
            account_id=ACCOUNT_B,
            operator_id=OPERATOR_B,
        )
        args = ("accepted_identities", "current_identities", (ACCOUNT, OPERATOR), value)
    elif component == "device":
        value = _recomputed(
            security.resolve_current_device(ACCOUNT, DEVICE),
            account_id=ACCOUNT_B,
            device_installation_id=DEVICE_B,
        )
        args = ("accepted_devices", "current_devices", (ACCOUNT, DEVICE), value)
    else:
        value = _recomputed(
            security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE),
            account_id=ACCOUNT_B,
            operator_id=OPERATOR_B,
            device_installation_id=DEVICE_B,
            runtime_session_id="run_018f0000-0000-7000-8000-000000000014",
        )
        args = (
            "accepted_sessions",
            "current_sessions",
            (ACCOUNT, OPERATOR, DEVICE),
            value,
        )
    _point_current_at(owner, *args)
    for action in (
        lambda: owner.derive_platform_biometric_challenge(req),
        lambda: owner.verify_platform_assertion(candidate, req, NOW),
    ):
        with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
            action()


@pytest.mark.parametrize("component", ["device", "session"])
def test_security_generation_drift_is_contract_inconsistent(tmp_path: Path, component: str) -> None:
    security, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = assertion(owner, req)
    _seed_external_platform(owner, candidate)
    if component == "device":
        _advance(
            owner,
            "accepted_devices",
            "current_devices",
            (ACCOUNT, DEVICE),
            security.resolve_current_device(ACCOUNT, DEVICE),
            trust_revision=2,
            security_generation=2,
            platform_enrollment_revision=1,
        )
    else:
        _advance(
            owner,
            "accepted_sessions",
            "current_sessions",
            (ACCOUNT, OPERATOR, DEVICE),
            security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE),
            security_generation=2,
        )
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
        owner.verify_platform_assertion(candidate, req, NOW)


def test_coherent_security_generation_advance_makes_old_challenge_stale(tmp_path: Path) -> None:
    security, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = assertion(owner, req)
    _seed_external_platform(owner, candidate)
    updates = (
        (
            "accepted_identities",
            "current_identities",
            (ACCOUNT, OPERATOR),
            security.resolve_current_identity(ACCOUNT, OPERATOR),
            {"identity_revision": 2, "security_generation": 2},
        ),
        (
            "accepted_devices",
            "current_devices",
            (ACCOUNT, DEVICE),
            security.resolve_current_device(ACCOUNT, DEVICE),
            {"trust_revision": 2, "security_generation": 2},
        ),
        (
            "accepted_sessions",
            "current_sessions",
            (ACCOUNT, OPERATOR, DEVICE),
            security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE),
            {"security_generation": 2},
        ),
    )
    for collection, current, scope, value, changes in updates:
        _advance(owner, collection, current, scope, value, **changes)
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
        owner.verify_platform_assertion(candidate, req, NOW)


def test_malformed_current_projection_is_contract_inconsistent(tmp_path: Path) -> None:
    security, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    malformed = _recomputed(security.resolve_current_device(ACCOUNT, DEVICE), trust_revision="1")
    _point_current_at(owner, "accepted_devices", "current_devices", (ACCOUNT, DEVICE), malformed)
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
        owner.derive_platform_biometric_challenge(req)


def test_missing_current_projection_is_authentication_failed(tmp_path: Path) -> None:
    _, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = assertion(owner, req)
    _seed_external_platform(owner, candidate)
    before = owner.snapshot
    owner._state.snapshot = replace(  # noqa: SLF001 - absent-authority fixture
        before, current_devices=MappingProxyType({})
    )
    for action in (
        lambda: owner.derive_platform_biometric_challenge(req),
        lambda: owner.verify_platform_assertion(candidate, req, NOW),
    ):
        with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
            action()


@pytest.mark.parametrize(
    ("component", "terminal"),
    [("device", "f" * 64), ("session", 123)],
)
def test_current_projection_terminal_fingerprint_is_intrinsic_authority(
    tmp_path: Path, component: str, terminal: object
) -> None:
    security, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    args: tuple[object, object, object, object]
    if component == "device":
        value = replace(
            security.resolve_current_device(ACCOUNT, DEVICE),
            content_fingerprint_sha256=terminal,
        )
        args = ("accepted_devices", "current_devices", (ACCOUNT, DEVICE), value)
    else:
        value = replace(
            security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE),
            content_fingerprint_sha256=terminal,
        )
        args = (
            "accepted_sessions",
            "current_sessions",
            (ACCOUNT, OPERATOR, DEVICE),
            value,
        )
    _point_current_at_recomputed_content(owner, *args)
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
        owner.derive_platform_biometric_challenge(req)


@pytest.mark.parametrize(
    ("now", "accepted"),
    [
        (NOW - timedelta(microseconds=1), False),
        (NOW, True),
        (NOW + timedelta(seconds=60), True),
        (NOW + timedelta(seconds=60, microseconds=1), False),
    ],
)
def test_inclusive_assertion_time_window(tmp_path: Path, now: datetime, accepted: bool) -> None:
    _, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = assertion(owner, req)
    _seed_external_platform(owner, candidate)
    if accepted:
        assert owner.verify_platform_assertion(candidate, req, now) == "BIOMETRIC_ACCEPTED"
    else:
        with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
            owner.verify_platform_assertion(candidate, req, now)


@pytest.mark.parametrize(
    "now", [datetime(2026, 8, 10), datetime(2026, 8, 10, tzinfo=timezone(timedelta(hours=1)))]
)
def test_non_utc_now_is_controlled(tmp_path: Path, now: datetime) -> None:
    _, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = assertion(owner, req)
    _seed_external_platform(owner, candidate)
    with pytest.raises(AuthenticationError, match="MALFORMED_UNTRUSTED_CONTEXT"):
        owner.verify_platform_assertion(candidate, req, now)


@pytest.mark.parametrize(
    "changes",
    [
        {"assertion_fingerprint_sha256": "bad"},
        {"platform_enrollment_revision": True},
        {"platform_enrollment_revision": "1"},
        {"verified_at_utc": "not-time"},
        {"platform_authenticator_source": object()},
    ],
)
def test_malformed_assertions_fail_closed(tmp_path: Path, changes: dict[str, object]) -> None:
    _, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = replace(assertion(owner, req), **changes)
    with pytest.raises(AuthenticationError, match="MALFORMED_UNTRUSTED_CONTEXT"):
        owner.verify_platform_assertion(candidate, req, NOW)


def test_verification_has_no_proof_or_transition_side_effect(tmp_path: Path) -> None:
    security, owner, comparator = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = assertion(owner, req)
    _seed_external_platform(owner, candidate)
    before_session = security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE)
    assert owner.verify_platform_assertion(candidate, req, NOW) == "BIOMETRIC_ACCEPTED"
    assert not owner.snapshot.accepted_authentication_proofs
    assert not owner.snapshot.accepted_authentication_proof_bindings
    assert security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE) == before_session
    with pytest.raises(AuthenticationError, match="OPERATION_UNSUPPORTED"):
        owner.issue_authentication_proof(req, "2468", NOW)
    assert comparator.calls == 0
