from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from bot_core.runtime.runtime_session import RuntimeSession
from bot_core.security.authentication import (
    AuthenticationError,
    CoreIssuedAuthenticationProofBinding,
    complete_authentication_proof_fingerprint,
)
from tests.security.test_authentication import (
    ACCOUNT,
    DEVICE,
    NOW,
    OPERATOR,
    RAW_PIN,
    SESSION,
    prepared,
    request,
)
from tests.security.test_platform_biometric_authentication import (
    _advance,
    _seed_external_platform,
    assertion,
)


def _genuine(owner, req, outcome: str = "SUCCESS"):  # type: ignore[no-untyped-def]
    candidate = assertion(owner, req, outcome)
    _seed_external_platform(owner, candidate)
    return candidate


def test_combined_factors_issue_exact_core_owned_accepted_proof(tmp_path: Path) -> None:
    security, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = _genuine(owner, req)
    identity_before = security.resolve_current_identity(ACCOUNT, OPERATOR)
    device_before = security.resolve_current_device(ACCOUNT, DEVICE)
    session_before = security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE)

    proof = owner.issue_authentication_proof(req, RAW_PIN, NOW, platform_assertion=candidate)

    assert proof.factor_set == ("PIN", "BIOMETRIC")
    assert owner.resolve_accepted_proof(proof) is proof
    assert owner.snapshot.accepted_authentication_proof_bindings[
        proof.proof_fingerprint_sha256
    ] == CoreIssuedAuthenticationProofBinding(
        proof.proof_fingerprint_sha256,
        complete_authentication_proof_fingerprint(proof),
        "CoreHost",
        ACCOUNT,
        OPERATOR,
        DEVICE,
        1,
        1,
        1,
        1,
        1,
        1,
    )
    assert security.resolve_current_identity(ACCOUNT, OPERATOR) == identity_before
    assert security.resolve_current_device(ACCOUNT, DEVICE) == device_before
    assert security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE) == session_before


def test_missing_or_nominal_biometric_evidence_cannot_downgrade_policy(tmp_path: Path) -> None:
    _, owner, comparator = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
        owner.issue_authentication_proof(req, RAW_PIN, NOW)
    nominal = assertion(owner, req)
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
        owner.issue_authentication_proof(req, RAW_PIN, NOW, platform_assertion=nominal)
    assert comparator.calls == 1
    assert not owner.snapshot.accepted_authentication_proofs


@pytest.mark.parametrize("include_biometric", [False, True])
def test_malformed_pin_is_an_absent_combined_factor_without_comparator_call(
    tmp_path: Path, include_biometric: bool
) -> None:
    _, owner, comparator = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = _genuine(owner, req) if include_biometric else None
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
        owner.issue_authentication_proof(
            req,
            object(),
            NOW,
            platform_assertion=candidate,
        )
    assert comparator.calls == 0
    assert not owner.snapshot.accepted_authentication_proofs
    assert not owner.snapshot.accepted_authentication_proof_bindings


@pytest.mark.parametrize(
    ("outcome", "reason"),
    [
        ("FAILED", "AUTHENTICATION_FAILED"),
        ("CANCELLED", "AUTHENTICATION_FAILED"),
        ("UNAVAILABLE", "AUTHENTICATION_FAILED"),
    ],
)
def test_combined_issuance_preserves_biometric_failure_taxonomy(
    tmp_path: Path, outcome: str, reason: str
) -> None:
    _, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = _genuine(owner, req, outcome)
    with pytest.raises(AuthenticationError, match=reason):
        owner.issue_authentication_proof(req, RAW_PIN, NOW, platform_assertion=candidate)
    assert not owner.snapshot.accepted_authentication_proofs


def test_unavailable_is_factor_level_but_aggregates_to_authentication_failure(
    tmp_path: Path,
) -> None:
    _, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = _genuine(owner, req, "UNAVAILABLE")
    with pytest.raises(AuthenticationError, match="FACTOR_UNAVAILABLE"):
        owner.verify_platform_assertion(candidate, req, NOW)
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
        owner.issue_authentication_proof(req, RAW_PIN, NOW, platform_assertion=candidate)


def test_combined_aggregation_does_not_mask_corrupt_core_authority(tmp_path: Path) -> None:
    security, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = _genuine(owner, req)
    session = security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE)
    _advance(
        owner,
        "accepted_sessions",
        "current_sessions",
        (ACCOUNT, OPERATOR, DEVICE),
        session,
        security_generation=2,
    )
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
        owner.verify_platform_assertion(candidate, req, NOW)
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
        owner.issue_authentication_proof(req, RAW_PIN, NOW, platform_assertion=candidate)


def test_wrong_pin_updates_failure_state_without_consuming_biometric_authority(
    tmp_path: Path,
) -> None:
    security, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = _genuine(owner, req)
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
        owner.issue_authentication_proof(req, "wrong", NOW, platform_assertion=candidate)
    assert security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE).failed_attempts == 1
    assert not owner.snapshot.accepted_authentication_proofs


def test_failed_biometric_does_not_publish_successful_pin_reset(tmp_path: Path) -> None:
    security, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    successful = _genuine(owner, req)
    with pytest.raises(AuthenticationError):
        owner.issue_authentication_proof(req, "wrong", NOW, platform_assertion=successful)
    failed = _genuine(owner, req, "FAILED")
    pin_after_failure = security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE)
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
        owner.issue_authentication_proof(req, RAW_PIN, NOW, platform_assertion=failed)
    assert security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE) == pin_after_failure
    assert not owner.snapshot.accepted_authentication_proofs


def test_successful_combined_authentication_resets_pin_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    security, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = _genuine(owner, req)
    with pytest.raises(AuthenticationError):
        owner.issue_authentication_proof(req, "wrong", NOW, platform_assertion=candidate)

    proof = owner.issue_authentication_proof(req, RAW_PIN, NOW, platform_assertion=candidate)

    current = security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE)
    assert (current.failed_attempts, current.lockout_until_utc) == (0, None)
    assert owner.snapshot.accepted_authentication_proofs[proof.proof_fingerprint_sha256] is proof
    assert proof.proof_fingerprint_sha256 in (owner.snapshot.accepted_authentication_proof_bindings)


@pytest.mark.parametrize("mutation", ["replace", "clone", "close"])
def test_final_runtime_fence_runs_after_both_combined_factors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    security, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = _genuine(owner, req)
    sessions = security._runtime_sessions  # noqa: SLF001 - adversarial Core-owner fixture
    original_verify = owner.verify_platform_assertion

    def verify_then_mutate(candidate, current_request, now):  # type: ignore[no-untyped-def]
        result = original_verify(candidate, current_request, now)
        if mutation == "replace":
            sessions.current_session = RuntimeSession(
                "run_018f0000-0000-7000-8000-000000000099", DEVICE
            )
        elif mutation == "clone":
            sessions.current_session = RuntimeSession(SESSION, DEVICE)
        else:
            sessions.current_session.close()
        return result

    monkeypatch.setattr(owner, "verify_platform_assertion", verify_then_mutate)
    before = owner.snapshot
    with pytest.raises(AuthenticationError, match="RUNTIME_SESSION_AUTHORITY_DENIED"):
        owner.issue_authentication_proof(req, RAW_PIN, NOW, platform_assertion=candidate)
    assert owner.snapshot == before


def test_assertion_is_bound_to_the_exact_combined_request(tmp_path: Path) -> None:
    _, owner, _ = prepared(tmp_path)
    req = request("UNLOCK_SESSION")
    candidate = _genuine(owner, req)
    for other in (
        request("TRUST_DEVICE"),
        replace(req, mutation_fingerprint_sha256="d" * 64),
        replace(req, correlation_id="other-correlation"),
    ):
        with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
            owner.issue_authentication_proof(other, RAW_PIN, NOW, platform_assertion=candidate)


def test_pin_only_policy_ignores_unneeded_assertion_and_keeps_factor_set(tmp_path: Path) -> None:
    _, owner, _ = prepared(tmp_path)
    req = request("LOCK_SESSION")
    assert owner.issue_authentication_proof(
        req, RAW_PIN, NOW, platform_assertion=object()
    ).factor_set == ("PIN",)
