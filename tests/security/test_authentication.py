from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Callable, cast

import pytest

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.runtime.runtime_session import RuntimeSession
from bot_core.security.authentication import (
    AuthenticationAuthority,
    AuthenticationError,
    AuthenticationProof,
    AuthorizationRequest,
    CoreIssuedAuthenticationProofBinding,
    authentication_proof_fingerprint,
    canonical_scope_fingerprint,
    complete_authentication_proof_fingerprint,
    session_mutation_fingerprint,
)
from bot_core.security.initial_security import PinVerifierRecord
from tests.security.test_initial_security import (
    ACCOUNT,
    DEVICE,
    OPERATOR,
    RAW_PIN,
    SESSION,
    authority as initial_authority,
)

NOW = datetime(2026, 8, 10, 10, 3, tzinfo=timezone.utc)
ACCOUNT_B = "acct_018f0000-0000-7000-8000-000000000011"
DEVICE_B = "dev_018f0000-0000-7000-8000-000000000012"
OPERATOR_B = "op_018f0000-0000-7000-8000-000000000013"


class Comparator:
    def __init__(self, *, leak: bool = False, action: Callable[[], None] | None = None) -> None:
        self.leak = leak
        self.action = action
        self.calls = 0

    def compare(self, raw_pin: str, record: PinVerifierRecord) -> bool:
        self.calls += 1
        if self.leak:
            raise RuntimeError(f"dependency exposed {raw_pin}")
        if self.action is not None:
            self.action()
        return cast(bool, raw_pin == RAW_PIN and record.verifier == "c" * 64)


def prepared(tmp_path: Path, comparator: Comparator | None = None):  # type: ignore[no-untyped-def]
    store, security, view = initial_authority(tmp_path / "state.sqlite3")
    with store:
        security.establish_initial_security(view, RAW_PIN)
    comparison = comparator or Comparator()
    return security, AuthenticationAuthority(security, comparison), comparison


def request(operation: str = "LOCK_SESSION", environment: str = "TESTNET", **changes: object):
    base: dict[str, object] = {
        "account_id": ACCOUNT,
        "operator_id": OPERATOR,
        "device_installation_id": DEVICE,
        "environment": environment,
        "operation": operation,
        "scope_fingerprint_sha256": "0" * 64,
        "mutation_fingerprint_sha256": "0" * 64,
        "causation_id": "cause-authentication",
        "correlation_id": "correlation-authentication",
    }
    base.update(changes)
    provisional = AuthorizationRequest(**base)  # type: ignore[arg-type]
    base["scope_fingerprint_sha256"] = changes.get(
        "scope_fingerprint_sha256", canonical_scope_fingerprint(provisional)
    )
    provisional = AuthorizationRequest(**base)  # type: ignore[arg-type]
    target = {
        "LOCK_SESSION": "LOCKED",
        "LOGOUT_SESSION": "LOGGED_OUT",
        "UNLOCK_SESSION": "UNLOCKED",
    }.get(operation)
    if target is not None:
        base["mutation_fingerprint_sha256"] = changes.get(
            "mutation_fingerprint_sha256",
            session_mutation_fingerprint(provisional, target, 1, 2),
        )
    return AuthorizationRequest(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize("operation", ["LOCK_SESSION", "LOGOUT_SESSION"])
def test_correct_pin_issues_exact_accepted_pin_only_proof(tmp_path: Path, operation: str) -> None:
    security, owner, _ = prepared(tmp_path)
    original_session = security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE)
    proof = owner.issue_authentication_proof(request(operation), RAW_PIN, NOW)

    assert tuple(AuthenticationProof.__dataclass_fields__) == (
        "account_id",
        "operator_id",
        "device_installation_id",
        "factor_set",
        "issued_at_utc",
        "expires_at_utc",
        "identity_revision",
        "device_trust_revision",
        "pin_revision",
        "platform_enrollment_revision",
        "security_generation",
        "session_generation",
        "environment",
        "operation",
        "scope_fingerprint_sha256",
        "mutation_fingerprint_sha256",
        "causation_id",
        "correlation_id",
        "proof_fingerprint_sha256",
    )
    assert proof.factor_set == ("PIN",)
    assert proof.issued_at_utc == "2026-08-10T10:03:00Z"
    assert proof.expires_at_utc == "2026-08-10T10:04:00Z"
    assert authentication_proof_fingerprint(proof) == proof.proof_fingerprint_sha256
    assert owner.resolve_accepted_proof(proof) is proof
    binding = owner.snapshot.accepted_authentication_proof_bindings[proof.proof_fingerprint_sha256]
    assert binding == CoreIssuedAuthenticationProofBinding(
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
    assert security.resolve_current_session(ACCOUNT, OPERATOR, DEVICE) == original_session
    assert "OperationEntitlement" not in repr(owner.snapshot)


def test_nominal_self_hashed_proof_has_no_accepted_authority(tmp_path: Path) -> None:
    _, owner, _ = prepared(tmp_path)
    genuine = owner.issue_authentication_proof(request(), RAW_PIN, NOW)
    manual = replace(genuine, correlation_id="other", proof_fingerprint_sha256="")
    manual = replace(manual, proof_fingerprint_sha256=authentication_proof_fingerprint(manual))
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_REQUIRED"):
        owner.resolve_accepted_proof(manual)


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ({"scope_fingerprint_sha256": "a" * 64}, "AUTHORIZATION_DENIED"),
        ({"mutation_fingerprint_sha256": "b" * 64}, "AUTHORIZATION_DENIED"),
        ({"environment": "LIVE"}, "AUTHORIZATION_DENIED"),
        ({"causation_id": ""}, "MALFORMED_UNTRUSTED_CONTEXT"),
        ({"correlation_id": ""}, "MALFORMED_UNTRUSTED_CONTEXT"),
    ],
)
def test_untrusted_request_must_be_exact(
    tmp_path: Path, change: dict[str, object], reason: str
) -> None:
    _, owner, comparator = prepared(tmp_path)
    with pytest.raises(AuthenticationError, match=reason):
        owner.issue_authentication_proof(request(**change), RAW_PIN, NOW)  # type: ignore[arg-type]
    assert comparator.calls == 0


def test_lock_and_logout_are_context_separated(tmp_path: Path) -> None:
    _, owner, _ = prepared(tmp_path)
    lock = owner.issue_authentication_proof(request("LOCK_SESSION"), RAW_PIN, NOW)
    logout = owner.issue_authentication_proof(request("LOGOUT_SESSION"), RAW_PIN, NOW)
    assert lock.proof_fingerprint_sha256 != logout.proof_fingerprint_sha256
    assert lock.scope_fingerprint_sha256 != logout.scope_fingerprint_sha256
    assert lock.mutation_fingerprint_sha256 != logout.mutation_fingerprint_sha256


@pytest.mark.parametrize("operation", ["UNLOCK_SESSION", "TRUST_DEVICE"])
def test_biometric_policies_never_downgrade_to_pin(tmp_path: Path, operation: str) -> None:
    _, owner, comparator = prepared(tmp_path)
    candidate = request(operation)
    with pytest.raises(AuthenticationError, match="OPERATION_UNSUPPORTED"):
        owner.issue_authentication_proof(candidate, RAW_PIN, NOW)
    assert comparator.calls == 0


def _install_tampered_current(security, collection: str, current: str, scope, value):  # type: ignore[no-untyped-def]
    snapshot = security.snapshot
    accepted = dict(getattr(snapshot, collection))
    fingerprint = canonical_json_sha256(
        {key: item for key, item in asdict(value).items() if key != "content_fingerprint_sha256"}
    )
    value = replace(value, content_fingerprint_sha256=fingerprint)
    accepted[fingerprint] = value
    currents = dict(getattr(snapshot, current))
    currents[scope] = fingerprint
    security._state.snapshot = replace(  # noqa: SLF001 -- adversarial semantic-state fixture
        snapshot,
        **{collection: MappingProxyType(accepted), current: MappingProxyType(currents)},
    )


@pytest.mark.parametrize(
    ("collection", "current", "resolver", "scope", "change", "reason"),
    [
        (
            "accepted_identities",
            "current_identities",
            "resolve_current_identity",
            (ACCOUNT, OPERATOR),
            {"state": "REVOKED"},
            "IDENTITY_INVALID",
        ),
        (
            "accepted_devices",
            "current_devices",
            "resolve_current_device",
            (ACCOUNT, DEVICE),
            {"state": "REVOKED"},
            "DEVICE_NOT_TRUSTED",
        ),
        (
            "accepted_pins",
            "current_pins",
            "resolve_current_pin",
            (ACCOUNT, OPERATOR, DEVICE),
            {"security_generation": 2},
            "CONTRACT_INCONSISTENT",
        ),
    ],
)
def test_current_authority_and_generation_are_enforced(
    tmp_path: Path,
    collection: str,
    current: str,
    resolver: str,
    scope: tuple[str, ...],
    change: dict[str, object],
    reason: str,
) -> None:
    security, owner, comparator = prepared(tmp_path)
    value = getattr(security, resolver)(*scope)
    _install_tampered_current(security, collection, current, scope, replace(value, **change))
    with pytest.raises(AuthenticationError, match=reason):
        owner.issue_authentication_proof(request(), RAW_PIN, NOW)
    assert comparator.calls == 0


@pytest.mark.parametrize(
    ("collection", "current", "resolver", "scope", "scope_changes"),
    [
        (
            "accepted_identities",
            "current_identities",
            "resolve_current_identity",
            (ACCOUNT, OPERATOR),
            {"account_id": ACCOUNT_B, "operator_id": OPERATOR_B},
        ),
        (
            "accepted_devices",
            "current_devices",
            "resolve_current_device",
            (ACCOUNT, DEVICE),
            {"account_id": ACCOUNT_B, "device_installation_id": DEVICE_B},
        ),
        (
            "accepted_pins",
            "current_pins",
            "resolve_current_pin",
            (ACCOUNT, OPERATOR, DEVICE),
            {
                "account_id": ACCOUNT_B,
                "operator_id": OPERATOR_B,
                "device_installation_id": DEVICE_B,
            },
        ),
        (
            "accepted_sessions",
            "current_sessions",
            "resolve_current_session",
            (ACCOUNT, OPERATOR, DEVICE),
            {
                "account_id": ACCOUNT_B,
                "operator_id": OPERATOR_B,
                "device_installation_id": DEVICE_B,
            },
        ),
    ],
)
def test_current_designation_cannot_reparent_intrinsic_valid_projection(
    tmp_path: Path,
    collection: str,
    current: str,
    resolver: str,
    scope: tuple[str, ...],
    scope_changes: dict[str, object],
) -> None:
    security, owner, comparator = prepared(tmp_path)
    value = getattr(security, resolver)(*scope)
    _install_tampered_current(security, collection, current, scope, replace(value, **scope_changes))
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
        owner.issue_authentication_proof(request(), RAW_PIN, NOW)
    assert comparator.calls == 0


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("pin_revision", "1"),
        ("failed_attempts", "0"),
        ("security_generation", "1"),
        ("pin_revision", True),
    ],
)
def test_malformed_pin_integer_is_controlled_before_comparison(
    tmp_path: Path, field: str, malformed: object
) -> None:
    security, owner, comparator = prepared(tmp_path)
    scope = (ACCOUNT, OPERATOR, DEVICE)
    pin = security.resolve_current_pin(*scope)
    _install_tampered_current(
        security,
        "accepted_pins",
        "current_pins",
        scope,
        replace(pin, **{field: malformed}),
    )
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT") as caught:
        owner.issue_authentication_proof(request(), RAW_PIN, NOW)
    assert not isinstance(caught.value, (TypeError, ValueError, AttributeError))
    assert comparator.calls == 0


def test_wrong_accepted_payload_type_is_controlled_for_issue_and_resolve(tmp_path: Path) -> None:
    security, owner, comparator = prepared(tmp_path)
    proof = owner.issue_authentication_proof(request(), RAW_PIN, NOW)
    snapshot = security.snapshot
    scope = (ACCOUNT, OPERATOR, DEVICE)
    bogus_fingerprint = "f" * 64
    accepted = dict(snapshot.accepted_pins)
    accepted[bogus_fingerprint] = "not-a-pin-projection"
    current = dict(snapshot.current_pins)
    current[scope] = bogus_fingerprint
    security._state.snapshot = replace(  # noqa: SLF001 -- corrupted authority fixture
        snapshot,
        accepted_pins=MappingProxyType(accepted),
        current_pins=MappingProxyType(current),
    )
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
        owner.issue_authentication_proof(request(), RAW_PIN, NOW)
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
        owner.resolve_accepted_proof(proof)
    assert comparator.calls == 1


def test_non_string_current_pin_verifier_is_controlled_for_issue_and_resolve(
    tmp_path: Path,
) -> None:
    security, owner, comparator = prepared(tmp_path)
    proof = owner.issue_authentication_proof(request(), RAW_PIN, NOW)
    scope = (ACCOUNT, OPERATOR, DEVICE)
    pin = security.resolve_current_pin(*scope)
    _install_tampered_current(
        security,
        "accepted_pins",
        "current_pins",
        scope,
        replace(pin, verifier=123),  # type: ignore[arg-type]
    )
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT") as issuance:
        owner.issue_authentication_proof(request(), RAW_PIN, NOW)
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT") as resolution:
        owner.resolve_accepted_proof(proof)
    assert not isinstance(issuance.value, TypeError)
    assert not isinstance(resolution.value, TypeError)
    assert comparator.calls == 1


@pytest.mark.parametrize(
    "changes",
    [
        {"factor_set": (object(),)},
        {"pin_revision": "1"},
        {"session_generation": True},
        {"proof_fingerprint_sha256": "NOT-SHA"},
    ],
)
def test_malformed_untrusted_proof_is_denied_before_fingerprinting(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    _, owner, _ = prepared(tmp_path)
    genuine = owner.issue_authentication_proof(request(), RAW_PIN, NOW)
    before = owner.snapshot
    malformed = replace(genuine, **changes)
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_REQUIRED"):
        owner.resolve_accepted_proof(malformed)
    assert owner.snapshot == before


def test_untrusted_proof_with_explicit_self_hash_mismatch_is_denied(tmp_path: Path) -> None:
    _, owner, _ = prepared(tmp_path)
    genuine = owner.issue_authentication_proof(request(), RAW_PIN, NOW)
    mismatched = replace(genuine, correlation_id="different")
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_REQUIRED"):
        owner.resolve_accepted_proof(mismatched)


@pytest.mark.parametrize(
    ("collection", "current", "resolver", "scope"),
    [
        ("accepted_pins", "current_pins", "resolve_current_pin", (ACCOUNT, OPERATOR, DEVICE)),
        ("accepted_devices", "current_devices", "resolve_current_device", (ACCOUNT, DEVICE)),
        (
            "accepted_sessions",
            "current_sessions",
            "resolve_current_session",
            (ACCOUNT, OPERATOR, DEVICE),
        ),
    ],
)
def test_proof_resolution_rejects_each_incoherent_security_generation(
    tmp_path: Path,
    collection: str,
    current: str,
    resolver: str,
    scope: tuple[str, ...],
) -> None:
    security, owner, _ = prepared(tmp_path)
    proof = owner.issue_authentication_proof(request(), RAW_PIN, NOW)
    value = getattr(security, resolver)(*scope)
    _install_tampered_current(
        security, collection, current, scope, replace(value, security_generation=2)
    )
    with pytest.raises(AuthenticationError, match="CONTRACT_INCONSISTENT"):
        owner.resolve_accepted_proof(proof)


def test_coherent_security_generation_advance_makes_old_proof_stale(tmp_path: Path) -> None:
    security, owner, _ = prepared(tmp_path)
    proof = owner.issue_authentication_proof(request(), RAW_PIN, NOW)
    changes = (
        (
            "accepted_identities",
            "current_identities",
            "resolve_current_identity",
            (ACCOUNT, OPERATOR),
        ),
        ("accepted_devices", "current_devices", "resolve_current_device", (ACCOUNT, DEVICE)),
        ("accepted_pins", "current_pins", "resolve_current_pin", (ACCOUNT, OPERATOR, DEVICE)),
        (
            "accepted_sessions",
            "current_sessions",
            "resolve_current_session",
            (ACCOUNT, OPERATOR, DEVICE),
        ),
    )
    for collection, current, resolver, scope in changes:
        value = getattr(security, resolver)(*scope)
        _install_tampered_current(
            security, collection, current, scope, replace(value, security_generation=2)
        )
    with pytest.raises(AuthenticationError, match="PROOF_STALE"):
        owner.resolve_accepted_proof(proof)


def test_authentication_reuses_exact_core_runtime_owner_and_rejects_injection(
    tmp_path: Path,
) -> None:
    security, owner, _ = prepared(tmp_path)
    assert owner._runtime_sessions is security._runtime_sessions  # noqa: SLF001
    fake = type(
        "FakeSessions",
        (),
        {"resolve_current": lambda self, account, device: RuntimeSession(SESSION, DEVICE)},
    )()
    with pytest.raises(TypeError):
        AuthenticationAuthority(security, Comparator(), fake)  # type: ignore[call-arg]


@pytest.mark.parametrize("mutation", ["replace", "clone", "close"])
def test_final_runtime_fence_rejects_changes_during_successful_comparison(
    tmp_path: Path, mutation: str
) -> None:
    security, _, _ = prepared(tmp_path)
    sessions = security._runtime_sessions  # noqa: SLF001 -- exact Core owner under test

    def mutate_runtime() -> None:
        if mutation == "replace":
            sessions.current_session = RuntimeSession(
                "run_018f0000-0000-7000-8000-000000000099", DEVICE
            )
        elif mutation == "clone":
            sessions.current_session = RuntimeSession(SESSION, DEVICE)
        else:
            sessions.current_session.close()

    owner = AuthenticationAuthority(security, Comparator(action=mutate_runtime))
    before = owner.snapshot
    with pytest.raises(AuthenticationError, match="RUNTIME_SESSION_AUTHORITY_DENIED"):
        owner.issue_authentication_proof(request(), RAW_PIN, NOW)
    assert owner.snapshot == before
    assert not owner.snapshot.accepted_authentication_proofs
    assert not owner.snapshot.accepted_authentication_proof_bindings


def test_final_runtime_failure_does_not_partially_publish_successful_pin_reset(
    tmp_path: Path,
) -> None:
    security, owner, _ = prepared(tmp_path)
    with pytest.raises(AuthenticationError, match="AUTHENTICATION_FAILED"):
        owner.issue_authentication_proof(request(), "wrong", NOW)
    pin_after_failure = security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE)
    sessions = security._runtime_sessions  # noqa: SLF001 -- exact Core owner under test

    def replace_runtime() -> None:
        sessions.current_session = RuntimeSession(SESSION, DEVICE)

    fenced = AuthenticationAuthority(security, Comparator(action=replace_runtime))
    with pytest.raises(AuthenticationError, match="RUNTIME_SESSION_AUTHORITY_DENIED"):
        fenced.issue_authentication_proof(request(), RAW_PIN, NOW)
    assert security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE) == pin_after_failure
    assert not fenced.snapshot.accepted_authentication_proofs
    assert not fenced.snapshot.accepted_authentication_proof_bindings


def test_wrong_pin_history_third_failure_lockout_and_active_lockout(tmp_path: Path) -> None:
    security, owner, comparator = prepared(tmp_path)
    verifier = security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE).verifier
    for attempt, reason in (
        (1, "AUTHENTICATION_FAILED"),
        (2, "AUTHENTICATION_FAILED"),
        (3, "PIN_LOCKED"),
    ):
        with pytest.raises(AuthenticationError, match=reason):
            owner.issue_authentication_proof(request(), "wrong", NOW)
        current = security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE)
        assert current.failed_attempts == attempt
        assert current.pin_revision == 1 and current.verifier == verifier
    assert current.lockout_until_utc == "2026-08-10T10:08:00Z"
    history_size = len(owner.snapshot.accepted_pins)
    with pytest.raises(AuthenticationError, match="PIN_LOCKED"):
        owner.issue_authentication_proof(request(), RAW_PIN, NOW + timedelta(seconds=299))
    assert comparator.calls == 3
    assert len(owner.snapshot.accepted_pins) == history_size


def test_success_at_lockout_expiry_atomically_resets_pin_and_issues_proof(tmp_path: Path) -> None:
    security, owner, _ = prepared(tmp_path)
    for _ in range(3):
        with pytest.raises(AuthenticationError):
            owner.issue_authentication_proof(request(), "wrong", NOW)
    before = security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE)
    proof = owner.issue_authentication_proof(request(), RAW_PIN, NOW + timedelta(seconds=300))
    current = security.resolve_current_pin(ACCOUNT, OPERATOR, DEVICE)
    assert (current.failed_attempts, current.lockout_until_utc) == (0, None)
    assert (current.pin_revision, current.verifier, current.salt_reference) == (
        before.pin_revision,
        before.verifier,
        before.salt_reference,
    )
    assert owner.resolve_accepted_proof(proof) == proof
    assert before.content_fingerprint_sha256 in owner.snapshot.accepted_pins


def test_raw_pin_and_leaking_dependency_exception_are_not_retained(tmp_path: Path) -> None:
    raw_pin = "do-not-retain-9876"
    _, owner, _ = prepared(tmp_path, Comparator(leak=True))
    with pytest.raises(AuthenticationError, match="PIN_VERIFIER_DEPENDENCY_FAILURE") as caught:
        owner.issue_authentication_proof(request(), raw_pin, NOW)
    assert raw_pin not in str(caught.value)
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None
    assert raw_pin not in repr(owner.snapshot)


@pytest.mark.parametrize(
    "now",
    [datetime(2026, 8, 10), datetime(2026, 8, 10, tzinfo=timezone(timedelta(hours=1)))],
)
def test_public_time_requires_exact_utc(tmp_path: Path, now: datetime) -> None:
    _, owner, comparator = prepared(tmp_path)
    with pytest.raises(AuthenticationError, match="MALFORMED_UNTRUSTED_CONTEXT"):
        owner.issue_authentication_proof(request(), RAW_PIN, now)
    assert comparator.calls == 0
