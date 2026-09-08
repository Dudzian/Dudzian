from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.security.current_projection_authority import (
    _accept_device_projection,
    _accept_identity_projection,
    _accept_pin_projection,
    _accept_session_projection,
)
from bot_core.security.initial_security import (
    DeviceTrustProjection,
    InitialSecurityError,
    InitialSecuritySemanticState,
    OperatorIdentitySecurityProjection,
    PinVerifierRecord,
    SessionSecurityState,
)

ACCOUNT = "acct_018f0000-0000-7000-8000-000000000001"
OPERATOR = "op_018f0000-0000-7000-8000-000000000002"
DEVICE = "dev_018f0000-0000-7000-8000-000000000003"
RUNTIME = "run_018f0000-0000-7000-8000-000000000004"


def _fingerprinted(value):
    payload = (
        {name: item for name, item in value.__dict__.items()}
        if hasattr(value, "__dict__")
        else {field: getattr(value, field) for field in value.__dataclass_fields__}
    )
    payload.pop("content_fingerprint_sha256")
    return replace(value, content_fingerprint_sha256=canonical_json_sha256(payload))


def identity(state="ACTIVE", revision=1, generation=1):
    return _fingerprinted(
        OperatorIdentitySecurityProjection(ACCOUNT, OPERATOR, state, revision, generation, "")
    )


def device(state="TRUSTED", revision=1, generation=1, enrollment=1):
    return _fingerprinted(
        DeviceTrustProjection(ACCOUNT, DEVICE, state, revision, generation, enrollment, "")
    )


def pin(revision=1, failures=0, lockout=None, verifier="a" * 64, generation=1):
    return _fingerprinted(
        PinVerifierRecord(
            ACCOUNT,
            OPERATOR,
            DEVICE,
            "ARGON2ID",
            1,
            "secure-store://pins/salt",
            verifier,
            revision,
            failures,
            lockout,
            generation,
            "",
        )
    )


def session(state="UNLOCKED", generation=1, security_generation=1, runtime=RUNTIME):
    return _fingerprinted(
        SessionSecurityState(
            ACCOUNT, OPERATOR, DEVICE, runtime, state, generation, security_generation, ""
        )
    )


def test_exact_current_is_idempotent_without_snapshot_republication():
    state = InitialSecuritySemanticState()
    assert _accept_identity_projection(state, identity())
    assert _accept_device_projection(state, device())
    before = state.snapshot
    assert _accept_identity_projection(state, identity())
    assert _accept_device_projection(state, device())
    assert state.snapshot is before


def test_pin_and_session_exact_current_are_idempotent_without_republication():
    state = InitialSecuritySemanticState()
    assert _accept_pin_projection(state, pin())
    assert _accept_session_projection(state, session())
    before = state.snapshot
    assert _accept_pin_projection(state, pin())
    assert _accept_session_projection(state, session())
    assert state.snapshot is before


def test_pin_same_revision_failure_update_and_reset_are_legal():
    state = InitialSecuritySemanticState()
    clean = pin()
    failed = pin(failures=1)
    assert _accept_pin_projection(state, clean)
    assert _accept_pin_projection(state, failed)
    assert (
        state.snapshot.current_pins[(ACCOUNT, OPERATOR, DEVICE)]
        == failed.content_fingerprint_sha256
    )
    assert _accept_pin_projection(state, clean)
    assert (
        state.snapshot.current_pins[(ACCOUNT, OPERATOR, DEVICE)] == clean.content_fingerprint_sha256
    )


@pytest.mark.parametrize("candidate", [pin(verifier="b" * 64), pin(generation=2)])
def test_pin_same_revision_credential_substitution_is_denied(candidate):
    state = InitialSecuritySemanticState()
    current = pin()
    assert _accept_pin_projection(state, current)
    assert not _accept_pin_projection(state, candidate)
    assert (
        state.snapshot.current_pins[(ACCOUNT, OPERATOR, DEVICE)]
        == current.content_fingerprint_sha256
    )


def test_pin_revision_rollback_is_denied_but_generic_revision_jump_is_accepted():
    state = InitialSecuritySemanticState()
    first = pin()
    third = pin(revision=3, verifier="c" * 64, generation=0x2)
    assert _accept_pin_projection(state, first)
    assert _accept_pin_projection(state, third)
    assert not _accept_pin_projection(state, first)
    assert (
        state.snapshot.current_pins[(ACCOUNT, OPERATOR, DEVICE)] == third.content_fingerprint_sha256
    )


def test_session_same_generation_substitution_and_rollback_are_denied():
    state = InitialSecuritySemanticState()
    first = session()
    second = session("LOCKED", 2)
    assert _accept_session_projection(state, first)
    assert not _accept_session_projection(state, session("LOCKED", 1))
    assert _accept_session_projection(state, second)
    assert not _accept_session_projection(state, first)
    assert (
        state.snapshot.current_sessions[(ACCOUNT, OPERATOR, DEVICE)]
        == second.content_fingerprint_sha256
    )


def test_generic_session_accepts_higher_generation_without_public_graph_policy():
    state = InitialSecuritySemanticState()
    assert _accept_session_projection(state, session("LOGGED_OUT", 1))
    candidate = session("UNLOCKED", 3, security_generation=1, runtime=RUNTIME.replace("4", "5"))
    assert _accept_session_projection(state, candidate)


@pytest.mark.parametrize(
    ("first", "candidate"),
    [
        (identity(), identity("REVOKED")),
        (identity(revision=2), identity(revision=1)),
        (identity(revision=2, generation=2), identity(revision=3, generation=1)),
        (identity("REVOKED", 2), identity("ACTIVE", 3)),
    ],
)
def test_identity_rollback_substitution_and_terminal_resurrection_are_denied(first, candidate):
    state = InitialSecuritySemanticState()
    assert _accept_identity_projection(state, first)
    assert not _accept_identity_projection(state, candidate)
    assert (
        state.snapshot.current_identities[(ACCOUNT, OPERATOR)] == first.content_fingerprint_sha256
    )


@pytest.mark.parametrize(
    ("first", "candidate"),
    [
        (device(), device("ENROLLED_UNTRUSTED")),
        (device(revision=2), device(revision=1)),
        (device(revision=2, generation=2), device(revision=3, generation=1)),
        (device(revision=2, enrollment=2), device(revision=3, enrollment=1)),
        (device("REVOKED", 2), device("TRUSTED", 3)),
        (device("REPLACED", 2), device("TRUSTED", 3)),
    ],
)
def test_device_rollback_substitution_and_terminal_resurrection_are_denied(first, candidate):
    state = InitialSecuritySemanticState()
    assert _accept_device_projection(state, first)
    assert not _accept_device_projection(state, candidate)
    assert state.snapshot.current_devices[(ACCOUNT, DEVICE)] == first.content_fingerprint_sha256


@pytest.mark.parametrize("kind", ["identity", "device"])
def test_corrupted_current_fails_closed_without_publication(kind):
    state = InitialSecuritySemanticState()
    before = state.snapshot
    if kind == "identity":
        state.snapshot = replace(
            before, current_identities=MappingProxyType({(ACCOUNT, OPERATOR): "0" * 64})
        )
        call = lambda: _accept_identity_projection(state, identity())
    else:
        state.snapshot = replace(
            before, current_devices=MappingProxyType({(ACCOUNT, DEVICE): "0" * 64})
        )
        call = lambda: _accept_device_projection(state, device())
    corrupted = state.snapshot
    with pytest.raises(InitialSecurityError, match="CONTRACT_INCONSISTENT"):
        call()
    assert state.snapshot is corrupted


@pytest.mark.parametrize("kind", ["pin", "session"])
def test_corrupted_pin_or_session_current_fails_closed_without_publication(kind):
    state = InitialSecuritySemanticState()
    before = state.snapshot
    scope = (ACCOUNT, OPERATOR, DEVICE)
    if kind == "pin":
        state.snapshot = replace(before, current_pins=MappingProxyType({scope: "0" * 64}))
        call = lambda: _accept_pin_projection(state, pin())
    else:
        state.snapshot = replace(before, current_sessions=MappingProxyType({scope: "0" * 64}))
        call = lambda: _accept_session_projection(state, session())
    corrupted = state.snapshot
    with pytest.raises(InitialSecurityError, match="CONTRACT_INCONSISTENT"):
        call()
    assert state.snapshot is corrupted


@pytest.mark.parametrize("kind", ["pin", "session"])
def test_pin_or_session_accepted_fingerprint_collision_fails_closed(kind):
    state = InitialSecuritySemanticState()
    if kind == "pin":
        candidate = pin()
        collision = replace(
            pin(revision=2), content_fingerprint_sha256=candidate.content_fingerprint_sha256
        )
        state.snapshot = replace(
            state.snapshot,
            accepted_pins=MappingProxyType({candidate.content_fingerprint_sha256: collision}),
        )
        call = lambda: _accept_pin_projection(state, candidate)
    else:
        candidate = session()
        collision = replace(
            session("LOCKED", 2), content_fingerprint_sha256=candidate.content_fingerprint_sha256
        )
        state.snapshot = replace(
            state.snapshot,
            accepted_sessions=MappingProxyType({candidate.content_fingerprint_sha256: collision}),
        )
        call = lambda: _accept_session_projection(state, candidate)
    with pytest.raises(InitialSecurityError, match="CONTRACT_INCONSISTENT"):
        call()


def test_accepted_fingerprint_collision_fails_closed():
    state = InitialSecuritySemanticState()
    candidate = identity()
    collision = identity(revision=2)
    collision = replace(collision, content_fingerprint_sha256=candidate.content_fingerprint_sha256)
    state.snapshot = replace(
        state.snapshot,
        accepted_identities=MappingProxyType({candidate.content_fingerprint_sha256: collision}),
    )
    with pytest.raises(InitialSecurityError, match="CONTRACT_INCONSISTENT"):
        _accept_identity_projection(state, candidate)
