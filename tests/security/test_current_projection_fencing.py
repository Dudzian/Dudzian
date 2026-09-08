from __future__ import annotations

from dataclasses import replace
from types import MappingProxyType

import pytest

from bot_core.persistence.fingerprints import canonical_json_sha256
from bot_core.security.current_projection_authority import (
    _accept_device_projection,
    _accept_identity_projection,
)
from bot_core.security.initial_security import (
    DeviceTrustProjection,
    InitialSecurityError,
    InitialSecuritySemanticState,
    OperatorIdentitySecurityProjection,
)

ACCOUNT = "acct_018f0000-0000-7000-8000-000000000001"
OPERATOR = "op_018f0000-0000-7000-8000-000000000002"
DEVICE = "dev_018f0000-0000-7000-8000-000000000003"


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


def test_exact_current_is_idempotent_without_snapshot_republication():
    state = InitialSecuritySemanticState()
    assert _accept_identity_projection(state, identity())
    assert _accept_device_projection(state, device())
    before = state.snapshot
    assert _accept_identity_projection(state, identity())
    assert _accept_device_projection(state, device())
    assert state.snapshot is before


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
