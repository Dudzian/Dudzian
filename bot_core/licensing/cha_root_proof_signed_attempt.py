"""Stage 9 stops at SIGNED_IMMUTABLE_DURABLE_NOT_SENT. No issuer transport."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from weakref import WeakKeyDictionary

from bot_core.cha_attempt_signatures import RetainedIssuanceRequest
from bot_core.cha_attempt_store import (
    AttemptAuthorization,
    AttemptIdentity,
    AttemptState,
    CurrentAttempt,
)
from bot_core.cha_issuance_request import (
    CLAIMANT_PROFILE,
    REQUESTER_PROFILE,
    IssuanceSignerIdentity,
    IssuanceSigningRole,
)
from bot_core.cha_root_proof_signing_custody import (
    LocalCHARequesterSigningCustody,
    LocalPreaccountClaimantAuthorizationCustody,
)
from bot_core.local_signing_custody import _custody_lock, _validate_directory

from . import cha_root_proof_attempt_reservation as reservation_boundary
from .canonical import parse_canonical


class SignedIssuanceAttemptError(RuntimeError):
    """A changed upstream authority or uncertain signing outcome fails closed."""


def _cut(name: str) -> None:
    """Fault injection seam; production never changes authority decisions here."""


def _custody_directories(auth: AttemptAuthorization) -> tuple[Path, Path]:
    base = reservation_boundary._attempt_store_path().parent
    principal_scope = hashlib.sha256(
        (
            auth.environment + "\x00" + auth.trust_domain + "\x00" + auth.provisioning_principal_id
        ).encode()
    ).hexdigest()
    return (
        base / "cha-root-proof-requester-custody",
        base / "preaccount-root-proof-claimant-custody" / principal_scope,
    )


def _installed_custody(auth):
    requester_path, claimant_path = _custody_directories(auth)
    return (
        LocalCHARequesterSigningCustody(requester_path, trust_domain=auth.trust_domain),
        LocalPreaccountClaimantAuthorizationCustody(
            claimant_path,
            trust_domain=auth.trust_domain,
            provisioning_principal=auth.provisioning_principal_id,
        ),
    )


def _qualified_pair(authorization):
    snapshot = reservation_boundary._authorization_snapshot(authorization)
    auth = snapshot.authorization
    requester, claimant = _installed_custody(auth)
    if (
        type(requester) is not LocalCHARequesterSigningCustody
        or type(claimant) is not LocalPreaccountClaimantAuthorizationCustody
    ):
        raise SignedIssuanceAttemptError("EXACT_INSTALLED_ISSUANCE_CUSTODY_REQUIRED")
    identities = requester.identity(), claimant.identity()
    evidence = parse_canonical(snapshot.evidence_raw)
    for identity, role, port, evidence_name in (
        (
            identities[0],
            IssuanceSigningRole.REQUESTER,
            snapshot.provider.requester_registry,
            "requester_credential",
        ),
        (
            identities[1],
            IssuanceSigningRole.CLAIMANT,
            snapshot.provider.claimant_registry,
            "claimant_credential",
        ),
    ):
        identity.require_authorization(auth)
        public = port.public_key(identity.key_id)
        if (
            identity.role is not role
            or public.hex() != identity.public_key_hex
            or evidence[evidence_name]["public_key_material_identity"]
            != identity.key_material_identity
        ):
            raise SignedIssuanceAttemptError("REGISTRY_PRIVATE_CUSTODY_MISMATCH")
    if identities[0].public_key_hex == identities[1].public_key_hex or any(
        getattr(identities[0], field) == getattr(identities[1], field)
        for field in (
            "service_namespace",
            "credential_namespace",
            "lifecycle_namespace",
            "key_handle",
        )
    ):
        raise SignedIssuanceAttemptError("REQUESTER_CLAIMANT_CUSTODY_ALIAS")
    return snapshot, (requester, claimant), identities


def _retained_pair_matches(
    value, identities: tuple[IssuanceSignerIdentity, IssuanceSignerIdentity]
) -> None:
    for retained, expected in ((value.requester, identities[0]), (value.claimant, identities[1])):
        if retained is not None:
            if retained[0] != expected:
                raise SignedIssuanceAttemptError("RETAINED_SIGNER_AUTHORIZATION_CHANGED")
            expected.verify(value.canonical_bytes, retained[1])


@dataclass(frozen=True, slots=True)
class _SignedSnapshot:
    binding: object
    authorization: object
    path: Path
    current: CurrentAttempt


class VerifiedSignedImmutableRootProofIssuanceAttempt:
    """Opaque exact current signed-attempt capability. It grants no send authority."""

    __slots__ = ("__weakref__",)

    def __init__(self) -> None:
        raise TypeError("signed attempt comes only from durable authority verification")

    def __init_subclass__(cls) -> None:
        raise TypeError("signed attempt cannot be subclassed")

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("signed attempt is immutable")

    def __copy__(self):
        raise TypeError("signed attempt cannot be copied")

    def __deepcopy__(self, memo: object):
        raise TypeError("signed attempt cannot be copied")

    @property
    def identity(self) -> AttemptIdentity:
        current, _ = _signed_snapshot(self)
        return cast(AttemptIdentity, current.identity)

    @property
    def issuance_attempt_id(self) -> str:
        return self.identity.issuance_attempt_id

    @property
    def state(self) -> AttemptState:
        return _signed_snapshot(self)[0].state

    @property
    def canonical_request_bytes(self) -> bytes:
        return _signed_snapshot(self)[1].canonical_bytes


_SIGNED: WeakKeyDictionary[VerifiedSignedImmutableRootProofIssuanceAttempt, _SignedSnapshot] = (
    WeakKeyDictionary()
)


def _signed_snapshot(value: object) -> tuple[CurrentAttempt, RetainedIssuanceRequest]:
    if type(value) is not VerifiedSignedImmutableRootProofIssuanceAttempt or value not in _SIGNED:
        raise SignedIssuanceAttemptError("VERIFIED_SIGNED_IMMUTABLE_ATTEMPT_REQUIRED")
    snapshot = _SIGNED[value]
    upstream = reservation_boundary._exact_binding_authorization(
        snapshot.binding, snapshot.authorization
    )
    _, _, identities = _qualified_pair(snapshot.authorization)
    if snapshot.path != reservation_boundary._attempt_store_path():
        raise SignedIssuanceAttemptError("ATTEMPT_STORE_OWNER_CHANGED")
    with reservation_boundary._open_store(upstream.authorization.trust_domain) as store:
        current = store.attempt(upstream.authorization.logical_operation_id)
        retained = store.signed_request(upstream.authorization.logical_operation_id)
    if (
        current != snapshot.current
        or current.state is not AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT
    ):
        raise SignedIssuanceAttemptError("SIGNED_IMMUTABLE_ATTEMPT_CHANGED")
    _retained_pair_matches(retained, identities)
    return current, retained


def require_verified_signed_immutable_root_proof_issuance_attempt(
    value: object,
) -> VerifiedSignedImmutableRootProofIssuanceAttempt:
    _signed_snapshot(value)
    return cast(VerifiedSignedImmutableRootProofIssuanceAttempt, value)


def _finish(
    binding: object, authorization: object, original: CurrentAttempt | None = None
) -> VerifiedSignedImmutableRootProofIssuanceAttempt:
    upstream = reservation_boundary._exact_binding_authorization(binding, authorization)
    auth = upstream.authorization
    path = reservation_boundary._attempt_store_path()
    # Separate SQLite commits cannot hold a SQLite writer lock across the durable
    # latch. The protected-host advisory lock serializes the entire ceremony.
    directory = _validate_directory(path.parent, create=False)
    with (
        _custody_lock(directory, exclusive=True),
        reservation_boundary._open_store(auth.trust_domain) as store,
    ):
        current = store.attempt(auth.logical_operation_id)
        if current.reservation.authorization != auth or (
            original is not None and current.reservation != original.reservation
        ):
            raise SignedIssuanceAttemptError("EXACT_CURRENT_RESERVED_LINEAGE_REQUIRED")
        for role, predecessor in (
            (IssuanceSigningRole.REQUESTER, AttemptState.RESERVED_AWAITING_SIGNATURES),
            (IssuanceSigningRole.CLAIMANT, AttemptState.REQUEST_SIGNED_BY_REQUESTER),
        ):
            if current.state is not predecessor:
                continue
            _, custody, identities = _qualified_pair(authorization)
            if role is IssuanceSigningRole.CLAIMANT:
                _retained_pair_matches(store.signed_request(auth.logical_operation_id), identities)
            offset = 0 if role is IssuanceSigningRole.REQUESTER else 1
            signer = identities[offset]
            _cut(role.value + ":before_intent")
            retained = store.prepare_signature(current, signer)
            _cut(role.value + ":intent_durable")
            # Recheck operation authorization after latch commit and immediately
            # before the one private operation. Failure leaves the latch spent.
            _qualified_pair(authorization)
            signature = (
                custody[0].sign_issuance_request(retained.canonical_bytes, signer)
                if role is IssuanceSigningRole.REQUESTER
                else custody[1].authorize_entitlement_claim(retained.canonical_bytes, signer)
            )
            _cut(role.value + ":signer_returned")
            _qualified_pair(authorization)
            current = store.persist_signature(current, signer, signature)
            _cut(role.value + ":checkpoint_durable")
        _, _, identities = _qualified_pair(authorization)
        retained = store.signed_request(auth.logical_operation_id)
        _retained_pair_matches(retained, identities)
        if retained.requester is None or retained.claimant is None:
            raise SignedIssuanceAttemptError("BOTH_DURABLE_SIGNATURES_REQUIRED")
        identity = AttemptIdentity(
            auth,
            current.reservation.issuance_attempt_id,
            retained.digest,
            retained.reference,
            retained.requester[1],
            retained.claimant[1],
            REQUESTER_PROFILE,
            CLAIMANT_PROFILE,
        )
        if current.state not in {
            AttemptState.CLAIMANT_AUTHORIZED,
            AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT,
        }:
            raise SignedIssuanceAttemptError("SIGNING_SCOPE_STOP_REQUIRED")
        _cut("before_finalization")
        current = store.finalize_attempt(identity, expected_fence=current.fence)
        _cut("finalization_durable")
    result = object.__new__(VerifiedSignedImmutableRootProofIssuanceAttempt)
    _SIGNED[result] = _SignedSnapshot(binding, authorization, path, current)
    _signed_snapshot(result)
    return result


def sign_root_proof_issuance_attempt(
    reservation: object,
) -> VerifiedSignedImmutableRootProofIssuanceAttempt:
    if (
        type(reservation) is not reservation_boundary.VerifiedRootProofIssuanceAttemptReservation
        or reservation not in reservation_boundary._RESERVATIONS
    ):
        raise SignedIssuanceAttemptError("VERIFIED_ATTEMPT_RESERVATION_REQUIRED")
    snapshot = reservation_boundary._RESERVATIONS[reservation]
    if snapshot.path != reservation_boundary._attempt_store_path():
        raise SignedIssuanceAttemptError("ATTEMPT_STORE_OWNER_CHANGED")
    return _finish(snapshot.binding, snapshot.authorization, snapshot.current)


def resume_root_proof_issuance_attempt(
    binding: object, authorization: object
) -> VerifiedSignedImmutableRootProofIssuanceAttempt:
    """Restart from existing upstream verified capabilities; never reserve/remint."""
    return _finish(binding, authorization)
