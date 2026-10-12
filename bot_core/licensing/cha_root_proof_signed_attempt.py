"""Stage 9 stops at SIGNED_IMMUTABLE_DURABLE_NOT_SENT. No issuer transport."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import cast
from weakref import WeakKeyDictionary

from bot_core.cha_attempt_signatures import RetainedIssuanceRequest, verify_final_identity
from bot_core.cha_attempt_store import (
    AttemptAuthorization,
    AttemptConflictError,
    AttemptCorruptError,
    AttemptIdentity,
    AttemptNotFoundError,
    AttemptState,
    CurrentAttempt,
    SQLiteCHAAttemptStore,
)
from bot_core.cha_issuance_execution import execute_local_issuance
from bot_core.cha_issuance_request import (
    CLAIMANT_PROFILE,
    REQUESTER_PROFILE,
    IssuanceSignerIdentity,
    IssuanceSigningRole,
    request_bytes,
    request_reference,
)
from bot_core.cha_root_proof_signing_custody import (
    LocalCHARequesterSigningCustody,
    LocalPreaccountClaimantAuthorizationCustody,
)
from bot_core.local_signing_custody import _custody_lock, _validate_directory
from bot_core.postgresql_root_proof_issuance_authority import PostgreSQLRootProofIssuanceAuthority

from . import cha_root_proof_attempt_reservation as reservation_boundary
from .canonical import canonical_json_bytes, parse_canonical
from .cha_account_reservation import VerifiedAccountGenesisInitialBinding


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
    _, current, retained = _read_issuer_preflight_source(value)
    return current, retained


def _signed_attempt_lock_selector(value: object) -> str:
    """Zwróć zachowany locator wyłącznie do wyboru blokady, bez autoryzacji.

    Ten odczyt pamięci nie potwierdza bieżącego stanu ani autentyczności źródeł.
    Konsument musi po uzyskaniu blokady ponownie zweryfikować pełne capability
    przez _read_issuer_preflight_source i porównać zweryfikowany attempt ID.
    Selector nie zezwala na odczyt proof, alokację ID, podpis ani zapis.
    """
    if type(value) is not VerifiedSignedImmutableRootProofIssuanceAttempt or value not in _SIGNED:
        raise SignedIssuanceAttemptError("VERIFIED_SIGNED_IMMUTABLE_ATTEMPT_REQUIRED")
    selector = _SIGNED[value].current.reservation.issuance_attempt_id
    if type(selector) is not str or not selector:
        raise SignedIssuanceAttemptError("EXACT_RETAINED_ATTEMPT_LOCK_SELECTOR_REQUIRED")
    return selector


def _read_issuer_preflight_source(
    value: object,
) -> tuple[reservation_boundary._AuthorizationSnapshot, CurrentAttempt, RetainedIssuanceRequest]:
    """Read genuine local CHA evidence without granting transport authority."""
    if type(value) is not VerifiedSignedImmutableRootProofIssuanceAttempt or value not in _SIGNED:
        raise SignedIssuanceAttemptError("VERIFIED_SIGNED_IMMUTABLE_ATTEMPT_REQUIRED")
    snapshot = _SIGNED[value]
    upstream = reservation_boundary._exact_binding_authorization(
        snapshot.binding, snapshot.authorization
    )
    qualified, _, identities = _qualified_pair(snapshot.authorization)
    if qualified != upstream:
        raise SignedIssuanceAttemptError("PREFLIGHT_AUTHORIZATION_CHANGED")
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
    return upstream, current, retained


@dataclass(frozen=True, slots=True)
class _IssuanceReconciliationSource:
    """Retained local observation, separate from live issuance authorization."""

    binding: VerifiedAccountGenesisInitialBinding
    binding_state_raw: bytes
    context_raw: bytes
    authorization: AttemptAuthorization
    current: CurrentAttempt
    retained: RetainedIssuanceRequest | None
    authority: PostgreSQLRootProofIssuanceAuthority


def _read_issuance_reconciliation_source(binding: object) -> _IssuanceReconciliationSource:
    """Resolve genuine retained CHA evidence through the deployment composition.

    This read deliberately does not require current UNBOUND entitlement or
    active private custody: those live preconditions belong to issuance, while
    reconciliation must also observe later BOUND and revoked states. Public
    credential provenance must be verified by the reconciliation consumer.
    """
    from deployment.windows_production_cha_account_reservation import _safe

    # This verifier rejects caller DTOs and revalidates the retained binding
    # against its genuine installed upstream authority on every invocation.
    state_raw, context = reservation_boundary._binding_context(binding)
    authority = reservation_boundary._issuance_authority_provider()
    if type(authority) is not PostgreSQLRootProofIssuanceAuthority:
        raise SignedIssuanceAttemptError("EXACT_PUBLIC_ISSUER_AUTHORITY_REQUIRED")
    authority.requalify()
    if (
        authority._entitlement_subject.environment != context["environment"]
        or authority._entitlement_subject.trust_domain != context["pdsa_trust_domain"]
    ):
        raise AttemptConflictError("retained initial binding has a different authority scope")
    path = reservation_boundary._attempt_store_path()
    if not path.is_absolute():
        raise SignedIssuanceAttemptError("UNSAFE_ATTEMPT_STORE_PATH")
    _safe(path.parent, directory=True)
    for target in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
        _safe(target)
    with SQLiteCHAAttemptStore.open_readonly(path, context["pdsa_trust_domain"]) as store:
        current = store.attempt(context["logical_operation_id"])
        try:
            retained = store.signed_request(context["logical_operation_id"])
        except AttemptNotFoundError:
            if current.state is not AttemptState.RESERVED_AWAITING_SIGNATURES:
                raise AttemptCorruptError(
                    "signed attempt has no retained canonical request"
                ) from None
            retained = None
    auth = current.reservation.authorization
    for name in (
        "environment",
        "product_scope",
        "logical_operation_id",
        "account_id",
        "canonical_genesis_request_fingerprint_sha256",
        "initial_binding_reference",
        "initial_binding_digest_sha256",
        "reservation_identity",
        "reservation_relation",
        "initial_binding_sha256",
    ):
        if getattr(auth, name) != context[name]:
            raise AttemptConflictError("retained attempt differs from exact initial binding")
    if auth.trust_domain != context["pdsa_trust_domain"]:
        raise AttemptConflictError("retained attempt has a different trust domain")
    if retained is not None:
        expected_raw = request_bytes(auth, current.reservation.issuance_attempt_id)
        if (
            retained.canonical_bytes != expected_raw
            or retained.reference != request_reference(expected_raw)
            or retained.digest != hashlib.sha256(expected_raw).hexdigest()
        ):
            raise AttemptCorruptError("retained signing request identity differs")
    if current.identity is not None:
        verify_final_identity(current.identity, retained)
    # Re-read the genuine InitialBinding after the independent local reads.
    # Equality detects change; it does not establish a global atomic snapshot.
    if reservation_boundary._binding_context(binding) != (state_raw, context):
        raise AttemptConflictError("retained initial binding changed during reconciliation")
    if path != reservation_boundary._attempt_store_path():
        raise AttemptConflictError("retained attempt store owner changed during reconciliation")
    return _IssuanceReconciliationSource(
        cast(VerifiedAccountGenesisInitialBinding, binding),
        state_raw,
        canonical_json_bytes(context),
        auth,
        current,
        retained,
        authority,
    )


def require_verified_signed_immutable_root_proof_issuance_attempt(
    value: object,
) -> VerifiedSignedImmutableRootProofIssuanceAttempt:
    _signed_snapshot(value)
    return cast(VerifiedSignedImmutableRootProofIssuanceAttempt, value)


def _finish(
    binding: object, authorization: object, original: CurrentAttempt | None = None
) -> VerifiedSignedImmutableRootProofIssuanceAttempt:
    from bot_core.cha_attempt_signatures import require_local_only_issuance_scope

    require_local_only_issuance_scope()
    upstream = reservation_boundary._exact_binding_authorization(binding, authorization)
    auth = upstream.authorization
    path = reservation_boundary._attempt_store_path()
    # Separate SQLite commits cannot hold a SQLite writer lock across the durable
    # intent. The protected-host advisory lock serializes the entire ceremony.
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
            retained = execute_local_issuance("PREPARE_SIGNATURE", store, current, signer)
            _cut(role.value + ":intent_durable")
            # Recheck live authorization for an exact operation, including a
            # recovery of an intent whose signature checkpoint never committed.
            _qualified_pair(authorization)
            signature = (
                execute_local_issuance(
                    "REQUESTER_SIGNATURE", custody[0], retained.canonical_bytes, signer
                )
                if role is IssuanceSigningRole.REQUESTER
                else execute_local_issuance(
                    "CLAIMANT_AUTHORIZATION", custody[1], retained.canonical_bytes, signer
                )
            )
            _cut(role.value + ":signer_returned")
            _qualified_pair(authorization)
            current = execute_local_issuance("PERSIST_SIGNATURE", store, current, signer, signature)
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
        current = execute_local_issuance("FINALIZE", store, identity, expected_fence=current.fence)
        _cut("finalization_durable")
    result = object.__new__(VerifiedSignedImmutableRootProofIssuanceAttempt)
    _SIGNED[result] = _SignedSnapshot(binding, authorization, path, current)
    _signed_snapshot(result)
    return result


def supersede_root_proof_issuance_attempt_pre_send(
    binding: object, authorization: object
) -> reservation_boundary.VerifiedRootProofIssuanceAttemptReservation:
    """Explicitly supersede an unsendable reservation for new eligible credentials.

    Fresh genuine public authorization qualifies the replacement; private
    custody must separately be provisioned and qualified before signing it.
    """
    upstream = reservation_boundary._exact_binding_authorization(binding, authorization)
    auth = upstream.authorization
    path = reservation_boundary._attempt_store_path()
    with (
        _custody_lock(_validate_directory(path.parent, create=False), exclusive=True),
        reservation_boundary._open_store(auth.trust_domain) as store,
    ):
        current = store.attempt(auth.logical_operation_id)
        reservation_boundary._exact_binding_authorization(binding, authorization)
        current = store.supersede_pre_send(current, auth)
    return reservation_boundary._issue_reservation(binding, authorization, current)


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
