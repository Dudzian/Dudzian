"""Read-only semantic issuer preflight of an exact local durable CHA request.

This first runtime slice neither transports requests nor binds entitlement or
issues proofs. Its result is an observation, never an issuance authorization.
Every invocation resolves genuine InitialBinding and live PostgreSQL authority
again; an earlier report cannot authorize a subsequent operation.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Literal

from bot_core.cha_attempt_signatures import RetainedIssuanceRequest, verify_final_identity
from bot_core.cha_attempt_store import AttemptIdentity, AttemptState, CurrentAttempt
from bot_core.cha_issuance_request import (
    CLAIMANT_PROFILE,
    REQUESTER_PROFILE,
    IssuanceSigningRole,
    request_bytes,
    request_reference,
)
from bot_core.entitlement_registry_contract import (
    AuthoritativeEntitlementState,
    EntitlementLifecycle,
    RegistryReadOutcome,
    RegistryReadResult,
    UnboundBinding,
    validate_exact_snapshot,
)
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.licensing.cha_root_proof_attempt_reservation import _AuthorizationSnapshot
from bot_core.licensing.cha_root_proof_signed_attempt import _read_issuer_preflight_source
from bot_core.postgresql_preaccount_credentials import CredentialGeneration, CredentialLifecycle
from bot_core.postgresql_root_proof_issuance_authority import PostgreSQLRootProofIssuanceAuthority
from bot_core.root_proof_issuer_substrate import (
    CredentialSemanticRole,
    public_key_material_identity,
)


class RootProofIssuerRuntimeError(RuntimeError):
    """Missing, changed or invalid authority evidence fails preflight closed."""


@dataclass(frozen=True, slots=True)
class RootProofIssuancePreflight:
    """Historical read-only report. No consumer may treat it as a capability."""

    disposition: Literal["VERIFIED_NOT_AUTHORIZED_TO_ISSUE"]
    security_profile: Literal["PRODUCTION_LOCAL"]
    environment: Literal["PRODUCTION"]
    trust_domain: str
    logical_operation_id: str
    account_id: str
    issuance_attempt_id: str
    request_reference: str
    request_digest_sha256: str
    initial_binding_reference: str
    initial_binding_digest_sha256: str
    entitlement_id: str
    entitlement_generation: int
    entitlement_registry_revision: int
    requester_registry_revision: int
    claimant_registry_revision: int
    attempt_identity_digest_sha256: str


def _cut(name: str) -> None:
    """Fault injection seam; it grants no authority and performs no writes."""


def _verify_request(
    source: _AuthorizationSnapshot, current: CurrentAttempt, retained: RetainedIssuanceRequest
) -> tuple[int, int, int]:
    """Independently authenticate immutable bytes against public authority."""
    authority = source.provider
    if type(authority) is not PostgreSQLRootProofIssuanceAuthority:
        raise RootProofIssuerRuntimeError("EXACT_PUBLIC_ISSUER_AUTHORITY_REQUIRED")
    auth = source.authorization
    if (
        auth.environment != "PRODUCTION"
        or auth.reservation_identity is None
        or current.state is not AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT
        or type(current.identity) is not AttemptIdentity
        or current.reservation.authorization != auth
    ):
        raise RootProofIssuerRuntimeError("EXACT_CURRENT_SIGNED_ATTEMPT_REQUIRED")
    identity = current.identity
    expected_raw = request_bytes(auth, identity.issuance_attempt_id)
    if (
        type(retained) is not RetainedIssuanceRequest
        or type(retained.canonical_bytes) is not bytes
        or retained.canonical_bytes != expected_raw
        or retained.reference != request_reference(expected_raw)
        or retained.digest != hashlib.sha256(expected_raw).hexdigest()
        or retained.requester is None
        or retained.claimant is None
    ):
        raise RootProofIssuerRuntimeError("EXACT_IMMUTABLE_REQUEST_REQUIRED")
    verify_final_identity(identity, retained)
    expected_identity = AttemptIdentity(
        auth,
        retained.attempt_id,
        retained.digest,
        retained.reference,
        retained.requester[1],
        retained.claimant[1],
        REQUESTER_PROFILE,
        CLAIMANT_PROFILE,
    )
    if identity != expected_identity or identity.digest_sha256 != expected_identity.digest_sha256:
        raise RootProofIssuerRuntimeError("EXACT_IMMUTABLE_ATTEMPT_IDENTITY_REQUIRED")

    evidence = parse_canonical(source.evidence_raw)
    result = authority.entitlement_registry.authoritative_state(authority._entitlement_subject)
    if (
        type(result) is not RegistryReadResult
        or result.outcome is not RegistryReadOutcome.FOUND
        or type(result.state) is not AuthoritativeEntitlementState
    ):
        raise RootProofIssuerRuntimeError("AUTHORITATIVE_ENTITLEMENT_REQUIRED")
    validate_exact_snapshot(result)
    state = result.state
    if (
        state.subject != authority._entitlement_subject
        or state.lifecycle is not EntitlementLifecycle.ACTIVE
        or type(state.binding) is not UnboundBinding
        or state.identity.environment != auth.environment
        or state.identity.trust_domain != auth.trust_domain
        or state.identity.product_scope != auth.product_scope
        or state.identity.bootstrap_entitlement_id != auth.bootstrap_entitlement_id
        or state.identity.entitlement_generation != auth.entitlement_generation
        or state.provenance.provisioning_principal_id != auth.provisioning_principal_id
        or state.provenance.claimant_key_id != auth.claimant_key_id
        or state.provenance.claimant_key_version != auth.claimant_key_version
        or canonical_json_bytes(asdict(state)) != canonical_json_bytes(evidence["entitlement"])
    ):
        raise RootProofIssuerRuntimeError("EXACT_ACTIVE_UNBOUND_ENTITLEMENT_REQUIRED")

    revisions = []
    public_material = []
    for port, checkpoint, role, semantic_role, principal, key_id, key_version, evidence_name in (
        (
            authority.requester_registry,
            retained.requester,
            IssuanceSigningRole.REQUESTER,
            CredentialSemanticRole.ROOT_PROOF_REQUESTER,
            auth.requester_principal_id,
            auth.requester_key_id,
            auth.requester_key_version,
            "requester",
        ),
        (
            authority.claimant_registry,
            retained.claimant,
            IssuanceSigningRole.CLAIMANT,
            CredentialSemanticRole.ROOT_PROOF_CLAIMANT,
            auth.provisioning_principal_id,
            auth.claimant_key_id,
            auth.claimant_key_version,
            "claimant",
        ),
    ):
        credential = port._current(principal)
        public = port.public_key(key_id)
        signer, signature = checkpoint
        signer.require_authorization(auth)
        if (
            type(credential) is not CredentialGeneration
            or credential.principal_id != principal
            or credential.semantic_role is not semantic_role
            or credential.key_id != key_id
            or credential.key_version != key_version
            or credential.lifecycle is not CredentialLifecycle.ACTIVE
            or credential.environment != auth.environment
            or credential.trust_domain != auth.trust_domain
            or credential.registry_revision != evidence[evidence_name]["registry_revision"]
            or public != credential.public_key
            or signer.role is not role
            or signer.public_key_hex != public.hex()
            or signer.key_material_identity != public_key_material_identity(public)
        ):
            raise RootProofIssuerRuntimeError("EXACT_ACTIVE_PUBLIC_SIGNER_REQUIRED")
        # The retained identity is used only after matching genuine public
        # authority. Existing domain-separated Ed25519 verification is retained.
        signer.verify(expected_raw, signature)
        revisions.append(credential.registry_revision)
        public_material.append(public)
    if auth.requester_key_id == auth.claimant_key_id or public_material[0] == public_material[1]:
        raise RootProofIssuerRuntimeError("REQUESTER_CLAIMANT_ALIAS")
    return state.authoritative_state_revision, revisions[0], revisions[1]


def preflight_root_proof_issuance_attempt(value: object) -> RootProofIssuancePreflight:
    """Evaluate a genuine local signed attempt; return no bind or send grant.

    InitialBinding comes from the installed CHA retained-history verifier. Raw
    requests, caller keys, provider DTOs and fabricated capability objects are
    rejected. Independent authority reads are observations, not a transaction
    across stores; any future mutation requires its own live checks and CAS.
    """
    try:
        source, current, retained = _read_issuer_preflight_source(value)
        entitlement_revision, requester_revision, claimant_revision = _verify_request(
            source, current, retained
        )
        _cut("REQUEST_VERIFIED")
        if _read_issuer_preflight_source(value) != (source, current, retained):
            raise RootProofIssuerRuntimeError("PREFLIGHT_EVIDENCE_CHANGED")
        auth = source.authorization
        identity = current.identity
        if identity is None:
            raise RootProofIssuerRuntimeError("EXACT_CURRENT_SIGNED_ATTEMPT_REQUIRED")
        return RootProofIssuancePreflight(
            "VERIFIED_NOT_AUTHORIZED_TO_ISSUE",
            "PRODUCTION_LOCAL",
            "PRODUCTION",
            auth.trust_domain,
            auth.logical_operation_id,
            auth.account_id,
            identity.issuance_attempt_id,
            retained.reference,
            retained.digest,
            auth.initial_binding_reference,
            auth.initial_binding_digest_sha256,
            auth.bootstrap_entitlement_id,
            auth.entitlement_generation,
            entitlement_revision,
            requester_revision,
            claimant_revision,
            identity.digest_sha256,
        )
    except RootProofIssuerRuntimeError:
        raise
    except Exception:
        # Database/custody errors can contain deployment selectors or secrets.
        # No uncertain read or verifier failure produces a successful report.
        raise RootProofIssuerRuntimeError("PREFLIGHT_AUTHORITY_OR_REQUEST_UNAVAILABLE") from None
