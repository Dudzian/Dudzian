"""Read-only requalification of genuine pre-migration signed attempt evidence.

The migrated entitlement reader deliberately loses its standalone CAS grant.
That capability is included in the original authorization evidence digest and
therefore in the already signed request. This internal bridge verifies the same
actual facts through the current qualified reader while preserving every byte
of the historical request. It never issues or updates an opaque capability.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import psycopg

from bot_core.cha_attempt_signatures import RetainedIssuanceRequest
from bot_core.cha_attempt_store import AttemptAuthorization, AttemptState, CurrentAttempt
from bot_core.entitlement_registry_contract import (
    AuthoritativeEntitlementState,
    EntitlementLifecycle,
    RegistryReadOutcome,
    UnboundBinding,
    validate_exact_snapshot,
)
from bot_core.licensing import (
    cha_root_proof_attempt_reservation as reservation_boundary,
    cha_root_proof_signed_attempt as signed_boundary,
)
from bot_core.licensing.canonical import canonical_json_bytes, parse_canonical
from bot_core.postgresql_atomic_root_proof_commit import (
    AtomicRootProofCommitConflict,
    PreparedAtomicRootProofCommit,
    _trusted_preparation,
)
from bot_core.postgresql_atomic_root_proof_qualification import (
    read_atomic_root_proof_physical_identity,
)
from bot_core.postgresql_root_proof_issuance_authority import PostgreSQLRootProofIssuanceAuthority
from bot_core.postgresql_root_proof_issuance_authority_v2 import (
    PostgreSQLRootProofIssuanceAuthorityV2,
)
from bot_core.root_proof_issuer_runtime import _verify_request


@dataclass(frozen=True, slots=True)
class RequalifiedOriginalHistorySuccessorEvidence:
    """Invocation-local evidence only; grants no preparation or write authority."""

    source: reservation_boundary._AuthorizationSnapshot
    current: CurrentAttempt
    request: RetainedIssuanceRequest
    predecessor: AuthoritativeEntitlementState


def _same_original_authorization_evidence(
    original_auth: AttemptAuthorization,
    original_raw: bytes,
    fresh_auth: AttemptAuthorization,
    fresh_raw: bytes,
    *,
    historical_entitlement_cas: bool,
) -> None:
    """Allow only the explicit removal of the legacy standalone BIND grant.

    The normalization affects a parsed comparison copy, never the retained
    evidence, signed authorization, genuine reader, or its capability result.
    All operation facts and all other provider capability fields stay exact.
    """
    if (
        type(original_auth) is not AttemptAuthorization
        or type(fresh_auth) is not AttemptAuthorization
        or type(original_raw) is not bytes
        or type(fresh_raw) is not bytes
        or type(historical_entitlement_cas) is not bool
        or original_auth.authorization_evidence_sha256 != hashlib.sha256(original_raw).hexdigest()
        or fresh_auth.authorization_evidence_sha256 != hashlib.sha256(fresh_raw).hexdigest()
    ):
        raise AtomicRootProofCommitConflict("EXACT_ORIGINAL_AUTHORIZATION_EVIDENCE_REQUIRED")
    original = parse_canonical(original_raw)
    fresh = parse_canonical(fresh_raw)
    expected_version = "RootProofOperationScopedAuthorizationEvidenceV1"
    if (
        original.get("schema_version") != expected_version
        or fresh.get("schema_version") != expected_version
        or original.get("scope") != "EXACT_OPERATION_SCOPED"
        or fresh.get("scope") != "EXACT_OPERATION_SCOPED"
    ):
        raise AtomicRootProofCommitConflict("EXACT_OPERATION_AUTHORIZATION_EVIDENCE_REQUIRED")
    original_capabilities = original["providers"]["entitlement"]["capabilities"]
    fresh_capabilities = fresh["providers"]["entitlement"]["capabilities"]
    if (
        original_capabilities["compare_and_swap"] is not historical_entitlement_cas
        or fresh_capabilities["compare_and_swap"] is not False
    ):
        raise AtomicRootProofCommitConflict("EXACT_HISTORICAL_ENTITLEMENT_GRANT_REQUIRED")
    original_capabilities["compare_and_swap"] = False
    original_authorization = asdict(original_auth)
    fresh_authorization = asdict(fresh_auth)
    original_authorization.pop("authorization_evidence_sha256")
    fresh_authorization.pop("authorization_evidence_sha256")
    if canonical_json_bytes(original) != fresh_raw or canonical_json_bytes(
        original_authorization
    ) != canonical_json_bytes(fresh_authorization):
        raise AtomicRootProofCommitConflict("MIGRATED_ORIGINAL_AUTHORIZATION_FACTS_CHANGED")


def _read_original_attempt(
    snapshot: signed_boundary._SignedSnapshot, auth: AttemptAuthorization
) -> tuple[CurrentAttempt, RetainedIssuanceRequest]:
    """Read the original SQLite authority without schema or state mutation."""
    from bot_core.cha_attempt_store import SQLiteCHAAttemptStore
    from deployment.windows_production_cha_account_reservation import _safe

    path = reservation_boundary._attempt_store_path()
    if path != snapshot.path or not path.is_absolute():
        raise AtomicRootProofCommitConflict("EXACT_ORIGINAL_ATTEMPT_STORE_OWNER_REQUIRED")
    _safe(path.parent, directory=True)
    for target in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
        _safe(target)
    with SQLiteCHAAttemptStore.open_readonly(path, auth.trust_domain) as store:
        current = store.attempt(auth.logical_operation_id)
        request = store.signed_request(auth.logical_operation_id)
        repeated = store.attempt(auth.logical_operation_id)
    if (
        current != snapshot.current
        or repeated != current
        or current.reservation.authorization != auth
        or current.state is not AttemptState.SIGNED_IMMUTABLE_DURABLE_NOT_SENT
        or path != reservation_boundary._attempt_store_path()
    ):
        raise AtomicRootProofCommitConflict("EXACT_ORIGINAL_SIGNED_ATTEMPT_REQUIRED")
    return current, request


def requalify_original_history_successor_evidence(
    original: PreparedAtomicRootProofCommit,
    binding: object,
    signed_attempt: object,
    *,
    issuer_authority: PostgreSQLRootProofIssuanceAuthorityV2,
) -> RequalifiedOriginalHistorySuccessorEvidence:
    """Reverify historical signed bytes with a current genuine V2 authority.

    The internal consumer must first qualify its explicit successor-reproposal
    deployment and must separately reread RESERVED/STAGED through that same
    deployment's qualified V2 ports. This result cannot register a preparation,
    authorize history signing, replace a request, or resolve an absent decision.
    """
    trusted = _trusted_preparation(original)
    if type(issuer_authority) is not PostgreSQLRootProofIssuanceAuthorityV2:
        raise AtomicRootProofCommitConflict("EXACT_MIGRATED_PUBLIC_AUTHORITY_REQUIRED")
    if (
        type(signed_attempt) is not signed_boundary.VerifiedSignedImmutableRootProofIssuanceAttempt
        or signed_attempt not in signed_boundary._SIGNED
    ):
        raise AtomicRootProofCommitConflict("GENUINE_ORIGINAL_SIGNED_ATTEMPT_REQUIRED")
    signed = signed_boundary._SIGNED[signed_attempt]
    if (
        type(signed.authorization)
        is not reservation_boundary.VerifiedRootProofIssuanceAuthorization
        or signed.authorization not in reservation_boundary._AUTHORIZATIONS
    ):
        raise AtomicRootProofCommitConflict("GENUINE_ORIGINAL_AUTHORIZATION_REQUIRED")
    retained = reservation_boundary._AUTHORIZATIONS[signed.authorization]
    if (
        signed.binding is not binding
        or retained.binding is not binding
        or type(retained.provider)
        not in (PostgreSQLRootProofIssuanceAuthority, PostgreSQLRootProofIssuanceAuthorityV2)
        or original.reserved.authorization != retained.authorization
        or original.prepared.authorization != retained.authorization
    ):
        raise AtomicRootProofCommitConflict("EXACT_ORIGINAL_BINDING_AUTHORIZATION_REQUIRED")
    config = issuer_authority.deployment.provisioning
    expected_schemas = (
        config.entitlement_schema,
        config.intent_schema,
        config.retention_schema,
        config.history_schema,
        config.requester_schema,
        config.claimant_schema,
    )
    if tuple(schema for _, schema in trusted.pins) != expected_schemas:
        raise AtomicRootProofCommitConflict("EXACT_ORIGINAL_AUTHORITY_SCHEMAS_REQUIRED")
    # The old qualification catalog is intentionally invalid after migration;
    # only physical identity is read through those already genuine retained pins.
    # Every actual public authority read below uses its live V2 qualifier.
    for connection, _ in trusted.pins:
        with psycopg.connect(connection.dsn, autocommit=True) as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
            conn.execute("SET LOCAL search_path=pg_catalog")
            if (
                read_atomic_root_proof_physical_identity(conn, config.instance_id)
                != issuer_authority.deployment.identity
            ):
                raise AtomicRootProofCommitConflict("ONE_ORIGINAL_PHYSICAL_AUTHORITY_REQUIRED")
    state_raw, context = reservation_boundary._binding_context(binding)
    if state_raw != retained.state_raw:
        raise AtomicRootProofCommitConflict("EXACT_ORIGINAL_INITIAL_BINDING_BYTES_REQUIRED")
    raw, fresh_auth, fresh_evidence = reservation_boundary._resolve_from_provider(
        binding, issuer_authority
    )
    if raw != state_raw:
        raise AtomicRootProofCommitConflict("INITIAL_BINDING_CHANGED_DURING_REQUALIFICATION")
    _same_original_authorization_evidence(
        retained.authorization,
        retained.evidence_raw,
        fresh_auth,
        fresh_evidence,
        historical_entitlement_cas=type(retained.provider) is PostgreSQLRootProofIssuanceAuthority,
    )
    source = reservation_boundary._AuthorizationSnapshot(
        retained.binding,
        issuer_authority,
        retained.state_raw,
        retained.authorization,
        retained.evidence_raw,
    )
    qualified, _, identities = signed_boundary._qualified_pair_from_snapshot(source)
    if qualified is not source:
        raise AtomicRootProofCommitConflict("EXACT_REQUALIFIED_PRIVATE_CUSTODY_REQUIRED")
    current, request = _read_original_attempt(signed, source.authorization)
    if (
        request != original.reserved.retained_request
        or request != original.prepared.retained_request
    ):
        raise AtomicRootProofCommitConflict("EXACT_ORIGINAL_RETAINED_REQUEST_BYTES_REQUIRED")
    signed_boundary._retained_pair_matches(request, identities)
    registry_revision, requester_revision, claimant_revision = _verify_request(
        source, current, request
    )
    result = issuer_authority.entitlement_registry.authoritative_state(
        issuer_authority._entitlement_subject
    )
    validate_exact_snapshot(result)
    predecessor = result.state
    if (
        result.outcome is not RegistryReadOutcome.FOUND
        or type(predecessor) is not AuthoritativeEntitlementState
        or predecessor.lifecycle is not EntitlementLifecycle.ACTIVE
        or type(predecessor.binding) is not UnboundBinding
        or predecessor.authoritative_state_revision != registry_revision
        or predecessor.authoritative_state_revision != original.bound_state.predecessor_revision
        or canonical_json_bytes(asdict(predecessor))
        != canonical_json_bytes(parse_canonical(retained.evidence_raw)["entitlement"])
        or original.requester_generation is None
        or original.claimant_generation is None
        or requester_revision != original.requester_generation.registry_revision
        or claimant_revision != original.claimant_generation.registry_revision
        or issuer_authority.requester_registry._current(source.authorization.requester_principal_id)
        != original.requester_generation
        or issuer_authority.claimant_registry._current(
            source.authorization.provisioning_principal_id
        )
        != original.claimant_generation
    ):
        raise AtomicRootProofCommitConflict("EXACT_ORIGINAL_PUBLIC_PREDECESSORS_REQUIRED")
    if reservation_boundary._resolve_from_provider(binding, issuer_authority) != (
        raw,
        fresh_auth,
        fresh_evidence,
    ) or reservation_boundary._binding_context(binding) != (state_raw, context):
        raise AtomicRootProofCommitConflict("ORIGINAL_AUTHORIZATION_CHANGED_DURING_REQUALIFICATION")
    if _read_original_attempt(signed, source.authorization) != (current, request):
        raise AtomicRootProofCommitConflict("ORIGINAL_ATTEMPT_CHANGED_DURING_REQUALIFICATION")
    signed_boundary._qualified_pair_from_snapshot(source)
    _verify_request(source, current, request)
    _trusted_preparation(original)
    return RequalifiedOriginalHistorySuccessorEvidence(source, current, request, predecessor)
