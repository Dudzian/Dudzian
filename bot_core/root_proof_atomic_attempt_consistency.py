"""Read-only comparison of genuine RESERVED and historically verified STAGED.

This internal diagnostic never qualifies atomic issuance. Independent, repeated
reads cannot prove a global transaction, BIND, history append, or issued proof.
The caller supplies genuine binding/attempt locators; authorization and the
signed INITIAL_BINDING request are reread through the pinned issuer authority.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import Literal, cast

from bot_core.authenticated_issuer_history import ReconciliationOutcome
from bot_core.cha_attempt_signatures import RetainedIssuanceRequest
from bot_core.cha_attempt_store import AttemptAuthorization
from bot_core.local_signing_custody import LocalRootProofSigningProvider, SigningKeyLifecycle
from bot_core.postgresql_root_proof_retention import (
    ROOT_PROOF_SIGNING_DOMAIN,
    PostgreSQLRootProofRetention,
    PreparedRootProof,
    RootProofRetentionConflict,
    RootProofRetentionCorrupt,
    RootProofRetentionIndeterminate,
    RootProofRetentionScope,
    RootProofRetentionUnavailable,
    _envelope as _prepared_envelope,
    _exact_inputs,
    _validate_payload,
    _validate_snapshot,
)
from bot_core.postgresql_root_proof_signing_intent import (
    PostgreSQLRootProofSigningIntent,
    RootProofSigningIntent,
    RootProofSigningIntentConflict,
    RootProofSigningIntentCorrupt,
    RootProofSigningIntentIndeterminate,
    RootProofSigningIntentUnavailable,
    _envelope as _intent_envelope,
    _payload as _intent_payload,
)
from bot_core.root_proof_issuance_reconciliation import _failure

_FROZEN_SIGNING_DOMAIN = b"CRYPTOHUNTER_ACCOUNT_GENESIS_ROOT_PROOF_V1\x00"
_INTENT_PROVIDER_PREFIX = "_PostgreSQLRootProofSigningIntent__"
_RETENTION_PROVIDER_PREFIX = "_PostgreSQLRootProofRetention__"
_BLOCKED_OPERATIONS = (
    "ROOT_PROOF_SIGN",
    "REGISTRY_BIND",
    "ISSUER_HISTORY_APPEND",
    "CHECKPOINT_ADVANCE",
    "CHA_FINALIZE",
    "ATTEMPT_REPLACEMENT",
    "ISSUER_TRANSPORT",
    "ACCOUNT_GENESIS_MUTATION",
)


class RootProofAtomicAttemptConsistencyOutcome(str, Enum):
    """Observed consistency only; EXACT_MATCH never means committed issuance."""

    RESERVED_WITHOUT_STAGED = "RESERVED_WITHOUT_STAGED"
    EXACT_MATCH = "EXACT_MATCH"
    NOT_FOUND = "NOT_FOUND"
    CONFLICT = "CONFLICT"
    CORRUPT = "CORRUPT"
    UNAVAILABLE = "UNAVAILABLE"
    INDETERMINATE = "INDETERMINATE"


@dataclass(frozen=True, slots=True)
class RootProofAtomicAttemptConsistencyObservation:
    reserved_root_proof_id: str | None
    staged_root_proof_id: str | None
    reserved_envelope_sha256: str | None
    reserved_payload_sha256: str | None
    staged_envelope_sha256: str | None
    staged_payload_sha256: str | None
    staged_signature_sha256: str | None


@dataclass(frozen=True, slots=True)
class RootProofAtomicAttemptConsistencyReport:
    """Detached observation without proof bytes, execution grants, or capabilities."""

    outcome: RootProofAtomicAttemptConsistencyOutcome
    root_proof_id: str | None
    observations: tuple[RootProofAtomicAttemptConsistencyObservation, ...]
    reasons: tuple[str, ...]
    disposition: Literal["NOT_READY"] = "NOT_READY"
    blocked_operations: tuple[str, ...] = _BLOCKED_OPERATIONS
    root_proof_issuance_available: Literal[False] = False
    authorizes_operations: Literal[False] = False
    atomic_global_snapshot: Literal[False] = False

    def __post_init__(self) -> None:
        if (
            self.disposition != "NOT_READY"
            or self.root_proof_issuance_available is not False
            or self.authorizes_operations is not False
            or self.atomic_global_snapshot is not False
            or self.blocked_operations != _BLOCKED_OPERATIONS
        ):
            raise TypeError("CONSISTENCY_OBSERVATION_NEVER_AUTHORIZES_OPERATIONS")


@dataclass(frozen=True, slots=True)
class _Sample:
    authorization: AttemptAuthorization
    retained_request: RetainedIssuanceRequest
    intent: RootProofSigningIntent | None
    prepared: PreparedRootProof | None
    observation: RootProofAtomicAttemptConsistencyObservation


def _cut(name: str) -> None:
    """Test interruption point, without persistence or authorization effects."""


def _intent_scope(signing_intent: PostgreSQLRootProofSigningIntent) -> RootProofRetentionScope:
    scope: object = getattr(signing_intent, _INTENT_PROVIDER_PREFIX + "scope")
    if type(scope) is not RootProofRetentionScope:
        raise RootProofSigningIntentConflict("EXACT_PINNED_SIGNING_INTENT_SCOPE_REQUIRED")
    return cast(RootProofRetentionScope, scope)


def _validate_providers(
    signing_intent: PostgreSQLRootProofSigningIntent, retention: PostgreSQLRootProofRetention
) -> None:
    if (
        type(signing_intent) is not PostgreSQLRootProofSigningIntent
        or type(retention) is not PostgreSQLRootProofRetention
    ):
        raise RootProofSigningIntentConflict("EXACT_PINNED_CONSISTENCY_PROVIDERS_REQUIRED")
    # These are the existing, immutable provider pins, not caller assertions of
    # origin. Equal public snapshots cannot substitute another custody authority.
    signer: object = getattr(signing_intent, _INTENT_PROVIDER_PREFIX + "signer")
    retained_signer: object = getattr(
        retention, _RETENTION_PROVIDER_PREFIX + "verification_authority"
    )
    scope = _intent_scope(signing_intent)
    retained_scope: object = getattr(retention, _RETENTION_PROVIDER_PREFIX + "scope")
    if (
        type(signer) is not LocalRootProofSigningProvider
        or signer is not retained_signer
        or type(retained_scope) is not RootProofRetentionScope
        or scope != retained_scope
    ):
        raise RootProofSigningIntentConflict("EXACT_SHARED_SCOPE_AND_CUSTODY_REQUIRED")
    if (
        type(ROOT_PROOF_SIGNING_DOMAIN) is not bytes
        or ROOT_PROOF_SIGNING_DOMAIN != _FROZEN_SIGNING_DOMAIN
    ):
        raise RootProofSigningIntentCorrupt("EXACT_FROZEN_SIGNING_DOMAIN_REQUIRED")


def _validate_records(
    sample_auth: AttemptAuthorization,
    sample_request: RetainedIssuanceRequest,
    intent: RootProofSigningIntent | None,
    prepared: PreparedRootProof | None,
    signing_intent: PostgreSQLRootProofSigningIntent,
) -> None:
    scope = _intent_scope(signing_intent)
    records: tuple[
        tuple[
            RootProofSigningIntent | PreparedRootProof | None,
            type[RootProofSigningIntent] | type[PreparedRootProof],
        ],
        ...,
    ] = ((intent, RootProofSigningIntent), (prepared, PreparedRootProof))
    for record, expected_type in records:
        if record is None:
            continue
        if (
            type(record) is not expected_type
            or record.authorizes_operations is not False
            or type(record.state) is not str
            or type(record.canonical_envelope) is not bytes
            or type(record.canonical_payload) is not bytes
        ):
            raise RootProofRetentionCorrupt("EXACT_NONAUTHORIZING_RETAINED_RECORD_REQUIRED")
        if (
            record.scope != scope
            or record.authorization != sample_auth
            or record.retained_request != sample_request
        ):
            raise RootProofRetentionConflict("EXACT_RETAINED_REQUEST_AND_AUTHORIZATION_REQUIRED")
        _validate_snapshot(record.credential_snapshot, scope)
        if record.credential_snapshot.lifecycle_state is not SigningKeyLifecycle.ACTIVE:
            raise RootProofRetentionCorrupt("ORIGINAL_ACTIVE_CREDENTIAL_SNAPSHOT_REQUIRED")
    if intent is not None:
        if (
            intent.state != "RESERVED"
            or intent.canonical_payload
            != _intent_payload(
                scope, sample_auth, sample_request, intent.credential_snapshot, intent.root_proof_id
            )
            or intent.canonical_envelope
            != _intent_envelope(
                scope,
                sample_auth,
                sample_request,
                intent.credential_snapshot,
                intent.root_proof_id,
                intent.canonical_payload,
            )
            or intent.bytes_to_sign != _FROZEN_SIGNING_DOMAIN + intent.canonical_payload
        ):
            raise RootProofSigningIntentCorrupt("EXACT_ORIGINAL_RESERVED_BYTES_REQUIRED")
    if prepared is not None:
        fields = _validate_payload(
            prepared.canonical_payload,
            prepared.signature,
            scope,
            sample_auth,
            sample_request,
            prepared.credential_snapshot,
        )
        if (
            prepared.state != "STAGED"
            or prepared.root_proof_id != fields["root_proof_id"]
            or prepared.canonical_envelope
            != _prepared_envelope(
                scope,
                sample_auth,
                sample_request,
                prepared.canonical_payload,
                prepared.signature,
                prepared.credential_snapshot,
                fields,
            )
        ):
            raise RootProofRetentionCorrupt("EXACT_ORIGINAL_STAGED_BYTES_REQUIRED")


def _capture(
    signing_intent: PostgreSQLRootProofSigningIntent,
    retention: PostgreSQLRootProofRetention,
    binding: object,
    signed_attempt: object,
) -> _Sample:
    # _inputs rereads the genuine signed attempt, InitialBinding and public
    # authority; no detached authorization DTO is accepted by this interface.
    auth, request = signing_intent._inputs(binding, signed_attempt)
    auth, request = _exact_inputs(_intent_scope(signing_intent), auth, request)
    intent = signing_intent.read_intent(binding, signed_attempt)
    # The existing reader performs both Ed25519 verification and the pinned
    # custody verify_historical check. A local public-key check cannot replace it.
    prepared = retention.read_prepared(authorization=auth, retained_request=request)
    _validate_records(auth, request, intent, prepared, signing_intent)

    def digest(value: bytes) -> str:
        return hashlib.sha256(value).hexdigest()

    observation = RootProofAtomicAttemptConsistencyObservation(
        None if intent is None else intent.root_proof_id,
        None if prepared is None else prepared.root_proof_id,
        None if intent is None else digest(intent.canonical_envelope),
        None if intent is None else digest(intent.canonical_payload),
        None if prepared is None else digest(prepared.canonical_envelope),
        None if prepared is None else digest(prepared.canonical_payload),
        None if prepared is None else digest(prepared.signature),
    )
    return _Sample(auth, request, intent, prepared, observation)


def _error_outcome(error: Exception) -> RootProofAtomicAttemptConsistencyOutcome:
    outcome = RootProofAtomicAttemptConsistencyOutcome
    if isinstance(error, (RootProofSigningIntentConflict, RootProofRetentionConflict)):
        return outcome.CONFLICT
    if isinstance(error, (RootProofSigningIntentCorrupt, RootProofRetentionCorrupt)):
        return outcome.CORRUPT
    if isinstance(error, (RootProofSigningIntentUnavailable, RootProofRetentionUnavailable)):
        return outcome.UNAVAILABLE
    if isinstance(error, (RootProofSigningIntentIndeterminate, RootProofRetentionIndeterminate)):
        return outcome.INDETERMINATE
    failure = _failure("ATOMIC_ATTEMPT_CONSISTENCY", error)
    if failure.outcome in {ReconciliationOutcome.CONFLICT, ReconciliationOutcome.CORRUPT}:
        return outcome(failure.outcome.value)
    # An unclassified refusal does not prove absence, corruption, or an outage.
    return outcome.INDETERMINATE


def validate_root_proof_atomic_attempt_consistency(
    signing_intent: PostgreSQLRootProofSigningIntent,
    retention: PostgreSQLRootProofRetention,
    *,
    binding: object,
    signed_attempt: object,
) -> RootProofAtomicAttemptConsistencyReport:
    """Compare trusted RESERVED/STAGED twice, always returning NOT_READY.

    read_intent preserves its exact ACTIVE lifecycle requirement. In particular,
    historical recovery of STAGED after VERIFY_ONLY does not relax RESERVED
    admission or establish atomic issuance. No write/sign/replacement is called.
    """
    outcome = RootProofAtomicAttemptConsistencyOutcome
    observations: tuple[RootProofAtomicAttemptConsistencyObservation, ...] = ()
    try:
        _validate_providers(signing_intent, retention)
        first = _capture(signing_intent, retention, binding, signed_attempt)
        observations = (first.observation,)
        _cut("FIRST_CONSISTENCY_OBSERVATION_READ")
        _validate_providers(signing_intent, retention)
        repeated = _capture(signing_intent, retention, binding, signed_attempt)
        observations += (repeated.observation,)
        if first != repeated:
            return RootProofAtomicAttemptConsistencyReport(
                outcome.INDETERMINATE, None, observations, ("AUTHORITATIVE_OBSERVATIONS_CHANGED",)
            )
        intent, prepared = repeated.intent, repeated.prepared
        if intent is None:
            selected = outcome.NOT_FOUND if prepared is None else outcome.CONFLICT
            reason = (
                "RESERVED_AND_STAGED_NOT_FOUND" if prepared is None else "STAGED_WITHOUT_RESERVED"
            )
            return RootProofAtomicAttemptConsistencyReport(selected, None, observations, (reason,))
        if prepared is None:
            return RootProofAtomicAttemptConsistencyReport(
                outcome.RESERVED_WITHOUT_STAGED,
                intent.root_proof_id,
                observations,
                ("RESERVED_IS_NOT_SIGNED_STAGED_OR_ISSUED",),
            )
        if (
            intent.root_proof_id != prepared.root_proof_id
            or intent.credential_snapshot != prepared.credential_snapshot
            or intent.canonical_payload != prepared.canonical_payload
        ):
            return RootProofAtomicAttemptConsistencyReport(
                outcome.CONFLICT,
                None,
                observations,
                ("RESERVED_STAGED_IDENTITY_OR_BYTES_CONFLICT",),
            )
        return RootProofAtomicAttemptConsistencyReport(
            outcome.EXACT_MATCH,
            intent.root_proof_id,
            observations,
            ("EXACT_RESERVED_STAGED_MATCH_IS_NOT_ATOMIC_ISSUANCE",),
        )
    except Exception as error:
        selected = _error_outcome(error)
        if observations and selected is outcome.UNAVAILABLE:
            selected = outcome.INDETERMINATE
        return RootProofAtomicAttemptConsistencyReport(
            selected,
            None,
            observations,
            ("TRUSTED_CONSISTENCY_READ_REFUSED",),
        )
