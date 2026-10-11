"""Odczytowe punkty odzyskiwania; nigdy uprawnienia do dalszej emisji.

Wyłącznie wewnętrzny kontroler przekazuje próbkę odczytaną przez przypięty
composition root. Zgodność BOUND, historii i checkpointu opisuje wcześniejsze
zapisy. Te magazyny nie zachowują podpisanych bajtów Root Proof, więc nawet
dokładne dopasowanie nie pozwala zakończyć odzyskiwania ani ponowić mutacji.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum

import bot_core.root_proof_issuance_reconciliation as reconciliation
from bot_core.authenticated_issuer_history import (
    NO_PREDECESSOR,
    AttestedHistoryHead,
    HistoryContractError,
    HistoryRecord,
    HistorySigningAuthorityUnavailable,
    ReconciliationOutcome,
    _snapshot_acceptance,
    _snapshot_checkpoint,
    _snapshot_head,
    _snapshot_record,
    verify_head,
)
from bot_core.cha_attempt_signatures import RetainedIssuanceRequest
from bot_core.cha_attempt_store import (
    AttemptState,
    _snapshot_current_attempt,
)
from bot_core.cha_issuance_request import request_bytes, request_reference
from bot_core.entitlement_registry_contract import (
    AuthoritativeEntitlementState,
    BoundBinding,
    EntitlementLifecycle,
    RegistryReadOutcome,
    UnboundBinding,
    validate_exact_snapshot,
)
from bot_core.licensing.cha_root_proof_signed_attempt import _IssuanceReconciliationSource
from bot_core.local_signing_custody import SigningKeyLifecycle


class RootProofRecoveryPoint(str, Enum):
    BEFORE_FIRST_MUTATION = "BEFORE_FIRST_MUTATION"
    UNKNOWN_PRIOR_OUTCOME = "UNKNOWN_PRIOR_OUTCOME"
    REGISTRY_BOUND = "REGISTRY_BOUND"
    HISTORY_RECORDED = "HISTORY_RECORDED"
    CHECKPOINT_ACCEPTED_CHA_PENDING = "CHECKPOINT_ACCEPTED_CHA_PENDING"
    CHA_RECOVERY_RECORDED = "CHA_RECOVERY_RECORDED"
    UNRESOLVED = "UNRESOLVED"


@dataclass(frozen=True, slots=True)
class RootProofRecoveryAssessment:
    recovery_point: RootProofRecoveryPoint
    outcome: ReconciliationOutcome
    reasons: tuple[str, ...]
    root_proof_id: str | None
    registry_binding_observed: bool
    history_record_observed: bool
    checkpoint_acceptance_observed: bool
    predecessor_proven: bool


def _unresolved(
    outcome: ReconciliationOutcome, reasons: tuple[str, ...]
) -> RootProofRecoveryAssessment:
    return RootProofRecoveryAssessment(
        RootProofRecoveryPoint.UNRESOLVED, outcome, reasons, None, False, False, False, False
    )


def _verify_history_and_checkpoint(
    sample: reconciliation._Sample, runtime: reconciliation._ReconciliationRuntime
) -> None:
    """Sprawdź podpisy oraz pełne prefiksy, również dla nieprzyjętego ogona.

    _classify odmawia przy HISTORY_AHEAD_OF_CHECKPOINT zanim sprawdzi payload
    emisji. Ten dodatkowy odczytowy audyt rozróżnia taki punkt przerwania od
    sprzecznego checkpointu; nie udostępnia operacji advance ani append.
    """
    history, checkpoint = sample.history, sample.checkpoint
    if history is None or checkpoint is None:
        raise HistoryContractError("incomplete retained snapshots")
    if (
        history.stream != runtime.history._stream
        or checkpoint.stream != history.stream
        or checkpoint.checkpoint_id != runtime.checkpoint._checkpoint_id
        or checkpoint.authority_identity != runtime.checkpoint._authority_identity
    ):
        raise HistoryContractError("retained authority scope mismatch")
    predecessor = NO_PREDECESSOR
    for sequence, record in enumerate(history.records, 1):
        _snapshot_record(record)
        if (
            record.stream != history.stream
            or record.sequence != sequence
            or record.predecessor_authenticated_digest != predecessor
        ):
            raise HistoryContractError("retained history predecessor mismatch")
        predecessor = record.authenticated_digest
    sequences: set[int] = set()
    for head in history.heads:
        _snapshot_head(head)
        if (
            head.sequence in sequences
            or head.sequence > len(history.records)
            or head.record_digest != history.records[head.sequence - 1].authenticated_digest
        ):
            raise HistoryContractError("retained history attestation mismatch")
        sequences.add(head.sequence)
        authorities = []
        for authority in runtime.verification_authorities:
            try:
                if head.signing_credential_identity in authority.credential_identities():
                    authorities.append(authority)
            except Exception as error:
                raise HistorySigningAuthorityUnavailable("selected verifier unavailable") from error
        if len(authorities) != 1:
            raise HistoryContractError("retained attestation signer is not unique")
        verify_head(
            head,
            authorities[0],
            expected_stream=history.stream,
            historical_evidence_authority=runtime.checkpoint,
        )
    if not (len(checkpoint.checkpoints) == len(checkpoint.acceptances) == len(checkpoint.heads)):
        raise HistoryContractError("checkpoint retained cardinality mismatch")
    predecessor, previous_sequence = NO_PREDECESSOR, 0
    for revision, (current, acceptance, head) in enumerate(
        zip(checkpoint.checkpoints, checkpoint.acceptances, checkpoint.heads, strict=True), 1
    ):
        _snapshot_checkpoint(current)
        _snapshot_acceptance(acceptance)
        _snapshot_head(head)
        retained_heads = tuple(item for item in history.heads if item.sequence == head.sequence)
        if (
            current.checkpoint_id != checkpoint.checkpoint_id
            or current.stream != history.stream
            or current.checkpoint_revision != revision
            or acceptance.checkpoint_authority_identity != checkpoint.authority_identity
            or acceptance.stream != history.stream
            or acceptance.checkpoint_revision != revision
            or acceptance.predecessor_acceptance_digest != predecessor
            or acceptance.lifecycle_state_at_acceptance is not SigningKeyLifecycle.ACTIVE
            or acceptance.accepted_history_sequence <= previous_sequence
            or len(retained_heads) != 1
            or retained_heads[0] != head
            or (
                current.history_sequence,
                current.authenticated_head_digest,
                current.history_signing_credential_identity,
            )
            != (head.sequence, head.record_digest, head.signing_credential_identity)
            or (
                acceptance.accepted_history_sequence,
                acceptance.accepted_authenticated_head_digest,
                acceptance.history_signing_credential_identity,
            )
            != (head.sequence, head.record_digest, head.signing_credential_identity)
        ):
            raise HistoryContractError("exact retained V2 acceptance mismatch")
        predecessor = acceptance.acceptance_digest
        previous_sequence = acceptance.accepted_history_sequence
    floor = runtime.checkpoint._highest_revision
    if len(checkpoint.acceptances) < floor or (
        floor > 0
        and checkpoint.acceptances[floor - 1].acceptance_digest
        != runtime.checkpoint._highest_acceptance_digest
    ):
        raise HistoryContractError("observed checkpoint prefix rolled back or forked")


def _bound_predecessor(
    sample: reconciliation._Sample, binding: BoundBinding
) -> tuple[AuthoritativeEntitlementState, AuthoritativeEntitlementState] | None:
    source, retained = sample.source, sample.registry_history
    if source is None or source.retained is None or retained is None:
        return None
    authorization = source.authorization
    expected = {
        name: getattr(authorization, name)
        for name in (
            "logical_operation_id",
            "account_id",
            "canonical_genesis_request_fingerprint_sha256",
            "entitlement_generation",
            "requester_principal_id",
            "requester_key_id",
            "requester_key_version",
            "provisioning_principal_id",
            "claimant_key_id",
            "claimant_key_version",
        )
    }
    expected.update(
        issuance_attempt_id=source.retained.attempt_id,
        signed_request_payload_digest_sha256=source.retained.digest,
        signed_request_canonical_bytes_reference=source.retained.reference,
    )
    if any(getattr(binding, name) != value for name, value in expected.items()):
        return None
    bound_index = next(
        (
            index
            for index, item in enumerate(retained.states)
            if item.binding == binding
            and item.identity.entitlement_generation == authorization.entitlement_generation
        ),
        None,
    )
    if bound_index is None or bound_index == 0:
        return None
    bound, predecessor = retained.states[bound_index], retained.states[bound_index - 1]
    if (
        type(predecessor.binding) is not UnboundBinding
        or predecessor.lifecycle is not EntitlementLifecycle.ACTIVE
        or bound.lifecycle is not EntitlementLifecycle.ACTIVE
        or bound.subject != predecessor.subject
        or bound.identity != predecessor.identity
        or bound.provenance != predecessor.provenance
        or bound.predecessor_revision != predecessor.authoritative_state_revision
        or bound.authoritative_state_revision != predecessor.authoritative_state_revision + 1
        or not reconciliation._authorization_anchor_matches(predecessor, authorization)
        or not reconciliation._authorization_anchor_matches(bound, authorization)
    ):
        return None
    return bound, predecessor


def _assess(
    sample: reconciliation._Sample, runtime: reconciliation._ReconciliationRuntime
) -> RootProofRecoveryAssessment:
    if type(sample) is not reconciliation._Sample:
        return _unresolved(ReconciliationOutcome.CORRUPT, ("EXACT_RECOVERY_SAMPLE_REQUIRED",))
    reconciliation._validate_runtime(runtime)
    outcome, disposition, reasons = reconciliation._classify(sample, runtime)
    if sample.observation.read_failures:
        return _unresolved(outcome, reasons)
    # Dopuszczone odmowy opisują wyłącznie brak kolejnego zachowanego zapisu.
    # Odmowa lifecycle zachowuje swój wynik; nie jest live admission.
    allowed_reasons = {
        "EXACT_ISSUANCE_HISTORY_NOT_FOUND",
        "EXACT_CHECKPOINT_NOT_FOUND",
        "HISTORY_AHEAD_OF_CHECKPOINT",
        "CHA_SENT_OUTCOME_UNKNOWN",
        "ENTITLEMENT_LIFECYCLE_NOT_ACTIVE",
    }
    if outcome in (
        ReconciliationOutcome.CORRUPT,
        ReconciliationOutcome.CONFLICT,
        ReconciliationOutcome.UNAVAILABLE,
    ) or any(reason not in allowed_reasons for reason in reasons):
        return _unresolved(outcome, reasons)
    source, registry, retained = sample.source, sample.registry, sample.registry_history
    history, checkpoint = sample.history, sample.checkpoint
    if (
        type(source) is not _IssuanceReconciliationSource
        or registry is None
        or retained is None
        or history is None
        or checkpoint is None
        or source.current.identity is None
        or type(source.retained) is not RetainedIssuanceRequest
    ):
        return _unresolved(ReconciliationOutcome.UNAVAILABLE, ("INCOMPLETE_RECOVERY_EVIDENCE",))
    validate_exact_snapshot(registry)
    validate_exact_snapshot(retained)
    current = _snapshot_current_attempt(source.current)
    if (
        source.authority is not runtime.issuer_authority
        or source.authorization != current.reservation.authorization
        or registry.outcome is not RegistryReadOutcome.FOUND
        or retained.outcome is not RegistryReadOutcome.FOUND
        or registry.state is None
        or not retained.states
        or retained.states[-1] != registry.state
    ):
        return _unresolved(ReconciliationOutcome.CONFLICT, ("EXACT_RETAINED_AUTHORITY_REQUIRED",))
    authorization, state = source.authorization, registry.state
    raw = request_bytes(authorization, current.reservation.issuance_attempt_id)
    if (
        source.retained.canonical_bytes != raw
        or source.retained.reference != request_reference(raw)
        or source.retained.digest != hashlib.sha256(raw).hexdigest()
        or not reconciliation._verify_retained_signatures(sample)
    ):
        return _unresolved(ReconciliationOutcome.CORRUPT, ("EXACT_SIGNED_REQUEST_REQUIRED",))
    if (
        state.subject != runtime.issuer_authority._entitlement_subject
        or not reconciliation._authorization_anchor_matches(state, authorization)
        or history.stream.product_scope != authorization.product_scope
    ):
        return _unresolved(ReconciliationOutcome.CONFLICT, ("EXACT_RECOVERY_SCOPE_REQUIRED",))
    _verify_history_and_checkpoint(sample, runtime)
    binding = state.binding
    matching = tuple(
        record
        for record in history.records
        if record.event_identity.logical_operation_id == authorization.logical_operation_id
        or record.event_identity.issuance_attempt_id == source.retained.attempt_id
        or (
            type(binding) is BoundBinding
            and record.event_identity.root_proof_id == binding.root_proof_id
        )
    )
    if type(binding) is UnboundBinding:
        if matching or state.lifecycle is not EntitlementLifecycle.ACTIVE:
            return _unresolved(ReconciliationOutcome.CONFLICT, ("UNBOUND_RECOVERY_CONFLICT",))
        if disposition == "PRE_ISSUANCE_OBSERVED":
            point = RootProofRecoveryPoint.BEFORE_FIRST_MUTATION
        elif current.state is AttemptState.MAY_HAVE_BEEN_SENT_OUTCOME_UNKNOWN and reasons == (
            "CHA_SENT_OUTCOME_UNKNOWN",
        ):
            point = RootProofRecoveryPoint.UNKNOWN_PRIOR_OUTCOME
        else:
            return _unresolved(outcome, reasons)
        return RootProofRecoveryAssessment(
            point, outcome, reasons, None, False, False, False, False
        )
    if type(binding) is not BoundBinding:
        return _unresolved(ReconciliationOutcome.CORRUPT, ("EXACT_BOUND_VARIANT_REQUIRED",))
    lineage = _bound_predecessor(sample, binding)
    if lineage is None:
        return _unresolved(ReconciliationOutcome.CORRUPT, ("EXACT_ACTIVE_UNBOUND_CAS_REQUIRED",))
    blocked = ("ROOT_PROOF_BYTES_NOT_RETAINED", "CHA_FINALIZATION_BLOCKED")
    if not matching:
        if current.state is AttemptState.EXACT_BOUND_RECOVERED:
            return _unresolved(ReconciliationOutcome.CONFLICT, ("CHA_RECOVERY_HISTORY_MISSING",))
        if history.records and (
            checkpoint.current is None
            or checkpoint.current.history_sequence != history.records[-1].sequence
            or checkpoint.current.authenticated_head_digest
            != history.records[-1].authenticated_digest
        ):
            return _unresolved(outcome, ("GLOBAL_HISTORY_CHECKPOINT_NOT_ALIGNED",))
        return RootProofRecoveryAssessment(
            RootProofRecoveryPoint.REGISTRY_BOUND,
            outcome,
            reasons + blocked,
            binding.root_proof_id,
            True,
            False,
            False,
            True,
        )
    if len(matching) != 1:
        return _unresolved(ReconciliationOutcome.CONFLICT, ("ISSUANCE_HISTORY_IDENTITY_CONFLICT",))
    record: HistoryRecord = matching[0]
    bound_state, predecessor = lineage
    expected = reconciliation._expected_issuance_history_payload(
        authorization, bound_state, predecessor, source.current.identity.digest_sha256
    )
    if (
        record.event_identity.logical_operation_id != authorization.logical_operation_id
        or record.event_identity.issuance_attempt_id != source.retained.attempt_id
        or record.event_identity.root_proof_id != binding.root_proof_id
        or reconciliation._plain(record.canonical_event_payload) != reconciliation._plain(expected)
    ):
        return _unresolved(ReconciliationOutcome.CONFLICT, ("EXACT_HISTORY_PAYLOAD_CAS_REQUIRED",))
    heads: tuple[AttestedHistoryHead, ...] = tuple(
        item for item in history.heads if item.sequence == record.sequence
    )
    if len(heads) != 1:
        return _unresolved(
            ReconciliationOutcome.INDETERMINATE, ("EXACT_SIGNED_HISTORY_HEAD_REQUIRED",)
        )
    accepted = tuple(
        item for item in checkpoint.acceptances if item.accepted_history_sequence == record.sequence
    )
    if not accepted:
        if (
            history.records[-1] != record
            or (checkpoint.current.history_sequence if checkpoint.current else 0)
            != record.sequence - 1
            or current.state is AttemptState.EXACT_BOUND_RECOVERED
        ):
            return _unresolved(
                ReconciliationOutcome.CONFLICT, ("CHECKPOINT_HISTORY_RECOVERY_CONFLICT",)
            )
        return RootProofRecoveryAssessment(
            RootProofRecoveryPoint.HISTORY_RECORDED,
            outcome,
            reasons + blocked,
            binding.root_proof_id,
            True,
            True,
            False,
            True,
        )
    # Dokładny wpis przyjęcia nie zastępuje zgodności bieżącego globalnego ogona.
    if (
        len(accepted) != 1
        or checkpoint.current is None
        or checkpoint.current.history_sequence != history.records[-1].sequence
        or checkpoint.current.authenticated_head_digest != history.records[-1].authenticated_digest
        or accepted[0].accepted_authenticated_head_digest != record.authenticated_digest
        or accepted[0].history_signing_credential_identity != heads[0].signing_credential_identity
    ):
        return _unresolved(
            ReconciliationOutcome.CONFLICT, ("EXACT_RETAINED_V2_ACCEPTANCE_REQUIRED",)
        )
    point = (
        RootProofRecoveryPoint.CHA_RECOVERY_RECORDED
        if current.state is AttemptState.EXACT_BOUND_RECOVERED
        else RootProofRecoveryPoint.CHECKPOINT_ACCEPTED_CHA_PENDING
    )
    return RootProofRecoveryAssessment(
        point, outcome, reasons + blocked, binding.root_proof_id, True, True, True, True
    )


def assess_root_proof_recovery(
    sample: reconciliation._Sample, runtime: reconciliation._ReconciliationRuntime
) -> RootProofRecoveryAssessment:
    """Sklasyfikuj zachowane zapisy; nie odtwarzaj proof ani nie zapisuj CHA.

    To API wewnętrzne: genuine guard, zaufana kompozycja i powtórne odczyty
    należą do kontrolera. Wynik jest wyłącznie odłączoną diagnostyką.
    """
    try:
        return _assess(sample, runtime)
    except Exception as error:
        failure = reconciliation._failure("RECOVERY", error)
        return _unresolved(failure.outcome, (failure.reason,))
