"""Pierwsza granica orkiestracji Stage 9: admission i odzyskiwanie bez zapisów.

Nie istnieje jeszcze wybrany autorytet atomowego BIND wraz z identycznym
podpisanym Root Proof i historią. Dlatego każda ścieżka kończy się odmową
wykonania mutacji. Żaden raport ani pomyślny odczyt nie staje się capability.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Literal

import bot_core.root_proof_issuance_reconciliation as reconciliation
from bot_core.authenticated_issuer_history import ReconciliationOutcome
from bot_core.licensing.cha_account_reservation import _initial_binding_snapshot
from bot_core.licensing.cha_root_proof_attempt_reservation import (
    _AuthorizationSnapshot,
    _resolve_from_provider,
)
from bot_core.root_proof_issuer_recovery import (
    RootProofRecoveryAssessment,
    RootProofRecoveryPoint,
    assess_root_proof_recovery,
)
from bot_core.root_proof_issuer_runtime import RootProofIssuerRuntimeError, _verify_request


class RootProofAdmissionDecision(str, Enum):
    """Bieżące warunki wstępne, bez uprawnienia do wykonania operacji."""

    NOT_EVALUATED = "NOT_EVALUATED"
    LIVE_ELIGIBLE_EXECUTION_BLOCKED = "LIVE_ELIGIBLE_EXECUTION_BLOCKED"
    REFUSED = "REFUSED"


class _LiveAdmissionIdentityError(RuntimeError):
    """Wykazana różnica tożsamości, odrębna od niepełnego odczytu autorytetu."""


# Blokady są właściwością tej implementacji, a nie flagami od callerów.
_EXECUTION_BLOCKERS = (
    "ATOMIC_SIGNED_PROOF_BIND_HISTORY_AUTHORITY_UNAVAILABLE",
    "BYTE_IDENTICAL_ROOT_PROOF_RECOVERY_UNAVAILABLE",
    "SEMANTIC_ISSUER_SIGNING_COMPOSITION_UNAVAILABLE",
)
_BLOCKED_OPERATIONS = (
    "ROOT_PROOF_SIGN",
    "REGISTRY_BIND",
    "ISSUER_HISTORY_APPEND",
    "CHECKPOINT_ADVANCE",
    "CHA_FINALIZE",
    "ATTEMPT_REPLACEMENT",
    "ISSUER_TRANSPORT",
)


@dataclass(frozen=True, slots=True)
class RootProofIssuanceOrchestrationResult:
    """Odłączony wynik ograniczonej iteracji; nigdy sukces emisji ani grant."""

    disposition: Literal["EXECUTION_BLOCKED", "REFUSED"]
    outcome: ReconciliationOutcome
    admission: RootProofAdmissionDecision
    recovery: RootProofRecoveryAssessment
    observations: tuple[reconciliation.ReconciliationObservation, ...]
    reasons: tuple[str, ...]
    blocked_operations: tuple[str, ...] = _BLOCKED_OPERATIONS
    root_proof_issuance_available: Literal[False] = False
    authorizes_operations: Literal[False] = False
    atomic_global_snapshot: Literal[False] = False


def _cut(name: str) -> None:
    """Punkt testowego przerwania; nie zmienia decyzji autorytetów."""


def _unresolved(
    outcome: ReconciliationOutcome, reasons: tuple[str, ...]
) -> RootProofRecoveryAssessment:
    return RootProofRecoveryAssessment(
        recovery_point=RootProofRecoveryPoint.UNRESOLVED,
        outcome=outcome,
        reasons=reasons,
        root_proof_id=None,
        registry_binding_observed=False,
        history_record_observed=False,
        checkpoint_acceptance_observed=False,
        predecessor_proven=False,
    )


def _refused(
    outcome: ReconciliationOutcome,
    reasons: tuple[str, ...],
    observations: tuple[reconciliation.ReconciliationObservation, ...] = (),
    *,
    recovery: RootProofRecoveryAssessment | None = None,
    admission: RootProofAdmissionDecision = RootProofAdmissionDecision.NOT_EVALUATED,
) -> RootProofIssuanceOrchestrationResult:
    return RootProofIssuanceOrchestrationResult(
        "REFUSED",
        outcome,
        admission,
        recovery if recovery is not None else _unresolved(outcome, reasons),
        observations,
        reasons + _EXECUTION_BLOCKERS,
    )


def _same_runtime(
    original: reconciliation._ReconciliationRuntime,
    selected: reconciliation._ReconciliationRuntime,
) -> bool:
    reconciliation._validate_runtime(selected)
    return all(
        getattr(original, name) is getattr(selected, name)
        for name in ("issuer_authority", "history", "checkpoint", "verification_authorities")
    )


def _live_admission(
    binding: object,
    sample: reconciliation._Sample,
    runtime: reconciliation._ReconciliationRuntime,
) -> None:
    """Rozwiąż żywe uprawnienia niezależnie od retained obserwacji.

    To jedynie konieczne warunki przed pierwszą mutacją. Wciąż brak kompletnego
    autorytetu emisji; nie wystawia się capability i nie otwiera writable CHA.
    """
    source = sample.source
    if (
        source is None
        or source.retained is None
        or source.authority is not runtime.issuer_authority
    ):
        raise _LiveAdmissionIdentityError("EXACT_SELECTED_RETAINED_ATTEMPT_REQUIRED")
    raw, authorization, evidence_raw = _resolve_from_provider(binding, runtime.issuer_authority)
    if raw != source.binding_state_raw or authorization != source.authorization:
        raise _LiveAdmissionIdentityError("LIVE_AUTHORIZATION_DIFFERS_FROM_RETAINED_ATTEMPT")
    _verify_request(
        _AuthorizationSnapshot(
            source.binding, runtime.issuer_authority, raw, authorization, evidence_raw
        ),
        source.current,
        source.retained,
    )


def orchestrate_root_proof_issuance(binding: object) -> RootProofIssuanceOrchestrationResult:
    """Przyjmij genuine InitialBinding, rozpoznaj poprzednika i zablokuj zapisy.

    Tożsamość operacji oraz istniejący issuance_attempt_id pochodzą wyłącznie
    z genuine binding i trwałej próby CHA. Caller nie wybiera portów, identyfikatora,
    podpisu ani raportu preflight/reconciliation jako uprawnienia do wznowienia.
    """
    try:
        _initial_binding_snapshot(binding)
    except Exception as error:
        failure = reconciliation._failure("INITIAL_BINDING", error)
        return _refused(failure.outcome, (failure.reason,))
    try:
        runtime = reconciliation._configured_reconciliation_runtime()
        reconciliation._validate_runtime(runtime)
    except Exception:
        return _refused(
            ReconciliationOutcome.UNAVAILABLE, ("PINNED_ISSUER_AUTHORITIES_UNAVAILABLE",)
        )
    return _orchestrate_selected_authorities(binding, runtime)


def _orchestrate_selected_authorities(
    binding: object, runtime: reconciliation._ReconciliationRuntime
) -> RootProofIssuanceOrchestrationResult:
    """Wewnętrzny silnik po genuine guard; żadne alternatywne publiczne wejście."""
    observations: tuple[reconciliation.ReconciliationObservation, ...] = ()
    recovery: RootProofRecoveryAssessment | None = None
    try:
        reconciliation._validate_runtime(runtime)
        first = reconciliation._capture(binding, runtime)
        observations = (first.observation,)
        _cut("RETAINED_STATE_OBSERVED")
        selected = reconciliation._configured_reconciliation_runtime()
        if not _same_runtime(runtime, selected):
            return _refused(
                ReconciliationOutcome.INDETERMINATE,
                ("PINNED_AUTHORITIES_CHANGED",),
                observations,
            )
        repeated = reconciliation._capture(binding, selected)
        observations += (repeated.observation,)
        if first.observation != repeated.observation:
            return _refused(
                ReconciliationOutcome.INDETERMINATE,
                ("AUTHORITATIVE_OBSERVATIONS_CHANGED",)
                + tuple(
                    "CHANGED:" + field.name
                    for field in fields(reconciliation.ReconciliationObservation)
                    if getattr(first.observation, field.name)
                    != getattr(repeated.observation, field.name)
                ),
                observations,
            )
        recovery = assess_root_proof_recovery(repeated, selected)
        if recovery.recovery_point is RootProofRecoveryPoint.UNRESOLVED:
            return _refused(recovery.outcome, recovery.reasons, observations, recovery=recovery)
        # Zachowany wynik ma pierwszeństwo przed nową decyzją admission. Nawet
        # EXACT_COMMITTED opisuje tylko zapisy: nie dowodzi zachowanych proof bytes.
        if recovery.recovery_point is not RootProofRecoveryPoint.BEFORE_FIRST_MUTATION:
            return RootProofIssuanceOrchestrationResult(
                "EXECUTION_BLOCKED",
                recovery.outcome,
                RootProofAdmissionDecision.NOT_EVALUATED,
                recovery,
                observations,
                recovery.reasons + _EXECUTION_BLOCKERS,
            )
        try:
            _live_admission(binding, repeated, selected)
        except Exception as error:
            failure = reconciliation._failure("LIVE_ADMISSION", error)
            outcome = failure.outcome
            if isinstance(error, _LiveAdmissionIdentityError):
                outcome = ReconciliationOutcome.CONFLICT
            elif isinstance(error, RootProofIssuerRuntimeError):
                # Istniejący preflight scala brak, konflikt i awarię odczytu
                # w jeden wyjątek. Nie przypisuj mu dokładniejszego rozstrzygnięcia.
                outcome = ReconciliationOutcome.INDETERMINATE
            return _refused(
                outcome,
                ("LIVE_ISSUANCE_ADMISSION_REFUSED", failure.reason),
                observations,
                recovery=recovery,
                admission=RootProofAdmissionDecision.REFUSED,
            )
        _cut("LIVE_ADMISSION_RECHECKED")
        final_runtime = reconciliation._configured_reconciliation_runtime()
        if not _same_runtime(selected, final_runtime):
            return _refused(
                ReconciliationOutcome.INDETERMINATE,
                ("PINNED_AUTHORITIES_CHANGED_BEFORE_MUTATION",),
                observations,
            )
        final = reconciliation._capture(binding, final_runtime)
        observations += (final.observation,)
        if final.observation != repeated.observation:
            return _refused(
                ReconciliationOutcome.INDETERMINATE,
                ("AUTHORITATIVE_PREDECESSOR_CHANGED_BEFORE_MUTATION",),
                observations,
                admission=RootProofAdmissionDecision.REFUSED,
            )
        _cut("BEFORE_FIRST_MUTATION")
        # Tu kończy się fragment wykonawczy tej iteracji. Nie ma fallthrough do
        # sign/BIND/append/advance, transportu, finalizacji ani nowego attempt_id.
        return RootProofIssuanceOrchestrationResult(
            "EXECUTION_BLOCKED",
            recovery.outcome,
            RootProofAdmissionDecision.LIVE_ELIGIBLE_EXECUTION_BLOCKED,
            recovery,
            observations,
            _EXECUTION_BLOCKERS,
        )
    except Exception as error:
        failure = reconciliation._failure("ISSUER_ORCHESTRATION", error)
        return _refused(
            failure.outcome,
            ("ORCHESTRATION_INTERRUPTED_OR_AUTHORITY_UNAVAILABLE", failure.reason),
            observations,
        )
