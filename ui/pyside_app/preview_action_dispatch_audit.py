"""Static-local BLOK D paper action dispatch audit envelope.

This module records deterministic audit evidence for a requested paper runtime
UI action intention.  It composes the action dispatch contract only; it does not
import PySide/QML, wire handlers, start runtime loops, dispatch commands,
execute lifecycle commands, generate or submit orders, read accounts or secrets,
fetch live/testnet data, export files, or access cloud paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from ui.pyside_app.preview_action_dispatch_contract import (
    SCHEMA_VERSION as DISPATCH_CONTRACT_SCHEMA_VERSION,
    FrozenMapping,
    PaperRuntimeActionDispatchEvidence,
    build_paper_runtime_action_dispatch_contract,
)

AUDIT_ENVELOPE_SCHEMA_VERSION: Final[str] = "paper_runtime_action_dispatch_audit_envelope.v1"
AUDIT_ENVELOPE_KIND: Final[str] = "block_d_paper_runtime_action_dispatch_audit_envelope"
ACCEPTED_INTENT_NOT_EXECUTED: Final[str] = "accepted_intent_not_executed"
REJECTED_INTENT: Final[str] = "rejected_intent"
_ACCEPTED_AUDIT_MESSAGE: Final[str] = (
    "Paper runtime action intent accepted for future UI binding only; "
    "execution is not allowed and no action was executed."
)
_REJECTED_AUDIT_MESSAGE_PREFIX: Final[str] = (
    "Paper runtime action intent rejected fail-closed; execution is not allowed "
    "and no action was executed."
)


@dataclass(frozen=True, slots=True)
class PaperRuntimeActionDispatchAuditEnvelope:
    """Immutable audit envelope for a paper runtime action dispatch intention."""

    schema_version: str
    envelope_kind: str
    dispatch_contract_schema_version: str
    requested_action: object
    normalized_action: str
    source_panel: str
    source_control: str
    operator_confirmation: bool
    operator_note: str
    classification: str
    safe_to_bind_from_ui: bool
    execution_allowed: bool
    execution_performed: bool
    runtime_mode: str
    paper_only: bool
    local_only: bool
    blocked_reason: str
    refusal_reason: str
    audit_status: str
    audit_message: str
    boundary_checks: Mapping[str, bool]
    dispatch_evidence: PaperRuntimeActionDispatchEvidence


def build_paper_runtime_action_dispatch_audit_envelope(
    requested_action_or_evidence: object,
    *,
    source_panel: object = "",
    source_control: object = "",
    operator_confirmation: bool = False,
    operator_note: object = "",
    reason: object = "",
) -> PaperRuntimeActionDispatchAuditEnvelope:
    """Build deterministic audit evidence for a requested action without execution."""

    dispatch_evidence = _coerce_dispatch_evidence(requested_action_or_evidence)
    normalized_operator_note = _normalize_text(operator_note) or _normalize_text(reason)
    classification = (
        ACCEPTED_INTENT_NOT_EXECUTED if dispatch_evidence.safe_to_bind_from_ui else REJECTED_INTENT
    )
    audit_message = _audit_message(dispatch_evidence, classification)

    return PaperRuntimeActionDispatchAuditEnvelope(
        schema_version=AUDIT_ENVELOPE_SCHEMA_VERSION,
        envelope_kind=AUDIT_ENVELOPE_KIND,
        dispatch_contract_schema_version=dispatch_evidence.schema_version,
        requested_action=dispatch_evidence.requested_action,
        normalized_action=dispatch_evidence.normalized_action,
        source_panel=_normalize_text(source_panel),
        source_control=_normalize_text(source_control),
        operator_confirmation=bool(operator_confirmation),
        operator_note=normalized_operator_note,
        classification=classification,
        safe_to_bind_from_ui=dispatch_evidence.safe_to_bind_from_ui,
        execution_allowed=False,
        execution_performed=False,
        runtime_mode=dispatch_evidence.runtime_mode,
        paper_only=dispatch_evidence.paper_only,
        local_only=dispatch_evidence.local_only,
        blocked_reason=dispatch_evidence.blocked_reason,
        refusal_reason=dispatch_evidence.refusal_reason,
        audit_status=classification,
        audit_message=audit_message,
        boundary_checks=FrozenMapping(dict(dispatch_evidence.boundary_checks)),
        dispatch_evidence=dispatch_evidence,
    )


def _coerce_dispatch_evidence(
    requested_action_or_evidence: object,
) -> PaperRuntimeActionDispatchEvidence:
    if isinstance(requested_action_or_evidence, PaperRuntimeActionDispatchEvidence):
        return requested_action_or_evidence
    return build_paper_runtime_action_dispatch_contract(requested_action_or_evidence)


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _audit_message(
    dispatch_evidence: PaperRuntimeActionDispatchEvidence,
    classification: str,
) -> str:
    if classification == ACCEPTED_INTENT_NOT_EXECUTED:
        return _ACCEPTED_AUDIT_MESSAGE
    reason = dispatch_evidence.refusal_reason or "fail_closed"
    return f"{_REJECTED_AUDIT_MESSAGE_PREFIX} refusal_reason={reason}"


__all__ = [
    "ACCEPTED_INTENT_NOT_EXECUTED",
    "AUDIT_ENVELOPE_KIND",
    "AUDIT_ENVELOPE_SCHEMA_VERSION",
    "PaperRuntimeActionDispatchAuditEnvelope",
    "REJECTED_INTENT",
    "build_paper_runtime_action_dispatch_audit_envelope",
]
