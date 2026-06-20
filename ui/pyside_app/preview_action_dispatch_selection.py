"""Static-local BLOK D paper action dispatch selection result contract.

This module models a future UI selecting a paper action or source control from
an immutable catalog.  It composes the catalog, audit envelope, and dispatch
contract only; it does not import PySide/QML, wire handlers, start runtime
loops, dispatch commands, execute lifecycle commands, generate or submit orders,
read accounts or secrets, fetch live/testnet data, export files, or access cloud
paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping

from ui.pyside_app.preview_action_dispatch_audit import (
    ACCEPTED_INTENT_NOT_EXECUTED,
    PaperRuntimeActionDispatchAuditEnvelope,
    build_paper_runtime_action_dispatch_audit_envelope,
)
from ui.pyside_app.preview_action_dispatch_catalog import (
    SOURCE_PANEL as CATALOG_SOURCE_PANEL,
    PaperRuntimeActionCatalogItem,
    PaperRuntimeActionDispatchCatalog,
    build_paper_runtime_action_dispatch_catalog,
)
from ui.pyside_app.preview_action_dispatch_contract import (
    RUNTIME_MODE,
    FrozenMapping,
    PaperRuntimeActionDispatchEvidence,
    build_paper_runtime_action_dispatch_contract,
)

SELECTION_RESULT_SCHEMA_VERSION: Final[str] = "paper_runtime_action_dispatch_selection_result.v1"
SELECTION_RESULT_KIND: Final[str] = "block_d_non_executing_paper_action_selection_result"
UNKNOWN_SELECTION_STATUS: Final[str] = "selection_rejected_fail_closed"
_ACCEPTED_SELECTION_MESSAGE: Final[str] = (
    "Paper runtime action selection accepted for future UI binding only; "
    "execution is not allowed and no action was executed."
)
_REJECTED_SELECTION_MESSAGE_PREFIX: Final[str] = (
    "Paper runtime action selection rejected fail-closed; execution is not allowed "
    "and no action was executed."
)


@dataclass(frozen=True, slots=True)
class PaperRuntimeActionDispatchSelectionResult:
    """Immutable result for a non-executing paper action/control selection."""

    schema_version: str
    result_kind: str
    requested_action_or_control: object
    resolved_action: str
    resolved_source_control: str
    source_panel: str
    source_control: str
    catalog_action_found: bool
    catalog_item: PaperRuntimeActionCatalogItem | None
    audit_envelope: PaperRuntimeActionDispatchAuditEnvelope
    dispatch_evidence: PaperRuntimeActionDispatchEvidence
    safe_to_bind_from_ui: bool
    execution_allowed: bool
    execution_performed: bool
    paper_only: bool
    local_only: bool
    runtime_mode: str
    blocked_reason: str
    refusal_reason: str
    result_status: str
    result_message: str
    boundary_checks: Mapping[str, bool]


def build_paper_runtime_action_dispatch_selection_result(
    requested_action_or_control: object,
    *,
    source_panel: object = "",
    source_control: object = "",
    operator_confirmation: bool = False,
    operator_note: object = "",
    reason: object = "",
    catalog: PaperRuntimeActionDispatchCatalog | None = None,
) -> PaperRuntimeActionDispatchSelectionResult:
    """Resolve a catalog action/control and build a non-executing result."""

    action_catalog = catalog or build_paper_runtime_action_dispatch_catalog()
    action_items = {item.action: item for item in action_catalog.actions}
    control_items = {item.source_control: item for item in action_catalog.actions}
    requested_text = _normalize_text(requested_action_or_control)
    explicit_source_control = _normalize_text(source_control)
    catalog_item = action_items.get(requested_text) or control_items.get(requested_text)
    catalog_action_found = catalog_item is not None
    resolved_action = catalog_item.action if catalog_item else requested_text
    resolved_source_control = catalog_item.source_control if catalog_item else ""
    effective_source_control = (
        resolved_source_control or explicit_source_control or requested_text
        if catalog_action_found
        else explicit_source_control
    )
    effective_source_panel = _normalize_text(source_panel) or (
        catalog_item.source_panel if catalog_item else CATALOG_SOURCE_PANEL
    )

    dispatch_request = (
        resolved_action
        if catalog_action_found or isinstance(requested_action_or_control, str)
        else requested_action_or_control
    )
    dispatch_evidence = build_paper_runtime_action_dispatch_contract(dispatch_request)
    audit_envelope = build_paper_runtime_action_dispatch_audit_envelope(
        dispatch_evidence,
        source_panel=effective_source_panel,
        source_control=effective_source_control,
        operator_confirmation=operator_confirmation,
        operator_note=operator_note,
        reason=reason,
    )
    result_status = (
        ACCEPTED_INTENT_NOT_EXECUTED
        if catalog_action_found and audit_envelope.safe_to_bind_from_ui
        else UNKNOWN_SELECTION_STATUS
    )
    boundary_checks = _build_selection_boundary_checks(
        dispatch_evidence.boundary_checks,
        catalog_action_found=catalog_action_found,
        audit_safe_to_bind=audit_envelope.safe_to_bind_from_ui,
    )
    safe_to_bind = catalog_action_found and audit_envelope.safe_to_bind_from_ui
    refusal_reason = (
        "" if safe_to_bind else audit_envelope.refusal_reason or "blocked_unknown_selection"
    )
    blocked_reason = "" if safe_to_bind else audit_envelope.blocked_reason or refusal_reason

    return PaperRuntimeActionDispatchSelectionResult(
        schema_version=SELECTION_RESULT_SCHEMA_VERSION,
        result_kind=SELECTION_RESULT_KIND,
        requested_action_or_control=requested_action_or_control,
        resolved_action=resolved_action,
        resolved_source_control=resolved_source_control,
        source_panel=effective_source_panel,
        source_control=effective_source_control,
        catalog_action_found=catalog_action_found,
        catalog_item=catalog_item,
        audit_envelope=audit_envelope,
        dispatch_evidence=dispatch_evidence,
        safe_to_bind_from_ui=safe_to_bind,
        execution_allowed=False,
        execution_performed=False,
        paper_only=True,
        local_only=True,
        runtime_mode=RUNTIME_MODE,
        blocked_reason=blocked_reason,
        refusal_reason=refusal_reason,
        result_status=result_status,
        result_message=_result_message(result_status, refusal_reason),
        boundary_checks=boundary_checks,
    )


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip().lower()
    return str(value).strip().lower()


def _build_selection_boundary_checks(
    dispatch_boundary_checks: Mapping[str, bool],
    *,
    catalog_action_found: bool,
    audit_safe_to_bind: bool,
) -> Mapping[str, bool]:
    values = dict(dispatch_boundary_checks)
    values.update(
        {
            "catalog_action_found": catalog_action_found,
            "selection_fail_closed": not (catalog_action_found and audit_safe_to_bind),
            "selection_safe_to_bind_from_ui": catalog_action_found and audit_safe_to_bind,
            "execution_disabled": True,
            "execution_not_performed": True,
        }
    )
    return FrozenMapping(values)


def _result_message(result_status: str, refusal_reason: str) -> str:
    if result_status == ACCEPTED_INTENT_NOT_EXECUTED:
        return _ACCEPTED_SELECTION_MESSAGE
    reason = refusal_reason or "fail_closed"
    return f"{_REJECTED_SELECTION_MESSAGE_PREFIX} refusal_reason={reason}"


__all__ = [
    "SELECTION_RESULT_KIND",
    "SELECTION_RESULT_SCHEMA_VERSION",
    "UNKNOWN_SELECTION_STATUS",
    "PaperRuntimeActionDispatchSelectionResult",
    "build_paper_runtime_action_dispatch_selection_result",
]
