"""Pure-data BLOK G paper order intent selection gate.

FUNCTIONAL-PREVIEW-9.5 selects one existing 9.3 paper order audit event as a
future preview candidate only. The gate never generates an intent/order,
submits an order, simulates fills, starts runtime execution, or exports audit.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_paper_order_audit_envelope import (
    AUDIT_ENVELOPE_DECISION,
    AUDIT_ENVELOPE_STATUS,
    PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_KIND,
    PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_4,
    build_preview_paper_order_audit_envelope,
)

PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_SCHEMA_VERSION: Final[str] = (
    "preview_paper_order_intent_selection_gate.v1"
)
PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_KIND: Final[str] = (
    "functional_preview_block_g_paper_order_intent_selection_gate"
)
BLOCK_ID: Final[str] = "G"
STEP_ID: Final[str] = "9.5"
SELECTION_GATE_STATUS: Final[str] = "paper_order_intent_selection_gate_ready_no_intent_generation"
SELECTION_GATE_DECISION: Final[str] = "BUILD_SELECTION_GATE_ONLY_NO_INTENT_GENERATION"
READY_FOR_BLOCK_G_6: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-9.6"
NEXT_STEP_TITLE: Final[str] = "CONTROLLED PAPER ORDER INTENT NO SUBMISSION"

AVAILABLE_SELECTION_STATUS: Final[str] = "available_for_controlled_preview_selection_only"
NO_SELECTION_STATUS: Final[str] = "no_selection_preview_only"
ACCEPTED_SELECTION_STATUS: Final[str] = "accepted_for_preview_only_no_intent_generation"
REJECTED_SELECTION_STATUS: Final[str] = "rejected_fail_closed_unknown_or_invalid_selection"
REJECTION_REASON: Final[str] = "unknown_or_invalid_selection"

ADDED_BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
    "paper order intent selection enabling execution",
    "selection-driven order intent generation",
    "selection-driven paper order generation",
    "paper order audit export",
    "paper order audit runtime dispatch",
    "paper order intent generation now",
    "paper order generation now",
    "paper order submission now",
    "paper fill simulation now",
    "paper runtime execution now",
    "risk governor execution now",
    "live/testnet/account/secrets/export/cloud",
    "TradingController / DecisionEnvelope",
    "QML changes / new QML calls",
    "EXE packaging",
)

FUTURE_STEPS: Final[tuple[str, ...]] = (
    "functional_preview_9_6_controlled_paper_order_intent_no_submission",
    "functional_preview_9_7_paper_fill_simulator_contract_static_only",
    "functional_preview_9_8_paper_order_lifecycle_audit",
    "functional_preview_9_9_block_g_closure_audit",
)

SELECTION_BOUNDARY_EXTENSIONS: Final[dict[str, bool]] = {
    "selection_gate_only": True,
    "ui_surface_read_only": True,
    "selection_gate_evaluated": True,
    "selection_acceptance_can_enable_intent_generation": False,
}


def _audit_envelope_reference() -> dict[str, Any]:
    return {
        "schema_version": PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_SCHEMA_VERSION,
        "envelope_kind": PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_KIND,
        "audit_envelope_status": AUDIT_ENVELOPE_STATUS,
        "audit_envelope_decision": AUDIT_ENVELOPE_DECISION,
        "ready_for_block_g_4": READY_FOR_BLOCK_G_4,
        "next_step": "FUNCTIONAL-PREVIEW-9.4",
    }


def _available_selection(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "case_id": event["case_id"],
        "event_id": event["event_id"],
        "label": event["case_description"],
        "selection_status": AVAILABLE_SELECTION_STATUS,
        "paper_only": True,
        "local_only": True,
        "intent_generation_allowed": False,
        "order_generation_allowed": False,
        "submission_allowed": False,
        "fills_allowed": False,
        "runtime_execution_allowed": False,
        "audit_export_allowed": False,
    }


def _empty_selected_audit_event_preview() -> dict[str, Any]:
    return {
        "selected": False,
        "case_id": None,
        "event_id": None,
        "input_snapshot": None,
        "unknown_input_keys": [],
        "paper_order_fixture_preview": None,
        "paper_order_audit_preview": None,
    }


def _selected_audit_event_preview(event: dict[str, Any]) -> dict[str, Any]:
    return deepcopy(
        {
            "selected": True,
            "case_id": event["case_id"],
            "event_id": event["event_id"],
            "input_snapshot": event["input_snapshot"],
            "unknown_input_keys": event["unknown_input_keys"],
            "paper_order_fixture_preview": event["paper_order_fixture_preview"],
            "paper_order_audit_preview": event["paper_order_audit_preview"],
        }
    )


def _selection_result(
    *,
    selection_status: str,
    selection_accepted: bool,
    selection_rejected: bool,
    selected_case_id: str | None,
    selected_event_id: str | None,
    rejection_reason: str | None,
) -> dict[str, Any]:
    return {
        "selection_status": selection_status,
        "selection_accepted": selection_accepted,
        "selection_rejected": selection_rejected,
        "rejection_reason": rejection_reason,
        "selected_case_id": selected_case_id,
        "selected_event_id": selected_event_id,
        "intent_generation_allowed": False,
        "order_generation_allowed": False,
        "submission_allowed": False,
        "fills_allowed": False,
        "runtime_execution_allowed": False,
    }


def _selection_no_execution_evidence() -> dict[str, bool]:
    return {
        "selection_gate_evaluated": True,
        "paper_order_intent_generated": False,
        "paper_order_generated": False,
        "paper_order_submitted": False,
        "paper_fill_simulated": False,
        "paper_runtime_execution_performed": False,
        "risk_governor_execution_performed": False,
        "audit_export_performed": False,
        "trading_controller_touched": False,
        "decision_envelope_touched": False,
        "live_execution_performed": False,
        "testnet_execution_performed": False,
        "account_fetch_performed": False,
        "secrets_read_performed": False,
        "export_performed": False,
    }


def _selection_gate_summary(
    available_selections: list[dict[str, Any]],
    selection_requested: bool,
    selection_accepted: bool,
    selection_rejected: bool,
) -> dict[str, Any]:
    return {
        "available_selection_count": len(available_selections),
        "selection_requested": selection_requested,
        "selection_accepted": selection_accepted,
        "selection_rejected": selection_rejected,
        "known_selection": selection_accepted,
        "ready_for_controlled_intent_preview_step": True,
        "all_available_selections_no_intent_generation": all(
            selection["intent_generation_allowed"] is False for selection in available_selections
        ),
        "all_available_selections_no_order_generation": all(
            selection["order_generation_allowed"] is False for selection in available_selections
        ),
        "all_available_selections_no_submission": all(
            selection["submission_allowed"] is False for selection in available_selections
        ),
        "all_available_selections_no_fills": all(
            selection["fills_allowed"] is False for selection in available_selections
        ),
        "all_available_selections_no_runtime_execution": all(
            selection["runtime_execution_allowed"] is False for selection in available_selections
        ),
        "all_available_selections_no_live_or_testnet": True,
        "all_available_selections_no_account_or_secrets": True,
        "all_available_selections_no_export": all(
            selection["audit_export_allowed"] is False for selection in available_selections
        ),
        "next_step": NEXT_STEP,
    }


def _boundary_checks(envelope: dict[str, Any]) -> dict[str, bool]:
    boundary_checks = dict(envelope["boundary_checks"])
    boundary_checks.update(SELECTION_BOUNDARY_EXTENSIONS)
    return boundary_checks


def _blocked_capabilities(envelope: dict[str, Any]) -> list[str]:
    blocked = list(envelope["blocked_capabilities"])
    for capability in ADDED_BLOCKED_CAPABILITIES:
        if capability not in blocked:
            blocked.append(capability)
    return blocked


def build_preview_paper_order_intent_selection_gate(
    selected_case_id: str | None = None,
) -> dict[str, Any]:
    """Return deterministic JSON-serializable 9.5 selection-gate data only."""

    envelope = build_preview_paper_order_audit_envelope()
    audit_events = envelope["audit_events"]
    available_selections = [_available_selection(event) for event in audit_events]
    events_by_case_id = {event["case_id"]: event for event in audit_events}

    selection_requested = selected_case_id is not None
    selected_case_id_output: str | None = None
    selected_preview = _empty_selected_audit_event_preview()

    if selected_case_id is None:
        result = _selection_result(
            selection_status=NO_SELECTION_STATUS,
            selection_accepted=False,
            selection_rejected=False,
            selected_case_id=None,
            selected_event_id=None,
            rejection_reason=None,
        )
    elif isinstance(selected_case_id, str) and selected_case_id in events_by_case_id:
        event = events_by_case_id[selected_case_id]
        selected_case_id_output = selected_case_id
        selected_preview = _selected_audit_event_preview(event)
        result = _selection_result(
            selection_status=ACCEPTED_SELECTION_STATUS,
            selection_accepted=True,
            selection_rejected=False,
            selected_case_id=selected_case_id,
            selected_event_id=event["event_id"],
            rejection_reason=None,
        )
    else:
        result = _selection_result(
            selection_status=REJECTED_SELECTION_STATUS,
            selection_accepted=False,
            selection_rejected=True,
            selected_case_id=None,
            selected_event_id=None,
            rejection_reason=REJECTION_REASON,
        )

    return deepcopy(
        {
            "schema_version": PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_SCHEMA_VERSION,
            "selection_gate_kind": PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_KIND,
            "block": BLOCK_ID,
            "step": STEP_ID,
            "selection_gate_status": SELECTION_GATE_STATUS,
            "selection_gate_decision": SELECTION_GATE_DECISION,
            "ready_for_block_g_6": READY_FOR_BLOCK_G_6,
            "next_step": NEXT_STEP,
            "next_step_title": NEXT_STEP_TITLE,
            "audit_envelope_reference": _audit_envelope_reference(),
            "available_selections": available_selections,
            "selected_case_id": selected_case_id_output,
            "selection_result": result,
            "selected_audit_event_preview": selected_preview,
            "selection_no_execution_evidence": _selection_no_execution_evidence(),
            "selection_gate_summary": _selection_gate_summary(
                available_selections,
                selection_requested,
                result["selection_accepted"],
                result["selection_rejected"],
            ),
            "boundary_checks": _boundary_checks(envelope),
            "blocked_capabilities": _blocked_capabilities(envelope),
            "source_boundaries": list(envelope["source_boundaries"]),
            "future_steps": list(FUTURE_STEPS),
            "status": "ready_for_functional_preview_9_6_no_intent_generation",
        }
    )


__all__ = [
    "BLOCK_ID",
    "NEXT_STEP",
    "NEXT_STEP_TITLE",
    "PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_KIND",
    "PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_SCHEMA_VERSION",
    "READY_FOR_BLOCK_G_6",
    "SELECTION_GATE_DECISION",
    "SELECTION_GATE_STATUS",
    "STEP_ID",
    "build_preview_paper_order_intent_selection_gate",
]
