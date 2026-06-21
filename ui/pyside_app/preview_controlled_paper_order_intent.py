"""Pure-data BLOK G controlled paper order intent preview.

FUNCTIONAL-PREVIEW-9.6 may build only a non-executable
``paper_order_intent_preview`` from an accepted 9.5 selection-gate result. It
never generates an executable intent/order, submits orders, simulates fills,
starts runtime execution, touches TradingController/DecisionEnvelope, or exports
audit data.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_paper_order_intent_selection_gate import (
    NEXT_STEP as SELECTION_GATE_NEXT_STEP,
    PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_KIND,
    PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_6,
    SELECTION_GATE_DECISION,
    SELECTION_GATE_STATUS,
    build_preview_paper_order_intent_selection_gate,
)

PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_SCHEMA_VERSION: Final[str] = (
    "preview_controlled_paper_order_intent.v1"
)
PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_KIND: Final[str] = (
    "functional_preview_block_g_controlled_paper_order_intent"
)
BLOCK_ID: Final[str] = "G"
STEP_ID: Final[str] = "9.6"
CONTROLLED_INTENT_STATUS: Final[str] = (
    "controlled_paper_order_intent_preview_ready_no_order_generation"
)
CONTROLLED_INTENT_DECISION: Final[str] = "BUILD_CONTROLLED_INTENT_PREVIEW_ONLY_NO_ORDER_GENERATION"
READY_FOR_BLOCK_G_7: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-9.7"
NEXT_STEP_TITLE: Final[str] = "PAPER FILL SIMULATOR CONTRACT STATIC ONLY"

FUTURE_STEPS: Final[tuple[str, ...]] = (
    "functional_preview_9_7_paper_fill_simulator_contract_static_only",
    "functional_preview_9_8_paper_order_lifecycle_audit",
    "functional_preview_9_9_block_g_closure_audit",
)

ADDED_BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
    "executable paper order intent generation",
    "preview intent conversion to order",
    "preview intent submission",
    "paper order generation now",
    "paper order submission now",
    "paper fill simulation now",
    "paper runtime execution now",
    "risk governor execution now",
    "audit export",
    "live/testnet/account/secrets/export/cloud",
    "TradingController / DecisionEnvelope",
    "QML changes / new QML calls",
    "EXE packaging",
)

BOUNDARY_EXTENSIONS: Final[dict[str, bool]] = {
    "controlled_intent_preview_only": True,
    "selection_gate_only": True,
    "ui_surface_read_only": True,
    "paper_order_intent_preview_allowed_now": True,
    "executable_intent_generation_allowed": False,
    "executable_intent_generated": False,
    "future_risk_governor_required_before_order": True,
}


def _selection_gate_reference() -> dict[str, Any]:
    return {
        "schema_version": PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_SCHEMA_VERSION,
        "selection_gate_kind": PREVIEW_PAPER_ORDER_INTENT_SELECTION_GATE_KIND,
        "selection_gate_status": SELECTION_GATE_STATUS,
        "selection_gate_decision": SELECTION_GATE_DECISION,
        "ready_for_block_g_6": READY_FOR_BLOCK_G_6,
        "next_step": SELECTION_GATE_NEXT_STEP,
    }


def _safe_selection_result(gate: dict[str, Any]) -> dict[str, Any]:
    result = gate["selection_result"]
    return {
        key: result[key]
        for key in (
            "selection_status",
            "selection_accepted",
            "selection_rejected",
            "rejection_reason",
            "selected_case_id",
            "selected_event_id",
            "intent_generation_allowed",
            "order_generation_allowed",
            "submission_allowed",
            "fills_allowed",
            "runtime_execution_allowed",
        )
    }


def _empty_paper_order_intent_preview() -> dict[str, Any]:
    return {
        "preview_available": False,
        "preview_only": True,
        "executable": False,
        "case_id": None,
        "event_id": None,
        "pair": None,
        "side": None,
        "order_type": None,
        "size_value": 0.0,
        "size_unit": "preview_only",
        "source": "no_selection_or_rejected_selection",
        "confidence_preview": 0.0,
        "risk_status_preview": "not_evaluated_no_selection",
        "unknown_input_keys": [],
        "order_generation_allowed": False,
        "submission_allowed": False,
        "fills_allowed": False,
        "runtime_execution_allowed": False,
    }


def _selected_paper_order_intent_preview(gate: dict[str, Any]) -> dict[str, Any]:
    selected = gate["selected_audit_event_preview"]
    snapshot = selected["input_snapshot"]
    size = snapshot["paper_order_intent_size_preview"]
    candidate = snapshot.get("operator_selected_candidate", {})
    dry_run = snapshot.get("dry_run_decision_preview", {})
    risk = snapshot.get("risk_check_preview", {})
    return deepcopy(
        {
            "preview_available": True,
            "preview_only": True,
            "executable": False,
            "case_id": selected["case_id"],
            "event_id": selected["event_id"],
            "pair": snapshot["operator_selected_pair"],
            "side": snapshot["paper_order_intent_side_preview"],
            "order_type": snapshot["paper_order_intent_type_preview"],
            "size_value": size["value"],
            "size_unit": size["unit"],
            "source": size["source"],
            "confidence_preview": candidate.get(
                "confidence", dry_run.get("confidence_preview", 0.0)
            ),
            "risk_status_preview": risk.get("risk_status", "not_evaluated_preview_only"),
            "unknown_input_keys": list(selected["unknown_input_keys"]),
            "order_generation_allowed": False,
            "submission_allowed": False,
            "fills_allowed": False,
            "runtime_execution_allowed": False,
        }
    )


def _controlled_intent_result(
    *, requested: bool, accepted: bool, rejected: bool, result: dict[str, Any]
) -> dict[str, Any]:
    if accepted:
        status = "built_for_preview_only_no_order_generation"
        reason = None
    elif rejected:
        status = "rejected_fail_closed_unknown_or_invalid_selection"
        reason = "unknown_or_invalid_selection"
    else:
        status = "not_built_no_selection"
        reason = None
    return {
        "intent_preview_status": status,
        "intent_preview_built": accepted,
        "intent_preview_rejected": rejected,
        "rejection_reason": reason,
        "selected_case_id": result["selected_case_id"] if accepted else None,
        "selected_event_id": result["selected_event_id"] if accepted else None,
        "intent_preview_executable": False,
        "order_generation_allowed": False,
        "submission_allowed": False,
        "fills_allowed": False,
        "runtime_execution_allowed": False,
    }


def _validation(accepted: bool, rejected: bool) -> dict[str, Any]:
    status = (
        "validated_for_preview_only_no_order_generation"
        if accepted
        else (
            "rejected_fail_closed_unknown_or_invalid_selection"
            if rejected
            else "not_evaluated_no_selection"
        )
    )
    return {
        "validation_status": status,
        "validated_for_preview_only": accepted,
        "executable_intent_allowed": False,
        "order_generation_allowed": False,
        "submission_allowed": False,
        "fills_allowed": False,
        "runtime_execution_allowed": False,
        "requires_manual_confirmation_before_any_future_order": True,
        "requires_kill_switch_before_any_future_order": True,
        "requires_risk_governor_before_any_future_order": True,
    }


def _refusal(accepted: bool, rejected: bool) -> dict[str, Any]:
    status = (
        "not_refused_preview_only"
        if accepted
        else ("refused_unknown_or_invalid_selection" if rejected else "refused_no_selection")
    )
    reason = None if accepted else ("unknown_or_invalid_selection" if rejected else "no_selection")
    return {
        "refusal_status": status,
        "refused": not accepted,
        "refusal_reason": reason,
        "order_generation_refused": True,
        "submission_refused": True,
        "fills_refused": True,
        "runtime_execution_refused": True,
        "audit_export_refused": True,
        "live_execution_refused": True,
        "testnet_execution_refused": True,
        "account_fetch_refused": True,
        "secrets_read_refused": True,
    }


def _no_execution_evidence(built: bool) -> dict[str, bool]:
    return {
        "controlled_intent_preview_evaluated": True,
        "paper_order_intent_preview_built": built,
        "executable_intent_generated": False,
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


def _summary(requested: bool, accepted: bool, built: bool) -> dict[str, Any]:
    return {
        "selection_requested": requested,
        "selection_accepted": accepted,
        "intent_preview_built": built,
        "intent_preview_executable": False,
        "ready_for_static_fill_contract_step": True,
        "order_generation_allowed": False,
        "submission_allowed": False,
        "fills_allowed": False,
        "runtime_execution_allowed": False,
        "live_or_testnet_allowed": False,
        "account_or_secrets_allowed": False,
        "audit_export_allowed": False,
        "requires_future_risk_governor_before_order": True,
        "requires_future_manual_confirmation_before_order": True,
        "requires_future_kill_switch_before_order": True,
        "next_step": NEXT_STEP,
    }


def _boundary_checks(gate: dict[str, Any], built: bool) -> dict[str, bool]:
    boundary_checks = dict(gate["boundary_checks"])
    boundary_checks.update(BOUNDARY_EXTENSIONS)
    boundary_checks["paper_order_intent_preview_built"] = built
    return boundary_checks


def _blocked_capabilities(gate: dict[str, Any]) -> list[str]:
    blocked = list(gate["blocked_capabilities"])
    for capability in ADDED_BLOCKED_CAPABILITIES:
        if capability not in blocked:
            blocked.append(capability)
    return blocked


def build_preview_controlled_paper_order_intent(
    selected_case_id: str | None = None,
) -> dict[str, Any]:
    """Return deterministic JSON-serializable 9.6 controlled preview data only."""

    gate = build_preview_paper_order_intent_selection_gate(selected_case_id)
    selection_result = _safe_selection_result(gate)
    requested = selected_case_id is not None
    accepted = selection_result["selection_accepted"] is True
    rejected = selection_result["selection_rejected"] is True
    built = accepted
    preview = (
        _selected_paper_order_intent_preview(gate)
        if accepted
        else _empty_paper_order_intent_preview()
    )

    return deepcopy(
        {
            "schema_version": PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_SCHEMA_VERSION,
            "controlled_intent_kind": PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_KIND,
            "block": BLOCK_ID,
            "step": STEP_ID,
            "controlled_intent_status": CONTROLLED_INTENT_STATUS,
            "controlled_intent_decision": CONTROLLED_INTENT_DECISION,
            "ready_for_block_g_7": READY_FOR_BLOCK_G_7,
            "next_step": NEXT_STEP,
            "next_step_title": NEXT_STEP_TITLE,
            "selection_gate_reference": _selection_gate_reference(),
            "selected_case_id": selection_result["selected_case_id"] if accepted else None,
            "selection_result": selection_result,
            "controlled_intent_result": _controlled_intent_result(
                requested=requested, accepted=accepted, rejected=rejected, result=selection_result
            ),
            "paper_order_intent_preview": preview,
            "intent_preview_validation": _validation(accepted, rejected),
            "intent_preview_refusal": _refusal(accepted, rejected),
            "intent_no_execution_evidence": _no_execution_evidence(built),
            "controlled_intent_summary": _summary(requested, accepted, built),
            "boundary_checks": _boundary_checks(gate, built),
            "blocked_capabilities": _blocked_capabilities(gate),
            "source_boundaries": list(gate["source_boundaries"]),
            "future_steps": list(FUTURE_STEPS),
            "status": "ready_for_functional_preview_9_7_no_order_generation",
        }
    )


__all__ = [
    "BLOCK_ID",
    "CONTROLLED_INTENT_DECISION",
    "CONTROLLED_INTENT_STATUS",
    "NEXT_STEP",
    "NEXT_STEP_TITLE",
    "PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_KIND",
    "PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_SCHEMA_VERSION",
    "READY_FOR_BLOCK_G_7",
    "STEP_ID",
    "build_preview_controlled_paper_order_intent",
]
