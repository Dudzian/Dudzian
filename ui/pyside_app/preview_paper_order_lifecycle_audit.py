"""Pure-data BLOK G paper order lifecycle audit preview.

FUNCTIONAL-PREVIEW-9.8 builds only a static, non-executable audit shape for a
future paper order lifecycle from the 9.7 paper fill simulator contract. It never
mutates lifecycle state, executes lifecycle transitions, simulates fills, creates
fill events, generates or submits paper orders, starts runtime execution, fetches
market/account data, touches TradingController/DecisionEnvelope, or exports audit
data.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_paper_fill_simulator_contract import (
    FILL_SIMULATOR_CONTRACT_DECISION,
    FILL_SIMULATOR_CONTRACT_STATUS,
    NEXT_STEP as FILL_SIMULATOR_NEXT_STEP,
    PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_KIND,
    PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_8,
    build_preview_paper_fill_simulator_contract,
)

PREVIEW_PAPER_ORDER_LIFECYCLE_AUDIT_SCHEMA_VERSION: Final[str] = (
    "preview_paper_order_lifecycle_audit.v1"
)
PREVIEW_PAPER_ORDER_LIFECYCLE_AUDIT_KIND: Final[str] = (
    "functional_preview_block_g_paper_order_lifecycle_audit"
)
BLOCK_ID: Final[str] = "G"
STEP_ID: Final[str] = "9.8"
LIFECYCLE_AUDIT_STATUS: Final[str] = (
    "paper_order_lifecycle_audit_static_ready_no_lifecycle_mutation"
)
LIFECYCLE_AUDIT_DECISION: Final[str] = "BUILD_LIFECYCLE_AUDIT_STATIC_ONLY_NO_LIFECYCLE_MUTATION"
READY_FOR_BLOCK_G_9: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-9.9"
NEXT_STEP_TITLE: Final[str] = "BLOCK G CLOSURE AUDIT"

FUTURE_STEPS: Final[tuple[str, ...]] = ("functional_preview_9_9_block_g_closure_audit",)

CONTROLLED_INTENT_RESULT_KEYS: Final[tuple[str, ...]] = (
    "intent_preview_status",
    "intent_preview_built",
    "intent_preview_rejected",
    "rejection_reason",
    "selected_case_id",
    "selected_event_id",
    "intent_preview_executable",
    "order_generation_allowed",
    "submission_allowed",
    "fills_allowed",
    "runtime_execution_allowed",
)

PAPER_ORDER_INTENT_PREVIEW_KEYS: Final[tuple[str, ...]] = (
    "preview_available",
    "preview_only",
    "executable",
    "case_id",
    "event_id",
    "pair",
    "side",
    "order_type",
    "size_value",
    "size_unit",
    "source",
    "confidence_preview",
    "risk_status_preview",
    "unknown_input_keys",
    "order_generation_allowed",
    "submission_allowed",
    "fills_allowed",
    "runtime_execution_allowed",
)

FILL_SIMULATOR_CONTRACT_KEYS: Final[tuple[str, ...]] = (
    "contract_available",
    "contract_static_only",
    "contract_executable",
    "case_id",
    "event_id",
    "pair",
    "side",
    "order_type",
    "size_value",
    "size_unit",
    "fill_policy",
    "fill_price_source",
    "market_data_required_for_future_execution",
    "risk_governor_required_for_future_execution",
    "manual_confirmation_required_for_future_execution",
    "kill_switch_required_for_future_execution",
    "fill_simulation_allowed_now",
    "fill_event_generation_allowed_now",
    "order_lifecycle_mutation_allowed_now",
    "runtime_execution_allowed",
)

ADDED_BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
    "paper order lifecycle mutation",
    "paper order lifecycle transition execution",
    "paper order lifecycle runtime dispatch",
    "paper fill simulation execution",
    "paper fill event generation",
    "market data fetch for fills",
    "executable paper order intent generation",
    "preview intent conversion to order",
    "preview intent submission",
    "paper order generation now",
    "paper order submission now",
    "paper runtime execution now",
    "risk governor execution now",
    "audit export",
    "live/testnet/account/secrets/export/cloud",
    "TradingController / DecisionEnvelope",
    "QML changes / new QML calls",
    "EXE packaging",
)


def _fill_simulator_contract_reference() -> dict[str, Any]:
    return {
        "schema_version": PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_SCHEMA_VERSION,
        "fill_simulator_contract_kind": PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_KIND,
        "fill_simulator_contract_status": FILL_SIMULATOR_CONTRACT_STATUS,
        "fill_simulator_contract_decision": FILL_SIMULATOR_CONTRACT_DECISION,
        "ready_for_block_g_8": READY_FOR_BLOCK_G_8,
        "next_step": FILL_SIMULATOR_NEXT_STEP,
    }


def _safe_subset(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return deepcopy({key: source[key] for key in keys})


def _lifecycle_audit(contract: dict[str, Any], accepted: bool) -> dict[str, Any]:
    if not accepted:
        return {
            "audit_available": False,
            "audit_static_only": True,
            "audit_executable": False,
            "case_id": None,
            "event_id": None,
            "pair": None,
            "side": None,
            "order_type": None,
            "size_value": 0.0,
            "size_unit": "preview_only",
            "initial_lifecycle_state": "not_available_no_selection_or_rejected_selection",
            "terminal_lifecycle_state": "not_available_no_selection_or_rejected_selection",
            "lifecycle_policy": "not_available_no_selection_or_rejected_selection",
            "lifecycle_mutation_allowed_now": False,
            "lifecycle_transition_allowed_now": False,
            "fill_event_generation_allowed_now": False,
            "fill_simulation_allowed_now": False,
            "order_generation_allowed_now": False,
            "submission_allowed_now": False,
            "runtime_execution_allowed": False,
            "audit_export_allowed": False,
        }
    return {
        "audit_available": True,
        "audit_static_only": True,
        "audit_executable": False,
        "case_id": contract["case_id"],
        "event_id": contract["event_id"],
        "pair": contract["pair"],
        "side": contract["side"],
        "order_type": contract["order_type"],
        "size_value": contract["size_value"],
        "size_unit": contract["size_unit"],
        "initial_lifecycle_state": "preview_intent_static_only",
        "terminal_lifecycle_state": "awaiting_future_order_lifecycle_contract",
        "lifecycle_policy": "static_lifecycle_audit_only_no_mutation",
        "lifecycle_mutation_allowed_now": False,
        "lifecycle_transition_allowed_now": False,
        "fill_event_generation_allowed_now": False,
        "fill_simulation_allowed_now": False,
        "order_generation_allowed_now": False,
        "submission_allowed_now": False,
        "runtime_execution_allowed": False,
        "audit_export_allowed": False,
    }


def _transition_preview(contract: dict[str, Any], accepted: bool, rejected: bool) -> dict[str, Any]:
    return {
        "transition_preview_available": accepted,
        "transition_static_only": True,
        "transition_executable": False,
        "transition_rejected": rejected,
        "rejection_reason": "unknown_or_invalid_selection" if rejected else None,
        "case_id": contract["case_id"] if accepted else None,
        "event_id": contract["event_id"] if accepted else None,
        "from_state": "preview_intent_static_only" if accepted else None,
        "to_state": "awaiting_future_order_lifecycle_contract" if accepted else None,
        "transition_reason": (
            "static_audit_only_no_lifecycle_mutation"
            if accepted
            else "no_selection_or_rejected_selection"
        ),
        "lifecycle_mutation_allowed_now": False,
        "fill_event_generation_allowed_now": False,
        "submission_allowed": False,
        "runtime_execution_allowed": False,
        "audit_export_allowed": False,
    }


def _mutation_refusal(accepted: bool, rejected: bool) -> dict[str, Any]:
    if accepted:
        status = "lifecycle_mutation_refused_static_audit_only"
        reason = "static_audit_only_no_lifecycle_mutation"
    elif rejected:
        status = "lifecycle_mutation_refused_unknown_or_invalid_selection"
        reason = "unknown_or_invalid_selection"
    else:
        status = "lifecycle_mutation_refused_no_selection"
        reason = "no_selection"
    return {
        "refusal_status": status,
        "refused": True,
        "refusal_reason": reason,
        "lifecycle_mutation_refused": True,
        "lifecycle_transition_refused": True,
        "fill_simulation_refused": True,
        "fill_event_generation_refused": True,
        "order_generation_refused": True,
        "submission_refused": True,
        "runtime_execution_refused": True,
        "market_data_fetch_refused": True,
        "audit_export_refused": True,
        "live_execution_refused": True,
        "testnet_execution_refused": True,
        "account_fetch_refused": True,
        "secrets_read_refused": True,
    }


def _no_execution_evidence(built: bool) -> dict[str, bool]:
    return {
        "lifecycle_audit_evaluated": True,
        "static_lifecycle_audit_built": built,
        "lifecycle_mutation_performed": False,
        "lifecycle_transition_performed": False,
        "fill_simulation_performed": False,
        "fill_event_generated": False,
        "paper_order_generated": False,
        "paper_order_submitted": False,
        "paper_runtime_execution_performed": False,
        "risk_governor_execution_performed": False,
        "market_data_fetch_performed": False,
        "audit_export_performed": False,
        "trading_controller_touched": False,
        "decision_envelope_touched": False,
        "live_execution_performed": False,
        "testnet_execution_performed": False,
        "account_fetch_performed": False,
        "secrets_read_performed": False,
        "export_performed": False,
    }


def _summary(requested: bool, built: bool) -> dict[str, Any]:
    return {
        "selection_requested": requested,
        "controlled_intent_preview_built": built,
        "fill_contract_available": built,
        "lifecycle_audit_available": built,
        "lifecycle_audit_executable": False,
        "lifecycle_mutation_allowed_now": False,
        "lifecycle_transition_allowed_now": False,
        "fill_simulation_allowed_now": False,
        "fill_event_generation_allowed_now": False,
        "order_generation_allowed": False,
        "submission_allowed": False,
        "runtime_execution_allowed": False,
        "market_data_fetch_allowed": False,
        "live_or_testnet_allowed": False,
        "account_or_secrets_allowed": False,
        "audit_export_allowed": False,
        "requires_future_order_lifecycle_contract_before_mutation": True,
        "requires_future_fill_event_contract_before_fill": True,
        "requires_future_market_data_before_fill": True,
        "requires_future_risk_governor_before_order": True,
        "requires_future_manual_confirmation_before_order": True,
        "requires_future_kill_switch_before_order": True,
        "ready_for_block_g_closure_audit": True,
        "next_step": NEXT_STEP,
    }


def _boundary_checks(fill: dict[str, Any], built: bool) -> dict[str, bool]:
    boundary_checks = dict(fill["boundary_checks"])
    boundary_checks.update(
        {
            "paper_order_lifecycle_audit_static_only": True,
            "paper_fill_simulator_contract_static_only": True,
            "paper_order_intent_preview_built": built,
            "order_lifecycle_audit_allowed_now": True,
            "order_lifecycle_mutation_allowed_now": False,
            "order_lifecycle_mutated": False,
            "lifecycle_transition_allowed_now": False,
            "lifecycle_transition_performed": False,
            "future_order_lifecycle_contract_required_before_mutation": True,
            "market_data_fetch_allowed": False,
            "market_data_fetch_performed": False,
            "audit_export_allowed": False,
            "audit_export_performed": False,
            "exe_direction_preserved": True,
        }
    )
    return boundary_checks


def _blocked_capabilities(fill: dict[str, Any]) -> list[str]:
    blocked = list(fill["blocked_capabilities"])
    for capability in ADDED_BLOCKED_CAPABILITIES:
        if capability not in blocked:
            blocked.append(capability)
    return blocked


def build_preview_paper_order_lifecycle_audit(
    selected_case_id: str | None = None,
) -> dict[str, Any]:
    """Return deterministic JSON-serializable 9.8 static lifecycle audit data only."""

    fill = build_preview_paper_fill_simulator_contract(selected_case_id)
    controlled_result = _safe_subset(
        fill["controlled_intent_result"], CONTROLLED_INTENT_RESULT_KEYS
    )
    paper_order_intent_preview = _safe_subset(
        fill["paper_order_intent_preview"], PAPER_ORDER_INTENT_PREVIEW_KEYS
    )
    fill_simulator_contract = _safe_subset(
        fill["fill_simulator_contract"], FILL_SIMULATOR_CONTRACT_KEYS
    )
    requested = selected_case_id is not None
    accepted = controlled_result["intent_preview_built"] is True
    rejected = controlled_result["intent_preview_rejected"] is True

    return deepcopy(
        {
            "schema_version": PREVIEW_PAPER_ORDER_LIFECYCLE_AUDIT_SCHEMA_VERSION,
            "lifecycle_audit_kind": PREVIEW_PAPER_ORDER_LIFECYCLE_AUDIT_KIND,
            "block": BLOCK_ID,
            "step": STEP_ID,
            "lifecycle_audit_status": LIFECYCLE_AUDIT_STATUS,
            "lifecycle_audit_decision": LIFECYCLE_AUDIT_DECISION,
            "ready_for_block_g_9": READY_FOR_BLOCK_G_9,
            "next_step": NEXT_STEP,
            "next_step_title": NEXT_STEP_TITLE,
            "fill_simulator_contract_reference": _fill_simulator_contract_reference(),
            "selected_case_id": controlled_result["selected_case_id"] if accepted else None,
            "controlled_intent_result": controlled_result,
            "paper_order_intent_preview": paper_order_intent_preview,
            "fill_simulator_contract": fill_simulator_contract,
            "lifecycle_audit": _lifecycle_audit(fill_simulator_contract, accepted),
            "lifecycle_transition_preview": _transition_preview(
                fill_simulator_contract, accepted, rejected
            ),
            "lifecycle_mutation_refusal": _mutation_refusal(accepted, rejected),
            "lifecycle_no_execution_evidence": _no_execution_evidence(accepted),
            "lifecycle_audit_summary": _summary(requested, accepted),
            "boundary_checks": _boundary_checks(fill, accepted),
            "blocked_capabilities": _blocked_capabilities(fill),
            "source_boundaries": list(fill["source_boundaries"]),
            "future_steps": list(FUTURE_STEPS),
            "status": "ready_for_functional_preview_9_9_no_lifecycle_mutation",
        }
    )


__all__ = [
    "BLOCK_ID",
    "LIFECYCLE_AUDIT_DECISION",
    "LIFECYCLE_AUDIT_STATUS",
    "NEXT_STEP",
    "NEXT_STEP_TITLE",
    "PREVIEW_PAPER_ORDER_LIFECYCLE_AUDIT_KIND",
    "PREVIEW_PAPER_ORDER_LIFECYCLE_AUDIT_SCHEMA_VERSION",
    "READY_FOR_BLOCK_G_9",
    "STEP_ID",
    "build_preview_paper_order_lifecycle_audit",
]
