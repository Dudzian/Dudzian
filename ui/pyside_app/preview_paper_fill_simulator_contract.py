"""Pure-data BLOK G paper fill simulator contract preview.

FUNCTIONAL-PREVIEW-9.7 builds only a static, non-executable contract shape for a
future paper fill simulator from the 9.6 controlled paper order intent preview.
It never simulates fills, creates fill events, mutates order lifecycle, generates
or submits orders, starts runtime execution, fetches market/account data, touches
TradingController/DecisionEnvelope, or exports audit data.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_controlled_paper_order_intent import (
    CONTROLLED_INTENT_DECISION,
    CONTROLLED_INTENT_STATUS,
    NEXT_STEP as CONTROLLED_INTENT_NEXT_STEP,
    PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_KIND,
    PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_7,
    build_preview_controlled_paper_order_intent,
)

PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "preview_paper_fill_simulator_contract.v1"
)
PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_KIND: Final[str] = (
    "functional_preview_block_g_paper_fill_simulator_contract"
)
BLOCK_ID: Final[str] = "G"
STEP_ID: Final[str] = "9.7"
FILL_SIMULATOR_CONTRACT_STATUS: Final[str] = (
    "paper_fill_simulator_contract_static_ready_no_fill_simulation"
)
FILL_SIMULATOR_CONTRACT_DECISION: Final[str] = (
    "BUILD_FILL_SIMULATOR_CONTRACT_STATIC_ONLY_NO_FILL_SIMULATION"
)
READY_FOR_BLOCK_G_8: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-9.8"
NEXT_STEP_TITLE: Final[str] = "PAPER ORDER LIFECYCLE AUDIT"

FUTURE_STEPS: Final[tuple[str, ...]] = (
    "functional_preview_9_8_paper_order_lifecycle_audit",
    "functional_preview_9_9_block_g_closure_audit",
)

ADDED_BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
    "paper fill simulation execution",
    "paper fill event generation",
    "paper order lifecycle mutation",
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


def _controlled_intent_reference() -> dict[str, Any]:
    return {
        "schema_version": PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_SCHEMA_VERSION,
        "controlled_intent_kind": PREVIEW_CONTROLLED_PAPER_ORDER_INTENT_KIND,
        "controlled_intent_status": CONTROLLED_INTENT_STATUS,
        "controlled_intent_decision": CONTROLLED_INTENT_DECISION,
        "ready_for_block_g_7": READY_FOR_BLOCK_G_7,
        "next_step": CONTROLLED_INTENT_NEXT_STEP,
    }


def _safe_subset(source: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    return deepcopy({key: source[key] for key in keys})


def _fill_simulator_contract(preview: dict[str, Any], accepted: bool) -> dict[str, Any]:
    if not accepted:
        return {
            "contract_available": False,
            "contract_static_only": True,
            "contract_executable": False,
            "case_id": None,
            "event_id": None,
            "pair": None,
            "side": None,
            "order_type": None,
            "size_value": 0.0,
            "size_unit": "preview_only",
            "fill_policy": "not_available_no_selection_or_rejected_selection",
            "fill_price_source": "none",
            "market_data_required_for_future_execution": True,
            "risk_governor_required_for_future_execution": True,
            "manual_confirmation_required_for_future_execution": True,
            "kill_switch_required_for_future_execution": True,
            "fill_simulation_allowed_now": False,
            "fill_event_generation_allowed_now": False,
            "order_lifecycle_mutation_allowed_now": False,
            "runtime_execution_allowed": False,
        }
    return {
        "contract_available": True,
        "contract_static_only": True,
        "contract_executable": False,
        "case_id": preview["case_id"],
        "event_id": preview["event_id"],
        "pair": preview["pair"],
        "side": preview["side"],
        "order_type": preview["order_type"],
        "size_value": preview["size_value"],
        "size_unit": preview["size_unit"],
        "fill_policy": "static_preview_contract_only_no_fill_simulation",
        "fill_price_source": "not_evaluated_no_market_data",
        "market_data_required_for_future_execution": True,
        "risk_governor_required_for_future_execution": True,
        "manual_confirmation_required_for_future_execution": True,
        "kill_switch_required_for_future_execution": True,
        "fill_simulation_allowed_now": False,
        "fill_event_generation_allowed_now": False,
        "order_lifecycle_mutation_allowed_now": False,
        "runtime_execution_allowed": False,
    }


def _fill_simulation_request_preview(
    preview: dict[str, Any], accepted: bool, rejected: bool
) -> dict[str, Any]:
    return {
        "request_available": accepted,
        "request_static_only": True,
        "request_executable": False,
        "request_rejected": rejected,
        "rejection_reason": "unknown_or_invalid_selection" if rejected else None,
        "case_id": preview["case_id"] if accepted else None,
        "event_id": preview["event_id"] if accepted else None,
        "pair": preview["pair"] if accepted else None,
        "side": preview["side"] if accepted else None,
        "size_value": preview["size_value"] if accepted else 0.0,
        "fill_simulation_allowed_now": False,
        "fill_event_generation_allowed_now": False,
        "submission_allowed": False,
        "runtime_execution_allowed": False,
    }


def _fill_simulation_refusal(accepted: bool, rejected: bool) -> dict[str, Any]:
    status = (
        "fill_simulation_refused_static_contract_only"
        if accepted
        else (
            "fill_simulation_refused_unknown_or_invalid_selection"
            if rejected
            else "fill_simulation_refused_no_selection"
        )
    )
    reason = (
        "static_contract_only_no_fill_simulation"
        if accepted
        else ("unknown_or_invalid_selection" if rejected else "no_selection")
    )
    return {
        "refusal_status": status,
        "refused": True,
        "refusal_reason": reason,
        "fill_simulation_refused": True,
        "fill_event_generation_refused": True,
        "order_lifecycle_mutation_refused": True,
        "order_generation_refused": True,
        "submission_refused": True,
        "runtime_execution_refused": True,
        "audit_export_refused": True,
        "live_execution_refused": True,
        "testnet_execution_refused": True,
        "market_data_fetch_refused": True,
        "account_fetch_refused": True,
        "secrets_read_refused": True,
    }


def _no_execution_evidence(built: bool) -> dict[str, bool]:
    return {
        "fill_simulator_contract_evaluated": True,
        "static_contract_built": built,
        "fill_simulation_performed": False,
        "fill_event_generated": False,
        "order_lifecycle_mutated": False,
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
        "fill_contract_executable": False,
        "fill_simulation_allowed_now": False,
        "fill_event_generation_allowed_now": False,
        "order_lifecycle_mutation_allowed_now": False,
        "order_generation_allowed": False,
        "submission_allowed": False,
        "runtime_execution_allowed": False,
        "market_data_fetch_allowed": False,
        "live_or_testnet_allowed": False,
        "account_or_secrets_allowed": False,
        "audit_export_allowed": False,
        "requires_future_market_data_before_fill": True,
        "requires_future_risk_governor_before_fill": True,
        "requires_future_manual_confirmation_before_order": True,
        "requires_future_kill_switch_before_order": True,
        "ready_for_order_lifecycle_audit_step": True,
        "next_step": NEXT_STEP,
    }


def _boundary_checks(controlled: dict[str, Any], built: bool) -> dict[str, bool]:
    boundary_checks = dict(controlled["boundary_checks"])
    boundary_checks.update(
        {
            "paper_fill_simulator_contract_static_only": True,
            "paper_order_intent_preview_built": built,
            "paper_fill_simulation_contract_allowed_now": True,
            "paper_fill_simulation_allowed_now": False,
            "paper_fill_simulated": False,
            "fill_event_generation_allowed_now": False,
            "fill_event_generated": False,
            "order_lifecycle_mutation_allowed_now": False,
            "order_lifecycle_mutated": False,
            "market_data_fetch_allowed": False,
            "market_data_fetch_performed": False,
            "future_market_data_required_before_fill": True,
            "exe_direction_preserved": True,
        }
    )
    return boundary_checks


def _blocked_capabilities(controlled: dict[str, Any]) -> list[str]:
    blocked = list(controlled["blocked_capabilities"])
    for capability in ADDED_BLOCKED_CAPABILITIES:
        if capability not in blocked:
            blocked.append(capability)
    return blocked


def build_preview_paper_fill_simulator_contract(
    selected_case_id: str | None = None,
) -> dict[str, Any]:
    """Return deterministic JSON-serializable 9.7 static fill contract data only."""

    controlled = build_preview_controlled_paper_order_intent(selected_case_id)
    controlled_result = _safe_subset(
        controlled["controlled_intent_result"], CONTROLLED_INTENT_RESULT_KEYS
    )
    paper_order_intent_preview = _safe_subset(
        controlled["paper_order_intent_preview"], PAPER_ORDER_INTENT_PREVIEW_KEYS
    )
    requested = selected_case_id is not None
    accepted = controlled_result["intent_preview_built"] is True
    rejected = controlled_result["intent_preview_rejected"] is True

    return deepcopy(
        {
            "schema_version": PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_SCHEMA_VERSION,
            "fill_simulator_contract_kind": PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_KIND,
            "block": BLOCK_ID,
            "step": STEP_ID,
            "fill_simulator_contract_status": FILL_SIMULATOR_CONTRACT_STATUS,
            "fill_simulator_contract_decision": FILL_SIMULATOR_CONTRACT_DECISION,
            "ready_for_block_g_8": READY_FOR_BLOCK_G_8,
            "next_step": NEXT_STEP,
            "next_step_title": NEXT_STEP_TITLE,
            "controlled_intent_reference": _controlled_intent_reference(),
            "selected_case_id": controlled_result["selected_case_id"] if accepted else None,
            "controlled_intent_result": controlled_result,
            "paper_order_intent_preview": paper_order_intent_preview,
            "fill_simulator_contract": _fill_simulator_contract(
                paper_order_intent_preview, accepted
            ),
            "fill_simulation_request_preview": _fill_simulation_request_preview(
                paper_order_intent_preview, accepted, rejected
            ),
            "fill_simulation_refusal": _fill_simulation_refusal(accepted, rejected),
            "fill_simulation_no_execution_evidence": _no_execution_evidence(accepted),
            "fill_contract_summary": _summary(requested, accepted),
            "boundary_checks": _boundary_checks(controlled, accepted),
            "blocked_capabilities": _blocked_capabilities(controlled),
            "source_boundaries": list(controlled["source_boundaries"]),
            "future_steps": list(FUTURE_STEPS),
            "status": "ready_for_functional_preview_9_8_no_fill_simulation",
        }
    )


__all__ = [
    "BLOCK_ID",
    "FILL_SIMULATOR_CONTRACT_DECISION",
    "FILL_SIMULATOR_CONTRACT_STATUS",
    "NEXT_STEP",
    "NEXT_STEP_TITLE",
    "PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_KIND",
    "PREVIEW_PAPER_FILL_SIMULATOR_CONTRACT_SCHEMA_VERSION",
    "READY_FOR_BLOCK_G_8",
    "STEP_ID",
    "build_preview_paper_fill_simulator_contract",
]
