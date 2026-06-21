"""Pure-data BLOK G paper order intent read model.

FUNCTIONAL-PREVIEW-9.1 normalizes a safe local/paper-only input snapshot
for a future paper order intent surface. It does not generate order intents,
orders, submissions, fills, runtime actions, account access, exports, QML calls,
or packaging artifacts.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_paper_decision_to_order_contract import (
    build_preview_paper_decision_to_order_contract,
)

PREVIEW_PAPER_ORDER_INTENT_READ_MODEL_SCHEMA_VERSION: Final[str] = (
    "preview_paper_order_intent_read_model.v1"
)
PREVIEW_PAPER_ORDER_INTENT_READ_MODEL_KIND: Final[str] = (
    "functional_preview_block_g_paper_order_intent_read_model"
)
BLOCK_ID: Final[str] = "G"
STEP_ID: Final[str] = "9.1"
READ_MODEL_STATUS: Final[str] = "paper_order_intent_read_model_ready_no_order_generation"
READ_MODEL_DECISION: Final[str] = "BUILD_PAPER_ORDER_INTENT_READ_MODEL_ONLY_NO_ORDER_GENERATION"
READY_FOR_BLOCK_G_2: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-9.2"
NEXT_STEP_TITLE: Final[str] = "PAPER ORDER STATIC FIXTURE"

DEFAULT_INPUT_SNAPSHOT: Final[dict[str, Any]] = {
    "paper_order_intent_context_id": "local-preview-paper-order-intent-context",
    "dry_run_decision_preview": {
        "decision_action": "NO_ORDER_DRY_RUN_PREVIEW",
        "decision_status": "not_executed",
        "confidence_preview": 0.0,
    },
    "decision_reason_summary": "read-model-only placeholder; no decision execution",
    "risk_check_preview": {
        "risk_status": "not_evaluated_read_model_only",
        "risk_engine_execution_performed": False,
    },
    "audit_event_preview": {
        "audit_event_status": "not_exported_read_model_only",
        "audit_export_allowed": False,
    },
    "operator_selected_pair": "BTC/USDT",
    "operator_selected_candidate": {
        "pair": "BTC/USDT",
        "source": "local_preview_default",
        "confidence": 0.0,
    },
    "paper_order_intent_size_preview": {
        "value": 0.0,
        "unit": "preview_only",
        "source": "read_model_default",
    },
    "paper_order_intent_side_preview": "none",
    "paper_order_intent_type_preview": "none",
}

FUTURE_STEPS: Final[tuple[str, ...]] = (
    "functional_preview_9_2_paper_order_static_fixture",
    "functional_preview_9_3_paper_order_audit_envelope",
    "functional_preview_9_4_ui_read_only_paper_order_surface",
    "functional_preview_9_5_controlled_paper_order_intent_selection_gate",
    "functional_preview_9_6_controlled_paper_order_intent_no_submission",
    "functional_preview_9_7_paper_fill_simulator_contract_static_only",
    "functional_preview_9_8_paper_order_lifecycle_audit",
    "functional_preview_9_9_block_g_closure_audit",
)

ADDED_BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
    "paper order intent generation now",
    "paper order validation now",
    "risk governor evaluation now",
    "paper order submission now",
    "paper fills now",
    "live/testnet/account/secrets/export/cloud",
    "TradingController / DecisionEnvelope",
    "QML changes / new QML calls",
    "EXE packaging",
)


def _contract_reference(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": contract["schema_version"],
        "contract_kind": contract["contract_kind"],
        "block_status": contract["block_status"],
        "contract_decision": contract["contract_decision"],
        "ready_for_block_g_1": contract["ready_for_block_g_1"],
        "next_step": contract["next_step"],
    }


def _normalize_input_snapshot(
    input_snapshot: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    defaults = deepcopy(DEFAULT_INPUT_SNAPSHOT)
    if input_snapshot is None:
        return defaults, []

    copied_input = deepcopy(input_snapshot)
    allowed_keys = set(DEFAULT_INPUT_SNAPSHOT)
    unknown_keys = sorted(str(key) for key in copied_input if key not in allowed_keys)
    for key in allowed_keys:
        if key in copied_input:
            defaults[key] = copied_input[key]
    return defaults, unknown_keys


def _boundary_checks_from_contract(contract: dict[str, Any]) -> dict[str, bool]:
    boundary_checks = dict(contract["boundary_checks"])
    boundary_checks.update(
        {
            "read_model_only": True,
            "contract_only": False,
            "paper_order_intent_generated": False,
            "risk_governor_execution_performed": False,
        }
    )
    return boundary_checks


def _blocked_capabilities_from_contract(contract: dict[str, Any]) -> list[str]:
    blocked = list(contract["blocked_capabilities"])
    for capability in ADDED_BLOCKED_CAPABILITIES:
        if capability not in blocked:
            blocked.append(capability)
    return blocked


def build_preview_paper_order_intent_read_model_snapshot(
    input_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return deterministic JSON-serializable 9.1 read-model data only."""

    contract = build_preview_paper_decision_to_order_contract()
    normalized_input, unknown_input_keys = _normalize_input_snapshot(input_snapshot)
    dry_run_preview = normalized_input["dry_run_decision_preview"]

    read_model = {
        "intent_context_id": normalized_input["paper_order_intent_context_id"],
        "operator_selected_pair": normalized_input["operator_selected_pair"],
        "operator_selected_candidate": deepcopy(normalized_input["operator_selected_candidate"]),
        "dry_run_decision_action": dry_run_preview.get("decision_action"),
        "dry_run_decision_status": dry_run_preview.get("decision_status"),
        "decision_reason_summary": normalized_input["decision_reason_summary"],
        "paper_order_intent_size_preview": deepcopy(
            normalized_input["paper_order_intent_size_preview"]
        ),
        "paper_order_intent_side_preview": normalized_input["paper_order_intent_side_preview"],
        "paper_order_intent_type_preview": normalized_input["paper_order_intent_type_preview"],
        "read_model_only": True,
        "order_intent_generated": False,
        "order_generation_allowed": False,
        "order_submission_allowed": False,
        "fill_simulation_allowed": False,
        "runtime_execution_allowed": False,
    }

    input_echo = deepcopy(normalized_input)
    input_echo["unknown_input_keys"] = unknown_input_keys

    return deepcopy(
        {
            "schema_version": PREVIEW_PAPER_ORDER_INTENT_READ_MODEL_SCHEMA_VERSION,
            "read_model_kind": PREVIEW_PAPER_ORDER_INTENT_READ_MODEL_KIND,
            "block": BLOCK_ID,
            "step": STEP_ID,
            "read_model_status": READ_MODEL_STATUS,
            "read_model_decision": READ_MODEL_DECISION,
            "ready_for_block_g_2": READY_FOR_BLOCK_G_2,
            "next_step": NEXT_STEP,
            "next_step_title": NEXT_STEP_TITLE,
            "contract_reference": _contract_reference(contract),
            "input_snapshot": deepcopy(normalized_input),
            "input_snapshot_echo": input_echo,
            "paper_order_intent_read_model": read_model,
            "paper_order_intent_preview": {
                "intent_status": "not_generated_read_model_only",
                "intent_action": "NO_PAPER_ORDER_INTENT_GENERATED",
                "paper_only": True,
                "local_only": True,
                "order_intent_generated": False,
                "order_generation_allowed": False,
                "order_submission_allowed": False,
                "fill_simulation_allowed": False,
                "runtime_execution_allowed": False,
            },
            "paper_order_validation_preview": {
                "validation_status": "not_evaluated_read_model_only",
                "validation_performed": False,
                "validation_passed": False,
                "risk_governor_evaluated": False,
                "manual_confirmation_present": False,
                "kill_switch_checked": False,
            },
            "paper_order_refusal_preview": {
                "refusal_status": "not_evaluated_read_model_only",
                "refusal_reason": "order path blocked in FUNCTIONAL-PREVIEW-9.1",
                "blocked_by_contract": True,
                "blocked_until_step": "FUNCTIONAL-PREVIEW-9.2+",
            },
            "paper_order_gate_status": {
                "paper_order_intent_read_model_ready": True,
                "paper_order_generation_gate_open": False,
                "paper_order_submission_gate_open": False,
                "paper_fill_simulation_gate_open": False,
                "runtime_execution_gate_open": False,
                "risk_governor_required": True,
                "manual_confirmation_required": True,
                "kill_switch_required": True,
                "live_credentials_refusal_required": True,
            },
            "paper_order_risk_gate_preview": {
                "risk_gate_status": "required_not_evaluated",
                "risk_governor_execution_allowed_now": False,
                "risk_governor_execution_performed": False,
                "risk_governor_required_before_any_order_execution": True,
            },
            "paper_order_no_execution_status": {
                "order_intent_generated": False,
                "order_generated": False,
                "order_submitted": False,
                "fill_simulated": False,
                "runtime_execution_performed": False,
                "live_execution_performed": False,
                "testnet_execution_performed": False,
                "account_fetch_performed": False,
                "secrets_read_performed": False,
                "export_performed": False,
            },
            "boundary_checks": _boundary_checks_from_contract(contract),
            "blocked_capabilities": _blocked_capabilities_from_contract(contract),
            "source_boundaries": list(contract["source_boundaries"]),
            "future_steps": list(FUTURE_STEPS),
            "status": "ready_for_functional_preview_9_2_no_order_generation",
        }
    )


__all__ = [
    "BLOCK_ID",
    "NEXT_STEP",
    "NEXT_STEP_TITLE",
    "PREVIEW_PAPER_ORDER_INTENT_READ_MODEL_KIND",
    "PREVIEW_PAPER_ORDER_INTENT_READ_MODEL_SCHEMA_VERSION",
    "READ_MODEL_DECISION",
    "READ_MODEL_STATUS",
    "READY_FOR_BLOCK_G_2",
    "STEP_ID",
    "build_preview_paper_order_intent_read_model_snapshot",
]
