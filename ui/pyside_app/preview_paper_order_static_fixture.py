"""Pure-data BLOK G paper order static fixture.

FUNCTIONAL-PREVIEW-9.2 builds deterministic static fixture cases on top of the
9.1 paper order intent read model. It remains read-only/static-only and does
not generate order intents, orders, submissions, fills, runtime actions,
account access, exports, QML calls, or packaging artifacts.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_paper_order_intent_read_model import (
    PREVIEW_PAPER_ORDER_INTENT_READ_MODEL_KIND,
    PREVIEW_PAPER_ORDER_INTENT_READ_MODEL_SCHEMA_VERSION,
    READ_MODEL_DECISION,
    READ_MODEL_STATUS,
    READY_FOR_BLOCK_G_2,
    build_preview_paper_order_intent_read_model_snapshot,
)

PREVIEW_PAPER_ORDER_STATIC_FIXTURE_SCHEMA_VERSION: Final[str] = (
    "preview_paper_order_static_fixture.v1"
)
PREVIEW_PAPER_ORDER_STATIC_FIXTURE_KIND: Final[str] = (
    "functional_preview_block_g_paper_order_static_fixture"
)
BLOCK_ID: Final[str] = "G"
STEP_ID: Final[str] = "9.2"
STATIC_FIXTURE_STATUS: Final[str] = "paper_order_static_fixture_ready_no_order_generation"
STATIC_FIXTURE_DECISION: Final[str] = "BUILD_PAPER_ORDER_STATIC_FIXTURE_ONLY_NO_ORDER_GENERATION"
READY_FOR_BLOCK_G_3: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-9.3"
NEXT_STEP_TITLE: Final[str] = "PAPER ORDER AUDIT ENVELOPE"

CASE_STATUS: Final[str] = "static_fixture_case_ready_no_intent_no_order_no_execution"

FUTURE_STEPS: Final[tuple[str, ...]] = (
    "functional_preview_9_3_paper_order_audit_envelope",
    "functional_preview_9_4_ui_read_only_paper_order_surface",
    "functional_preview_9_5_controlled_paper_order_intent_selection_gate",
    "functional_preview_9_6_controlled_paper_order_intent_no_submission",
    "functional_preview_9_7_paper_fill_simulator_contract_static_only",
    "functional_preview_9_8_paper_order_lifecycle_audit",
    "functional_preview_9_9_block_g_closure_audit",
)

ADDED_BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
    "static fixture execution",
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

BOUNDARY_FALSE_EXTENSIONS: Final[tuple[str, ...]] = (
    "paper_order_generated",
    "paper_order_submitted",
    "paper_fill_simulated",
    "paper_runtime_execution_performed",
)

BOUNDARY_SNAPSHOT_KEYS: Final[tuple[str, ...]] = (
    "local_only",
    "paper_only",
    "read_model_only",
    "paper_order_intent_generated",
    "paper_order_generation_allowed_now",
    "paper_order_submission_allowed_now",
    "paper_fill_simulation_allowed_now",
    "paper_runtime_execution_allowed_now",
    "live_mode_allowed",
    "testnet_mode_allowed_initially",
    "account_fetch_allowed",
    "live_credentials_allowed",
    "secrets_read_allowed",
    "external_export_allowed",
    "qml_changes_allowed",
    "exe_packaging_in_scope",
    "exe_direction_preserved",
)


def _read_model_reference(read_model_snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": read_model_snapshot["schema_version"],
        "read_model_kind": read_model_snapshot["read_model_kind"],
        "read_model_status": read_model_snapshot["read_model_status"],
        "read_model_decision": read_model_snapshot["read_model_decision"],
        "ready_for_block_g_2": read_model_snapshot["ready_for_block_g_2"],
        "next_step": read_model_snapshot["next_step"],
    }


def _paper_order_fixture_preview() -> dict[str, bool | str]:
    return {
        "fixture_status": "static_fixture_only_no_order_generation",
        "paper_only": True,
        "local_only": True,
        "read_model_only": True,
        "static_fixture_only": True,
        "order_intent_generated": False,
        "order_generated": False,
        "order_submission_allowed": False,
        "fill_simulation_allowed": False,
        "runtime_execution_allowed": False,
        "live_execution_allowed": False,
        "testnet_execution_allowed": False,
    }


def _fixture_no_execution_evidence() -> dict[str, bool]:
    return {
        "decision_engine_execution_performed": False,
        "paper_order_intent_generated": False,
        "paper_order_generated": False,
        "paper_order_submitted": False,
        "paper_fill_simulated": False,
        "paper_runtime_execution_performed": False,
        "risk_governor_execution_performed": False,
        "trading_controller_touched": False,
        "decision_envelope_touched": False,
        "live_execution_performed": False,
        "testnet_execution_performed": False,
        "account_fetch_performed": False,
        "secrets_read_performed": False,
        "export_performed": False,
    }


def _boundary_checks_from_read_model(read_model_snapshot: dict[str, Any]) -> dict[str, bool]:
    boundary_checks = dict(read_model_snapshot["boundary_checks"])
    boundary_checks.update(
        {
            "static_fixture_only": True,
            "contract_only": False,
            "paper_order_intent_allowed_now": False,
            "paper_order_intent_generated": False,
            "risk_governor_execution_allowed_now": False,
            "risk_governor_execution_performed": False,
            "trading_controller_allowed": False,
            "decision_envelope_allowed": False,
            "strategy_execution_allowed": False,
            "ai_scoring_execution_allowed": False,
            "model_inference_execution_allowed": False,
            "runtime_loop_allowed": False,
            "command_dispatch_execution_allowed": False,
            "lifecycle_execution_allowed": False,
            "dynamic_action_dispatch_allowed": False,
            "new_qml_method_calls_allowed": False,
            "qml_changes_allowed": False,
            "exe_packaging_in_scope": False,
            "bat_productization_allowed": False,
            "exe_direction_preserved": True,
        }
    )
    for key in BOUNDARY_FALSE_EXTENSIONS:
        boundary_checks[key] = False
    return boundary_checks


def _boundary_snapshot(boundary_checks: dict[str, bool]) -> dict[str, bool]:
    return {key: boundary_checks[key] for key in BOUNDARY_SNAPSHOT_KEYS}


def _blocked_capabilities_from_read_model(read_model_snapshot: dict[str, Any]) -> list[str]:
    blocked = list(read_model_snapshot["blocked_capabilities"])
    for capability in ADDED_BLOCKED_CAPABILITIES:
        if capability not in blocked:
            blocked.append(capability)
    return blocked


def _build_fixture_case(
    case_id: str,
    case_description: str,
    input_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    read_model_snapshot = build_preview_paper_order_intent_read_model_snapshot(input_snapshot)
    boundary_checks = _boundary_checks_from_read_model(read_model_snapshot)
    return {
        "case_id": case_id,
        "case_description": case_description,
        "input_snapshot": deepcopy(read_model_snapshot["input_snapshot"]),
        "read_model_snapshot": read_model_snapshot,
        "paper_order_fixture_preview": _paper_order_fixture_preview(),
        "fixture_no_execution_evidence": _fixture_no_execution_evidence(),
        "boundary_snapshot": _boundary_snapshot(boundary_checks),
        "case_status": CASE_STATUS,
    }


def _fixture_summary(fixture_cases: list[dict[str, Any]]) -> dict[str, bool | int | str]:
    return {
        "case_count": len(fixture_cases),
        "all_cases_static_only": all(
            case["paper_order_fixture_preview"]["static_fixture_only"] is True
            for case in fixture_cases
        ),
        "all_cases_no_intent_generated": all(
            case["paper_order_fixture_preview"]["order_intent_generated"] is False
            and case["fixture_no_execution_evidence"]["paper_order_intent_generated"] is False
            for case in fixture_cases
        ),
        "all_cases_no_order_generated": all(
            case["paper_order_fixture_preview"]["order_generated"] is False
            and case["fixture_no_execution_evidence"]["paper_order_generated"] is False
            for case in fixture_cases
        ),
        "all_cases_no_submission": all(
            case["fixture_no_execution_evidence"]["paper_order_submitted"] is False
            for case in fixture_cases
        ),
        "all_cases_no_fills": all(
            case["fixture_no_execution_evidence"]["paper_fill_simulated"] is False
            for case in fixture_cases
        ),
        "all_cases_no_runtime_execution": all(
            case["fixture_no_execution_evidence"]["paper_runtime_execution_performed"] is False
            for case in fixture_cases
        ),
        "all_cases_no_live_or_testnet": all(
            case["fixture_no_execution_evidence"]["live_execution_performed"] is False
            and case["fixture_no_execution_evidence"]["testnet_execution_performed"] is False
            for case in fixture_cases
        ),
        "all_cases_no_account_or_secrets": all(
            case["fixture_no_execution_evidence"]["account_fetch_performed"] is False
            and case["fixture_no_execution_evidence"]["secrets_read_performed"] is False
            for case in fixture_cases
        ),
        "all_cases_no_export": all(
            case["fixture_no_execution_evidence"]["export_performed"] is False
            for case in fixture_cases
        ),
        "ready_for_audit_envelope_step": True,
        "next_step": NEXT_STEP,
    }


def build_preview_paper_order_static_fixture() -> dict[str, Any]:
    """Return deterministic JSON-serializable 9.2 paper static fixture data only."""

    baseline = _build_fixture_case(
        "baseline_btc_no_intent_no_order",
        "Default BTC/USDT read-model fixture with no intent, no order, and no execution.",
    )
    eth_size_preview = _build_fixture_case(
        "eth_size_preview_no_intent_no_order",
        "ETH/USDT size preview fixture with no intent, no order, and no execution.",
        {
            "operator_selected_pair": "ETH/USDT",
            "operator_selected_candidate": {
                "pair": "ETH/USDT",
                "source": "static_fixture_preview",
                "confidence": 0.15,
            },
            "dry_run_decision_preview": {
                "decision_action": "NO_ORDER_DRY_RUN_PREVIEW",
                "decision_status": "not_executed",
                "confidence_preview": 0.15,
            },
            "paper_order_intent_size_preview": {
                "value": 25.0,
                "unit": "preview_only",
                "source": "static_fixture_eth_size_preview",
            },
            "paper_order_intent_side_preview": "buy_preview_only",
            "paper_order_intent_type_preview": "market_preview_only",
        },
    )
    sol_risk_blocked = _build_fixture_case(
        "sol_risk_blocked_preview_no_intent_no_order",
        "SOL/USDT risk-blocked preview fixture without risk governor execution or order path.",
        {
            "operator_selected_pair": "SOL/USDT",
            "operator_selected_candidate": {
                "pair": "SOL/USDT",
                "source": "static_fixture_preview",
                "confidence": 0.05,
            },
            "dry_run_decision_preview": {
                "decision_action": "NO_ORDER_DRY_RUN_PREVIEW",
                "decision_status": "not_executed",
                "confidence_preview": 0.05,
            },
            "risk_check_preview": {
                "risk_status": "blocked_preview_only",
                "risk_engine_execution_performed": False,
            },
            "paper_order_intent_size_preview": {
                "value": 0.0,
                "unit": "preview_only",
                "source": "static_fixture_risk_blocked_preview",
            },
            "paper_order_intent_side_preview": "none",
            "paper_order_intent_type_preview": "none",
        },
    )
    unknown_keys = _build_fixture_case(
        "unknown_input_keys_reported_no_execution",
        "Unknown input keys are reported by the 9.1 read model and never executed.",
        {
            "operator_selected_pair": "BTC/USDT",
            "unsafe_submit_order_request": True,
            "live_credentials_reference": "blocked",
        },
    )

    fixture_cases = [baseline, eth_size_preview, sol_risk_blocked, unknown_keys]
    reference_snapshot = baseline["read_model_snapshot"]
    boundary_checks = _boundary_checks_from_read_model(reference_snapshot)

    return deepcopy(
        {
            "schema_version": PREVIEW_PAPER_ORDER_STATIC_FIXTURE_SCHEMA_VERSION,
            "fixture_kind": PREVIEW_PAPER_ORDER_STATIC_FIXTURE_KIND,
            "block": BLOCK_ID,
            "step": STEP_ID,
            "static_fixture_status": STATIC_FIXTURE_STATUS,
            "static_fixture_decision": STATIC_FIXTURE_DECISION,
            "ready_for_block_g_3": READY_FOR_BLOCK_G_3,
            "next_step": NEXT_STEP,
            "next_step_title": NEXT_STEP_TITLE,
            "read_model_reference": _read_model_reference(reference_snapshot),
            "fixture_cases": fixture_cases,
            "fixture_summary": _fixture_summary(fixture_cases),
            "boundary_checks": boundary_checks,
            "blocked_capabilities": _blocked_capabilities_from_read_model(reference_snapshot),
            "source_boundaries": list(reference_snapshot["source_boundaries"]),
            "future_steps": list(FUTURE_STEPS),
            "status": "ready_for_functional_preview_9_3_no_order_generation",
        }
    )


__all__ = [
    "BLOCK_ID",
    "NEXT_STEP",
    "NEXT_STEP_TITLE",
    "PREVIEW_PAPER_ORDER_STATIC_FIXTURE_KIND",
    "PREVIEW_PAPER_ORDER_STATIC_FIXTURE_SCHEMA_VERSION",
    "READY_FOR_BLOCK_G_3",
    "STATIC_FIXTURE_DECISION",
    "STATIC_FIXTURE_STATUS",
    "STEP_ID",
    "build_preview_paper_order_static_fixture",
]
