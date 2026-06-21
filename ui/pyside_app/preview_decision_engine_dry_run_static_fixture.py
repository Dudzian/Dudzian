"""Pure-data BLOK F decision engine dry-run static fixture.

This module is intentionally inert. It builds deterministic JSON-serializable
fixture cases for a future local/paper decision preview by reusing the 8.1 read
model helper. It does not execute or import the real decision engine, runtime
loops, controllers, adapters, orders, secrets, or UI/QML bindings.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_decision_engine_dry_run_contract import (
    BLOCKED_CAPABILITIES as CONTRACT_BLOCKED_CAPABILITIES,
    CONTRACT_DECISION,
    DRY_RUN_MODE,
    PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_KIND,
    PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_SCHEMA_VERSION,
    REQUIRED_BOUNDARIES,
    SOURCE_BOUNDARIES,
)
from ui.pyside_app.preview_decision_engine_dry_run_read_model import (
    PREVIEW_DECISION_ENGINE_DRY_RUN_READ_MODEL_KIND,
    PREVIEW_DECISION_ENGINE_DRY_RUN_READ_MODEL_SCHEMA_VERSION,
    READ_MODEL_DECISION,
    READ_MODEL_STATUS,
    build_preview_decision_engine_dry_run_read_model_snapshot,
)

PREVIEW_DECISION_ENGINE_DRY_RUN_STATIC_FIXTURE_SCHEMA_VERSION: Final[str] = (
    "preview_decision_engine_dry_run_static_fixture.v1"
)
PREVIEW_DECISION_ENGINE_DRY_RUN_STATIC_FIXTURE_KIND: Final[str] = (
    "functional_preview_block_f_decision_engine_dry_run_static_fixture"
)
BLOCK_ID: Final[str] = "F"
STEP_ID: Final[str] = "8.2"
STATIC_FIXTURE_STATUS: Final[str] = "static_fixture_ready_no_engine_execution"
STATIC_FIXTURE_DECISION: Final[str] = "BUILD_STATIC_FIXTURE_ONLY_NO_ENGINE_EXECUTION"
READY_FOR_BLOCK_F_3: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-8.3"
NEXT_STEP_TITLE: Final[str] = "DECISION ENGINE DRY-RUN AUDIT ENVELOPE"

EXPECTED_DECISION_ACTION: Final[str] = "NO_ORDER_DRY_RUN_PREVIEW"
EXPECTED_DECISION_STATUS: Final[str] = "not_executed"
CASE_STATUS: Final[str] = "ready_no_engine_execution_no_orders"

STATIC_FIXTURE_CASE_DEFINITIONS: Final[tuple[dict[str, Any], ...]] = (
    {
        "case_id": "baseline_btc_no_order",
        "description": "Baseline BTC/USDT local paper dry-run fixture; no order path is allowed.",
        "input_snapshot": {
            "dry_run_context_id": "static-fixture-baseline-btc",
            "operator_selected_pair": "BTC/USDT",
            "operator_selected_candidate": {
                "pair": "BTC/USDT",
                "source": "static_fixture_baseline",
                "confidence": 0.0,
            },
            "scanner_candidate_snapshot": {
                "source": "static_fixture_baseline",
                "available": True,
                "pair": "BTC/USDT",
                "confidence": 0.0,
            },
        },
    },
    {
        "case_id": "scanner_eth_candidate_no_order",
        "description": "ETH/USDT scanner candidate dry-run fixture; candidate is data-only.",
        "input_snapshot": {
            "dry_run_context_id": "static-fixture-scanner-eth",
            "operator_selected_pair": "ETH/USDT",
            "operator_selected_candidate": {
                "pair": "ETH/USDT",
                "source": "static_fixture_scanner_candidate",
                "confidence": 0.25,
            },
            "scanner_candidate_snapshot": {
                "source": "static_fixture_scanner_candidate",
                "available": True,
                "pair": "ETH/USDT",
                "confidence": 0.25,
            },
        },
    },
    {
        "case_id": "risk_blocked_sol_no_order",
        "description": "SOL/USDT risk-blocked placeholder; risk is not evaluated contract-only.",
        "input_snapshot": {
            "dry_run_context_id": "static-fixture-risk-blocked-sol",
            "operator_selected_pair": "SOL/USDT",
            "operator_selected_candidate": {
                "pair": "SOL/USDT",
                "source": "static_fixture_risk_blocked",
                "confidence": 0.0,
            },
            "scanner_candidate_snapshot": {
                "source": "static_fixture_risk_blocked",
                "available": True,
                "pair": "SOL/USDT",
                "confidence": 0.0,
            },
            "risk_preview_snapshot": {
                "source": "static_fixture_risk_blocked",
                "available": True,
                "risk_status": "blocked_not_evaluated_contract_only",
                "blocked_reason_preview": "blocked/not evaluated contract-only",
            },
        },
    },
)

STATIC_FIXTURE_ADDITIONAL_BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
    "real decision recommendation",
    "model inference",
    "risk engine evaluation",
)

STATIC_FIXTURE_FUTURE_STEPS: Final[tuple[str, ...]] = (
    "functional_preview_8_3_decision_engine_dry_run_audit_envelope",
    "functional_preview_8_4_decision_engine_dry_run_ui_read_only_surface",
    "functional_preview_8_5_block_f_closure_audit",
)


def _build_contract_reference() -> dict[str, str]:
    return {
        "schema_version": PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_SCHEMA_VERSION,
        "contract_kind": PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_KIND,
        "block_status": "decision_engine_dry_run_contract_ready_no_execution",
        "contract_decision": CONTRACT_DECISION,
    }


def _build_read_model_reference() -> dict[str, str]:
    return {
        "schema_version": PREVIEW_DECISION_ENGINE_DRY_RUN_READ_MODEL_SCHEMA_VERSION,
        "read_model_kind": PREVIEW_DECISION_ENGINE_DRY_RUN_READ_MODEL_KIND,
        "read_model_status": READ_MODEL_STATUS,
        "read_model_decision": READ_MODEL_DECISION,
    }


def _build_fixture_case(case_definition: dict[str, Any]) -> dict[str, Any]:
    input_snapshot = deepcopy(case_definition["input_snapshot"])
    return {
        "case_id": case_definition["case_id"],
        "description": case_definition["description"],
        "input_snapshot": deepcopy(input_snapshot),
        "read_model_snapshot": build_preview_decision_engine_dry_run_read_model_snapshot(
            input_snapshot
        ),
        "expected_decision_action": EXPECTED_DECISION_ACTION,
        "expected_decision_status": EXPECTED_DECISION_STATUS,
        "expected_order_generation_allowed": False,
        "expected_order_submission_allowed": False,
        "expected_execution_performed": False,
        "expected_risk_engine_execution_performed": False,
        "expected_audit_export_allowed": False,
        "case_status": CASE_STATUS,
    }


def build_preview_decision_engine_dry_run_static_fixture() -> dict[str, Any]:
    """Return deterministic plain-data 8.2 static fixture for dry-run preview."""

    fixture_cases = [_build_fixture_case(case) for case in STATIC_FIXTURE_CASE_DEFINITIONS]
    boundary_checks = dict(REQUIRED_BOUNDARIES)
    boundary_checks["model_inference_execution_allowed"] = False
    fixture: dict[str, Any] = {
        "schema_version": PREVIEW_DECISION_ENGINE_DRY_RUN_STATIC_FIXTURE_SCHEMA_VERSION,
        "fixture_kind": PREVIEW_DECISION_ENGINE_DRY_RUN_STATIC_FIXTURE_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "static_fixture_status": STATIC_FIXTURE_STATUS,
        "dry_run_mode": DRY_RUN_MODE,
        "static_fixture_decision": STATIC_FIXTURE_DECISION,
        "ready_for_block_f_3": READY_FOR_BLOCK_F_3,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "contract_reference": _build_contract_reference(),
        "read_model_reference": _build_read_model_reference(),
        "fixture_cases": fixture_cases,
        "fixture_summary": {
            "fixture_case_count": len(fixture_cases),
            "all_cases_no_order": all(
                not case["expected_order_generation_allowed"]
                and not case["expected_order_submission_allowed"]
                for case in fixture_cases
            ),
            "all_cases_no_execution": all(
                not case["expected_execution_performed"] for case in fixture_cases
            ),
            "all_cases_no_risk_engine_execution": all(
                not case["expected_risk_engine_execution_performed"] for case in fixture_cases
            ),
            "all_cases_no_export": all(
                not case["expected_audit_export_allowed"] for case in fixture_cases
            ),
            "all_cases_json_serializable": True,
            "engine_execution_performed": False,
            "order_generation_allowed": False,
            "order_submission_allowed": False,
            "live_mode_allowed": False,
            "testnet_mode_allowed": False,
        },
        "boundary_checks": boundary_checks,
        "blocked_capabilities": list(
            dict.fromkeys(
                (*CONTRACT_BLOCKED_CAPABILITIES, *STATIC_FIXTURE_ADDITIONAL_BLOCKED_CAPABILITIES)
            )
        ),
        "source_boundaries": list(SOURCE_BOUNDARIES),
        "future_steps": list(STATIC_FIXTURE_FUTURE_STEPS),
        "status": CASE_STATUS,
    }
    return deepcopy(fixture)


__all__ = [
    "BLOCK_ID",
    "DRY_RUN_MODE",
    "NEXT_STEP",
    "NEXT_STEP_TITLE",
    "PREVIEW_DECISION_ENGINE_DRY_RUN_STATIC_FIXTURE_KIND",
    "PREVIEW_DECISION_ENGINE_DRY_RUN_STATIC_FIXTURE_SCHEMA_VERSION",
    "READY_FOR_BLOCK_F_3",
    "STATIC_FIXTURE_DECISION",
    "STATIC_FIXTURE_STATUS",
    "STEP_ID",
    "build_preview_decision_engine_dry_run_static_fixture",
]
