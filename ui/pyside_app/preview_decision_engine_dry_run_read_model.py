"""Pure-data BLOK F decision engine dry-run read model snapshot.

This module is intentionally inert. It builds a deterministic JSON-serializable
snapshot for a future local/paper decision preview without importing or running
the decision engine, runtime loops, controllers, adapters, orders, secrets, or
UI/QML bindings.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_decision_engine_dry_run_contract import (
    ALLOWED_DRY_RUN_INPUTS,
    ALLOWED_DRY_RUN_OUTPUTS,
    BLOCKED_CAPABILITIES as CONTRACT_BLOCKED_CAPABILITIES,
    DRY_RUN_MODE,
    PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_KIND,
    PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_SCHEMA_VERSION,
    REQUIRED_BOUNDARIES,
    SOURCE_BOUNDARIES,
    build_preview_decision_engine_dry_run_contract,
)

PREVIEW_DECISION_ENGINE_DRY_RUN_READ_MODEL_SCHEMA_VERSION: Final[str] = (
    "preview_decision_engine_dry_run_read_model.v1"
)
PREVIEW_DECISION_ENGINE_DRY_RUN_READ_MODEL_KIND: Final[str] = (
    "functional_preview_block_f_decision_engine_dry_run_read_model"
)
BLOCK_ID: Final[str] = "F"
STEP_ID: Final[str] = "8.1"
READ_MODEL_STATUS: Final[str] = "read_model_snapshot_ready_no_engine_execution"
READ_MODEL_DECISION: Final[str] = "BUILD_READ_MODEL_ONLY_NO_ENGINE_EXECUTION"
READY_FOR_BLOCK_F_2: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-8.2"
NEXT_STEP_TITLE: Final[str] = "DECISION ENGINE DRY-RUN STATIC FIXTURE"

DEFAULT_INPUT_SNAPSHOT: Final[dict[str, Any]] = {
    "dry_run_context_id": "local-preview-dry-run-context",
    "operator_selected_pair": "BTC/USDT",
    "operator_selected_candidate": {
        "pair": "BTC/USDT",
        "source": "local_preview_default",
        "confidence": 0.0,
    },
    "local_preview_state_snapshot": {
        "source": "local_preview_state_snapshot",
        "available": False,
    },
    "paper_runtime_snapshot": {
        "source": "paper_runtime_snapshot",
        "available": False,
    },
    "scanner_candidate_snapshot": {
        "source": "scanner_candidate_snapshot",
        "available": False,
    },
    "risk_preview_snapshot": {
        "source": "risk_preview_snapshot",
        "available": False,
    },
    "portfolio_preview_snapshot": {
        "source": "portfolio_preview_snapshot",
        "available": False,
    },
}

READ_MODEL_FUTURE_STEPS: Final[tuple[str, ...]] = (
    "functional_preview_8_2_decision_engine_dry_run_static_fixture",
    "functional_preview_8_3_decision_engine_dry_run_audit_envelope",
    "functional_preview_8_4_decision_engine_dry_run_ui_read_only_surface",
    "functional_preview_8_5_block_f_closure_audit",
)

READ_MODEL_ADDITIONAL_BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
    "real decision recommendation",
    "model inference",
    "risk engine evaluation",
)

READ_MODEL_ALLOWED_OUTPUT_KEYS: Final[tuple[str, ...]] = (
    *ALLOWED_DRY_RUN_OUTPUTS,
    "read_model",
    "decision_preview",
    "risk_check_preview",
    "audit_preview",
)


def _normalize_input_snapshot(
    input_snapshot: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str]]:
    defaults = deepcopy(DEFAULT_INPUT_SNAPSHOT)
    if input_snapshot is None:
        return defaults, []

    allowed = set(ALLOWED_DRY_RUN_INPUTS)
    ignored_input_keys = sorted(key for key in input_snapshot if key not in allowed)
    for key in ALLOWED_DRY_RUN_INPUTS:
        if key in input_snapshot:
            defaults[key] = deepcopy(input_snapshot[key])
    return defaults, ignored_input_keys


def _build_contract_reference() -> dict[str, Any]:
    contract = build_preview_decision_engine_dry_run_contract()
    return {
        "schema_version": PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_SCHEMA_VERSION,
        "contract_kind": PREVIEW_DECISION_ENGINE_DRY_RUN_CONTRACT_KIND,
        "block_status": contract["block_status"],
        "contract_decision": contract["contract_decision"],
    }


def build_preview_decision_engine_dry_run_read_model_snapshot(
    input_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return deterministic plain-data 8.1 read model snapshot.

    The returned data is a read-model snapshot only. It echoes normalized input
    shape and records no-execution placeholders; it never scores, infers,
    submits orders, starts runtime loops, performs I/O, or mutates caller data.
    """

    normalized_input_snapshot, ignored_input_keys = _normalize_input_snapshot(input_snapshot)
    input_keys_present = [key for key in ALLOWED_DRY_RUN_INPUTS if key in normalized_input_snapshot]
    input_keys_missing: list[str] = []
    boundary_checks = dict(REQUIRED_BOUNDARIES)
    boundary_checks["model_inference_execution_allowed"] = False

    snapshot: dict[str, Any] = {
        "schema_version": PREVIEW_DECISION_ENGINE_DRY_RUN_READ_MODEL_SCHEMA_VERSION,
        "read_model_kind": PREVIEW_DECISION_ENGINE_DRY_RUN_READ_MODEL_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "read_model_status": READ_MODEL_STATUS,
        "dry_run_mode": DRY_RUN_MODE,
        "read_model_decision": READ_MODEL_DECISION,
        "ready_for_block_f_2": READY_FOR_BLOCK_F_2,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "contract_reference": _build_contract_reference(),
        "input_snapshot": deepcopy(normalized_input_snapshot),
        "input_snapshot_echo": deepcopy(normalized_input_snapshot),
        "read_model": {
            "read_model_ready": True,
            "engine_execution_required": False,
            "engine_execution_performed": False,
            "model_source": "contract_only_static_read_model",
            "input_keys_present": input_keys_present,
            "input_keys_missing": input_keys_missing,
            "ignored_input_keys": ignored_input_keys,
            "allowed_input_keys": list(ALLOWED_DRY_RUN_INPUTS),
            "allowed_output_keys": list(READ_MODEL_ALLOWED_OUTPUT_KEYS),
        },
        "decision_preview": {
            "decision_preview_ready": True,
            "decision_source": "dry_run_read_model_no_engine",
            "decision_action": "NO_ORDER_DRY_RUN_PREVIEW",
            "decision_status": "not_executed",
            "confidence_preview": 0.0,
            "reason_summary": ("Decision engine execution is disabled in FUNCTIONAL-PREVIEW-8.1."),
            "order_generation_allowed": False,
            "order_submission_allowed": False,
            "execution_performed": False,
        },
        "risk_check_preview": {
            "risk_check_preview_ready": True,
            "risk_engine_execution_performed": False,
            "risk_status": "not_evaluated_contract_only",
            "blocked_reason_preview": (
                "Risk engine execution is disabled in FUNCTIONAL-PREVIEW-8.1."
            ),
        },
        "audit_preview": {
            "audit_event_preview_ready": True,
            "audit_event_type": "decision_engine_dry_run_read_model_snapshot",
            "audit_export_allowed": False,
            "cloud_export_allowed": False,
            "external_export_allowed": False,
        },
        "boundary_checks": boundary_checks,
        "blocked_capabilities": list(
            dict.fromkeys(
                (*CONTRACT_BLOCKED_CAPABILITIES, *READ_MODEL_ADDITIONAL_BLOCKED_CAPABILITIES)
            )
        ),
        "source_boundaries": list(SOURCE_BOUNDARIES),
        "future_steps": list(READ_MODEL_FUTURE_STEPS),
        "status": "ready_no_engine_execution_no_orders",
    }
    return deepcopy(snapshot)


__all__ = [
    "BLOCK_ID",
    "DRY_RUN_MODE",
    "NEXT_STEP",
    "NEXT_STEP_TITLE",
    "PREVIEW_DECISION_ENGINE_DRY_RUN_READ_MODEL_KIND",
    "PREVIEW_DECISION_ENGINE_DRY_RUN_READ_MODEL_SCHEMA_VERSION",
    "READ_MODEL_DECISION",
    "READ_MODEL_STATUS",
    "READY_FOR_BLOCK_F_2",
    "STEP_ID",
    "build_preview_decision_engine_dry_run_read_model_snapshot",
]
