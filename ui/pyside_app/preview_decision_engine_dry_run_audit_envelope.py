"""Pure-data BLOK F decision engine dry-run audit envelope.

This module is intentionally inert. It builds deterministic JSON-serializable
audit events from the 8.2 static fixture for a future read-only UI surface. It
does not execute or import the real decision engine, runtime loops, controllers,
adapters, orders, secrets, exports, or UI/QML bindings.
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
)
from ui.pyside_app.preview_decision_engine_dry_run_static_fixture import (
    PREVIEW_DECISION_ENGINE_DRY_RUN_STATIC_FIXTURE_KIND,
    PREVIEW_DECISION_ENGINE_DRY_RUN_STATIC_FIXTURE_SCHEMA_VERSION,
    STATIC_FIXTURE_DECISION,
    STATIC_FIXTURE_STATUS,
    build_preview_decision_engine_dry_run_static_fixture,
)

PREVIEW_DECISION_ENGINE_DRY_RUN_AUDIT_ENVELOPE_SCHEMA_VERSION: Final[str] = (
    "preview_decision_engine_dry_run_audit_envelope.v1"
)
PREVIEW_DECISION_ENGINE_DRY_RUN_AUDIT_ENVELOPE_KIND: Final[str] = (
    "functional_preview_block_f_decision_engine_dry_run_audit_envelope"
)
BLOCK_ID: Final[str] = "F"
STEP_ID: Final[str] = "8.3"
AUDIT_ENVELOPE_STATUS: Final[str] = "audit_envelope_ready_no_engine_execution"
AUDIT_ENVELOPE_DECISION: Final[str] = "BUILD_AUDIT_ENVELOPE_ONLY_NO_ENGINE_EXECUTION"
READY_FOR_BLOCK_F_4: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-8.4"
NEXT_STEP_TITLE: Final[str] = "DECISION ENGINE DRY-RUN UI READ-ONLY SURFACE"

AUDIT_EVENT_TYPE: Final[str] = "decision_engine_dry_run_fixture_case_audit"
AUDIT_EVENT_STATUS: Final[str] = "ready_no_engine_execution_no_orders"

AUDIT_ENVELOPE_ADDITIONAL_BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
    "real decision recommendation",
    "model inference",
    "risk engine evaluation",
)

AUDIT_ENVELOPE_FUTURE_STEPS: Final[tuple[str, ...]] = (
    "functional_preview_8_4_decision_engine_dry_run_ui_read_only_surface",
    "functional_preview_8_5_block_f_closure_audit",
)

NO_EXECUTION_EVIDENCE: Final[dict[str, bool]] = {
    "decision_engine_execution_performed": False,
    "risk_engine_execution_performed": False,
    "model_inference_execution_allowed": False,
    "order_generation_allowed": False,
    "order_submission_allowed": False,
    "fills_allowed": False,
    "live_mode_allowed": False,
    "testnet_mode_allowed": False,
    "audit_export_allowed": False,
    "cloud_export_allowed": False,
    "external_export_allowed": False,
}


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


def _build_static_fixture_reference() -> dict[str, str]:
    return {
        "schema_version": PREVIEW_DECISION_ENGINE_DRY_RUN_STATIC_FIXTURE_SCHEMA_VERSION,
        "fixture_kind": PREVIEW_DECISION_ENGINE_DRY_RUN_STATIC_FIXTURE_KIND,
        "static_fixture_status": STATIC_FIXTURE_STATUS,
        "static_fixture_decision": STATIC_FIXTURE_DECISION,
    }


def _build_audit_event(index: int, fixture_case: dict[str, Any]) -> dict[str, Any]:
    read_model_snapshot = fixture_case["read_model_snapshot"]
    input_snapshot = fixture_case["input_snapshot"]
    case_id = fixture_case["case_id"]
    return {
        "audit_event_id": f"dry-run-audit-{index:04d}-{case_id.replace('_', '-')}",
        "audit_event_type": AUDIT_EVENT_TYPE,
        "case_id": case_id,
        "case_description": fixture_case["description"],
        "dry_run_context_id": input_snapshot["dry_run_context_id"],
        "operator_selected_pair": input_snapshot["operator_selected_pair"],
        "operator_selected_candidate": deepcopy(input_snapshot["operator_selected_candidate"]),
        "decision_preview": deepcopy(read_model_snapshot["decision_preview"]),
        "risk_check_preview": deepcopy(read_model_snapshot["risk_check_preview"]),
        "audit_preview": deepcopy(read_model_snapshot["audit_preview"]),
        "boundary_snapshot": deepcopy(read_model_snapshot["boundary_checks"]),
        "no_execution_evidence": dict(NO_EXECUTION_EVIDENCE),
        "event_status": AUDIT_EVENT_STATUS,
    }


def build_preview_decision_engine_dry_run_audit_envelope() -> dict[str, Any]:
    """Return deterministic plain-data 8.3 audit envelope for dry-run preview."""

    static_fixture = build_preview_decision_engine_dry_run_static_fixture()
    fixture_cases = static_fixture["fixture_cases"]
    audit_events = [
        _build_audit_event(index, fixture_case)
        for index, fixture_case in enumerate(fixture_cases, start=1)
    ]
    boundary_checks = dict(REQUIRED_BOUNDARIES)
    boundary_checks["model_inference_execution_allowed"] = False
    envelope: dict[str, Any] = {
        "schema_version": PREVIEW_DECISION_ENGINE_DRY_RUN_AUDIT_ENVELOPE_SCHEMA_VERSION,
        "envelope_kind": PREVIEW_DECISION_ENGINE_DRY_RUN_AUDIT_ENVELOPE_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "audit_envelope_status": AUDIT_ENVELOPE_STATUS,
        "dry_run_mode": DRY_RUN_MODE,
        "audit_envelope_decision": AUDIT_ENVELOPE_DECISION,
        "ready_for_block_f_4": READY_FOR_BLOCK_F_4,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "contract_reference": _build_contract_reference(),
        "read_model_reference": _build_read_model_reference(),
        "static_fixture_reference": _build_static_fixture_reference(),
        "audit_events": audit_events,
        "audit_summary": {
            "audit_event_count": len(audit_events),
            "fixture_case_count": len(fixture_cases),
            "all_events_have_deterministic_ids": all(
                event["audit_event_id"]
                == f"dry-run-audit-{index:04d}-{event['case_id'].replace('_', '-')}"
                for index, event in enumerate(audit_events, start=1)
            ),
            "all_events_no_engine_execution": all(
                not event["no_execution_evidence"]["decision_engine_execution_performed"]
                for event in audit_events
            ),
            "all_events_no_risk_engine_execution": all(
                not event["no_execution_evidence"]["risk_engine_execution_performed"]
                for event in audit_events
            ),
            "all_events_no_model_inference": all(
                not event["no_execution_evidence"]["model_inference_execution_allowed"]
                for event in audit_events
            ),
            "all_events_no_order_generation": all(
                not event["no_execution_evidence"]["order_generation_allowed"]
                for event in audit_events
            ),
            "all_events_no_order_submission": all(
                not event["no_execution_evidence"]["order_submission_allowed"]
                for event in audit_events
            ),
            "all_events_no_fills": all(
                not event["no_execution_evidence"]["fills_allowed"] for event in audit_events
            ),
            "all_events_no_live": all(
                not event["no_execution_evidence"]["live_mode_allowed"] for event in audit_events
            ),
            "all_events_no_testnet": all(
                not event["no_execution_evidence"]["testnet_mode_allowed"] for event in audit_events
            ),
            "all_events_no_export": all(
                not event["no_execution_evidence"]["audit_export_allowed"]
                and not event["no_execution_evidence"]["cloud_export_allowed"]
                and not event["no_execution_evidence"]["external_export_allowed"]
                for event in audit_events
            ),
            "all_events_json_serializable": True,
            "engine_execution_performed": False,
            "order_generation_allowed": False,
            "order_submission_allowed": False,
            "live_mode_allowed": False,
            "testnet_mode_allowed": False,
        },
        "boundary_checks": boundary_checks,
        "blocked_capabilities": list(
            dict.fromkeys(
                (
                    *CONTRACT_BLOCKED_CAPABILITIES,
                    *AUDIT_ENVELOPE_ADDITIONAL_BLOCKED_CAPABILITIES,
                )
            )
        ),
        "source_boundaries": list(SOURCE_BOUNDARIES),
        "future_steps": list(AUDIT_ENVELOPE_FUTURE_STEPS),
        "status": AUDIT_EVENT_STATUS,
    }
    return deepcopy(envelope)


__all__ = [
    "AUDIT_ENVELOPE_DECISION",
    "AUDIT_ENVELOPE_STATUS",
    "BLOCK_ID",
    "DRY_RUN_MODE",
    "NEXT_STEP",
    "NEXT_STEP_TITLE",
    "PREVIEW_DECISION_ENGINE_DRY_RUN_AUDIT_ENVELOPE_KIND",
    "PREVIEW_DECISION_ENGINE_DRY_RUN_AUDIT_ENVELOPE_SCHEMA_VERSION",
    "READY_FOR_BLOCK_F_4",
    "STEP_ID",
    "build_preview_decision_engine_dry_run_audit_envelope",
]
