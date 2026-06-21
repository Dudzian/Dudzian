"""Pure-data BLOK G paper order audit envelope.

FUNCTIONAL-PREVIEW-9.3 builds deterministic audit events on top of the 9.2
paper order static fixture. It is audit-envelope-only: no order intent, order,
submission, fill, runtime, account, export, QML, or packaging path is executed.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_paper_order_static_fixture import (
    PREVIEW_PAPER_ORDER_STATIC_FIXTURE_KIND,
    PREVIEW_PAPER_ORDER_STATIC_FIXTURE_SCHEMA_VERSION,
    READY_FOR_BLOCK_G_3,
    STATIC_FIXTURE_DECISION,
    STATIC_FIXTURE_STATUS,
    build_preview_paper_order_static_fixture,
)

PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_SCHEMA_VERSION: Final[str] = (
    "preview_paper_order_audit_envelope.v1"
)
PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_KIND: Final[str] = (
    "functional_preview_block_g_paper_order_audit_envelope"
)
BLOCK_ID: Final[str] = "G"
STEP_ID: Final[str] = "9.3"
AUDIT_ENVELOPE_STATUS: Final[str] = "paper_order_audit_envelope_ready_no_order_generation"
AUDIT_ENVELOPE_DECISION: Final[str] = "BUILD_PAPER_ORDER_AUDIT_ENVELOPE_ONLY_NO_ORDER_GENERATION"
READY_FOR_BLOCK_G_4: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-9.4"
NEXT_STEP_TITLE: Final[str] = "PAPER ORDER UI READ-ONLY SURFACE"

AUDIT_EVENT_TYPE: Final[str] = "paper_order_static_fixture_case_audit"
AUDIT_EVENT_STATUS: Final[str] = "ready_no_intent_no_order_no_execution"
AUDIT_EVENT_IDS: Final[tuple[str, ...]] = (
    "paper-order-audit-0001-baseline-btc-no-intent-no-order",
    "paper-order-audit-0002-eth-size-preview-no-intent-no-order",
    "paper-order-audit-0003-sol-risk-blocked-preview-no-intent-no-order",
    "paper-order-audit-0004-unknown-input-keys-reported-no-execution",
)

FUTURE_STEPS: Final[tuple[str, ...]] = (
    "functional_preview_9_4_ui_read_only_paper_order_surface",
    "functional_preview_9_5_controlled_paper_order_intent_selection_gate",
    "functional_preview_9_6_controlled_paper_order_intent_no_submission",
    "functional_preview_9_7_paper_fill_simulator_contract_static_only",
    "functional_preview_9_8_paper_order_lifecycle_audit",
    "functional_preview_9_9_block_g_closure_audit",
)

ADDED_BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
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

AUDIT_BOUNDARY_EXTENSIONS: Final[dict[str, bool]] = {
    "audit_envelope_only": True,
    "static_fixture_only": True,
    "read_model_only": True,
    "contract_only": False,
    "decision_engine_execution_allowed": False,
    "decision_engine_execution_performed": False,
    "paper_order_intent_allowed_now": False,
    "paper_order_intent_generated": False,
    "paper_order_generation_allowed_now": False,
    "paper_order_generated": False,
    "paper_order_submission_allowed_now": False,
    "paper_order_submitted": False,
    "paper_fill_simulation_allowed_now": False,
    "paper_fill_simulated": False,
    "paper_runtime_execution_allowed_now": False,
    "paper_runtime_execution_performed": False,
    "risk_governor_execution_allowed_now": False,
    "risk_governor_execution_performed": False,
    "audit_event_generation_allowed": True,
    "audit_export_allowed": False,
    "audit_export_performed": False,
    "trading_controller_allowed": False,
    "decision_envelope_allowed": False,
    "strategy_execution_allowed": False,
    "ai_scoring_execution_allowed": False,
    "model_inference_execution_allowed": False,
    "runtime_loop_allowed": False,
    "command_dispatch_execution_allowed": False,
    "lifecycle_execution_allowed": False,
    "live_mode_allowed": False,
    "testnet_mode_allowed_initially": False,
    "account_fetch_allowed": False,
    "market_account_fetch_allowed": False,
    "real_account_balance_allowed": False,
    "live_credentials_allowed": False,
    "secrets_read_allowed": False,
    "secrets_export_allowed": False,
    "cloud_export_allowed": False,
    "external_export_allowed": False,
    "dynamic_action_dispatch_allowed": False,
    "new_qml_method_calls_allowed": False,
    "qml_changes_allowed": False,
    "exe_packaging_in_scope": False,
    "bat_productization_allowed": False,
    "exe_direction_preserved": True,
}


def _static_fixture_reference() -> dict[str, Any]:
    return {
        "schema_version": PREVIEW_PAPER_ORDER_STATIC_FIXTURE_SCHEMA_VERSION,
        "fixture_kind": PREVIEW_PAPER_ORDER_STATIC_FIXTURE_KIND,
        "static_fixture_status": STATIC_FIXTURE_STATUS,
        "static_fixture_decision": STATIC_FIXTURE_DECISION,
        "ready_for_block_g_3": READY_FOR_BLOCK_G_3,
        "next_step": "FUNCTIONAL-PREVIEW-9.3",
    }


def _read_model_reference(read_model_snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": read_model_snapshot["schema_version"],
        "read_model_kind": read_model_snapshot["read_model_kind"],
        "read_model_status": read_model_snapshot["read_model_status"],
        "read_model_decision": read_model_snapshot["read_model_decision"],
        "ready_for_block_g_2": read_model_snapshot["ready_for_block_g_2"],
        "next_step": read_model_snapshot["next_step"],
    }


def _paper_order_audit_preview() -> dict[str, bool | str]:
    return {
        "audit_preview_status": "static_audit_only_no_export",
        "paper_only": True,
        "local_only": True,
        "static_fixture_only": True,
        "audit_event_generated": True,
        "audit_export_allowed": False,
        "audit_export_performed": False,
        "order_intent_generated": False,
        "order_generated": False,
        "order_submitted": False,
        "fill_simulated": False,
        "runtime_execution_performed": False,
        "live_execution_performed": False,
        "testnet_execution_performed": False,
    }


def _audit_export_status() -> dict[str, bool]:
    return {
        "export_allowed": False,
        "export_performed": False,
        "cloud_export_allowed": False,
        "external_export_allowed": False,
        "secrets_export_allowed": False,
    }


def _unknown_input_keys(case: dict[str, Any]) -> list[str]:
    return list(case["read_model_snapshot"]["input_snapshot_echo"]["unknown_input_keys"])


def _build_audit_event(event_id: str, case: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "event_type": AUDIT_EVENT_TYPE,
        "event_status": AUDIT_EVENT_STATUS,
        "case_id": case["case_id"],
        "case_description": case["case_description"],
        "input_snapshot": deepcopy(case["input_snapshot"]),
        "read_model_reference": _read_model_reference(case["read_model_snapshot"]),
        "paper_order_fixture_preview": deepcopy(case["paper_order_fixture_preview"]),
        "paper_order_audit_preview": _paper_order_audit_preview(),
        "no_execution_evidence": deepcopy(case["fixture_no_execution_evidence"]),
        "boundary_snapshot": deepcopy(case["boundary_snapshot"]),
        "unknown_input_keys": _unknown_input_keys(case),
        "audit_export_status": _audit_export_status(),
    }


def _boundary_checks(static_fixture: dict[str, Any]) -> dict[str, bool]:
    boundary_checks = dict(static_fixture["boundary_checks"])
    boundary_checks.update(AUDIT_BOUNDARY_EXTENSIONS)
    return boundary_checks


def _blocked_capabilities(static_fixture: dict[str, Any]) -> list[str]:
    blocked = list(static_fixture["blocked_capabilities"])
    for capability in ADDED_BLOCKED_CAPABILITIES:
        if capability not in blocked:
            blocked.append(capability)
    return blocked


def _audit_summary(audit_events: list[dict[str, Any]]) -> dict[str, bool | int | str]:
    return {
        "event_count": len(audit_events),
        "all_events_static_audit_only": all(
            event["paper_order_audit_preview"]["static_fixture_only"] is True
            and event["paper_order_audit_preview"]["audit_event_generated"] is True
            for event in audit_events
        ),
        "all_events_no_intent_generated": all(
            event["paper_order_audit_preview"]["order_intent_generated"] is False
            and event["no_execution_evidence"]["paper_order_intent_generated"] is False
            for event in audit_events
        ),
        "all_events_no_order_generated": all(
            event["paper_order_audit_preview"]["order_generated"] is False
            and event["no_execution_evidence"]["paper_order_generated"] is False
            for event in audit_events
        ),
        "all_events_no_submission": all(
            event["paper_order_audit_preview"]["order_submitted"] is False
            and event["no_execution_evidence"]["paper_order_submitted"] is False
            for event in audit_events
        ),
        "all_events_no_fills": all(
            event["paper_order_audit_preview"]["fill_simulated"] is False
            and event["no_execution_evidence"]["paper_fill_simulated"] is False
            for event in audit_events
        ),
        "all_events_no_runtime_execution": all(
            event["paper_order_audit_preview"]["runtime_execution_performed"] is False
            and event["no_execution_evidence"]["paper_runtime_execution_performed"] is False
            for event in audit_events
        ),
        "all_events_no_live_or_testnet": all(
            event["paper_order_audit_preview"]["live_execution_performed"] is False
            and event["paper_order_audit_preview"]["testnet_execution_performed"] is False
            and event["no_execution_evidence"]["live_execution_performed"] is False
            and event["no_execution_evidence"]["testnet_execution_performed"] is False
            for event in audit_events
        ),
        "all_events_no_account_or_secrets": all(
            event["no_execution_evidence"]["account_fetch_performed"] is False
            and event["no_execution_evidence"]["secrets_read_performed"] is False
            for event in audit_events
        ),
        "all_events_no_export": all(
            event["audit_export_status"]["export_performed"] is False
            and event["audit_export_status"]["export_allowed"] is False
            and event["no_execution_evidence"]["export_performed"] is False
            for event in audit_events
        ),
        "unknown_input_key_events": sum(1 for event in audit_events if event["unknown_input_keys"]),
        "ready_for_ui_read_only_surface_step": True,
        "next_step": NEXT_STEP,
    }


def build_preview_paper_order_audit_envelope() -> dict[str, Any]:
    """Return deterministic JSON-serializable 9.3 paper audit envelope data only."""

    static_fixture = build_preview_paper_order_static_fixture()
    audit_events = [
        _build_audit_event(event_id, case)
        for event_id, case in zip(AUDIT_EVENT_IDS, static_fixture["fixture_cases"], strict=True)
    ]

    return deepcopy(
        {
            "schema_version": PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_SCHEMA_VERSION,
            "envelope_kind": PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_KIND,
            "block": BLOCK_ID,
            "step": STEP_ID,
            "audit_envelope_status": AUDIT_ENVELOPE_STATUS,
            "audit_envelope_decision": AUDIT_ENVELOPE_DECISION,
            "ready_for_block_g_4": READY_FOR_BLOCK_G_4,
            "next_step": NEXT_STEP,
            "next_step_title": NEXT_STEP_TITLE,
            "static_fixture_reference": _static_fixture_reference(),
            "audit_events": audit_events,
            "audit_summary": _audit_summary(audit_events),
            "boundary_checks": _boundary_checks(static_fixture),
            "blocked_capabilities": _blocked_capabilities(static_fixture),
            "source_boundaries": list(static_fixture["source_boundaries"]),
            "future_steps": list(FUTURE_STEPS),
            "status": "ready_for_functional_preview_9_4_no_order_generation",
        }
    )


__all__ = [
    "AUDIT_ENVELOPE_DECISION",
    "AUDIT_ENVELOPE_STATUS",
    "BLOCK_ID",
    "NEXT_STEP",
    "NEXT_STEP_TITLE",
    "PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_KIND",
    "PREVIEW_PAPER_ORDER_AUDIT_ENVELOPE_SCHEMA_VERSION",
    "READY_FOR_BLOCK_G_4",
    "STEP_ID",
    "build_preview_paper_order_audit_envelope",
]
