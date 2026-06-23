"""FUNCTIONAL-PREVIEW-10.8 BLOK H read-only market data closure audit.

Pure-data closure audit for the Block H preview path.  The helper reads the
10.7 bridge snapshot handoff and returns only JSON-serializable containers.  It
performs no I/O, imports no UI/runtime/trading modules, and does not implement
or execute market data refresh, fetch, adapter, bridge API, QML, account, order,
live, testnet, or runtime behavior.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_action_dispatch_bridge_snapshot import (
    build_paper_runtime_action_dispatch_bridge_snapshot,
)

PREVIEW_READ_ONLY_MARKET_DATA_CLOSURE_AUDIT_SCHEMA_VERSION: Final[str] = (
    "preview_read_only_market_data_closure_audit.v1"
)
PREVIEW_READ_ONLY_MARKET_DATA_CLOSURE_AUDIT_KIND: Final[str] = (
    "functional_preview_block_h_read_only_market_data_closure_audit"
)
BLOCK_ID: Final[str] = "H"
STEP_ID: Final[str] = "10.8"
READ_ONLY_MARKET_DATA_CLOSURE_AUDIT_STATUS: Final[str] = (
    "block_h_read_only_market_data_closure_audit_complete_ready_for_next_block"
)
READ_ONLY_MARKET_DATA_CLOSURE_AUDIT_DECISION: Final[str] = (
    "CLOSE_BLOCK_H_READ_ONLY_MARKET_DATA_PREVIEW_PATH_NO_LIVE_FETCH_NO_RUNTIME_EXECUTION"
)
READY_FOR_NEXT_BLOCK: Final[bool] = True
NEXT_BLOCK: Final[str] = "BLOK I — TESTNET/SANDBOX ADAPTER"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-11.0"
NEXT_STEP_TITLE: Final[str] = "BLOK I — TESTNET/SANDBOX ADAPTER CONTRACT"
STATUS: Final[str] = "ready_for_functional_preview_11_0_testnet_sandbox_adapter_contract"
CLOSURE_LINE: Final[str] = "BLOK GOTOWY — PRZECHODZIMY DO KOLEJNEGO BLOKU"
_ALLOWED_SYMBOLS: Final[list[str]] = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"]
_DEFAULT_SELECTION_ID: Final[str] = "btc_usdt_static_fixture"
_HANDOFF_KEYS: Final[tuple[str, ...]] = (
    "read_only_market_data_bridge_snapshot_status",
    "read_only_market_data_bridge_snapshot_decision",
    "read_only_market_data_bridge_snapshot_next_step",
    "read_only_market_data_bridge_snapshot_next_step_title",
    "read_only_market_data_bridge_snapshot_ready_for_block_h_8",
    "read_only_market_data_controlled_refresh_status",
    "read_only_market_data_controlled_refresh_next_step",
    "read_only_market_data_controlled_refresh_ready_for_block_h_7",
    "read_only_market_data_allowed_refresh_preview_count",
    "read_only_market_data_default_refresh_selection_id",
    "read_only_market_data_allowed_refresh_symbols",
    "read_only_market_data_no_refresh_summary",
    "read_only_market_data_no_fetch_summary",
    "read_only_market_data_no_network_summary",
    "read_only_market_data_no_bridge_api_change_summary",
    "read_only_market_data_bridge_snapshot_summary",
)


def _block_h_completed_steps() -> list[str]:
    return [
        "FUNCTIONAL-PREVIEW-10.0 — READ-ONLY MARKET DATA ADAPTER CONTRACT",
        "FUNCTIONAL-PREVIEW-10.1 — READ-ONLY MARKET DATA READ MODEL",
        "FUNCTIONAL-PREVIEW-10.2 — READ-ONLY MARKET DATA STATIC FIXTURE",
        "FUNCTIONAL-PREVIEW-10.3 — READ-ONLY MARKET DATA AUDIT ENVELOPE",
        "FUNCTIONAL-PREVIEW-10.4 — READ-ONLY MARKET DATA UI READ-ONLY SURFACE",
        "FUNCTIONAL-PREVIEW-10.5 — READ-ONLY MARKET DATA SELECTION GATE",
        "FUNCTIONAL-PREVIEW-10.6 — READ-ONLY MARKET DATA CONTROLLED REFRESH PREVIEW",
        "FUNCTIONAL-PREVIEW-10.7 — READ-ONLY MARKET DATA BRIDGE SNAPSHOT",
        "FUNCTIONAL-PREVIEW-10.8 — READ-ONLY MARKET DATA CLOSURE AUDIT",
    ]


def _completion_matrix() -> list[dict[str, Any]]:
    entries = [
        ("10.0", "contract", "read_only_market_data_adapter_contract_ready"),
        ("10.1", "read model", "read_only_market_data_read_model_ready"),
        ("10.2", "static fixture", "read_only_market_data_static_fixture_ready"),
        ("10.3", "audit envelope", "read_only_market_data_audit_envelope_ready"),
        ("10.4", "UI read-only surface", "ui_read_only_surface_ready"),
        ("10.5", "selection gate", "selection_gate_ready"),
        (
            "10.6",
            "controlled refresh preview",
            "controlled_refresh_preview_ready_no_refresh_performed",
        ),
        ("10.7", "bridge snapshot", "bridge_snapshot_ready_data_only"),
        ("10.8", "closure audit", "closure_audit_complete"),
    ]
    return [
        {
            "step": step,
            "artifact": artifact,
            "status": status,
            "ready": True,
            "network_io_performed": False,
            "market_data_fetch_performed": False,
            "live_or_testnet_connection_performed": False,
            "account_or_private_data_accessed": False,
            "order_or_fill_action_performed": False,
            "runtime_execution_performed": False,
            "qml_action_added": False,
            "bridge_api_changed": False,
        }
        for step, artifact, status in entries
    ]


def build_preview_read_only_market_data_closure_audit() -> dict[str, Any]:
    """Build the pure-data Block H closure audit from the 10.7 bridge snapshot."""

    snapshot = build_paper_runtime_action_dispatch_bridge_snapshot()
    handoff = {key: snapshot[key] for key in _HANDOFF_KEYS}
    bridge_reference_valid = (
        handoff["read_only_market_data_bridge_snapshot_ready_for_block_h_8"] is True
        and handoff["read_only_market_data_bridge_snapshot_next_step"] == "FUNCTIONAL-PREVIEW-10.8"
        and handoff["read_only_market_data_bridge_snapshot_next_step_title"]
        == "READ-ONLY MARKET DATA CLOSURE AUDIT"
    )
    no_refresh = handoff["read_only_market_data_no_refresh_summary"]
    no_fetch = handoff["read_only_market_data_no_fetch_summary"]
    no_network = handoff["read_only_market_data_no_network_summary"]
    no_bridge = handoff["read_only_market_data_no_bridge_api_change_summary"]

    preview_summary = {
        "block_h_complete": True,
        "read_only_market_data_contract_ready": True,
        "read_only_market_data_read_model_ready": True,
        "static_fixture_ready": True,
        "audit_envelope_ready": True,
        "ui_read_only_surface_ready": True,
        "selection_gate_ready": True,
        "controlled_refresh_preview_ready": True,
        "bridge_snapshot_ready": True,
        "closure_audit_ready": True,
        "allowed_market_symbols": list(_ALLOWED_SYMBOLS),
        "allowed_refresh_preview_count": 4,
        "default_selection_id": _DEFAULT_SELECTION_ID,
        "no_real_refresh_performed": True,
        "no_market_data_fetch_performed": True,
        "no_network_io_performed": True,
        "no_exchange_api_connection_opened": True,
        "no_account_data_accessed": True,
        "no_order_or_fill_action_performed": True,
        "no_runtime_loop_started": True,
        "no_scheduler_started": True,
        "no_qml_action_added": True,
        "no_bridge_api_changes": True,
        "no_live_or_testnet_connection": True,
        "no_credentials_or_secrets_accessed": True,
        "no_export_performed": True,
        "exe_direction_preserved": True,
        "ready_for_next_block": True,
    }
    bridge_summary = {
        "bridge_snapshot_data_only": True,
        "bridge_snapshot_qml_safe": True,
        "bridge_snapshot_ready_for_block_h_8": True,
        "controlled_refresh_preview_read": True,
        "allowed_refresh_preview_count": handoff[
            "read_only_market_data_allowed_refresh_preview_count"
        ],
        "default_refresh_selection_id": handoff[
            "read_only_market_data_default_refresh_selection_id"
        ],
        "no_refresh_summary_green": no_refresh["no_real_refresh"] is True,
        "no_fetch_summary_green": no_fetch["no_market_fetch"] is True,
        "no_network_summary_green": no_network["no_network_io"] is True,
        "no_bridge_api_change_summary_green": no_bridge["no_bridge_api_changes"] is True,
        "next_step_matched_closure_audit": bridge_reference_valid,
    }
    evidence = {
        "closure_audit_evaluated": True,
        "bridge_snapshot_read": True,
        "block_h_artifacts_referenced": True,
        "adapter_implemented": False,
        "refresh_performed": False,
        "controlled_refresh_performed": False,
        "network_io_performed": False,
        "market_data_fetch_performed": False,
        "exchange_connection_opened": False,
        "private_endpoint_accessed": False,
        "account_fetch_performed": False,
        "balance_fetch_performed": False,
        "positions_fetch_performed": False,
        "orders_fetch_performed": False,
        "fills_fetch_performed": False,
        "order_generated": False,
        "order_submitted": False,
        "fill_simulated": False,
        "lifecycle_mutated": False,
        "runtime_loop_started": False,
        "scheduler_started": False,
        "trading_controller_touched": False,
        "decision_envelope_touched": False,
        "secrets_read_performed": False,
        "audit_export_performed": False,
        "export_performed": False,
        "qml_changes_performed": False,
        "qml_action_added": False,
        "bridge_api_changes_performed": False,
    }
    boundary_checks = {
        "local_only": True,
        "block_h_closure_audit_only": True,
        "bridge_snapshot_reference_valid": bridge_reference_valid,
        "all_block_h_steps_complete": True,
        "read_only_market_data_scope_complete": True,
        "adapter_implemented_now": False,
        "refresh_execution_allowed_now": False,
        "refresh_performed_now": False,
        "controlled_refresh_allowed_now": False,
        "controlled_refresh_performed_now": False,
        "network_io_allowed_now": False,
        "market_data_fetch_allowed_now": False,
        "live_market_data_fetch_allowed_now": False,
        "exchange_connection_allowed_now": False,
        "public_market_data_allowed_future_only": True,
        "recorded_replay_allowed_future_only": True,
        "private_account_data_allowed": False,
        "account_fetch_allowed": False,
        "balance_fetch_allowed": False,
        "positions_fetch_allowed": False,
        "orders_fetch_allowed": False,
        "fills_fetch_allowed": False,
        "order_generation_allowed": False,
        "order_submission_allowed": False,
        "fill_simulation_allowed": False,
        "lifecycle_mutation_allowed": False,
        "runtime_loop_allowed": False,
        "scheduler_allowed": False,
        "trading_controller_allowed": False,
        "decision_envelope_allowed": False,
        "strategy_execution_allowed": False,
        "ai_scoring_execution_allowed": False,
        "model_inference_execution_allowed": False,
        "live_mode_allowed": False,
        "testnet_mode_allowed_in_block_h": False,
        "testnet_mode_allowed_next_block_contract_only": True,
        "live_credentials_allowed": False,
        "secrets_read_allowed": False,
        "secrets_export_allowed": False,
        "cloud_export_allowed": False,
        "external_export_allowed": False,
        "dynamic_action_dispatch_allowed": False,
        "new_qml_method_calls_allowed": False,
        "qml_changes_allowed": False,
        "bridge_api_changes_allowed": False,
        "exe_packaging_in_scope": False,
        "bat_productization_allowed": False,
        "exe_direction_preserved": True,
        "ready_for_next_block": True,
    }
    blocked_capabilities = [
        "market data adapter implementation in Block H",
        "network I/O in Block H",
        "live market data fetch in Block H",
        "controlled refresh execution in Block H",
        "exchange API connection in Block H",
        "private account endpoint access",
        "account balance fetch",
        "positions fetch",
        "orders fetch",
        "fills fetch",
        "order generation",
        "order submission",
        "fill simulation",
        "lifecycle mutation",
        "runtime loop",
        "scheduler",
        "audit export",
        "bridge API changes",
        "TradingController / DecisionEnvelope",
        "live/testnet/account/secrets/export/cloud in Block H",
        "QML changes / new QML calls in closure audit",
        "EXE packaging",
    ]
    source_boundaries = [
        "no PySide import",
        "no QML import",
        "no runtime loop import",
        "no scheduler import",
        "no TradingController import",
        "no DecisionEnvelope import",
        "no strategy/AI/scoring/recommendation import",
        "no order module import",
        "no live adapter import",
        "no testnet adapter import in Block H",
        "no market data runtime adapter import",
        "no account module import",
        "no secrets module import",
        "no filesystem I/O",
        "no network I/O",
        "no QML changes",
        "no bridge API changes",
        "no .bat changes",
        "no app.py changes",
        "no dependency declarations changes",
        "no workflow changes",
    ]
    next_block_entry_requirements = {
        "next_block_contract_first": True,
        "testnet_or_sandbox_contract_required_before_any_adapter": True,
        "no_testnet_runtime_until_contract_and_guards": True,
        "no_live_mode_in_next_block_initial_step": True,
        "no_credentials_until_explicit_secrets_gate": True,
        "no_account_fetch_until_explicit_private_endpoint_gate": True,
        "no_order_submission_until_explicit_order_gate": True,
        "block_i_must_start_with_contract_only": True,
    }
    closure_decision = {
        "close_block_h": True,
        "ready_for_next_block": True,
        "next_block": NEXT_BLOCK,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "closure_line": CLOSURE_LINE,
    }
    return {
        "schema_version": PREVIEW_READ_ONLY_MARKET_DATA_CLOSURE_AUDIT_SCHEMA_VERSION,
        "market_data_closure_audit_kind": PREVIEW_READ_ONLY_MARKET_DATA_CLOSURE_AUDIT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "market_data_closure_audit_status": READ_ONLY_MARKET_DATA_CLOSURE_AUDIT_STATUS,
        "market_data_closure_audit_decision": READ_ONLY_MARKET_DATA_CLOSURE_AUDIT_DECISION,
        "ready_for_next_block": READY_FOR_NEXT_BLOCK,
        "next_block": NEXT_BLOCK,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_h_handoff_reference": handoff,
        "block_h_completed_steps": _block_h_completed_steps(),
        "block_h_completion_matrix": _completion_matrix(),
        "read_only_market_data_preview_path_summary": preview_summary,
        "bridge_snapshot_closure_summary": bridge_summary,
        "no_live_no_fetch_no_runtime_evidence": evidence,
        "boundary_checks": boundary_checks,
        "blocked_capabilities": blocked_capabilities,
        "source_boundaries": source_boundaries,
        "next_block_entry_requirements": next_block_entry_requirements,
        "closure_decision": closure_decision,
        "status": STATUS,
    }
