"""Pure-data FUNCTIONAL-PREVIEW-10.5 BLOK H market data selection gate.

This module defines an inert, static allowlist/fail-closed selection gate for
read-only market-data preview choices. It intentionally does not implement a
market-data adapter, refresh market data, fetch market data, perform network I/O,
access accounts, read credentials, start runtimes, export audit data, or integrate
with QML/trading/action-dispatch paths.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_read_only_market_data_audit_envelope import (
    build_preview_read_only_market_data_audit_envelope,
)

PREVIEW_READ_ONLY_MARKET_DATA_SELECTION_GATE_SCHEMA_VERSION: Final[str] = (
    "preview_read_only_market_data_selection_gate.v1"
)
PREVIEW_READ_ONLY_MARKET_DATA_SELECTION_GATE_KIND: Final[str] = (
    "functional_preview_block_h_read_only_market_data_selection_gate"
)
BLOCK_ID: Final[str] = "H"
STEP_ID: Final[str] = "10.5"
READ_ONLY_MARKET_DATA_SELECTION_GATE_STATUS: Final[str] = (
    "read_only_market_data_selection_gate_ready_no_refresh_no_network_io"
)
READ_ONLY_MARKET_DATA_SELECTION_GATE_DECISION: Final[str] = (
    "BUILD_SELECTION_GATE_ONLY_NO_REFRESH_NO_NETWORK_IO_NO_QML_ACTIONS"
)
READY_FOR_BLOCK_H_6: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-10.6"
NEXT_STEP_TITLE: Final[str] = "READ-ONLY MARKET DATA CONTROLLED REFRESH PREVIEW"

_FALSE_SELECTION_FLAGS: Final[dict[str, bool]] = {
    "network_io_allowed": False,
    "market_data_fetch_allowed": False,
    "controlled_refresh_allowed": False,
    "execution_allowed": False,
    "account_data_allowed": False,
    "order_or_fill_data_allowed": False,
    "credentials_or_secrets_allowed": False,
}
_ALLOWED_SELECTIONS: Final[tuple[dict[str, str | bool], ...]] = (
    {
        "selection_id": "btc_usdt_static_fixture",
        "symbol": "BTC/USDT",
        "source": "static_fixture",
        "quality_status": "ok_preview_only",
        "selection_status": "allowed_preview_only_no_refresh",
        "selection_label": "BTC/USDT static fixture",
        **_FALSE_SELECTION_FLAGS,
    },
    {
        "selection_id": "eth_usdt_static_fixture",
        "symbol": "ETH/USDT",
        "source": "static_fixture",
        "quality_status": "ok_preview_only",
        "selection_status": "allowed_preview_only_no_refresh",
        "selection_label": "ETH/USDT static fixture",
        **_FALSE_SELECTION_FLAGS,
    },
    {
        "selection_id": "sol_usdt_low_liquidity_preview",
        "symbol": "SOL/USDT",
        "source": "static_fixture_low_liquidity_preview",
        "quality_status": "low_liquidity_preview_only",
        "selection_status": "allowed_preview_only_no_refresh",
        "selection_label": "SOL/USDT low-liquidity preview",
        **_FALSE_SELECTION_FLAGS,
    },
    {
        "selection_id": "ada_usdt_stale_preview",
        "symbol": "ADA/USDT",
        "source": "static_fixture_stale_preview",
        "quality_status": "stale_preview_only",
        "selection_status": "allowed_preview_only_no_refresh",
        "selection_label": "ADA/USDT stale preview",
        **_FALSE_SELECTION_FLAGS,
    },
)
_REJECTED_SELECTION_EXAMPLES: Final[tuple[dict[str, str | bool], ...]] = (
    {
        "selection_id": "",
        "selection_status": "rejected_fail_closed_no_refresh",
        "reason": "empty_selection_id_rejected_fail_closed",
        **_FALSE_SELECTION_FLAGS,
    },
    {
        "selection_id": "unknown_pair",
        "selection_status": "rejected_fail_closed_no_refresh",
        "reason": "unknown_selection_not_in_static_allowlist",
        **_FALSE_SELECTION_FLAGS,
    },
    {
        "selection_id": "live_btc_usdt",
        "selection_status": "rejected_fail_closed_no_refresh",
        "reason": "live_market_data_selection_blocked_for_10_5",
        **_FALSE_SELECTION_FLAGS,
    },
    {
        "selection_id": "fetch_market_data",
        "selection_status": "rejected_fail_closed_no_refresh",
        "reason": "market_data_fetch_request_blocked_for_10_5",
        **_FALSE_SELECTION_FLAGS,
    },
    {
        "selection_id": "account_balance",
        "selection_status": "rejected_fail_closed_no_refresh",
        "reason": "account_or_balance_selection_blocked_for_10_5",
        **_FALSE_SELECTION_FLAGS,
    },
)
_BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
    "market data adapter implementation now",
    "network I/O",
    "live market data fetch now",
    "controlled refresh now",
    "exchange API connection",
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
    "TradingController / DecisionEnvelope",
    "live/testnet/account/secrets/export/cloud",
    "QML changes / new QML calls",
    "EXE packaging",
)
_SOURCE_BOUNDARIES: Final[tuple[str, ...]] = (
    "no PySide import",
    "no QML import",
    "no runtime loop import",
    "no scheduler import",
    "no TradingController import",
    "no DecisionEnvelope import",
    "no strategy/AI/scoring/recommendation import",
    "no order module import",
    "no live adapter import",
    "no testnet adapter import",
    "no market data runtime adapter import",
    "no account module import",
    "no secrets module import",
    "no filesystem I/O",
    "no network I/O",
    "no QML changes",
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
)


def _audit_envelope_reference() -> dict[str, Any]:
    envelope = build_preview_read_only_market_data_audit_envelope()
    return {
        "schema_version": envelope["schema_version"],
        "market_data_audit_envelope_kind": envelope["market_data_audit_envelope_kind"],
        "market_data_audit_envelope_status": envelope["market_data_audit_envelope_status"],
        "market_data_audit_envelope_decision": envelope["market_data_audit_envelope_decision"],
        "ready_for_block_h_4": envelope["ready_for_block_h_4"],
        "next_step": envelope["next_step"],
        "next_step_title": envelope["next_step_title"],
        "status": envelope["status"],
    }


def build_preview_read_only_market_data_selection_gate() -> dict[str, Any]:
    """Build the inert FUNCTIONAL-PREVIEW-10.5 Block H selection gate."""

    return {
        "schema_version": PREVIEW_READ_ONLY_MARKET_DATA_SELECTION_GATE_SCHEMA_VERSION,
        "market_data_selection_gate_kind": PREVIEW_READ_ONLY_MARKET_DATA_SELECTION_GATE_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "market_data_selection_gate_status": READ_ONLY_MARKET_DATA_SELECTION_GATE_STATUS,
        "market_data_selection_gate_decision": READ_ONLY_MARKET_DATA_SELECTION_GATE_DECISION,
        "ready_for_block_h_6": READY_FOR_BLOCK_H_6,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "audit_envelope_reference": _audit_envelope_reference(),
        "selection_gate_scope": {
            "scope_name": "read_only_market_data_selection_gate",
            "selection_gate_only": True,
            "audit_envelope_reference_required": True,
            "allowed_selections_static_only": True,
            "selection_result_preview_only": True,
            "selection_execution_allowed_now": False,
            "controlled_refresh_allowed_now": False,
            "adapter_implemented_now": False,
            "network_io_allowed_now": False,
            "market_data_fetch_allowed_now": False,
            "live_market_data_fetch_allowed_now": False,
            "exchange_connection_allowed_now": False,
            "audit_export_allowed_now": False,
            "qml_changes_allowed": False,
            "new_qml_method_calls_allowed": False,
            "account_data_allowed": False,
            "balance_data_allowed": False,
            "orders_allowed": False,
            "fills_allowed": False,
            "trading_allowed": False,
            "credentials_allowed": False,
            "secrets_allowed": False,
            "export_allowed": False,
        },
        "selection_gate": {
            "selection_gate_available": True,
            "selection_gate_static_only": True,
            "selection_gate_executable": False,
            "selection_gate_name": "read_only_market_data_preview_selection_gate",
            "selection_source_mode": "local_static_audit_selection_gate_no_network_io",
            "allowed_selection_count": 4,
            "default_selection_id": "btc_usdt_static_fixture",
            "unknown_selection_status": "rejected_unknown_selection_no_refresh",
            "selection_execution_allowed_now": False,
            "controlled_refresh_allowed_now": False,
            "network_io_allowed_now": False,
            "market_fetch_performed": False,
            "runtime_execution_allowed": False,
            "trading_execution_allowed": False,
            "account_access_allowed": False,
            "credentials_access_allowed": False,
            "qml_actions_allowed_now": False,
        },
        "allowed_selections": [dict(selection) for selection in _ALLOWED_SELECTIONS],
        "rejected_selection_examples": [
            dict(selection) for selection in _REJECTED_SELECTION_EXAMPLES
        ],
        "selection_boundary_summary": {
            "allowed_selection_count": 4,
            "rejected_example_count": 5,
            "allowed_symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"],
            "normal_preview_symbols": ["BTC/USDT", "ETH/USDT"],
            "low_liquidity_preview_symbols": ["SOL/USDT"],
            "stale_preview_symbols": ["ADA/USDT"],
            "all_allowed_selections_preview_only": True,
            "all_allowed_selections_no_refresh": True,
            "all_allowed_selections_no_network": True,
            "all_allowed_selections_no_fetch": True,
            "all_allowed_selections_no_execution": True,
            "all_allowed_selections_no_account_data": True,
            "all_allowed_selections_no_order_or_fill_data": True,
            "all_allowed_selections_no_credentials_or_secrets": True,
            "unknown_selections_rejected_fail_closed": True,
            "live_or_private_selections_rejected_fail_closed": True,
        },
        "no_refresh_no_fetch_no_execution_evidence": {
            "selection_gate_evaluated": True,
            "audit_envelope_read": True,
            "allowed_selections_built_from_static_audit_events": True,
            "adapter_implemented": False,
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
        },
        "boundary_checks": {
            "local_only": True,
            "block_h_selection_gate_only": True,
            "audit_envelope_reference_valid": True,
            "read_only_market_data_scope_defined": True,
            "allowed_selections_static_only": True,
            "selection_execution_allowed_now": False,
            "controlled_refresh_allowed_now": False,
            "adapter_implemented_now": False,
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
            "testnet_mode_allowed_initially": False,
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
            "ready_for_block_h_6": True,
        },
        "blocked_capabilities": list(_BLOCKED_CAPABILITIES),
        "source_boundaries": list(_SOURCE_BOUNDARIES),
        "future_steps": [
            "functional_preview_10_6_read_only_market_data_controlled_refresh_preview",
            "functional_preview_10_7_read_only_market_data_bridge_snapshot",
            "functional_preview_10_8_read_only_market_data_closure_audit",
        ],
        "status": "ready_for_functional_preview_10_6_read_only_market_data_controlled_refresh_preview",
    }


def build_preview_read_only_market_data_selection_result(
    selection_id: object = None,
) -> dict[str, Any]:
    """Build an inert preview-only result for a static selection id."""

    selection_text = "" if selection_id is None else str(selection_id)
    for selection in _ALLOWED_SELECTIONS:
        if selection_text == selection["selection_id"]:
            return {
                "result_status": "accepted_preview_only_no_refresh",
                "selection_id": selection["selection_id"],
                "symbol": selection["symbol"],
                "source": selection["source"],
                "quality_status": selection["quality_status"],
                "selection_allowed": True,
                "controlled_refresh_allowed": False,
                "controlled_refresh_performed": False,
                "network_io_allowed": False,
                "network_io_performed": False,
                "market_data_fetch_allowed": False,
                "market_data_fetch_performed": False,
                "execution_allowed": False,
                "execution_performed": False,
                "account_data_allowed": False,
                "order_or_fill_data_allowed": False,
                "credentials_or_secrets_allowed": False,
            }
    return {
        "result_status": "rejected_fail_closed_no_refresh",
        "selection_id": selection_text,
        "selection_allowed": False,
        "reason": "selection_id_not_allowed_by_static_preview_gate",
        "controlled_refresh_allowed": False,
        "controlled_refresh_performed": False,
        "network_io_allowed": False,
        "network_io_performed": False,
        "market_data_fetch_allowed": False,
        "market_data_fetch_performed": False,
        "execution_allowed": False,
        "execution_performed": False,
        "account_data_allowed": False,
        "order_or_fill_data_allowed": False,
        "credentials_or_secrets_allowed": False,
    }


__all__ = [
    "BLOCK_ID",
    "NEXT_STEP",
    "NEXT_STEP_TITLE",
    "PREVIEW_READ_ONLY_MARKET_DATA_SELECTION_GATE_KIND",
    "PREVIEW_READ_ONLY_MARKET_DATA_SELECTION_GATE_SCHEMA_VERSION",
    "READ_ONLY_MARKET_DATA_SELECTION_GATE_DECISION",
    "READ_ONLY_MARKET_DATA_SELECTION_GATE_STATUS",
    "READY_FOR_BLOCK_H_6",
    "STEP_ID",
    "build_preview_read_only_market_data_selection_gate",
    "build_preview_read_only_market_data_selection_result",
]
