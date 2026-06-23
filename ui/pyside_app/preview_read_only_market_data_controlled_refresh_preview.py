"""Pure-data FUNCTIONAL-PREVIEW-10.6 Block H controlled refresh preview.

This module builds a static preview result for future read-only market-data
refresh handling. It is intentionally inert: no adapter, no refresh execution,
no market-data fetch, no network I/O, no account/order/fill access, no runtime,
no QML action, and no bridge API change.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_read_only_market_data_selection_gate import (
    build_preview_read_only_market_data_selection_gate,
)

PREVIEW_READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_SCHEMA_VERSION: Final[str] = (
    "preview_read_only_market_data_controlled_refresh_preview.v1"
)
PREVIEW_READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_KIND: Final[str] = (
    "functional_preview_block_h_read_only_market_data_controlled_refresh_preview"
)
BLOCK_ID: Final[str] = "H"
STEP_ID: Final[str] = "10.6"
READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_STATUS: Final[str] = (
    "read_only_market_data_controlled_refresh_preview_ready_no_refresh_performed"
)
READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_DECISION: Final[str] = (
    "BUILD_CONTROLLED_REFRESH_PREVIEW_ONLY_NO_REFRESH_NO_FETCH_NO_NETWORK_IO"
)
READY_FOR_BLOCK_H_7: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-10.7"
NEXT_STEP_TITLE: Final[str] = "READ-ONLY MARKET DATA BRIDGE SNAPSHOT"

_FALSE_REFRESH_FLAGS: Final[dict[str, bool]] = {
    "refresh_allowed": False,
    "refresh_performed": False,
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
_ALLOWED_REFRESH_PREVIEWS: Final[tuple[dict[str, str | bool], ...]] = (
    {
        "selection_id": "btc_usdt_static_fixture",
        "symbol": "BTC/USDT",
        "source": "static_fixture",
        "quality_status": "ok_preview_only",
        "refresh_preview_status": "accepted_preview_only_no_refresh_performed",
        "refresh_preview_label": "BTC/USDT static fixture refresh preview",
        **_FALSE_REFRESH_FLAGS,
    },
    {
        "selection_id": "eth_usdt_static_fixture",
        "symbol": "ETH/USDT",
        "source": "static_fixture",
        "quality_status": "ok_preview_only",
        "refresh_preview_status": "accepted_preview_only_no_refresh_performed",
        "refresh_preview_label": "ETH/USDT static fixture refresh preview",
        **_FALSE_REFRESH_FLAGS,
    },
    {
        "selection_id": "sol_usdt_low_liquidity_preview",
        "symbol": "SOL/USDT",
        "source": "static_fixture_low_liquidity_preview",
        "quality_status": "low_liquidity_preview_only",
        "refresh_preview_status": "accepted_preview_only_no_refresh_performed",
        "refresh_preview_label": "SOL/USDT low-liquidity refresh preview",
        **_FALSE_REFRESH_FLAGS,
    },
    {
        "selection_id": "ada_usdt_stale_preview",
        "symbol": "ADA/USDT",
        "source": "static_fixture_stale_preview",
        "quality_status": "stale_preview_only",
        "refresh_preview_status": "accepted_preview_only_no_refresh_performed",
        "refresh_preview_label": "ADA/USDT stale refresh preview",
        **_FALSE_REFRESH_FLAGS,
    },
)
_REJECTED_REFRESH_PREVIEW_EXAMPLES: Final[tuple[dict[str, str | bool], ...]] = tuple(
    {
        "selection_id": selection_id,
        "refresh_preview_status": "rejected_fail_closed_no_refresh_performed",
        "reason": reason,
        **_FALSE_REFRESH_FLAGS,
    }
    for selection_id, reason in (
        ("", "empty_selection_id_rejected_fail_closed"),
        ("unknown_pair", "unknown_selection_not_in_static_allowlist"),
        ("live_btc_usdt", "live_market_data_selection_blocked_for_10_6"),
        ("fetch_market_data", "market_data_fetch_request_blocked_for_10_6"),
        ("refresh_market_data", "market_data_refresh_request_blocked_for_10_6"),
        ("account_balance", "account_or_balance_selection_blocked_for_10_6"),
    )
)
_BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
    "market data adapter implementation now",
    "network I/O",
    "live market data fetch now",
    "controlled refresh execution now",
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
    "bridge API changes",
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
    "no bridge API changes",
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
)


def _selection_gate_reference() -> dict[str, Any]:
    gate = build_preview_read_only_market_data_selection_gate()
    return {
        "schema_version": gate["schema_version"],
        "market_data_selection_gate_kind": gate["market_data_selection_gate_kind"],
        "market_data_selection_gate_status": gate["market_data_selection_gate_status"],
        "market_data_selection_gate_decision": gate["market_data_selection_gate_decision"],
        "ready_for_block_h_6": gate["ready_for_block_h_6"],
        "next_step": gate["next_step"],
        "next_step_title": gate["next_step_title"],
        "status": gate["status"],
    }


def build_preview_read_only_market_data_controlled_refresh_preview() -> dict[str, Any]:
    """Build the inert FUNCTIONAL-PREVIEW-10.6 controlled refresh preview."""

    return {
        "schema_version": PREVIEW_READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_SCHEMA_VERSION,
        "market_data_controlled_refresh_preview_kind": PREVIEW_READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "market_data_controlled_refresh_preview_status": READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_STATUS,
        "market_data_controlled_refresh_preview_decision": READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_DECISION,
        "ready_for_block_h_7": READY_FOR_BLOCK_H_7,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "selection_gate_reference": _selection_gate_reference(),
        "controlled_refresh_preview_scope": {
            "scope_name": "read_only_market_data_controlled_refresh_preview",
            "controlled_refresh_preview_only": True,
            "selection_gate_reference_required": True,
            "allowed_refresh_previews_static_only": True,
            "refresh_result_preview_only": True,
            "refresh_execution_allowed_now": False,
            "refresh_performed_now": False,
            "adapter_implemented_now": False,
            "network_io_allowed_now": False,
            "market_data_fetch_allowed_now": False,
            "live_market_data_fetch_allowed_now": False,
            "exchange_connection_allowed_now": False,
            "audit_export_allowed_now": False,
            "qml_changes_allowed": False,
            "new_qml_method_calls_allowed": False,
            "bridge_api_changes_allowed": False,
            "account_data_allowed": False,
            "balance_data_allowed": False,
            "orders_allowed": False,
            "fills_allowed": False,
            "trading_allowed": False,
            "credentials_allowed": False,
            "secrets_allowed": False,
            "export_allowed": False,
        },
        "controlled_refresh_preview": {
            "controlled_refresh_preview_available": True,
            "controlled_refresh_preview_static_only": True,
            "controlled_refresh_preview_executable": False,
            "controlled_refresh_preview_name": "read_only_market_data_controlled_refresh_preview",
            "refresh_source_mode": "local_static_selection_preview_no_network_io",
            "allowed_refresh_preview_count": 4,
            "default_selection_id": "btc_usdt_static_fixture",
            "unknown_selection_status": "rejected_fail_closed_no_refresh",
            "refresh_execution_allowed_now": False,
            "refresh_performed_now": False,
            "network_io_allowed_now": False,
            "market_fetch_performed": False,
            "runtime_execution_allowed": False,
            "trading_execution_allowed": False,
            "account_access_allowed": False,
            "credentials_access_allowed": False,
            "qml_actions_allowed_now": False,
            "bridge_api_changes_allowed_now": False,
        },
        "allowed_refresh_previews": [dict(preview) for preview in _ALLOWED_REFRESH_PREVIEWS],
        "rejected_refresh_preview_examples": [
            dict(preview) for preview in _REJECTED_REFRESH_PREVIEW_EXAMPLES
        ],
        "refresh_preview_boundary_summary": {
            "allowed_refresh_preview_count": 4,
            "rejected_example_count": 6,
            "allowed_symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT", "ADA/USDT"],
            "normal_preview_symbols": ["BTC/USDT", "ETH/USDT"],
            "low_liquidity_preview_symbols": ["SOL/USDT"],
            "stale_preview_symbols": ["ADA/USDT"],
            "all_allowed_previews_preview_only": True,
            "all_allowed_previews_no_refresh_performed": True,
            "all_allowed_previews_no_network": True,
            "all_allowed_previews_no_fetch": True,
            "all_allowed_previews_no_execution": True,
            "all_allowed_previews_no_account_data": True,
            "all_allowed_previews_no_order_or_fill_data": True,
            "all_allowed_previews_no_credentials_or_secrets": True,
            "unknown_previews_rejected_fail_closed": True,
            "live_or_private_previews_rejected_fail_closed": True,
            "fetch_or_refresh_requests_rejected_fail_closed": True,
        },
        "no_refresh_no_fetch_no_execution_evidence": {
            "controlled_refresh_preview_evaluated": True,
            "selection_gate_read": True,
            "allowed_refresh_previews_built_from_static_selection_gate": True,
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
            "bridge_api_changes_performed": False,
        },
        "boundary_checks": {
            "local_only": True,
            "block_h_controlled_refresh_preview_only": True,
            "selection_gate_reference_valid": True,
            "read_only_market_data_scope_defined": True,
            "allowed_refresh_previews_static_only": True,
            "refresh_execution_allowed_now": False,
            "refresh_performed_now": False,
            "controlled_refresh_allowed_now": False,
            "controlled_refresh_performed_now": False,
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
            "bridge_api_changes_allowed": False,
            "exe_packaging_in_scope": False,
            "bat_productization_allowed": False,
            "exe_direction_preserved": True,
            "ready_for_block_h_7": True,
        },
        "blocked_capabilities": list(_BLOCKED_CAPABILITIES),
        "source_boundaries": list(_SOURCE_BOUNDARIES),
        "future_steps": [
            "functional_preview_10_7_read_only_market_data_bridge_snapshot",
            "functional_preview_10_8_read_only_market_data_closure_audit",
            "functional_preview_10_9_block_h_closure",
        ],
        "status": "ready_for_functional_preview_10_7_read_only_market_data_bridge_snapshot",
    }


def build_preview_read_only_market_data_controlled_refresh_result(
    selection_id: object = None,
) -> dict[str, Any]:
    """Build an inert preview-only result for an allowlisted selection id."""

    selection_text = "" if selection_id is None else str(selection_id)
    for preview in _ALLOWED_REFRESH_PREVIEWS:
        if selection_text == preview["selection_id"]:
            return {
                "result_status": "accepted_preview_only_no_refresh_performed",
                "selection_id": preview["selection_id"],
                "symbol": preview["symbol"],
                "source": preview["source"],
                "quality_status": preview["quality_status"],
                "selection_allowed": True,
                "refresh_allowed": False,
                "refresh_performed": False,
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
        "result_status": "rejected_fail_closed_no_refresh_performed",
        "selection_id": selection_text,
        "selection_allowed": False,
        "reason": "selection_id_not_allowed_by_static_preview_gate",
        "refresh_allowed": False,
        "refresh_performed": False,
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
    "PREVIEW_READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_KIND",
    "PREVIEW_READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_SCHEMA_VERSION",
    "READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_DECISION",
    "READ_ONLY_MARKET_DATA_CONTROLLED_REFRESH_PREVIEW_STATUS",
    "READY_FOR_BLOCK_H_7",
    "STEP_ID",
    "build_preview_read_only_market_data_controlled_refresh_preview",
    "build_preview_read_only_market_data_controlled_refresh_result",
]
