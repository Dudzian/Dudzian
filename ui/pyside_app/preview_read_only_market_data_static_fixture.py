"""Pure-data FUNCTIONAL-PREVIEW-10.2 BLOK H market data static fixture.

This module builds local, static, read-only market-data sample rows for the
10.1 read model. It intentionally does not implement a market-data adapter,
fetch market data, perform network I/O, access accounts, read credentials,
start runtimes, export audit data, or integrate with QML/trading paths.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_read_only_market_data_read_model import (
    build_preview_read_only_market_data_read_model,
)

PREVIEW_READ_ONLY_MARKET_DATA_STATIC_FIXTURE_SCHEMA_VERSION: Final[str] = (
    "preview_read_only_market_data_static_fixture.v1"
)
PREVIEW_READ_ONLY_MARKET_DATA_STATIC_FIXTURE_KIND: Final[str] = (
    "functional_preview_block_h_read_only_market_data_static_fixture"
)
BLOCK_ID: Final[str] = "H"
STEP_ID: Final[str] = "10.2"
READ_ONLY_MARKET_DATA_STATIC_FIXTURE_STATUS: Final[str] = (
    "read_only_market_data_static_fixture_ready_no_market_data_fetch"
)
READ_ONLY_MARKET_DATA_STATIC_FIXTURE_DECISION: Final[str] = (
    "BUILD_STATIC_FIXTURE_ONLY_NO_NETWORK_IO_NO_MARKET_DATA_FETCH"
)
READY_FOR_BLOCK_H_3: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-10.3"
NEXT_STEP_TITLE: Final[str] = "READ-ONLY MARKET DATA AUDIT ENVELOPE"

_MARKET_DATA_FIELD_ALLOWLIST: Final[tuple[str, ...]] = (
    "symbol",
    "timestamp",
    "bid",
    "ask",
    "last",
    "volume",
    "source",
    "latency_ms_preview",
)
_MARKET_DATA_PRIVATE_FIELD_DENYLIST: Final[tuple[str, ...]] = (
    "account_id",
    "balance",
    "position",
    "order_id",
    "fill_id",
    "api_key",
    "secret",
    "private_endpoint",
)
_BLOCKED_CAPABILITIES: Final[tuple[str, ...]] = (
    "market data adapter implementation now",
    "network I/O",
    "live market data fetch now",
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
_FIXTURE_ROWS: Final[tuple[dict[str, str | float], ...]] = (
    {
        "symbol": "BTC/USDT",
        "timestamp": "2026-01-01T00:00:00Z",
        "bid": 43000.0,
        "ask": 43010.0,
        "last": 43005.0,
        "volume": 123.45,
        "source": "static_fixture",
        "latency_ms_preview": 0.0,
    },
    {
        "symbol": "ETH/USDT",
        "timestamp": "2026-01-01T00:00:01Z",
        "bid": 2300.0,
        "ask": 2301.0,
        "last": 2300.5,
        "volume": 456.78,
        "source": "static_fixture",
        "latency_ms_preview": 0.0,
    },
    {
        "symbol": "SOL/USDT",
        "timestamp": "2026-01-01T00:00:02Z",
        "bid": 98.5,
        "ask": 98.7,
        "last": 98.6,
        "volume": 12.34,
        "source": "static_fixture_low_liquidity_preview",
        "latency_ms_preview": 0.0,
    },
    {
        "symbol": "ADA/USDT",
        "timestamp": "2026-01-01T00:00:03Z",
        "bid": 0.52,
        "ask": 0.53,
        "last": 0.525,
        "volume": 999.0,
        "source": "static_fixture_stale_preview",
        "latency_ms_preview": 0.0,
    },
)


def _read_model_reference() -> dict[str, Any]:
    read_model = build_preview_read_only_market_data_read_model()
    return {
        "schema_version": read_model["schema_version"],
        "market_data_read_model_kind": read_model["market_data_read_model_kind"],
        "market_data_read_model_status": read_model["market_data_read_model_status"],
        "market_data_read_model_decision": read_model["market_data_read_model_decision"],
        "ready_for_block_h_2": read_model["ready_for_block_h_2"],
        "next_step": read_model["next_step"],
        "next_step_title": read_model["next_step_title"],
        "status": read_model["status"],
    }


def build_preview_read_only_market_data_static_fixture() -> dict[str, Any]:
    """Build the inert FUNCTIONAL-PREVIEW-10.2 Block H static fixture."""

    fixture: dict[str, Any] = {
        "schema_version": PREVIEW_READ_ONLY_MARKET_DATA_STATIC_FIXTURE_SCHEMA_VERSION,
        "market_data_static_fixture_kind": PREVIEW_READ_ONLY_MARKET_DATA_STATIC_FIXTURE_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "market_data_static_fixture_status": READ_ONLY_MARKET_DATA_STATIC_FIXTURE_STATUS,
        "market_data_static_fixture_decision": READ_ONLY_MARKET_DATA_STATIC_FIXTURE_DECISION,
        "ready_for_block_h_3": READY_FOR_BLOCK_H_3,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "read_model_reference": _read_model_reference(),
        "fixture_scope": {
            "scope_name": "read_only_market_data_static_fixture",
            "static_fixture_only": True,
            "read_model_reference_required": True,
            "adapter_implemented_now": False,
            "network_io_allowed_now": False,
            "market_data_fetch_allowed_now": False,
            "live_market_data_fetch_allowed_now": False,
            "exchange_connection_allowed_now": False,
            "fixture_data_available_now": True,
            "fixture_data_static_only": True,
            "fixture_data_executable": False,
            "recorded_replay_allowed_future_only": True,
            "public_market_data_allowed_future_only": True,
            "account_data_allowed": False,
            "balance_data_allowed": False,
            "orders_allowed": False,
            "fills_allowed": False,
            "trading_allowed": False,
            "credentials_allowed": False,
            "secrets_allowed": False,
            "export_allowed": False,
        },
        "market_data_static_fixture": {
            "fixture_available": True,
            "fixture_static_only": True,
            "fixture_executable": False,
            "fixture_name": "read_only_market_data_preview_static_fixture",
            "data_source_mode": "local_static_fixture_no_network_io",
            "rows_available_now": True,
            "row_count": 4,
            "fixture_source": "functional_preview_static_fixture",
            "fixture_version": "10.2",
            "fixture_generated_from_network": False,
            "fixture_generated_from_account": False,
            "fixture_contains_private_data": False,
            "fixture_contains_orders_or_fills": False,
            "fixture_contains_credentials_or_secrets": False,
            "network_io_allowed_now": False,
            "market_fetch_performed": False,
            "runtime_execution_allowed": False,
            "trading_execution_allowed": False,
            "account_access_allowed": False,
            "credentials_access_allowed": False,
        },
        "fixture_rows": [dict(row) for row in _FIXTURE_ROWS],
        "fixture_validation_preview": {
            "validation_available": True,
            "validation_static_only": True,
            "validation_executable": False,
            "row_count": 4,
            "allowed_fields_only": True,
            "private_fields_absent": True,
            "account_fields_absent": True,
            "order_fields_absent": True,
            "fill_fields_absent": True,
            "credentials_fields_absent": True,
            "network_required_for_validation": False,
            "market_data_fetch_required_for_validation": False,
            "all_rows_have_symbol": True,
            "all_rows_have_timestamp": True,
            "all_rows_have_bid_ask_last": True,
            "all_rows_have_volume": True,
            "all_rows_have_source": True,
            "all_rows_have_latency_preview": True,
            "all_rows_have_bid_less_than_or_equal_ask": True,
            "all_rows_have_last_within_bid_ask": True,
            "all_rows_have_non_negative_volume": True,
            "all_rows_have_non_negative_latency": True,
        },
        "market_data_field_allowlist": list(_MARKET_DATA_FIELD_ALLOWLIST),
        "market_data_private_field_denylist": list(_MARKET_DATA_PRIVATE_FIELD_DENYLIST),
        "no_fetch_no_execution_evidence": {
            "static_fixture_evaluated": True,
            "read_model_read": True,
            "fixture_rows_built_from_static_literals": True,
            "adapter_implemented": False,
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
            "export_performed": False,
            "qml_changes_performed": False,
        },
        "boundary_checks": {
            "local_only": True,
            "block_h_static_fixture_only": True,
            "read_model_reference_valid": True,
            "read_only_market_data_scope_defined": True,
            "adapter_implemented_now": False,
            "fixture_data_available_now": True,
            "fixture_data_static_only": True,
            "fixture_data_executable": False,
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
            "ready_for_block_h_3": True,
        },
        "blocked_capabilities": list(_BLOCKED_CAPABILITIES),
        "source_boundaries": list(_SOURCE_BOUNDARIES),
        "future_steps": [
            "functional_preview_10_3_read_only_market_data_audit_envelope",
            "functional_preview_10_4_read_only_market_data_ui_read_only_surface",
            "functional_preview_10_5_read_only_market_data_selection_gate",
        ],
        "status": "ready_for_functional_preview_10_3_read_only_market_data_audit_envelope",
    }
    return deepcopy(fixture)
