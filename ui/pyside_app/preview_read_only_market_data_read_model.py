"""Pure-data FUNCTIONAL-PREVIEW-10.1 BLOK H market data read model.

This module defines a static, inert read model for the future read-only market
adapter. It references the accepted 10.0 contract and intentionally does not
implement an adapter, populate fixture data, fetch market data, open network
connections, access accounts, read credentials, start runtimes, or integrate
with QML/trading execution paths.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_read_only_market_data_adapter_contract import (
    build_preview_read_only_market_data_adapter_contract,
)

PREVIEW_READ_ONLY_MARKET_DATA_READ_MODEL_SCHEMA_VERSION: Final[str] = (
    "preview_read_only_market_data_read_model.v1"
)
PREVIEW_READ_ONLY_MARKET_DATA_READ_MODEL_KIND: Final[str] = (
    "functional_preview_block_h_read_only_market_data_read_model"
)
BLOCK_ID: Final[str] = "H"
STEP_ID: Final[str] = "10.1"
READ_ONLY_MARKET_DATA_READ_MODEL_STATUS: Final[str] = (
    "read_only_market_data_read_model_ready_no_market_data_fetch"
)
READ_ONLY_MARKET_DATA_READ_MODEL_DECISION: Final[str] = (
    "BUILD_READ_MODEL_ONLY_NO_NETWORK_IO_NO_MARKET_DATA_FETCH"
)
READY_FOR_BLOCK_H_2: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-10.2"
NEXT_STEP_TITLE: Final[str] = "READ-ONLY MARKET DATA STATIC FIXTURE"

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
    "read model population from market data now",
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


def _contract_reference() -> dict[str, Any]:
    contract = build_preview_read_only_market_data_adapter_contract()
    return {
        "schema_version": contract["schema_version"],
        "market_data_adapter_contract_kind": contract["market_data_adapter_contract_kind"],
        "market_data_adapter_contract_status": contract["market_data_adapter_contract_status"],
        "market_data_adapter_contract_decision": contract["market_data_adapter_contract_decision"],
        "ready_for_block_h_1": contract["ready_for_block_h_1"],
        "next_step": contract["next_step"],
        "next_step_title": contract["next_step_title"],
        "status": contract["status"],
    }


def build_preview_read_only_market_data_read_model() -> dict[str, Any]:
    """Build the inert FUNCTIONAL-PREVIEW-10.1 Block H read model."""

    read_model: dict[str, Any] = {
        "schema_version": PREVIEW_READ_ONLY_MARKET_DATA_READ_MODEL_SCHEMA_VERSION,
        "market_data_read_model_kind": PREVIEW_READ_ONLY_MARKET_DATA_READ_MODEL_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "market_data_read_model_status": READ_ONLY_MARKET_DATA_READ_MODEL_STATUS,
        "market_data_read_model_decision": READ_ONLY_MARKET_DATA_READ_MODEL_DECISION,
        "ready_for_block_h_2": READY_FOR_BLOCK_H_2,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "contract_reference": _contract_reference(),
        "read_model_scope": {
            "scope_name": "read_only_market_data_read_model",
            "read_model_only": True,
            "contract_reference_required": True,
            "adapter_implemented_now": False,
            "network_io_allowed_now": False,
            "market_data_fetch_allowed_now": False,
            "live_market_data_fetch_allowed_now": False,
            "exchange_connection_allowed_now": False,
            "fixture_data_allowed_next_step": True,
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
        "market_data_read_model": {
            "read_model_available": True,
            "read_model_static_only": True,
            "read_model_executable": False,
            "read_model_name": "read_only_market_data_preview_read_model",
            "data_source_mode": "not_connected_no_network_io",
            "rows_available_now": False,
            "row_count": 0,
            "sample_rows": [],
            "empty_state_reason": "no_market_data_fixture_until_functional_preview_10_2",
            "allowed_row_fields": list(_MARKET_DATA_FIELD_ALLOWLIST),
            "forbidden_row_fields": list(_MARKET_DATA_PRIVATE_FIELD_DENYLIST),
            "network_io_allowed_now": False,
            "market_fetch_performed": False,
            "runtime_execution_allowed": False,
            "trading_execution_allowed": False,
            "account_access_allowed": False,
            "credentials_access_allowed": False,
        },
        "market_data_field_allowlist": list(_MARKET_DATA_FIELD_ALLOWLIST),
        "market_data_private_field_denylist": list(_MARKET_DATA_PRIVATE_FIELD_DENYLIST),
        "read_model_quality_preview": {
            "quality_preview_available": True,
            "quality_static_only": True,
            "quality_executable": False,
            "freshness_status": "not_evaluated_no_data",
            "latency_status": "not_evaluated_no_data",
            "schema_status": "defined_by_contract_only",
            "source_status": "not_connected",
            "row_validation_allowed_next_step": True,
            "market_data_fetch_required_for_quality": False,
            "network_required_now": False,
            "account_required_now": False,
        },
        "no_fetch_no_execution_evidence": {
            "read_model_evaluated": True,
            "contract_read": True,
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
            "block_h_read_model_only": True,
            "contract_reference_valid": True,
            "read_only_market_data_scope_defined": True,
            "adapter_implemented_now": False,
            "read_model_populated_now": False,
            "sample_rows_allowed_now": False,
            "fixture_data_allowed_next_step": True,
            "network_io_allowed_now": False,
            "market_data_fetch_allowed_now": False,
            "live_market_data_fetch_allowed_now": False,
            "exchange_connection_allowed_now": False,
            "public_market_data_allowed_future_only": True,
            "recorded_fixture_allowed_next_step": True,
            "local_replay_allowed_future_only": True,
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
            "ready_for_block_h_2": True,
        },
        "blocked_capabilities": list(_BLOCKED_CAPABILITIES),
        "source_boundaries": list(_SOURCE_BOUNDARIES),
        "future_steps": [
            "functional_preview_10_2_read_only_market_data_static_fixture",
            "functional_preview_10_3_read_only_market_data_audit_envelope",
            "functional_preview_10_4_read_only_market_data_ui_read_only_surface",
        ],
        "status": "ready_for_functional_preview_10_2_read_only_market_data_static_fixture",
    }
    return deepcopy(read_model)
