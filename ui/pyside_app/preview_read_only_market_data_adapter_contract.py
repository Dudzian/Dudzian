"""Pure-data FUNCTIONAL-PREVIEW-10.0 BLOK H market data adapter contract.

This module defines the static contract for a future read-only market data
adapter. It intentionally does not implement an adapter, fetch market data,
open network connections, access accounts, read credentials, start runtimes, or
integrate with QML/trading execution paths.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from ui.pyside_app.preview_block_g_closure_audit import build_preview_block_g_closure_audit

PREVIEW_READ_ONLY_MARKET_DATA_ADAPTER_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "preview_read_only_market_data_adapter_contract.v1"
)
PREVIEW_READ_ONLY_MARKET_DATA_ADAPTER_CONTRACT_KIND: Final[str] = (
    "functional_preview_block_h_read_only_market_data_adapter_contract"
)
BLOCK_ID: Final[str] = "H"
STEP_ID: Final[str] = "10.0"
READ_ONLY_MARKET_DATA_ADAPTER_CONTRACT_STATUS: Final[str] = (
    "read_only_market_data_adapter_contract_ready_no_network_io"
)
READ_ONLY_MARKET_DATA_ADAPTER_CONTRACT_DECISION: Final[str] = (
    "START_BLOCK_H_WITH_CONTRACT_ONLY_NO_MARKET_DATA_FETCH"
)
READY_FOR_BLOCK_H_1: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-10.1"
NEXT_STEP_TITLE: Final[str] = "READ-ONLY MARKET DATA ADAPTER READ MODEL"

_ALLOWED_FUTURE_INPUT_TYPES: Final[tuple[str, ...]] = (
    "recorded_fixture",
    "local_replay",
    "read_only_public_market_data",
)
_FORBIDDEN_INPUT_TYPES: Final[tuple[str, ...]] = (
    "live_account",
    "live_balance",
    "live_orders",
    "live_fills",
    "private_account_endpoint",
    "credential_backed_private_api",
)
_ALLOWED_FUTURE_MARKET_DATA_FIELDS: Final[tuple[str, ...]] = (
    "symbol",
    "timestamp",
    "bid",
    "ask",
    "last",
    "volume",
    "source",
    "latency_ms_preview",
)
_FORBIDDEN_FIELDS: Final[tuple[str, ...]] = (
    "account_id",
    "balance",
    "position",
    "order_id",
    "fill_id",
    "api_key",
    "secret",
    "private_endpoint",
)

_ALLOWED_MARKET_DATA_CAPABILITIES: Final[tuple[str, ...]] = (
    "define read-only market data adapter contract",
    "define future recorded fixture input shape",
    "define future local replay input shape",
    "define future public read-only market data shape",
    "define market data field allowlist",
    "define market data private-field denylist",
    "prepare read-only market data read model next step",
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


def _block_g_handoff_reference() -> dict[str, Any]:
    audit = build_preview_block_g_closure_audit()
    return {
        "schema_version": audit["schema_version"],
        "closure_audit_kind": audit["closure_audit_kind"],
        "block_g_closure_status": audit["block_g_closure_status"],
        "block_g_closure_decision": audit["block_g_closure_decision"],
        "ready_for_block_h": audit["ready_for_block_h"],
        "next_step": audit["next_step"],
        "next_step_title": audit["next_step_title"],
        "status": audit["status"],
    }


def build_preview_read_only_market_data_adapter_contract() -> dict[str, Any]:
    """Build the inert FUNCTIONAL-PREVIEW-10.0 Block H contract."""

    contract: dict[str, Any] = {
        "schema_version": PREVIEW_READ_ONLY_MARKET_DATA_ADAPTER_CONTRACT_SCHEMA_VERSION,
        "market_data_adapter_contract_kind": PREVIEW_READ_ONLY_MARKET_DATA_ADAPTER_CONTRACT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "market_data_adapter_contract_status": READ_ONLY_MARKET_DATA_ADAPTER_CONTRACT_STATUS,
        "market_data_adapter_contract_decision": READ_ONLY_MARKET_DATA_ADAPTER_CONTRACT_DECISION,
        "ready_for_block_h_1": READY_FOR_BLOCK_H_1,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_g_handoff_reference": _block_g_handoff_reference(),
        "read_only_market_data_scope": {
            "scope_name": "read_only_market_data",
            "contract_only": True,
            "adapter_implemented_now": False,
            "network_io_allowed_now": False,
            "market_data_fetch_allowed_now": False,
            "live_market_data_fetch_allowed_now": False,
            "fixture_market_data_allowed_in_future": True,
            "recorded_market_data_replay_allowed_in_future": True,
            "read_only_exchange_market_data_allowed_in_future": True,
            "account_data_allowed": False,
            "balance_data_allowed": False,
            "orders_allowed": False,
            "fills_allowed": False,
            "trading_allowed": False,
            "credentials_allowed": False,
            "secrets_allowed": False,
            "export_allowed": False,
        },
        "adapter_contract": {
            "contract_available": True,
            "contract_static_only": True,
            "contract_executable": False,
            "adapter_name": "read_only_market_data_adapter_preview_contract",
            "adapter_mode": "contract_only_no_network_io",
            "allowed_future_input_types": list(_ALLOWED_FUTURE_INPUT_TYPES),
            "forbidden_input_types": list(_FORBIDDEN_INPUT_TYPES),
            "allowed_future_market_data_fields": list(_ALLOWED_FUTURE_MARKET_DATA_FIELDS),
            "forbidden_fields": list(_FORBIDDEN_FIELDS),
            "network_io_allowed_now": False,
            "market_fetch_performed": False,
            "runtime_execution_allowed": False,
            "trading_execution_allowed": False,
            "account_access_allowed": False,
            "credentials_access_allowed": False,
        },
        "allowed_market_data_capabilities": list(_ALLOWED_MARKET_DATA_CAPABILITIES),
        "blocked_capabilities": list(_BLOCKED_CAPABILITIES),
        "no_network_no_execution_evidence": {
            "contract_evaluated": True,
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
            "paper_only_previous_block_preserved": True,
            "block_h_contract_only": True,
            "read_only_market_data_scope_defined": True,
            "adapter_implemented_now": False,
            "network_io_allowed_now": False,
            "market_data_fetch_allowed_now": False,
            "live_market_data_fetch_allowed_now": False,
            "exchange_connection_allowed_now": False,
            "public_market_data_allowed_future_only": True,
            "recorded_fixture_allowed_future_only": True,
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
            "ready_for_block_h_1": True,
        },
        "source_boundaries": list(_SOURCE_BOUNDARIES),
        "future_steps": [
            "functional_preview_10_1_read_only_market_data_adapter_read_model",
            "functional_preview_10_2_read_only_market_data_static_fixture",
            "functional_preview_10_3_read_only_market_data_audit_envelope",
        ],
        "status": "ready_for_functional_preview_10_1_read_only_market_data_adapter_read_model",
    }
    return deepcopy(contract)
