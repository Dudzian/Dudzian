"""FUNCTIONAL-PREVIEW-11.6 Block I public market data probe preview.

Pure-data Testnet/Sandbox public market data probe preview derived from the
11.5 credentials gate contract. This module intentionally does not fetch market
data, open network connections, read configuration, read credentials, or activate
runtime/adapter code.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_testnet_sandbox_credentials_gate_contract import (
    build_preview_testnet_sandbox_credentials_gate_contract,
)

PREVIEW_TESTNET_SANDBOX_PUBLIC_MARKET_DATA_PROBE_PREVIEW_SCHEMA_VERSION: Final[str] = (
    "preview_testnet_sandbox_public_market_data_probe_preview.v1"
)
PREVIEW_TESTNET_SANDBOX_PUBLIC_MARKET_DATA_PROBE_PREVIEW_KIND: Final[str] = (
    "functional_preview_block_i_testnet_sandbox_public_market_data_probe_preview"
)
BLOCK_ID: Final[str] = "I"
STEP_ID: Final[str] = "11.6"
TESTNET_SANDBOX_PUBLIC_MARKET_DATA_PROBE_PREVIEW_STATUS: Final[str] = (
    "testnet_sandbox_public_market_data_probe_preview_ready_no_fetch_no_network"
)
TESTNET_SANDBOX_PUBLIC_MARKET_DATA_PROBE_PREVIEW_DECISION: Final[str] = (
    "BUILD_PUBLIC_MARKET_DATA_PROBE_PREVIEW_ONLY_NO_FETCH_NO_NETWORK_NO_RUNTIME"
)
READY_FOR_BLOCK_I_7: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-11.7"
NEXT_STEP_TITLE: Final[str] = "TESTNET/SANDBOX PRIVATE ENDPOINT GATE"
STATUS: Final[str] = "ready_for_functional_preview_11_7_testnet_sandbox_private_endpoint_gate"

_ENTRY_FIELDS: Final[list[str]] = [
    "public_probe_preview_id",
    "source_capability",
    "display_name",
    "probe_preview_classification",
    "probe_surface_type",
    "planned_probe_category",
    "planned_symbol_scope",
    "planned_timeframe_scope",
    "planned_rate_limit_profile",
    "required_prior_gate",
    "eligible_for_11_7_private_endpoint_gate",
    "real_probe_allowed_now",
    "real_market_data_fetch_allowed_now",
    "network_io_allowed_now",
    "dns_lookup_allowed_now",
    "http_request_allowed_now",
    "websocket_allowed_now",
    "adapter_instantiation_allowed_now",
    "runtime_allowed_now",
    "credentials_allowed_now",
    "secrets_allowed_now",
    "private_endpoint_allowed_now",
    "account_fetch_allowed_now",
    "balance_fetch_allowed_now",
    "positions_fetch_allowed_now",
    "orders_fetch_allowed_now",
    "fills_fetch_allowed_now",
    "order_submission_allowed_now",
    "probe_preview_safe_for_offline_tests",
    "operator_visibility",
    "notes",
]
_PLANNED_SYMBOLS: Final[list[str]] = ["BTC/USDT", "ETH/USDT"]
_PLANNED_TIMEFRAMES: Final[list[str]] = ["1m", "5m"]

_BLOCKED_PUBLIC_PROBE_CAPABILITIES: Final[list[str]] = [
    "real public market data probe",
    "real market data fetch",
    "adapter instantiation",
    "adapter config application",
    "credential read",
    "secret read",
    "secure store read",
    "testnet connection",
    "sandbox connection",
    "live connection",
    "DNS lookup",
    "HTTP request",
    "WebSocket connection",
    "network I/O",
    "private endpoint access",
    "account fetch",
    "balance fetch",
    "positions fetch",
    "orders fetch",
    "fills fetch",
    "order submission",
    "fill simulation",
    "runtime loop",
    "scheduler",
    "QML action dispatch",
    "bridge API changes",
    "EXE packaging",
]
_SOURCE_BOUNDARIES: Final[list[str]] = [
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
    "no sandbox adapter import",
    "no exchange adapter runtime import",
    "no account module import",
    "no secrets module import",
    "no security store import",
    "no filesystem I/O",
    "no config file read",
    "no config discovery",
    "no YAML parse",
    "no JSON parse",
    "no environment variable read",
    "no credential read",
    "no credential validation",
    "no secret material handling",
    "no secure store read",
    "no secure store write",
    "no real market data fetch",
    "no network I/O",
    "no DNS lookup",
    "no HTTP request",
    "no WebSocket connection",
    "no private endpoint access",
    "no QML changes",
    "no bridge API changes",
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
]
_FUTURE_STEPS: Final[list[str]] = [
    "functional_preview_11_7_testnet_sandbox_private_endpoint_gate",
    "functional_preview_11_8_block_i_closure_audit",
]


def _credentials_gate_contract_reference(contract: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": contract["schema_version"],
        "testnet_sandbox_credentials_gate_contract_kind": contract[
            "testnet_sandbox_credentials_gate_contract_kind"
        ],
        "testnet_sandbox_credentials_gate_contract_status": contract[
            "testnet_sandbox_credentials_gate_contract_status"
        ],
        "testnet_sandbox_credentials_gate_contract_decision": contract[
            "testnet_sandbox_credentials_gate_contract_decision"
        ],
        "ready_for_block_i_6": contract["ready_for_block_i_6"],
        "next_step": contract["next_step"],
        "next_step_title": contract["next_step_title"],
        "status": contract["status"],
    }


def _entry(values: dict[str, Any]) -> dict[str, Any]:
    base = {
        "planned_symbol_scope": _PLANNED_SYMBOLS,
        "planned_timeframe_scope": _PLANNED_TIMEFRAMES,
        "real_probe_allowed_now": False,
        "real_market_data_fetch_allowed_now": False,
        "network_io_allowed_now": False,
        "dns_lookup_allowed_now": False,
        "http_request_allowed_now": False,
        "websocket_allowed_now": False,
        "adapter_instantiation_allowed_now": False,
        "runtime_allowed_now": False,
        "credentials_allowed_now": False,
        "secrets_allowed_now": False,
        "private_endpoint_allowed_now": False,
        "account_fetch_allowed_now": False,
        "balance_fetch_allowed_now": False,
        "positions_fetch_allowed_now": False,
        "orders_fetch_allowed_now": False,
        "fills_fetch_allowed_now": False,
        "order_submission_allowed_now": False,
        "probe_preview_safe_for_offline_tests": True,
    }
    base.update(values)
    return {field: base[field] for field in _ENTRY_FIELDS}


def _public_probe_preview_entries() -> list[dict[str, Any]]:
    return [
        _entry(
            {
                "public_probe_preview_id": "public_probe_preview_read_only_market_data_provider",
                "source_capability": "read_only_market_data_provider",
                "display_name": "Read-only market data provider public probe preview",
                "probe_preview_classification": "lowest_risk_public_market_data_probe_preview",
                "probe_surface_type": "public_market_data_probe_preview",
                "planned_probe_category": "public_market_data",
                "planned_rate_limit_profile": "public_read_only_preview",
                "required_prior_gate": "adapter_config_gate_read_only_market_data_provider",
                "eligible_for_11_7_private_endpoint_gate": False,
                "operator_visibility": "read_only_future",
                "notes": "Static read-only public market data preview only; no fetch, network, credentials, adapter, or runtime activation.",
            }
        ),
        _entry(
            {
                "public_probe_preview_id": "public_probe_preview_exchange_adapter_layer",
                "source_capability": "exchange_adapter_layer",
                "display_name": "Exchange adapter layer public probe preview",
                "probe_preview_classification": "exchange_adapter_public_market_data_probe_preview",
                "probe_surface_type": "exchange_adapter_public_probe_preview",
                "planned_probe_category": "exchange_public_market_data",
                "planned_rate_limit_profile": "exchange_adapter_public_preview_guarded",
                "required_prior_gate": "credentials_gate_exchange_adapter_layer",
                "eligible_for_11_7_private_endpoint_gate": True,
                "operator_visibility": "blocked_until_probe_gate",
                "notes": "Static exchange adapter public probe preview only; no adapter instantiation, credential read, network, private endpoint, or runtime activation.",
            }
        ),
    ]


def _count_enabled(entries: list[dict[str, Any]], field: str) -> int:
    return sum(1 for entry in entries if entry[field] is True)


def build_preview_testnet_sandbox_public_market_data_probe_preview() -> dict[str, Any]:
    """Build the pure-data Block I public market data probe preview shape."""

    credentials_gate = build_preview_testnet_sandbox_credentials_gate_contract()
    entries = _public_probe_preview_entries()
    preview_ids = [entry["public_probe_preview_id"] for entry in entries]
    symbols_by_id = {
        entry["public_probe_preview_id"]: entry["planned_symbol_scope"] for entry in entries
    }
    timeframes_by_id = {
        entry["public_probe_preview_id"]: entry["planned_timeframe_scope"] for entry in entries
    }
    return {
        "schema_version": PREVIEW_TESTNET_SANDBOX_PUBLIC_MARKET_DATA_PROBE_PREVIEW_SCHEMA_VERSION,
        "testnet_sandbox_public_market_data_probe_preview_kind": PREVIEW_TESTNET_SANDBOX_PUBLIC_MARKET_DATA_PROBE_PREVIEW_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "testnet_sandbox_public_market_data_probe_preview_status": TESTNET_SANDBOX_PUBLIC_MARKET_DATA_PROBE_PREVIEW_STATUS,
        "testnet_sandbox_public_market_data_probe_preview_decision": TESTNET_SANDBOX_PUBLIC_MARKET_DATA_PROBE_PREVIEW_DECISION,
        "ready_for_block_i_7": READY_FOR_BLOCK_I_7,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "credentials_gate_contract_reference": _credentials_gate_contract_reference(
            credentials_gate
        ),
        "public_probe_preview_scope": {
            "scope_name": "testnet_sandbox_public_market_data_probe_preview",
            "probe_preview_only": True,
            "derived_from_credentials_gate_11_5": True,
            "real_probe_allowed_now": False,
            "real_market_data_fetch_allowed_now": False,
            "network_io_allowed_now": False,
            "dns_lookup_allowed_now": False,
            "http_request_allowed_now": False,
            "websocket_allowed_now": False,
            "adapter_instantiation_allowed_now": False,
            "adapter_wiring_allowed_now": False,
            "runtime_execution_allowed_now": False,
            "scheduler_allowed_now": False,
            "config_file_read_allowed_now": False,
            "config_discovery_allowed_now": False,
            "yaml_parse_allowed_now": False,
            "json_parse_allowed_now": False,
            "environment_variable_read_allowed_now": False,
            "credential_secret_read_allowed_now": False,
            "credential_discovery_allowed_now": False,
            "credential_validation_allowed_now": False,
            "credential_material_handling_allowed_now": False,
            "secret_material_handling_allowed_now": False,
            "secure_store_read_allowed_now": False,
            "secure_store_write_allowed_now": False,
            "private_endpoint_access_allowed_now": False,
            "account_fetch_allowed_now": False,
            "balance_fetch_allowed_now": False,
            "positions_fetch_allowed_now": False,
            "orders_fetch_allowed_now": False,
            "fills_fetch_allowed_now": False,
            "order_submission_allowed_now": False,
            "fill_simulation_allowed_now": False,
            "qml_changes_allowed": False,
            "new_qml_method_calls_allowed": False,
            "bridge_api_changes_allowed": False,
            "exe_packaging_in_scope": False,
            "bat_productization_allowed": False,
            "exe_direction_preserved": True,
        },
        "public_probe_preview_entries": entries,
        "default_public_probe_preview_selection": {
            "public_probe_preview_id": "public_probe_preview_read_only_market_data_provider",
            "source_capability": "read_only_market_data_provider",
            "reason": "lowest-risk public market data probe preview; no real fetch, no network I/O, no credentials, no private endpoint, no runtime activation",
            "real_probe_allowed_now": False,
            "real_market_data_fetch_allowed_now": False,
            "network_io_allowed_now": False,
        },
        "public_probe_preview_summary": {
            "entry_count": len(entries),
            "default_selection_id": "public_probe_preview_read_only_market_data_provider",
            "real_probe_enabled_entry_count": _count_enabled(entries, "real_probe_allowed_now"),
            "real_market_data_fetch_enabled_entry_count": _count_enabled(
                entries, "real_market_data_fetch_allowed_now"
            ),
            "network_enabled_entry_count": _count_enabled(entries, "network_io_allowed_now"),
            "dns_lookup_enabled_entry_count": _count_enabled(entries, "dns_lookup_allowed_now"),
            "http_request_enabled_entry_count": _count_enabled(entries, "http_request_allowed_now"),
            "websocket_enabled_entry_count": _count_enabled(entries, "websocket_allowed_now"),
            "adapter_instantiation_enabled_entry_count": _count_enabled(
                entries, "adapter_instantiation_allowed_now"
            ),
            "runtime_enabled_entry_count": _count_enabled(entries, "runtime_allowed_now"),
            "credentials_enabled_entry_count": _count_enabled(entries, "credentials_allowed_now"),
            "secrets_enabled_entry_count": _count_enabled(entries, "secrets_allowed_now"),
            "private_endpoint_enabled_entry_count": _count_enabled(
                entries, "private_endpoint_allowed_now"
            ),
            "account_fetch_enabled_entry_count": _count_enabled(
                entries, "account_fetch_allowed_now"
            ),
            "balance_fetch_enabled_entry_count": _count_enabled(
                entries, "balance_fetch_allowed_now"
            ),
            "positions_fetch_enabled_entry_count": _count_enabled(
                entries, "positions_fetch_allowed_now"
            ),
            "orders_fetch_enabled_entry_count": _count_enabled(entries, "orders_fetch_allowed_now"),
            "fills_fetch_enabled_entry_count": _count_enabled(entries, "fills_fetch_allowed_now"),
            "order_submission_enabled_entry_count": _count_enabled(
                entries, "order_submission_allowed_now"
            ),
            "offline_safe_entry_count": _count_enabled(
                entries, "probe_preview_safe_for_offline_tests"
            ),
            "entries_eligible_for_11_7_private_endpoint_gate": _count_enabled(
                entries, "eligible_for_11_7_private_endpoint_gate"
            ),
            "safe_to_render_in_future_ui_as_read_only": True,
            "safe_for_runtime_execution_now": False,
        },
        "public_probe_preview_matrix": {
            "public_probe_preview_ids": preview_ids,
            "public_market_data_only_preview_ids": [
                "public_probe_preview_read_only_market_data_provider"
            ],
            "exchange_adapter_public_probe_preview_ids": [
                "public_probe_preview_exchange_adapter_layer"
            ],
            "previews_eligible_for_private_endpoint_gate_later": [
                "public_probe_preview_exchange_adapter_layer"
            ],
            "previews_never_runtime_enabled_in_11_6": preview_ids,
            "planned_symbols_by_preview_id": symbols_by_id,
            "planned_timeframes_by_preview_id": timeframes_by_id,
        },
        "blocked_public_probe_capabilities": _BLOCKED_PUBLIC_PROBE_CAPABILITIES,
        "public_probe_boundaries": {
            "public_probe_preview_is_static": True,
            "public_probe_preview_is_derived_from_11_5": True,
            "public_probe_preview_can_feed_11_7_private_endpoint_gate": True,
            "public_probe_preview_cannot_feed_runtime_directly": True,
            "public_probe_preview_cannot_fetch_market_data": True,
            "public_probe_preview_cannot_open_network_connection": True,
            "public_probe_preview_cannot_perform_dns_lookup": True,
            "public_probe_preview_cannot_perform_http_request": True,
            "public_probe_preview_cannot_open_websocket": True,
            "public_probe_preview_cannot_read_credentials": True,
            "public_probe_preview_cannot_read_secrets": True,
            "public_probe_preview_cannot_read_secure_store": True,
            "public_probe_preview_cannot_access_private_endpoints": True,
            "public_probe_preview_cannot_fetch_account": True,
            "public_probe_preview_balance_read_blocked": True,
            "public_probe_preview_cannot_fetch_positions": True,
            "public_probe_preview_cannot_fetch_orders": True,
            "public_probe_preview_cannot_fetch_fills": True,
            "public_probe_preview_cannot_submit_orders": True,
            "public_probe_preview_cannot_change_qml_or_bridge": True,
        },
        "non_activation_evidence": {
            "credentials_gate_contract_11_5_read": True,
            "public_market_data_probe_preview_built": True,
            "real_public_market_data_probe_performed": False,
            "real_market_data_fetch_performed": False,
            "config_files_read": False,
            "config_discovery_performed": False,
            "yaml_parsed": False,
            "json_parsed": False,
            "environment_variables_read": False,
            "credential_secret_read": False,
            "credential_discovery_performed": False,
            "credential_validation_performed": False,
            "credential_material_handled": False,
            "secret_material_handled": False,
            "secure_store_read": False,
            "secure_store_write": False,
            "backend_modules_imported": False,
            "backend_modules_activated": False,
            "adapter_instantiated": False,
            "adapter_config_applied": False,
            "runtime_started": False,
            "scheduler_started": False,
            "dns_lookup_performed": False,
            "http_request_performed": False,
            "websocket_opened": False,
            "network_io_performed": False,
            "credentials_read": False,
            "secrets_read": False,
            "private_endpoint_accessed": False,
            "account_fetch_performed": False,
            "balance_fetch_performed": False,
            "positions_fetch_performed": False,
            "orders_fetch_performed": False,
            "fills_fetch_performed": False,
            "order_submitted": False,
            "order_generated": False,
            "fill_simulated": False,
            "qml_changed": False,
            "bridge_api_changed": False,
        },
        "source_boundaries": _SOURCE_BOUNDARIES,
        "future_steps": _FUTURE_STEPS,
        "status": STATUS,
    }
