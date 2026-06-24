"""FUNCTIONAL-PREVIEW-11.7 Block I private endpoint gate contract.

Pure-data Testnet/Sandbox private endpoint gate contract derived from the 11.6
public market data probe preview. This module intentionally does not access
private endpoints, fetch account/balance/positions/orders/fills, submit orders,
open network connections, read configuration, read credentials, or activate
runtime/adapter code.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_testnet_sandbox_public_market_data_probe_preview import (
    build_preview_testnet_sandbox_public_market_data_probe_preview,
)

PREVIEW_TESTNET_SANDBOX_PRIVATE_ENDPOINT_GATE_SCHEMA_VERSION: Final[str] = (
    "preview_testnet_sandbox_private_endpoint_gate.v1"
)
PREVIEW_TESTNET_SANDBOX_PRIVATE_ENDPOINT_GATE_KIND: Final[str] = (
    "functional_preview_block_i_testnet_sandbox_private_endpoint_gate"
)
BLOCK_ID: Final[str] = "I"
STEP_ID: Final[str] = "11.7"
TESTNET_SANDBOX_PRIVATE_ENDPOINT_GATE_STATUS: Final[str] = (
    "testnet_sandbox_private_endpoint_gate_ready_no_private_endpoint_access"
)
TESTNET_SANDBOX_PRIVATE_ENDPOINT_GATE_DECISION: Final[str] = (
    "BUILD_PRIVATE_ENDPOINT_GATE_ONLY_NO_PRIVATE_CALLS_NO_ACCOUNT_FETCH_NO_ORDER_FLOW"
)
READY_FOR_BLOCK_I_8: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-11.8"
NEXT_STEP_TITLE: Final[str] = "BLOK I — TESTNET/SANDBOX ADAPTER CLOSURE AUDIT"
STATUS: Final[str] = "ready_for_functional_preview_11_8_testnet_sandbox_adapter_closure_audit"

_ENTRY_FIELDS: Final[list[str]] = [
    "private_endpoint_gate_id",
    "source_public_probe_preview_id",
    "source_capability",
    "display_name",
    "private_endpoint_gate_classification",
    "private_endpoint_surface_type",
    "planned_private_endpoint_categories",
    "required_prior_gate",
    "required_future_risk_gate",
    "required_future_observability_gate",
    "allowed_future_private_read_categories",
    "forbidden_private_endpoint_categories",
    "eligible_for_11_8_closure_audit",
    "private_endpoint_access_allowed_now",
    "private_endpoint_probe_allowed_now",
    "private_endpoint_validation_allowed_now",
    "account_fetch_allowed_now",
    "balance_fetch_allowed_now",
    "positions_fetch_allowed_now",
    "orders_fetch_allowed_now",
    "fills_fetch_allowed_now",
    "order_submission_allowed_now",
    "order_generation_allowed_now",
    "fill_simulation_allowed_now",
    "real_market_data_fetch_allowed_now",
    "network_io_allowed_now",
    "dns_lookup_allowed_now",
    "http_request_allowed_now",
    "websocket_allowed_now",
    "adapter_instantiation_allowed_now",
    "runtime_allowed_now",
    "credentials_allowed_now",
    "secrets_allowed_now",
    "gate_safe_for_offline_tests",
    "operator_visibility",
    "notes",
]
_ALLOWED_READ_CATEGORIES: Final[list[str]] = [
    "account_read",
    "balance_read",
    "positions_read",
    "orders_read",
    "fills_read",
]
_FORBIDDEN_PRIVATE_ENDPOINT_CATEGORIES: Final[list[str]] = [
    "order_submission",
    "order_cancel",
    "order_replace",
    "withdrawal",
    "transfer",
    "deposit_address_generation",
    "live_trading",
    "margin_or_leverage_mutation",
]
_BLOCKED_PRIVATE_ENDPOINT_CAPABILITIES: Final[list[str]] = [
    "real private endpoint access",
    "private endpoint probe",
    "private endpoint validation",
    "account fetch",
    "balance fetch",
    "positions fetch",
    "orders fetch",
    "fills fetch",
    "order submission",
    "order generation",
    "order cancel",
    "order replace",
    "withdrawal",
    "transfer",
    "deposit address generation",
    "margin or leverage mutation",
    "live trading",
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
    "no private endpoint access",
    "no account fetch",
    "no balance fetch",
    "no positions fetch",
    "no orders fetch",
    "no fills fetch",
    "no order submission",
    "no order generation",
    "no order cancel",
    "no order replace",
    "no withdrawal",
    "no transfer",
    "no margin/leverage mutation",
    "no network I/O",
    "no DNS lookup",
    "no HTTP request",
    "no WebSocket connection",
    "no QML changes",
    "no bridge API changes",
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
]
_FUTURE_STEPS: Final[list[str]] = ["functional_preview_11_8_testnet_sandbox_adapter_closure_audit"]
_GATE_ID: Final[str] = "private_endpoint_gate_exchange_adapter_layer"


def _public_market_data_probe_preview_reference(preview: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": preview["schema_version"],
        "testnet_sandbox_public_market_data_probe_preview_kind": preview[
            "testnet_sandbox_public_market_data_probe_preview_kind"
        ],
        "testnet_sandbox_public_market_data_probe_preview_status": preview[
            "testnet_sandbox_public_market_data_probe_preview_status"
        ],
        "testnet_sandbox_public_market_data_probe_preview_decision": preview[
            "testnet_sandbox_public_market_data_probe_preview_decision"
        ],
        "ready_for_block_i_7": preview["ready_for_block_i_7"],
        "next_step": preview["next_step"],
        "next_step_title": preview["next_step_title"],
        "status": preview["status"],
    }


def _private_endpoint_gate_entries() -> list[dict[str, Any]]:
    base = {
        "private_endpoint_gate_id": _GATE_ID,
        "source_public_probe_preview_id": "public_probe_preview_exchange_adapter_layer",
        "source_capability": "exchange_adapter_layer",
        "display_name": "Exchange adapter layer private endpoint gate",
        "private_endpoint_gate_classification": "private_endpoint_contract_only",
        "private_endpoint_surface_type": "exchange_adapter_private_endpoint_gate_contract",
        "planned_private_endpoint_categories": _ALLOWED_READ_CATEGORIES,
        "required_prior_gate": "public_probe_preview_exchange_adapter_layer",
        "required_future_risk_gate": "BLOK J — RISK GOVERNOR / LIMITS / KILL SWITCH",
        "required_future_observability_gate": "BLOK K — OBSERVABILITY / AUDIT / SOAK",
        "allowed_future_private_read_categories": _ALLOWED_READ_CATEGORIES,
        "forbidden_private_endpoint_categories": _FORBIDDEN_PRIVATE_ENDPOINT_CATEGORIES,
        "eligible_for_11_8_closure_audit": True,
        "private_endpoint_access_allowed_now": False,
        "private_endpoint_probe_allowed_now": False,
        "private_endpoint_validation_allowed_now": False,
        "account_fetch_allowed_now": False,
        "balance_fetch_allowed_now": False,
        "positions_fetch_allowed_now": False,
        "orders_fetch_allowed_now": False,
        "fills_fetch_allowed_now": False,
        "order_submission_allowed_now": False,
        "order_generation_allowed_now": False,
        "fill_simulation_allowed_now": False,
        "real_market_data_fetch_allowed_now": False,
        "network_io_allowed_now": False,
        "dns_lookup_allowed_now": False,
        "http_request_allowed_now": False,
        "websocket_allowed_now": False,
        "adapter_instantiation_allowed_now": False,
        "runtime_allowed_now": False,
        "credentials_allowed_now": False,
        "secrets_allowed_now": False,
        "gate_safe_for_offline_tests": True,
        "operator_visibility": "blocked_until_private_endpoint_gate",
        "notes": "Static private endpoint gate contract only; no account, balance, positions, orders, fills, network, credential, adapter, order, or runtime access.",
    }
    return [{field: base[field] for field in _ENTRY_FIELDS}]


def _count_enabled(entries: list[dict[str, Any]], field: str) -> int:
    return sum(1 for entry in entries if entry[field] is True)


def build_preview_testnet_sandbox_private_endpoint_gate() -> dict[str, Any]:
    """Build the pure-data Block I private endpoint gate contract shape."""

    public_probe_preview = build_preview_testnet_sandbox_public_market_data_probe_preview()
    entries = _private_endpoint_gate_entries()
    gate_ids = [entry["private_endpoint_gate_id"] for entry in entries]
    return {
        "schema_version": PREVIEW_TESTNET_SANDBOX_PRIVATE_ENDPOINT_GATE_SCHEMA_VERSION,
        "testnet_sandbox_private_endpoint_gate_kind": PREVIEW_TESTNET_SANDBOX_PRIVATE_ENDPOINT_GATE_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "testnet_sandbox_private_endpoint_gate_status": TESTNET_SANDBOX_PRIVATE_ENDPOINT_GATE_STATUS,
        "testnet_sandbox_private_endpoint_gate_decision": TESTNET_SANDBOX_PRIVATE_ENDPOINT_GATE_DECISION,
        "ready_for_block_i_8": READY_FOR_BLOCK_I_8,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "public_market_data_probe_preview_reference": _public_market_data_probe_preview_reference(
            public_probe_preview
        ),
        "private_endpoint_gate_scope": {
            "scope_name": "testnet_sandbox_private_endpoint_gate",
            "private_endpoint_gate_contract_only": True,
            "derived_from_public_probe_preview_11_6": True,
            "private_endpoint_access_allowed_now": False,
            "private_endpoint_probe_allowed_now": False,
            "private_endpoint_validation_allowed_now": False,
            "account_fetch_allowed_now": False,
            "balance_fetch_allowed_now": False,
            "positions_fetch_allowed_now": False,
            "orders_fetch_allowed_now": False,
            "fills_fetch_allowed_now": False,
            "order_submission_allowed_now": False,
            "order_generation_allowed_now": False,
            "fill_simulation_allowed_now": False,
            "real_market_data_fetch_allowed_now": False,
            "real_public_probe_allowed_now": False,
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
            "qml_changes_allowed": False,
            "new_qml_method_calls_allowed": False,
            "bridge_api_changes_allowed": False,
            "exe_packaging_in_scope": False,
            "bat_productization_allowed": False,
            "exe_direction_preserved": True,
        },
        "private_endpoint_gate_entries": entries,
        "default_private_endpoint_gate_selection": {
            "private_endpoint_gate_id": _GATE_ID,
            "source_capability": "exchange_adapter_layer",
            "reason": "only 11.7-eligible public probe preview; private endpoint contract only, no account/balance/positions/orders/fills fetch, no order submission, no network I/O",
            "private_endpoint_access_allowed_now": False,
            "network_io_allowed_now": False,
            "order_submission_allowed_now": False,
        },
        "private_endpoint_gate_summary": {
            "entry_count": len(entries),
            "default_selection_id": _GATE_ID,
            "private_endpoint_access_enabled_entry_count": _count_enabled(
                entries, "private_endpoint_access_allowed_now"
            ),
            "private_endpoint_probe_enabled_entry_count": _count_enabled(
                entries, "private_endpoint_probe_allowed_now"
            ),
            "private_endpoint_validation_enabled_entry_count": _count_enabled(
                entries, "private_endpoint_validation_allowed_now"
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
            "order_generation_enabled_entry_count": _count_enabled(
                entries, "order_generation_allowed_now"
            ),
            "fill_simulation_enabled_entry_count": _count_enabled(
                entries, "fill_simulation_allowed_now"
            ),
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
            "offline_safe_entry_count": _count_enabled(entries, "gate_safe_for_offline_tests"),
            "entries_eligible_for_11_8_closure_audit": _count_enabled(
                entries, "eligible_for_11_8_closure_audit"
            ),
            "safe_to_render_in_future_ui_as_read_only": True,
            "safe_for_runtime_execution_now": False,
            "safe_for_order_execution_now": False,
            "safe_for_private_endpoint_access_now": False,
        },
        "private_endpoint_contract_requirements": {
            "planned_private_read_categories_by_source_capability": {
                "exchange_adapter_layer": _ALLOWED_READ_CATEGORIES
            },
            "forbidden_private_endpoint_categories_by_source_capability": {
                "exchange_adapter_layer": _FORBIDDEN_PRIVATE_ENDPOINT_CATEGORIES
            },
            "required_future_gates_by_source_capability": {
                "exchange_adapter_layer": {
                    "risk_gate": "BLOK J — RISK GOVERNOR / LIMITS / KILL SWITCH",
                    "observability_gate": "BLOK K — OBSERVABILITY / AUDIT / SOAK",
                }
            },
            "global_allowed_future_private_read_categories": _ALLOWED_READ_CATEGORIES,
            "global_forbidden_private_endpoint_categories": _FORBIDDEN_PRIVATE_ENDPOINT_CATEGORIES,
            "private_endpoint_read_only_contract": True,
            "private_endpoint_mutation_forbidden": True,
            "order_submission_forbidden": True,
            "live_trading_forbidden": True,
            "risk_governor_required_before_any_order_flow": True,
            "observability_required_before_any_soak": True,
        },
        "private_endpoint_gate_matrix": {
            "private_endpoint_gate_ids": gate_ids,
            "gates_eligible_for_11_8_closure_audit": gate_ids,
            "gates_requiring_risk_governor_later": gate_ids,
            "gates_requiring_observability_soak_later": gate_ids,
            "gates_never_runtime_enabled_in_11_7": gate_ids,
            "gates_never_order_enabled_in_11_7": gate_ids,
            "gates_never_private_endpoint_enabled_in_11_7": gate_ids,
        },
        "blocked_private_endpoint_capabilities": _BLOCKED_PRIVATE_ENDPOINT_CAPABILITIES,
        "private_endpoint_gate_boundaries": {
            "private_endpoint_gate_is_static": True,
            "private_endpoint_gate_is_derived_from_11_6": True,
            "private_endpoint_gate_can_feed_11_8_closure_audit": True,
            "private_endpoint_gate_cannot_feed_runtime_directly": True,
            "private_endpoint_gate_cannot_access_private_endpoints": True,
            "private_endpoint_gate_cannot_probe_private_endpoints": True,
            "private_endpoint_gate_cannot_validate_private_endpoints": True,
            "private_endpoint_gate_cannot_fetch_account": True,
            "private_endpoint_gate_balance_read_blocked": True,
            "private_endpoint_gate_cannot_fetch_positions": True,
            "private_endpoint_gate_cannot_fetch_orders": True,
            "private_endpoint_gate_cannot_fetch_fills": True,
            "private_endpoint_gate_cannot_submit_orders": True,
            "private_endpoint_gate_cannot_generate_orders": True,
            "private_endpoint_gate_cannot_cancel_orders": True,
            "private_endpoint_gate_cannot_replace_orders": True,
            "private_endpoint_gate_cannot_withdraw": True,
            "private_endpoint_gate_cannot_transfer": True,
            "private_endpoint_gate_cannot_mutate_margin_or_leverage": True,
            "private_endpoint_gate_cannot_fetch_market_data": True,
            "private_endpoint_gate_cannot_open_network_connection": True,
            "private_endpoint_gate_cannot_perform_dns_lookup": True,
            "private_endpoint_gate_cannot_perform_http_request": True,
            "private_endpoint_gate_cannot_open_websocket": True,
            "private_endpoint_gate_cannot_read_credentials": True,
            "private_endpoint_gate_cannot_read_secrets": True,
            "private_endpoint_gate_cannot_read_secure_store": True,
            "private_endpoint_gate_cannot_change_qml_or_bridge": True,
        },
        "non_activation_evidence": {
            "public_market_data_probe_preview_11_6_read": True,
            "private_endpoint_gate_built": True,
            "real_private_endpoint_access_performed": False,
            "private_endpoint_probe_performed": False,
            "private_endpoint_validation_performed": False,
            "account_fetch_performed": False,
            "balance_fetch_performed": False,
            "positions_fetch_performed": False,
            "orders_fetch_performed": False,
            "fills_fetch_performed": False,
            "order_submitted": False,
            "order_generated": False,
            "order_cancelled": False,
            "order_replaced": False,
            "withdrawal_performed": False,
            "transfer_performed": False,
            "deposit_address_generated": False,
            "margin_or_leverage_mutated": False,
            "live_trading_performed": False,
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
            "qml_changed": False,
            "bridge_api_changed": False,
        },
        "source_boundaries": _SOURCE_BOUNDARIES,
        "future_steps": _FUTURE_STEPS,
        "status": STATUS,
    }
