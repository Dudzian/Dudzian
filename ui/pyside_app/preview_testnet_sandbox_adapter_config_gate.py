"""FUNCTIONAL-PREVIEW-11.4 Block I adapter config gate shape.

Pure-data Testnet/Sandbox adapter config gate derived from the 11.3 static
connectivity fixture. This module intentionally does not read configuration
files, parse config formats, import runtime/exchange adapters, touch secrets, or
perform network activity.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_testnet_sandbox_static_connectivity_fixture import (
    build_preview_testnet_sandbox_static_connectivity_fixture,
)

PREVIEW_TESTNET_SANDBOX_ADAPTER_CONFIG_GATE_SCHEMA_VERSION: Final[str] = (
    "preview_testnet_sandbox_adapter_config_gate.v1"
)
PREVIEW_TESTNET_SANDBOX_ADAPTER_CONFIG_GATE_KIND: Final[str] = (
    "functional_preview_block_i_testnet_sandbox_adapter_config_gate"
)
BLOCK_ID: Final[str] = "I"
STEP_ID: Final[str] = "11.4"
TESTNET_SANDBOX_ADAPTER_CONFIG_GATE_STATUS: Final[str] = (
    "testnet_sandbox_adapter_config_gate_ready_shape_only_no_config_read"
)
TESTNET_SANDBOX_ADAPTER_CONFIG_GATE_DECISION: Final[str] = (
    "BUILD_ADAPTER_CONFIG_GATE_SHAPE_ONLY_NO_CONFIG_READ_NO_NETWORK_NO_RUNTIME"
)
READY_FOR_BLOCK_I_5: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-11.5"
NEXT_STEP_TITLE: Final[str] = "TESTNET/SANDBOX CREDENTIALS GATE CONTRACT"
STATUS: Final[str] = "ready_for_functional_preview_11_5_testnet_sandbox_credentials_gate_contract"

_ENTRY_FIELDS: Final[list[str]] = [
    "config_gate_id",
    "source_connectivity_fixture_id",
    "source_capability",
    "display_name",
    "config_gate_classification",
    "config_surface_type",
    "required_config_shape",
    "optional_config_shape",
    "forbidden_config_material",
    "eligible_for_11_5_credentials_gate",
    "eligible_for_11_6_public_market_data_probe_preview",
    "eligible_for_11_7_private_endpoint_gate",
    "config_file_read_allowed_now",
    "config_discovery_allowed_now",
    "yaml_parse_allowed_now",
    "json_parse_allowed_now",
    "environment_variable_read_allowed_now",
    "credentials_allowed_now",
    "secrets_allowed_now",
    "network_io_allowed_now",
    "dns_lookup_allowed_now",
    "http_request_allowed_now",
    "websocket_allowed_now",
    "private_endpoint_allowed_now",
    "order_submission_allowed_now",
    "runtime_allowed_now",
    "gate_safe_for_offline_tests",
    "operator_visibility",
    "notes",
]

_ENTRY_ORDER: Final[list[str]] = [
    "read_only_market_data_provider",
    "exchange_adapter_layer",
    "exchange_network_guard",
]

_ENTRY_OVERRIDES: Final[dict[str, dict[str, Any]]] = {
    "read_only_market_data_provider": {
        "config_gate_id": "adapter_config_gate_read_only_market_data_provider",
        "source_connectivity_fixture_id": "static_connectivity_fixture_read_only_market_data_provider",
        "display_name": "Read-only market data provider config gate",
        "config_gate_classification": "public_market_data_config_shape",
        "config_surface_type": "public_market_data_config_contract",
        "required_config_shape": [
            "mode",
            "provider_id",
            "symbols_allowlist",
            "timeframe",
            "rate_limit_profile",
        ],
        "optional_config_shape": [
            "fixture_profile",
            "recorded_replay_profile",
            "public_endpoint_profile_name",
        ],
        "forbidden_config_material": [
            "api_key",
            "api_secret",
            "passphrase",
            "account_id",
            "private_endpoint_url",
            "order_endpoint_url",
        ],
        "eligible_for_11_5_credentials_gate": False,
        "eligible_for_11_6_public_market_data_probe_preview": True,
        "eligible_for_11_7_private_endpoint_gate": False,
        "operator_visibility": "read_only_future",
    },
    "exchange_adapter_layer": {
        "config_gate_id": "adapter_config_gate_exchange_adapter_layer",
        "source_connectivity_fixture_id": "static_connectivity_fixture_exchange_adapter_layer",
        "display_name": "Exchange adapter layer config gate",
        "config_gate_classification": "exchange_adapter_config_shape",
        "config_surface_type": "exchange_adapter_config_contract",
        "required_config_shape": [
            "mode",
            "exchange_id",
            "adapter_family",
            "environment",
            "sandbox_or_testnet_flag",
            "symbols_allowlist",
            "rate_limit_profile",
            "network_guard_profile",
        ],
        "optional_config_shape": [
            "public_endpoint_profile_name",
            "credential_profile_reference",
            "private_endpoint_profile_name",
            "paper_oracle_comparison_profile",
        ],
        "forbidden_config_material": [
            "api_key_value",
            "api_secret_value",
            "passphrase_value",
            "raw_secret",
            "live_endpoint_override",
            "order_submission_enabled",
        ],
        "eligible_for_11_5_credentials_gate": True,
        "eligible_for_11_6_public_market_data_probe_preview": True,
        "eligible_for_11_7_private_endpoint_gate": True,
        "operator_visibility": "blocked_until_gated",
    },
    "exchange_network_guard": {
        "config_gate_id": "adapter_config_gate_exchange_network_guard",
        "source_connectivity_fixture_id": "static_connectivity_fixture_exchange_network_guard",
        "display_name": "Exchange network guard config gate",
        "config_gate_classification": "network_guard_config_shape",
        "config_surface_type": "network_guard_config_contract",
        "required_config_shape": [
            "mode",
            "network_policy",
            "allowed_endpoint_categories",
            "blocked_endpoint_categories",
            "rate_limit_profile",
            "audit_profile",
        ],
        "optional_config_shape": [
            "public_probe_policy_name",
            "private_endpoint_policy_name",
            "circuit_breaker_profile",
        ],
        "forbidden_config_material": [
            "api_key",
            "api_secret",
            "passphrase",
            "raw_secret",
            "account_id",
            "order_endpoint_url",
        ],
        "eligible_for_11_5_credentials_gate": False,
        "eligible_for_11_6_public_market_data_probe_preview": False,
        "eligible_for_11_7_private_endpoint_gate": True,
        "operator_visibility": "safety_guard_future",
    },
}

_BLOCKED_CONFIG_GATE_CAPABILITIES: Final[list[str]] = [
    "real config file read",
    "config discovery",
    "YAML parse",
    "JSON parse",
    "environment variable read",
    "credential material handling",
    "secret material handling",
    "adapter instantiation",
    "adapter config application",
    "testnet connection",
    "sandbox connection",
    "live connection",
    "DNS lookup",
    "HTTP request",
    "WebSocket connection",
    "network I/O",
    "private endpoint access",
    "public market data fetch",
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
    "no filesystem I/O",
    "no config file read",
    "no config discovery",
    "no YAML parse",
    "no JSON parse",
    "no environment variable read",
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

_FUTURE_STEPS: Final[list[str]] = [
    "functional_preview_11_5_testnet_sandbox_credentials_gate_contract",
    "functional_preview_11_6_testnet_sandbox_public_market_data_probe_preview",
    "functional_preview_11_7_testnet_sandbox_private_endpoint_gate",
    "functional_preview_11_8_block_i_closure_audit",
]


def _static_connectivity_fixture_reference(fixture: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": fixture["schema_version"],
        "testnet_sandbox_static_connectivity_fixture_kind": fixture[
            "testnet_sandbox_static_connectivity_fixture_kind"
        ],
        "testnet_sandbox_static_connectivity_fixture_status": fixture[
            "testnet_sandbox_static_connectivity_fixture_status"
        ],
        "testnet_sandbox_static_connectivity_fixture_decision": fixture[
            "testnet_sandbox_static_connectivity_fixture_decision"
        ],
        "ready_for_block_i_4": fixture["ready_for_block_i_4"],
        "next_step": fixture["next_step"],
        "next_step_title": fixture["next_step_title"],
        "status": fixture["status"],
    }


def _adapter_config_gate_entries(fixture: dict[str, Any]) -> list[dict[str, Any]]:
    fixture_entries = {
        entry["source_capability"]: entry
        for entry in fixture["static_connectivity_fixture_entries"]
        if entry["eligible_for_11_4_config_gate"] is True
    }
    entries: list[dict[str, Any]] = []
    for source_capability in _ENTRY_ORDER:
        source_entry = fixture_entries[source_capability]
        entry = {
            "source_capability": source_capability,
            "config_file_read_allowed_now": False,
            "config_discovery_allowed_now": False,
            "yaml_parse_allowed_now": False,
            "json_parse_allowed_now": False,
            "environment_variable_read_allowed_now": False,
            "credentials_allowed_now": False,
            "secrets_allowed_now": False,
            "network_io_allowed_now": False,
            "dns_lookup_allowed_now": False,
            "http_request_allowed_now": False,
            "websocket_allowed_now": False,
            "private_endpoint_allowed_now": False,
            "order_submission_allowed_now": False,
            "runtime_allowed_now": False,
            "gate_safe_for_offline_tests": True,
            "notes": source_entry["notes"],
        }
        entry.update(_ENTRY_OVERRIDES[source_capability])
        entries.append({field: entry[field] for field in _ENTRY_FIELDS})
    return entries


def _count_enabled(entries: list[dict[str, Any]], field: str) -> int:
    return sum(1 for entry in entries if entry[field] is True)


def build_preview_testnet_sandbox_adapter_config_gate() -> dict[str, Any]:
    """Build the pure-data Block I adapter config gate shape."""

    fixture = build_preview_testnet_sandbox_static_connectivity_fixture()
    entries = _adapter_config_gate_entries(fixture)
    gate_ids = [entry["config_gate_id"] for entry in entries]
    required_by_capability = {
        entry["source_capability"]: entry["required_config_shape"] for entry in entries
    }
    forbidden_by_capability = {
        entry["source_capability"]: entry["forbidden_config_material"] for entry in entries
    }
    return {
        "schema_version": PREVIEW_TESTNET_SANDBOX_ADAPTER_CONFIG_GATE_SCHEMA_VERSION,
        "testnet_sandbox_adapter_config_gate_kind": PREVIEW_TESTNET_SANDBOX_ADAPTER_CONFIG_GATE_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "testnet_sandbox_adapter_config_gate_status": TESTNET_SANDBOX_ADAPTER_CONFIG_GATE_STATUS,
        "testnet_sandbox_adapter_config_gate_decision": TESTNET_SANDBOX_ADAPTER_CONFIG_GATE_DECISION,
        "ready_for_block_i_5": READY_FOR_BLOCK_I_5,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "static_connectivity_fixture_reference": _static_connectivity_fixture_reference(fixture),
        "config_gate_scope": {
            "scope_name": "testnet_sandbox_adapter_config_gate",
            "config_gate_shape_only": True,
            "derived_from_static_connectivity_fixture_11_3": True,
            "config_file_read_allowed_now": False,
            "config_discovery_allowed_now": False,
            "yaml_parse_allowed_now": False,
            "json_parse_allowed_now": False,
            "environment_variable_read_allowed_now": False,
            "credential_material_allowed_now": False,
            "secret_material_allowed_now": False,
            "adapter_selection_for_runtime_allowed_now": False,
            "adapter_instantiation_allowed_now": False,
            "adapter_wiring_allowed_now": False,
            "runtime_execution_allowed_now": False,
            "network_io_allowed_now": False,
            "dns_lookup_allowed_now": False,
            "http_request_allowed_now": False,
            "websocket_allowed_now": False,
            "credentials_allowed_now": False,
            "secrets_allowed_now": False,
            "private_endpoint_access_allowed_now": False,
            "public_market_data_fetch_allowed_now": False,
            "account_fetch_allowed_now": False,
            "balance_fetch_allowed_now": False,
            "positions_fetch_allowed_now": False,
            "orders_fetch_allowed_now": False,
            "fills_fetch_allowed_now": False,
            "order_submission_allowed_now": False,
            "fill_simulation_allowed_now": False,
            "scheduler_allowed_now": False,
            "qml_changes_allowed": False,
            "new_qml_method_calls_allowed": False,
            "bridge_api_changes_allowed": False,
            "exe_packaging_in_scope": False,
            "bat_productization_allowed": False,
            "exe_direction_preserved": True,
        },
        "adapter_config_gate_entries": entries,
        "default_adapter_config_gate_selection": {
            "config_gate_id": "adapter_config_gate_read_only_market_data_provider",
            "source_capability": "read_only_market_data_provider",
            "reason": "lowest-risk config gate; public-market-data shape only, no config read, no credentials, no network I/O",
            "config_file_read_allowed_now": False,
            "network_io_allowed_now": False,
        },
        "adapter_config_gate_summary": {
            "entry_count": len(entries),
            "default_selection_id": "adapter_config_gate_read_only_market_data_provider",
            "config_file_read_enabled_entry_count": _count_enabled(
                entries, "config_file_read_allowed_now"
            ),
            "config_discovery_enabled_entry_count": _count_enabled(
                entries, "config_discovery_allowed_now"
            ),
            "yaml_parse_enabled_entry_count": _count_enabled(entries, "yaml_parse_allowed_now"),
            "json_parse_enabled_entry_count": _count_enabled(entries, "json_parse_allowed_now"),
            "environment_variable_read_enabled_entry_count": _count_enabled(
                entries, "environment_variable_read_allowed_now"
            ),
            "credentials_enabled_entry_count": _count_enabled(entries, "credentials_allowed_now"),
            "secrets_enabled_entry_count": _count_enabled(entries, "secrets_allowed_now"),
            "network_enabled_entry_count": _count_enabled(entries, "network_io_allowed_now"),
            "dns_lookup_enabled_entry_count": _count_enabled(entries, "dns_lookup_allowed_now"),
            "http_request_enabled_entry_count": _count_enabled(entries, "http_request_allowed_now"),
            "websocket_enabled_entry_count": _count_enabled(entries, "websocket_allowed_now"),
            "private_endpoint_enabled_entry_count": _count_enabled(
                entries, "private_endpoint_allowed_now"
            ),
            "order_submission_enabled_entry_count": _count_enabled(
                entries, "order_submission_allowed_now"
            ),
            "runtime_enabled_entry_count": _count_enabled(entries, "runtime_allowed_now"),
            "offline_safe_entry_count": _count_enabled(entries, "gate_safe_for_offline_tests"),
            "entries_eligible_for_11_5_credentials_gate": _count_enabled(
                entries, "eligible_for_11_5_credentials_gate"
            ),
            "entries_eligible_for_11_6_public_market_data_probe_preview": _count_enabled(
                entries, "eligible_for_11_6_public_market_data_probe_preview"
            ),
            "entries_eligible_for_11_7_private_endpoint_gate": _count_enabled(
                entries, "eligible_for_11_7_private_endpoint_gate"
            ),
            "safe_to_render_in_future_ui_as_read_only": True,
            "safe_for_runtime_execution_now": False,
        },
        "config_shape_requirements": {
            "required_shapes_by_source_capability": required_by_capability,
            "forbidden_material_by_source_capability": forbidden_by_capability,
            "global_required_shape_fields": ["mode", "rate_limit_profile"],
            "global_forbidden_material": [
                "api_key_value",
                "api_secret_value",
                "passphrase_value",
                "raw_secret",
                "live_endpoint_override",
                "order_submission_enabled",
                "account_id",
                "private_endpoint_url",
                "order_endpoint_url",
            ],
        },
        "config_gate_matrix": {
            "public_market_data_config_gate_ids": [
                "adapter_config_gate_read_only_market_data_provider"
            ],
            "exchange_adapter_config_gate_ids": ["adapter_config_gate_exchange_adapter_layer"],
            "network_guard_config_gate_ids": ["adapter_config_gate_exchange_network_guard"],
            "gates_requiring_credentials_gate_later": [
                "adapter_config_gate_exchange_adapter_layer"
            ],
            "gates_requiring_private_endpoint_gate_later": [
                "adapter_config_gate_exchange_adapter_layer",
                "adapter_config_gate_exchange_network_guard",
            ],
            "gates_eligible_for_public_market_data_probe_preview_later": [
                "adapter_config_gate_read_only_market_data_provider",
                "adapter_config_gate_exchange_adapter_layer",
            ],
            "gates_never_runtime_enabled_in_11_4": gate_ids,
        },
        "blocked_config_gate_capabilities": _BLOCKED_CONFIG_GATE_CAPABILITIES,
        "config_gate_boundaries": {
            "config_gate_is_static": True,
            "config_gate_is_derived_from_11_3": True,
            "config_gate_can_feed_11_5_credentials_gate": True,
            "config_gate_can_feed_11_6_public_market_data_probe_preview": True,
            "config_gate_can_feed_11_7_private_endpoint_gate": True,
            "config_gate_cannot_feed_runtime_directly": True,
            "config_gate_cannot_read_config_files": True,
            "config_gate_cannot_discover_configs": True,
            "config_gate_cannot_parse_yaml": True,
            "config_gate_cannot_parse_json": True,
            "config_gate_cannot_read_environment_variables": True,
            "config_gate_cannot_read_credentials": True,
            "config_gate_cannot_read_secrets": True,
            "config_gate_cannot_open_network_connection": True,
            "config_gate_cannot_perform_dns_lookup": True,
            "config_gate_cannot_perform_http_request": True,
            "config_gate_cannot_open_websocket": True,
            "config_gate_cannot_submit_orders": True,
            "config_gate_cannot_change_qml_or_bridge": True,
        },
        "non_activation_evidence": {
            "static_connectivity_fixture_11_3_read": True,
            "adapter_config_gate_built": True,
            "config_files_read": False,
            "config_discovery_performed": False,
            "yaml_parsed": False,
            "json_parsed": False,
            "environment_variables_read": False,
            "credential_material_handled": False,
            "secret_material_handled": False,
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
            "public_market_data_fetch_performed": False,
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
