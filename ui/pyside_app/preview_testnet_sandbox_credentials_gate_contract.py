"""FUNCTIONAL-PREVIEW-11.5 Block I credentials gate contract shape.

Pure-data Testnet/Sandbox credentials gate contract derived from the 11.4
adapter config gate. This module intentionally does not read credentials,
secrets, environment variables, configuration files, secure stores, or perform
network/runtime activity.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_testnet_sandbox_adapter_config_gate import (
    build_preview_testnet_sandbox_adapter_config_gate,
)

PREVIEW_TESTNET_SANDBOX_CREDENTIALS_GATE_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "preview_testnet_sandbox_credentials_gate_contract.v1"
)
PREVIEW_TESTNET_SANDBOX_CREDENTIALS_GATE_CONTRACT_KIND: Final[str] = (
    "functional_preview_block_i_testnet_sandbox_credentials_gate_contract"
)
BLOCK_ID: Final[str] = "I"
STEP_ID: Final[str] = "11.5"
TESTNET_SANDBOX_CREDENTIALS_GATE_CONTRACT_STATUS: Final[str] = (
    "testnet_sandbox_credentials_gate_contract_ready_no_secret_read"
)
TESTNET_SANDBOX_CREDENTIALS_GATE_CONTRACT_DECISION: Final[str] = (
    "BUILD_CREDENTIALS_GATE_CONTRACT_ONLY_NO_SECRET_READ_NO_ENV_READ_NO_NETWORK"
)
READY_FOR_BLOCK_I_6: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-11.6"
NEXT_STEP_TITLE: Final[str] = "TESTNET/SANDBOX PUBLIC MARKET DATA PROBE PREVIEW"
STATUS: Final[str] = (
    "ready_for_functional_preview_11_6_testnet_sandbox_public_market_data_probe_preview"
)

_ENTRY_FIELDS: Final[list[str]] = [
    "credentials_gate_id",
    "source_config_gate_id",
    "source_capability",
    "display_name",
    "credentials_gate_classification",
    "credential_surface_type",
    "required_credential_reference_shape",
    "optional_credential_reference_shape",
    "forbidden_credential_material",
    "allowed_credential_reference_material",
    "eligible_for_11_6_public_market_data_probe_preview",
    "eligible_for_11_7_private_endpoint_gate",
    "credential_secret_read_allowed_now",
    "credential_discovery_allowed_now",
    "credential_validation_allowed_now",
    "credential_material_handling_allowed_now",
    "secret_material_handling_allowed_now",
    "environment_variable_read_allowed_now",
    "config_file_read_allowed_now",
    "secure_store_read_allowed_now",
    "secure_store_write_allowed_now",
    "api_key_value_allowed_in_payload",
    "api_secret_value_allowed_in_payload",
    "passphrase_value_allowed_in_payload",
    "raw_secret_allowed_in_payload",
    "account_identifier_allowed_in_payload",
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

_REQUIRED_REFERENCE_SHAPE: Final[list[str]] = [
    "credential_profile_reference",
    "credential_scope",
    "environment",
    "exchange_id",
    "rotation_policy_reference",
    "redaction_policy",
]
_OPTIONAL_REFERENCE_SHAPE: Final[list[str]] = [
    "operator_approval_reference",
    "secret_store_profile_name",
    "credential_health_policy_name",
    "dry_run_credential_profile_name",
]
_FORBIDDEN_CREDENTIAL_MATERIAL: Final[list[str]] = [
    "api_key",
    "api_key_value",
    "api_secret",
    "api_secret_value",
    "passphrase",
    "passphrase_value",
    "raw_secret",
    "private_key",
    "mnemonic",
    "account_id",
    "account_number",
    "wallet_address",
    "live_endpoint_override",
    "order_submission_enabled",
]
_ALLOWED_REFERENCE_MATERIAL: Final[list[str]] = [
    "credential_profile_reference",
    "secret_store_profile_name",
    "rotation_policy_reference",
    "redaction_policy",
    "operator_approval_reference",
]
_GATE_ID: Final[str] = "credentials_gate_exchange_adapter_layer"

_BLOCKED_CREDENTIALS_GATE_CAPABILITIES: Final[list[str]] = [
    "real credential read",
    "credential discovery",
    "credential validation",
    "credential material handling",
    "secret material handling",
    "environment variable read",
    "real config file read",
    "secure store read",
    "secure store write",
    "API key value in payload",
    "API secret value in payload",
    "passphrase value in payload",
    "raw secret in payload",
    "account identifier in payload",
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
    "functional_preview_11_6_testnet_sandbox_public_market_data_probe_preview",
    "functional_preview_11_7_testnet_sandbox_private_endpoint_gate",
    "functional_preview_11_8_block_i_closure_audit",
]


def _adapter_config_gate_reference(gate: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": gate["schema_version"],
        "testnet_sandbox_adapter_config_gate_kind": gate[
            "testnet_sandbox_adapter_config_gate_kind"
        ],
        "testnet_sandbox_adapter_config_gate_status": gate[
            "testnet_sandbox_adapter_config_gate_status"
        ],
        "testnet_sandbox_adapter_config_gate_decision": gate[
            "testnet_sandbox_adapter_config_gate_decision"
        ],
        "ready_for_block_i_5": gate["ready_for_block_i_5"],
        "next_step": gate["next_step"],
        "next_step_title": gate["next_step_title"],
        "status": gate["status"],
    }


def _credentials_gate_entries(gate: dict[str, Any]) -> list[dict[str, Any]]:
    source = {
        entry["source_capability"]: entry
        for entry in gate["adapter_config_gate_entries"]
        if entry["eligible_for_11_5_credentials_gate"] is True
    }["exchange_adapter_layer"]
    entry = {
        "credentials_gate_id": _GATE_ID,
        "source_config_gate_id": source["config_gate_id"],
        "source_capability": "exchange_adapter_layer",
        "display_name": "Exchange adapter layer credentials gate",
        "credentials_gate_classification": "credential_reference_contract_only",
        "credential_surface_type": "exchange_adapter_credential_reference_contract",
        "required_credential_reference_shape": _REQUIRED_REFERENCE_SHAPE,
        "optional_credential_reference_shape": _OPTIONAL_REFERENCE_SHAPE,
        "forbidden_credential_material": _FORBIDDEN_CREDENTIAL_MATERIAL,
        "allowed_credential_reference_material": _ALLOWED_REFERENCE_MATERIAL,
        "eligible_for_11_6_public_market_data_probe_preview": True,
        "eligible_for_11_7_private_endpoint_gate": True,
        "credential_secret_read_allowed_now": False,
        "credential_discovery_allowed_now": False,
        "credential_validation_allowed_now": False,
        "credential_material_handling_allowed_now": False,
        "secret_material_handling_allowed_now": False,
        "environment_variable_read_allowed_now": False,
        "config_file_read_allowed_now": False,
        "secure_store_read_allowed_now": False,
        "secure_store_write_allowed_now": False,
        "api_key_value_allowed_in_payload": False,
        "api_secret_value_allowed_in_payload": False,
        "passphrase_value_allowed_in_payload": False,
        "raw_secret_allowed_in_payload": False,
        "account_identifier_allowed_in_payload": False,
        "network_io_allowed_now": False,
        "dns_lookup_allowed_now": False,
        "http_request_allowed_now": False,
        "websocket_allowed_now": False,
        "private_endpoint_allowed_now": False,
        "order_submission_allowed_now": False,
        "runtime_allowed_now": False,
        "gate_safe_for_offline_tests": True,
        "operator_visibility": "blocked_until_credentials_gate",
        "notes": "Credential references only; no secret, env, config, secure-store, network, or runtime activation in 11.5.",
    }
    return [{field: entry[field] for field in _ENTRY_FIELDS}]


def _count_enabled(entries: list[dict[str, Any]], field: str) -> int:
    return sum(1 for entry in entries if entry[field] is True)


def build_preview_testnet_sandbox_credentials_gate_contract() -> dict[str, Any]:
    """Build the pure-data Block I credentials gate contract shape."""

    adapter_gate = build_preview_testnet_sandbox_adapter_config_gate()
    entries = _credentials_gate_entries(adapter_gate)
    return {
        "schema_version": PREVIEW_TESTNET_SANDBOX_CREDENTIALS_GATE_CONTRACT_SCHEMA_VERSION,
        "testnet_sandbox_credentials_gate_contract_kind": PREVIEW_TESTNET_SANDBOX_CREDENTIALS_GATE_CONTRACT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "testnet_sandbox_credentials_gate_contract_status": TESTNET_SANDBOX_CREDENTIALS_GATE_CONTRACT_STATUS,
        "testnet_sandbox_credentials_gate_contract_decision": TESTNET_SANDBOX_CREDENTIALS_GATE_CONTRACT_DECISION,
        "ready_for_block_i_6": READY_FOR_BLOCK_I_6,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "adapter_config_gate_reference": _adapter_config_gate_reference(adapter_gate),
        "credentials_gate_scope": {
            "scope_name": "testnet_sandbox_credentials_gate_contract",
            "credentials_gate_contract_only": True,
            "derived_from_adapter_config_gate_11_4": True,
            "credential_secret_read_allowed_now": False,
            "credential_discovery_allowed_now": False,
            "credential_validation_allowed_now": False,
            "credential_material_handling_allowed_now": False,
            "secret_material_handling_allowed_now": False,
            "environment_variable_read_allowed_now": False,
            "config_file_read_allowed_now": False,
            "secure_store_read_allowed_now": False,
            "secure_store_write_allowed_now": False,
            "api_key_value_allowed_in_payload": False,
            "api_secret_value_allowed_in_payload": False,
            "passphrase_value_allowed_in_payload": False,
            "raw_secret_allowed_in_payload": False,
            "account_identifier_allowed_in_payload": False,
            "adapter_selection_for_runtime_allowed_now": False,
            "adapter_instantiation_allowed_now": False,
            "adapter_wiring_allowed_now": False,
            "runtime_execution_allowed_now": False,
            "network_io_allowed_now": False,
            "dns_lookup_allowed_now": False,
            "http_request_allowed_now": False,
            "websocket_allowed_now": False,
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
        "credentials_gate_entries": entries,
        "default_credentials_gate_selection": {
            "credentials_gate_id": _GATE_ID,
            "source_capability": "exchange_adapter_layer",
            "reason": "only 11.5-eligible config gate; credential references only, no secret read, no env read, no network I/O",
            "credential_secret_read_allowed_now": False,
            "network_io_allowed_now": False,
        },
        "credentials_gate_summary": {
            "entry_count": len(entries),
            "default_selection_id": _GATE_ID,
            "credential_secret_read_enabled_entry_count": _count_enabled(
                entries, "credential_secret_read_allowed_now"
            ),
            "credential_discovery_enabled_entry_count": _count_enabled(
                entries, "credential_discovery_allowed_now"
            ),
            "credential_validation_enabled_entry_count": _count_enabled(
                entries, "credential_validation_allowed_now"
            ),
            "credential_material_handling_enabled_entry_count": _count_enabled(
                entries, "credential_material_handling_allowed_now"
            ),
            "secret_material_handling_enabled_entry_count": _count_enabled(
                entries, "secret_material_handling_allowed_now"
            ),
            "environment_variable_read_enabled_entry_count": _count_enabled(
                entries, "environment_variable_read_allowed_now"
            ),
            "config_file_read_enabled_entry_count": _count_enabled(
                entries, "config_file_read_allowed_now"
            ),
            "secure_store_read_enabled_entry_count": _count_enabled(
                entries, "secure_store_read_allowed_now"
            ),
            "secure_store_write_enabled_entry_count": _count_enabled(
                entries, "secure_store_write_allowed_now"
            ),
            "api_key_value_payload_enabled_entry_count": _count_enabled(
                entries, "api_key_value_allowed_in_payload"
            ),
            "api_secret_value_payload_enabled_entry_count": _count_enabled(
                entries, "api_secret_value_allowed_in_payload"
            ),
            "passphrase_value_payload_enabled_entry_count": _count_enabled(
                entries, "passphrase_value_allowed_in_payload"
            ),
            "raw_secret_payload_enabled_entry_count": _count_enabled(
                entries, "raw_secret_allowed_in_payload"
            ),
            "account_identifier_payload_enabled_entry_count": _count_enabled(
                entries, "account_identifier_allowed_in_payload"
            ),
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
            "entries_eligible_for_11_6_public_market_data_probe_preview": _count_enabled(
                entries, "eligible_for_11_6_public_market_data_probe_preview"
            ),
            "entries_eligible_for_11_7_private_endpoint_gate": _count_enabled(
                entries, "eligible_for_11_7_private_endpoint_gate"
            ),
            "safe_to_render_in_future_ui_as_read_only": True,
            "safe_for_runtime_execution_now": False,
        },
        "credential_contract_requirements": {
            "required_reference_shapes_by_source_capability": {
                "exchange_adapter_layer": _REQUIRED_REFERENCE_SHAPE
            },
            "forbidden_material_by_source_capability": {
                "exchange_adapter_layer": _FORBIDDEN_CREDENTIAL_MATERIAL
            },
            "allowed_reference_material_by_source_capability": {
                "exchange_adapter_layer": _ALLOWED_REFERENCE_MATERIAL
            },
            "global_required_reference_fields": [
                "credential_profile_reference",
                "credential_scope",
                "environment",
                "exchange_id",
                "redaction_policy",
            ],
            "global_forbidden_credential_material": _FORBIDDEN_CREDENTIAL_MATERIAL,
            "credential_redaction_required": True,
            "credential_logging_forbidden": True,
            "credential_payload_secret_values_forbidden": True,
            "credential_reference_only": True,
        },
        "credential_gate_matrix": {
            "credential_gate_ids": [_GATE_ID],
            "gates_requiring_secret_store_later": [_GATE_ID],
            "gates_eligible_for_public_market_data_probe_preview_later": [_GATE_ID],
            "gates_eligible_for_private_endpoint_gate_later": [_GATE_ID],
            "gates_never_runtime_enabled_in_11_5": [_GATE_ID],
        },
        "blocked_credentials_gate_capabilities": _BLOCKED_CREDENTIALS_GATE_CAPABILITIES,
        "credentials_gate_boundaries": {
            key: True
            for key in [
                "credentials_gate_is_static",
                "credentials_gate_is_derived_from_11_4",
                "credentials_gate_can_feed_11_6_public_market_data_probe_preview",
                "credentials_gate_can_feed_11_7_private_endpoint_gate",
                "credentials_gate_cannot_feed_runtime_directly",
                "credentials_gate_cannot_read_credentials",
                "credentials_gate_cannot_discover_credentials",
                "credentials_gate_cannot_validate_credentials",
                "credentials_gate_cannot_handle_credential_material",
                "credentials_gate_cannot_handle_secret_material",
                "credentials_gate_cannot_read_environment_variables",
                "credentials_gate_cannot_read_config_files",
                "credentials_gate_cannot_read_secure_store",
                "credentials_gate_cannot_write_secure_store",
                "credentials_gate_cannot_include_secret_values_in_payload",
                "credentials_gate_cannot_log_credentials",
                "credentials_gate_cannot_open_network_connection",
                "credentials_gate_cannot_perform_dns_lookup",
                "credentials_gate_cannot_perform_http_request",
                "credentials_gate_cannot_open_websocket",
                "credentials_gate_cannot_submit_orders",
                "credentials_gate_cannot_change_qml_or_bridge",
            ]
        },
        "non_activation_evidence": {
            "adapter_config_gate_11_4_read": True,
            "credentials_gate_contract_built": True,
            **{
                key: False
                for key in [
                    "config_files_read",
                    "config_discovery_performed",
                    "yaml_parsed",
                    "json_parsed",
                    "environment_variables_read",
                    "credential_secret_read",
                    "credential_discovery_performed",
                    "credential_validation_performed",
                    "credential_material_handled",
                    "secret_material_handled",
                    "secure_store_read",
                    "secure_store_write",
                    "api_key_value_in_payload",
                    "api_secret_value_in_payload",
                    "passphrase_value_in_payload",
                    "raw_secret_in_payload",
                    "account_identifier_in_payload",
                    "backend_modules_imported",
                    "backend_modules_activated",
                    "adapter_instantiated",
                    "adapter_config_applied",
                    "runtime_started",
                    "scheduler_started",
                    "dns_lookup_performed",
                    "http_request_performed",
                    "websocket_opened",
                    "network_io_performed",
                    "credentials_read",
                    "secrets_read",
                    "private_endpoint_accessed",
                    "public_market_data_fetch_performed",
                    "account_fetch_performed",
                    "balance_fetch_performed",
                    "positions_fetch_performed",
                    "orders_fetch_performed",
                    "fills_fetch_performed",
                    "order_submitted",
                    "order_generated",
                    "fill_simulated",
                    "qml_changed",
                    "bridge_api_changed",
                ]
            },
        },
        "source_boundaries": _SOURCE_BOUNDARIES,
        "future_steps": _FUTURE_STEPS,
        "status": STATUS,
    }
