"""FUNCTIONAL-PREVIEW-11.3 Block I static connectivity fixture.

Pure-data Testnet/Sandbox static connectivity fixture derived from the 11.2
adapter read model. This module intentionally does not import or activate
backend, exchange, runtime, account, credentials, secrets, order, UI, bridge,
network, or filesystem code.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_testnet_sandbox_adapter_read_model import (
    build_preview_testnet_sandbox_adapter_read_model,
)

PREVIEW_TESTNET_SANDBOX_STATIC_CONNECTIVITY_FIXTURE_SCHEMA_VERSION: Final[str] = (
    "preview_testnet_sandbox_static_connectivity_fixture.v1"
)
PREVIEW_TESTNET_SANDBOX_STATIC_CONNECTIVITY_FIXTURE_KIND: Final[str] = (
    "functional_preview_block_i_testnet_sandbox_static_connectivity_fixture"
)
BLOCK_ID: Final[str] = "I"
STEP_ID: Final[str] = "11.3"
TESTNET_SANDBOX_STATIC_CONNECTIVITY_FIXTURE_STATUS: Final[str] = (
    "testnet_sandbox_static_connectivity_fixture_ready_no_network"
)
TESTNET_SANDBOX_STATIC_CONNECTIVITY_FIXTURE_DECISION: Final[str] = (
    "BUILD_STATIC_CONNECTIVITY_FIXTURE_ONLY_NO_PROBE_NO_CONFIG_NO_NETWORK"
)
READY_FOR_BLOCK_I_4: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-11.4"
NEXT_STEP_TITLE: Final[str] = "TESTNET/SANDBOX ADAPTER CONFIG GATE"
STATUS: Final[str] = "ready_for_functional_preview_11_4_testnet_sandbox_adapter_config_gate"

_ENTRY_FIELDS: Final[list[str]] = [
    "connectivity_fixture_id",
    "source_adapter_read_model_id",
    "source_capability",
    "display_name",
    "fixture_classification",
    "connectivity_surface_type",
    "static_connectivity_state",
    "simulated_endpoint_category",
    "eligible_for_11_4_config_gate",
    "eligible_for_11_5_credentials_gate",
    "eligible_for_11_6_public_market_data_probe_preview",
    "eligible_for_11_7_private_endpoint_gate",
    "real_probe_allowed_now",
    "network_io_allowed_now",
    "dns_lookup_allowed_now",
    "http_request_allowed_now",
    "websocket_allowed_now",
    "config_read_allowed_now",
    "credentials_allowed_now",
    "private_endpoint_allowed_now",
    "order_submission_allowed_now",
    "runtime_allowed_now",
    "fixture_safe_for_offline_tests",
    "operator_visibility",
    "notes",
]

_ENTRY_ORDER: Final[list[str]] = [
    "read_only_market_data_provider",
    "exchange_adapter_layer",
    "exchange_network_guard",
    "paper_execution_oracle",
]

_ENTRY_OVERRIDES: Final[dict[str, dict[str, Any]]] = {
    "read_only_market_data_provider": {
        "connectivity_fixture_id": "static_connectivity_fixture_read_only_market_data_provider",
        "source_adapter_read_model_id": "adapter_read_model_read_only_market_data_provider",
        "display_name": "Read-only market data provider connectivity fixture",
        "fixture_classification": "lowest_risk_public_market_data_fixture",
        "connectivity_surface_type": "public_market_data_static_fixture",
        "simulated_endpoint_category": "public_market_data",
        "eligible_for_11_4_config_gate": True,
        "eligible_for_11_5_credentials_gate": False,
        "eligible_for_11_6_public_market_data_probe_preview": True,
        "eligible_for_11_7_private_endpoint_gate": False,
        "operator_visibility": "read_only_future",
    },
    "exchange_adapter_layer": {
        "connectivity_fixture_id": "static_connectivity_fixture_exchange_adapter_layer",
        "source_adapter_read_model_id": "adapter_read_model_exchange_adapter_layer",
        "display_name": "Exchange adapter layer connectivity fixture",
        "fixture_classification": "high_risk_exchange_adapter_fixture",
        "connectivity_surface_type": "exchange_adapter_static_fixture",
        "simulated_endpoint_category": "exchange_adapter_inventory",
        "eligible_for_11_4_config_gate": True,
        "eligible_for_11_5_credentials_gate": True,
        "eligible_for_11_6_public_market_data_probe_preview": True,
        "eligible_for_11_7_private_endpoint_gate": True,
        "operator_visibility": "blocked_until_gated",
    },
    "exchange_network_guard": {
        "connectivity_fixture_id": "static_connectivity_fixture_exchange_network_guard",
        "source_adapter_read_model_id": "adapter_read_model_exchange_network_guard",
        "display_name": "Exchange network guard connectivity fixture",
        "fixture_classification": "network_guard_fixture",
        "connectivity_surface_type": "network_guard_static_fixture",
        "simulated_endpoint_category": "network_safety_guard",
        "eligible_for_11_4_config_gate": True,
        "eligible_for_11_5_credentials_gate": False,
        "eligible_for_11_6_public_market_data_probe_preview": False,
        "eligible_for_11_7_private_endpoint_gate": True,
        "operator_visibility": "safety_guard_future",
    },
    "paper_execution_oracle": {
        "connectivity_fixture_id": "static_connectivity_fixture_paper_execution_oracle",
        "source_adapter_read_model_id": "adapter_read_model_paper_execution_oracle",
        "display_name": "Paper execution oracle connectivity fixture",
        "fixture_classification": "paper_comparison_oracle_fixture",
        "connectivity_surface_type": "paper_oracle_static_fixture",
        "simulated_endpoint_category": "paper_comparison_oracle",
        "eligible_for_11_4_config_gate": False,
        "eligible_for_11_5_credentials_gate": False,
        "eligible_for_11_6_public_market_data_probe_preview": False,
        "eligible_for_11_7_private_endpoint_gate": False,
        "operator_visibility": "comparison_only_future",
    },
}

_BLOCKED_CONNECTIVITY_CAPABILITIES: Final[list[str]] = [
    "real connectivity probe",
    "adapter instantiation",
    "adapter config read",
    "testnet connection",
    "sandbox connection",
    "live connection",
    "DNS lookup",
    "HTTP request",
    "WebSocket connection",
    "network I/O",
    "credentials read",
    "secrets read",
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
    "functional_preview_11_4_testnet_sandbox_adapter_config_gate",
    "functional_preview_11_5_testnet_sandbox_credentials_gate_contract",
    "functional_preview_11_6_testnet_sandbox_public_market_data_probe_preview",
    "functional_preview_11_7_testnet_sandbox_private_endpoint_gate",
    "functional_preview_11_8_block_i_closure_audit",
]


def _adapter_read_model_reference(read_model: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": read_model["schema_version"],
        "testnet_sandbox_adapter_read_model_kind": read_model[
            "testnet_sandbox_adapter_read_model_kind"
        ],
        "testnet_sandbox_adapter_read_model_status": read_model[
            "testnet_sandbox_adapter_read_model_status"
        ],
        "testnet_sandbox_adapter_read_model_decision": read_model[
            "testnet_sandbox_adapter_read_model_decision"
        ],
        "ready_for_block_i_3": read_model["ready_for_block_i_3"],
        "next_step": read_model["next_step"],
        "next_step_title": read_model["next_step_title"],
        "status": read_model["status"],
    }


def _static_connectivity_fixture_entries(read_model: dict[str, Any]) -> list[dict[str, Any]]:
    read_model_entries = {
        entry["source_capability"]: entry for entry in read_model["adapter_read_model_entries"]
    }
    entries: list[dict[str, Any]] = []
    for source_capability in _ENTRY_ORDER:
        source_entry = read_model_entries[source_capability]
        entry = {
            "source_capability": source_capability,
            "static_connectivity_state": "fixture_available_no_probe",
            "real_probe_allowed_now": False,
            "network_io_allowed_now": False,
            "dns_lookup_allowed_now": False,
            "http_request_allowed_now": False,
            "websocket_allowed_now": False,
            "config_read_allowed_now": False,
            "credentials_allowed_now": False,
            "private_endpoint_allowed_now": False,
            "order_submission_allowed_now": False,
            "runtime_allowed_now": False,
            "fixture_safe_for_offline_tests": True,
            "notes": source_entry["notes"],
        }
        entry.update(_ENTRY_OVERRIDES[source_capability])
        entries.append({field: entry[field] for field in _ENTRY_FIELDS})
    return entries


def _count_enabled(entries: list[dict[str, Any]], field: str) -> int:
    return sum(1 for entry in entries if entry[field] is True)


def build_preview_testnet_sandbox_static_connectivity_fixture() -> dict[str, Any]:
    """Build the pure-data Block I static connectivity fixture."""

    read_model = build_preview_testnet_sandbox_adapter_read_model()
    entries = _static_connectivity_fixture_entries(read_model)
    fixture_ids = [entry["connectivity_fixture_id"] for entry in entries]
    return {
        "schema_version": PREVIEW_TESTNET_SANDBOX_STATIC_CONNECTIVITY_FIXTURE_SCHEMA_VERSION,
        "testnet_sandbox_static_connectivity_fixture_kind": PREVIEW_TESTNET_SANDBOX_STATIC_CONNECTIVITY_FIXTURE_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "testnet_sandbox_static_connectivity_fixture_status": TESTNET_SANDBOX_STATIC_CONNECTIVITY_FIXTURE_STATUS,
        "testnet_sandbox_static_connectivity_fixture_decision": TESTNET_SANDBOX_STATIC_CONNECTIVITY_FIXTURE_DECISION,
        "ready_for_block_i_4": READY_FOR_BLOCK_I_4,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "adapter_read_model_reference": _adapter_read_model_reference(read_model),
        "fixture_scope": {
            "scope_name": "testnet_sandbox_static_connectivity_fixture",
            "static_fixture_only": True,
            "derived_from_adapter_read_model_11_2": True,
            "connectivity_probe_allowed_now": False,
            "real_connectivity_check_allowed_now": False,
            "config_read_allowed_now": False,
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
        "static_connectivity_fixture_entries": entries,
        "default_static_connectivity_fixture_selection": {
            "connectivity_fixture_id": "static_connectivity_fixture_read_only_market_data_provider",
            "source_capability": "read_only_market_data_provider",
            "reason": "lowest-risk static connectivity fixture; no config read, no credentials, no network I/O, no probe, no runtime activation",
            "real_probe_allowed_now": False,
            "network_io_allowed_now": False,
        },
        "static_connectivity_fixture_summary": {
            "entry_count": len(entries),
            "default_selection_id": "static_connectivity_fixture_read_only_market_data_provider",
            "real_probe_enabled_entry_count": _count_enabled(entries, "real_probe_allowed_now"),
            "network_enabled_entry_count": _count_enabled(entries, "network_io_allowed_now"),
            "dns_lookup_enabled_entry_count": _count_enabled(entries, "dns_lookup_allowed_now"),
            "http_request_enabled_entry_count": _count_enabled(entries, "http_request_allowed_now"),
            "websocket_enabled_entry_count": _count_enabled(entries, "websocket_allowed_now"),
            "config_read_enabled_entry_count": _count_enabled(entries, "config_read_allowed_now"),
            "credentials_enabled_entry_count": _count_enabled(entries, "credentials_allowed_now"),
            "private_endpoint_enabled_entry_count": _count_enabled(
                entries, "private_endpoint_allowed_now"
            ),
            "order_submission_enabled_entry_count": _count_enabled(
                entries, "order_submission_allowed_now"
            ),
            "runtime_enabled_entry_count": _count_enabled(entries, "runtime_allowed_now"),
            "offline_safe_entry_count": _count_enabled(entries, "fixture_safe_for_offline_tests"),
            "entries_eligible_for_11_4_config_gate": _count_enabled(
                entries, "eligible_for_11_4_config_gate"
            ),
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
        "connectivity_fixture_matrix": {
            "public_market_data_fixture_ids": [
                "static_connectivity_fixture_read_only_market_data_provider"
            ],
            "exchange_adapter_fixture_ids": ["static_connectivity_fixture_exchange_adapter_layer"],
            "network_guard_fixture_ids": ["static_connectivity_fixture_exchange_network_guard"],
            "paper_oracle_fixture_ids": ["static_connectivity_fixture_paper_execution_oracle"],
            "fixtures_requiring_credentials_gate_later": [
                "static_connectivity_fixture_exchange_adapter_layer"
            ],
            "fixtures_requiring_private_endpoint_gate_later": [
                "static_connectivity_fixture_exchange_adapter_layer",
                "static_connectivity_fixture_exchange_network_guard",
            ],
            "fixtures_eligible_for_public_market_data_probe_preview_later": [
                "static_connectivity_fixture_read_only_market_data_provider",
                "static_connectivity_fixture_exchange_adapter_layer",
            ],
            "fixtures_never_runtime_enabled_in_11_3": fixture_ids,
        },
        "blocked_connectivity_capabilities": _BLOCKED_CONNECTIVITY_CAPABILITIES,
        "fixture_boundaries": {
            "fixture_is_static": True,
            "fixture_is_derived_from_11_2": True,
            "fixture_can_feed_11_4_config_gate": True,
            "fixture_can_feed_11_5_credentials_gate": True,
            "fixture_can_feed_11_6_public_market_data_probe_preview": True,
            "fixture_can_feed_11_7_private_endpoint_gate": True,
            "fixture_cannot_feed_runtime_directly": True,
            "fixture_cannot_read_configs": True,
            "fixture_cannot_read_credentials": True,
            "fixture_cannot_open_network_connection": True,
            "fixture_cannot_perform_dns_lookup": True,
            "fixture_cannot_perform_http_request": True,
            "fixture_cannot_open_websocket": True,
            "fixture_cannot_submit_orders": True,
            "fixture_cannot_change_qml_or_bridge": True,
        },
        "non_activation_evidence": {
            "adapter_read_model_11_2_read": True,
            "static_connectivity_fixture_built": True,
            "config_files_read": False,
            "backend_modules_imported": False,
            "backend_modules_activated": False,
            "adapter_instantiated": False,
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
