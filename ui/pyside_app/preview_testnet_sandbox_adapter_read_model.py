"""FUNCTIONAL-PREVIEW-11.2 Block I testnet/sandbox adapter read model.

Pure-data static adapter read model derived from the 11.1 backend capability
handoff. This module intentionally does not import or activate backend,
exchange, runtime, account, credentials, secrets, order, UI, bridge, network, or
filesystem code.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_testnet_sandbox_backend_capability_handoff import (
    build_preview_testnet_sandbox_backend_capability_handoff,
)

PREVIEW_TESTNET_SANDBOX_ADAPTER_READ_MODEL_SCHEMA_VERSION: Final[str] = (
    "preview_testnet_sandbox_adapter_read_model.v1"
)
PREVIEW_TESTNET_SANDBOX_ADAPTER_READ_MODEL_KIND: Final[str] = (
    "functional_preview_block_i_testnet_sandbox_adapter_read_model"
)
BLOCK_ID: Final[str] = "I"
STEP_ID: Final[str] = "11.2"
TESTNET_SANDBOX_ADAPTER_READ_MODEL_STATUS: Final[str] = (
    "testnet_sandbox_adapter_read_model_ready_static_no_adapter_runtime"
)
TESTNET_SANDBOX_ADAPTER_READ_MODEL_DECISION: Final[str] = (
    "BUILD_STATIC_ADAPTER_READ_MODEL_ONLY_NO_CONFIG_NO_NETWORK_NO_RUNTIME"
)
READY_FOR_BLOCK_I_3: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-11.3"
NEXT_STEP_TITLE: Final[str] = "TESTNET/SANDBOX STATIC CONNECTIVITY FIXTURE"
STATUS: Final[str] = "ready_for_functional_preview_11_3_testnet_sandbox_static_connectivity_fixture"

_ENTRY_FIELDS: Final[list[str]] = [
    "adapter_read_model_id",
    "source_capability",
    "display_name",
    "read_model_classification",
    "adapter_surface_type",
    "evidence_paths",
    "main_symbols",
    "eligible_for_11_3_static_connectivity_fixture",
    "eligible_for_11_4_config_gate",
    "eligible_for_11_5_credentials_gate",
    "eligible_for_11_6_public_market_data_probe_preview",
    "eligible_for_11_7_private_endpoint_gate",
    "runtime_allowed_now",
    "network_io_allowed_now",
    "credentials_allowed_now",
    "private_endpoint_allowed_now",
    "order_submission_allowed_now",
    "requires_risk_governor_before_execution",
    "requires_observability_before_soak",
    "requires_live_gate_before_live",
    "operator_visibility",
    "notes",
]

_BLOCKED_ADAPTER_RUNTIME_CAPABILITIES: Final[list[str]] = [
    "adapter instantiation",
    "adapter config read",
    "testnet connection",
    "sandbox connection",
    "live connection",
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
    "no QML changes",
    "no bridge API changes",
    "no .bat changes",
    "no app.py changes",
    "no dependency declarations changes",
    "no workflow changes",
]

_FUTURE_STEPS: Final[list[str]] = [
    "functional_preview_11_3_testnet_sandbox_static_connectivity_fixture",
    "functional_preview_11_4_testnet_sandbox_adapter_config_gate",
    "functional_preview_11_5_testnet_sandbox_credentials_gate_contract",
    "functional_preview_11_6_testnet_sandbox_public_market_data_probe_preview",
    "functional_preview_11_7_testnet_sandbox_private_endpoint_gate",
    "functional_preview_11_8_block_i_closure_audit",
]

_ENTRY_OVERRIDES: Final[dict[str, dict[str, Any]]] = {
    "read_only_market_data_provider": {
        "adapter_read_model_id": "adapter_read_model_read_only_market_data_provider",
        "display_name": "Read-only market data provider",
        "read_model_classification": "ready_for_contract_gate",
        "adapter_surface_type": "public_market_data_read_model",
        "eligible_for_11_4_config_gate": True,
        "eligible_for_11_5_credentials_gate": False,
        "eligible_for_11_6_public_market_data_probe_preview": True,
        "eligible_for_11_7_private_endpoint_gate": False,
        "requires_risk_governor_before_execution": False,
        "requires_observability_before_soak": True,
        "requires_live_gate_before_live": False,
        "operator_visibility": "read_only_future",
    },
    "exchange_adapter_layer": {
        "adapter_read_model_id": "adapter_read_model_exchange_adapter_layer",
        "display_name": "Exchange adapter layer",
        "read_model_classification": "high_risk_requires_gate",
        "adapter_surface_type": "exchange_adapter_inventory_read_model",
        "eligible_for_11_4_config_gate": True,
        "eligible_for_11_5_credentials_gate": True,
        "eligible_for_11_6_public_market_data_probe_preview": True,
        "eligible_for_11_7_private_endpoint_gate": True,
        "requires_risk_governor_before_execution": True,
        "requires_observability_before_soak": True,
        "requires_live_gate_before_live": True,
        "operator_visibility": "blocked_until_gated",
    },
    "exchange_network_guard": {
        "adapter_read_model_id": "adapter_read_model_exchange_network_guard",
        "display_name": "Exchange network guard",
        "read_model_classification": "implemented_not_wired",
        "adapter_surface_type": "network_guard_read_model",
        "eligible_for_11_4_config_gate": True,
        "eligible_for_11_5_credentials_gate": False,
        "eligible_for_11_6_public_market_data_probe_preview": False,
        "eligible_for_11_7_private_endpoint_gate": True,
        "requires_risk_governor_before_execution": True,
        "requires_observability_before_soak": True,
        "requires_live_gate_before_live": True,
        "operator_visibility": "safety_guard_future",
    },
    "paper_execution_oracle": {
        "adapter_read_model_id": "adapter_read_model_paper_execution_oracle",
        "display_name": "Paper execution oracle",
        "read_model_classification": "implemented_not_wired",
        "adapter_surface_type": "paper_comparison_oracle_read_model",
        "eligible_for_11_4_config_gate": False,
        "eligible_for_11_5_credentials_gate": False,
        "eligible_for_11_6_public_market_data_probe_preview": False,
        "eligible_for_11_7_private_endpoint_gate": False,
        "requires_risk_governor_before_execution": True,
        "requires_observability_before_soak": True,
        "requires_live_gate_before_live": False,
        "operator_visibility": "comparison_only_future",
    },
}

_ENTRY_ORDER: Final[list[str]] = [
    "read_only_market_data_provider",
    "exchange_adapter_layer",
    "exchange_network_guard",
    "paper_execution_oracle",
]


def _handoff_reference(handoff: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": handoff["schema_version"],
        "testnet_sandbox_backend_capability_handoff_kind": handoff[
            "testnet_sandbox_backend_capability_handoff_kind"
        ],
        "testnet_sandbox_backend_capability_handoff_status": handoff[
            "testnet_sandbox_backend_capability_handoff_status"
        ],
        "testnet_sandbox_backend_capability_handoff_decision": handoff[
            "testnet_sandbox_backend_capability_handoff_decision"
        ],
        "ready_for_block_i_2": handoff["ready_for_block_i_2"],
        "next_step": handoff["next_step"],
        "next_step_title": handoff["next_step_title"],
        "status": handoff["status"],
    }


def _adapter_read_model_entries(handoff: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = {
        candidate["capability"]: candidate
        for candidate in handoff["testnet_sandbox_candidate_capabilities"]
    }
    entries: list[dict[str, Any]] = []
    for source_capability in _ENTRY_ORDER:
        candidate = candidates[source_capability]
        overrides = _ENTRY_OVERRIDES[source_capability]
        entry = {
            "source_capability": source_capability,
            "evidence_paths": candidate["evidence_paths"],
            "main_symbols": candidate["main_symbols"],
            "eligible_for_11_3_static_connectivity_fixture": True,
            "runtime_allowed_now": False,
            "network_io_allowed_now": False,
            "credentials_allowed_now": False,
            "private_endpoint_allowed_now": False,
            "order_submission_allowed_now": False,
            "notes": candidate["notes"],
        }
        entry.update(overrides)
        entries.append({field: entry[field] for field in _ENTRY_FIELDS})
    return entries


def _count_enabled(entries: list[dict[str, Any]], field: str) -> int:
    return sum(1 for entry in entries if entry[field] is True)


def build_preview_testnet_sandbox_adapter_read_model() -> dict[str, Any]:
    """Build the pure-data Block I static adapter read model."""

    handoff = build_preview_testnet_sandbox_backend_capability_handoff()
    entries = _adapter_read_model_entries(handoff)
    return {
        "schema_version": PREVIEW_TESTNET_SANDBOX_ADAPTER_READ_MODEL_SCHEMA_VERSION,
        "testnet_sandbox_adapter_read_model_kind": PREVIEW_TESTNET_SANDBOX_ADAPTER_READ_MODEL_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "testnet_sandbox_adapter_read_model_status": TESTNET_SANDBOX_ADAPTER_READ_MODEL_STATUS,
        "testnet_sandbox_adapter_read_model_decision": TESTNET_SANDBOX_ADAPTER_READ_MODEL_DECISION,
        "ready_for_block_i_3": READY_FOR_BLOCK_I_3,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "handoff_reference": _handoff_reference(handoff),
        "read_model_scope": {
            "scope_name": "testnet_sandbox_adapter_read_model",
            "read_model_only": True,
            "static_model_only": True,
            "derived_from_handoff_11_1": True,
            "config_read_allowed_now": False,
            "adapter_selection_for_runtime_allowed_now": False,
            "adapter_instantiation_allowed_now": False,
            "adapter_wiring_allowed_now": False,
            "runtime_execution_allowed_now": False,
            "network_io_allowed_now": False,
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
        "adapter_read_model_entries": entries,
        "default_adapter_read_model_selection": {
            "adapter_read_model_id": "adapter_read_model_read_only_market_data_provider",
            "source_capability": "read_only_market_data_provider",
            "reason": "lowest-risk read-model candidate; no credentials, private endpoint, order submission, or runtime activation",
            "runtime_allowed_now": False,
            "network_io_allowed_now": False,
        },
        "adapter_read_model_summary": {
            "entry_count": len(entries),
            "default_selection_id": "adapter_read_model_read_only_market_data_provider",
            "runtime_enabled_entry_count": _count_enabled(entries, "runtime_allowed_now"),
            "network_enabled_entry_count": _count_enabled(entries, "network_io_allowed_now"),
            "credentials_enabled_entry_count": _count_enabled(entries, "credentials_allowed_now"),
            "private_endpoint_enabled_entry_count": _count_enabled(
                entries, "private_endpoint_allowed_now"
            ),
            "order_submission_enabled_entry_count": _count_enabled(
                entries, "order_submission_allowed_now"
            ),
            "entries_eligible_for_11_3_static_connectivity_fixture": _count_enabled(
                entries, "eligible_for_11_3_static_connectivity_fixture"
            ),
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
        "blocked_adapter_runtime_capabilities": _BLOCKED_ADAPTER_RUNTIME_CAPABILITIES,
        "read_model_boundaries": {
            "model_is_static": True,
            "model_is_derived_from_11_1": True,
            "model_can_feed_11_3_static_fixture": True,
            "model_can_feed_11_4_config_gate": True,
            "model_can_feed_11_5_credentials_gate": True,
            "model_can_feed_11_6_public_market_data_probe_preview": True,
            "model_can_feed_11_7_private_endpoint_gate": True,
            "model_cannot_feed_runtime_directly": True,
            "model_cannot_read_configs": True,
            "model_cannot_read_credentials": True,
            "model_cannot_open_network_connection": True,
            "model_cannot_submit_orders": True,
            "model_cannot_change_qml_or_bridge": True,
        },
        "non_activation_evidence": {
            "handoff_11_1_read": True,
            "adapter_read_model_built": True,
            "config_files_read": False,
            "backend_modules_imported": False,
            "backend_modules_activated": False,
            "adapter_instantiated": False,
            "runtime_started": False,
            "scheduler_started": False,
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
