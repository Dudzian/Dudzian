"""FUNCTIONAL-PREVIEW-13.2 Block K audit envelope read model.

Pure-data read model for planned Block K audit envelope categories. This module
only assembles static dictionaries from the accepted Block K 13.0 contract and
13.1 observability read model safe subsets. It does not write or export audits,
read files, collect metrics, perform filesystem/network I/O, touch QML/bridge
surfaces, or activate any runtime behavior.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_observability_audit_rollback_soak_contract import (
    build_preview_observability_audit_rollback_soak_contract,
)
from ui.pyside_app.preview_observability_read_model import build_preview_observability_read_model

PREVIEW_AUDIT_ENVELOPE_READ_MODEL_SCHEMA_VERSION: Final[str] = (
    "preview_audit_envelope_read_model.v1"
)
PREVIEW_AUDIT_ENVELOPE_READ_MODEL_KIND: Final[str] = (
    "functional_preview_block_k_audit_envelope_read_model"
)
BLOCK_ID: Final[str] = "K"
STEP_ID: Final[str] = "13.2"
AUDIT_ENVELOPE_READ_MODEL_STATUS: Final[str] = (
    "audit_envelope_read_model_ready_no_audit_writer_no_exports"
)
AUDIT_ENVELOPE_READ_MODEL_DECISION: Final[str] = (
    "BUILD_AUDIT_ENVELOPE_READ_MODEL_ONLY_NO_WRITER_NO_EXPORTS_NO_IO"
)
READY_FOR_BLOCK_K_3: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-13.3"
NEXT_STEP_TITLE: Final[str] = "ROLLBACK READ MODEL"
STATUS: Final[str] = "ready_for_functional_preview_13_3_rollback_read_model"

_BLOCK_K_CONTRACT_SAFE_KEYS: Final[list[str]] = [
    "schema_version",
    "observability_audit_rollback_soak_contract_kind",
    "observability_audit_rollback_soak_contract_status",
    "observability_audit_rollback_soak_contract_decision",
    "ready_for_block_k_1",
    "next_step",
    "next_step_title",
    "status",
]

_OBSERVABILITY_READ_MODEL_SAFE_KEYS: Final[list[str]] = [
    "schema_version",
    "observability_read_model_kind",
    "observability_read_model_status",
    "observability_read_model_decision",
    "ready_for_block_k_2",
    "next_step",
    "next_step_title",
    "status",
]

_FALSE_SCOPE_FLAGS: Final[list[str]] = [
    "audit_writer_allowed_now",
    "audit_export_allowed_now",
    "audit_file_read_allowed_now",
    "audit_file_write_allowed_now",
    "log_file_read_allowed_now",
    "log_file_write_allowed_now",
    "filesystem_io_allowed_now",
    "observability_runtime_allowed_now",
    "runtime_metrics_collection_allowed_now",
    "metrics_export_allowed_now",
    "rollback_execution_allowed_now",
    "soak_runtime_allowed_now",
    "runtime_enforcement_allowed_now",
    "risk_decision_runtime_allowed_now",
    "limit_enforcement_runtime_allowed_now",
    "kill_switch_runtime_allowed_now",
    "manual_trigger_allowed_now",
    "automatic_trigger_allowed_now",
    "kill_switch_state_mutation_allowed_now",
    "order_generation_allowed_now",
    "order_submission_allowed_now",
    "order_cancel_allowed_now",
    "order_replace_allowed_now",
    "position_mutation_allowed_now",
    "private_endpoint_access_allowed_now",
    "account_read_allowed_now",
    "balance_read_allowed_now",
    "positions_read_allowed_now",
    "orders_read_allowed_now",
    "fills_read_allowed_now",
    "market_data_read_allowed_now",
    "network_io_allowed_now",
    "dns_lookup_allowed_now",
    "http_request_allowed_now",
    "websocket_allowed_now",
    "adapter_instantiation_allowed_now",
    "adapter_wiring_allowed_now",
    "scheduler_allowed_now",
    "config_file_read_allowed_now",
    "config_discovery_allowed_now",
    "yaml_parse_allowed_now",
    "json_parse_allowed_now",
    "environment_variable_read_allowed_now",
    "credential_secret_read_allowed_now",
    "credential_validation_allowed_now",
    "secure_store_read_allowed_now",
    "secure_store_write_allowed_now",
    "qml_changes_allowed",
    "new_qml_method_calls_allowed",
    "bridge_api_changes_allowed",
    "exe_packaging_in_scope",
    "bat_productization_allowed",
]

_ENTRY_DEFINITIONS: Final[list[dict[str, str]]] = [
    {
        "source_audit_event_id": "runtime_lifecycle_event",
        "display_name": "Runtime lifecycle event",
        "audit_domain": "runtime",
        "planned_event_source": "future_runtime_lifecycle_audit",
        "planned_event_type": "lifecycle",
        "required_envelope_profile": "runtime_lifecycle_profile",
        "planned_retention_class": "local_short_retention_planned_only",
    },
    {
        "source_audit_event_id": "decision_event",
        "display_name": "Decision event",
        "audit_domain": "decision",
        "planned_event_source": "future_decision_audit",
        "planned_event_type": "decision",
        "required_envelope_profile": "decision_profile",
        "planned_retention_class": "local_standard_retention_planned_only",
    },
    {
        "source_audit_event_id": "risk_gate_event",
        "display_name": "Risk gate event",
        "audit_domain": "risk",
        "planned_event_source": "future_risk_gate_audit",
        "planned_event_type": "gate_state",
        "required_envelope_profile": "risk_gate_profile",
        "planned_retention_class": "local_standard_retention_planned_only",
    },
    {
        "source_audit_event_id": "order_flow_event",
        "display_name": "Order flow event",
        "audit_domain": "order_flow",
        "planned_event_source": "future_order_flow_audit",
        "planned_event_type": "order_flow",
        "required_envelope_profile": "order_flow_profile",
        "planned_retention_class": "local_extended_retention_planned_only",
    },
    {
        "source_audit_event_id": "private_endpoint_gate_event",
        "display_name": "Private endpoint gate event",
        "audit_domain": "private_endpoint",
        "planned_event_source": "future_private_endpoint_gate_audit",
        "planned_event_type": "gate_state",
        "required_envelope_profile": "private_endpoint_gate_profile",
        "planned_retention_class": "local_standard_retention_planned_only",
    },
    {
        "source_audit_event_id": "network_gate_event",
        "display_name": "Network gate event",
        "audit_domain": "network",
        "planned_event_source": "future_network_gate_audit",
        "planned_event_type": "gate_state",
        "required_envelope_profile": "network_gate_profile",
        "planned_retention_class": "local_standard_retention_planned_only",
    },
    {
        "source_audit_event_id": "rollback_event",
        "display_name": "Rollback event",
        "audit_domain": "rollback",
        "planned_event_source": "future_rollback_audit",
        "planned_event_type": "rollback",
        "required_envelope_profile": "rollback_profile",
        "planned_retention_class": "local_extended_retention_planned_only",
    },
    {
        "source_audit_event_id": "soak_event",
        "display_name": "Soak event",
        "audit_domain": "soak",
        "planned_event_source": "future_soak_audit",
        "planned_event_type": "soak",
        "required_envelope_profile": "soak_profile",
        "planned_retention_class": "local_short_retention_planned_only",
    },
]

_REQUIRED_FIELD_NAMES: Final[list[str]] = [
    "event_id",
    "event_type",
    "event_time",
    "source_component",
    "block_id",
    "step_id",
    "mode",
    "safety_state",
    "risk_state",
    "operator_action_id",
    "correlation_id",
    "payload_summary",
]

BLOCKED_AUDIT_ENVELOPE_READ_MODEL_CAPABILITIES: Final[list[str]] = [
    "audit writer",
    "audit export",
    "audit file read",
    "audit file write",
    "log file read",
    "log file write",
    "filesystem I/O",
    "observability runtime collection",
    "metrics collection",
    "metrics export",
    "rollback execution",
    "runtime shutdown",
    "soak runtime",
    "soak scheduler",
    "risk runtime enforcement",
    "limit runtime enforcement",
    "kill switch runtime trigger",
    "manual kill switch trigger",
    "automatic kill switch trigger",
    "kill switch state mutation",
    "order generation",
    "order submission",
    "order cancel",
    "order replace",
    "position mutation",
    "private endpoint access",
    "account read",
    "balance read",
    "positions read",
    "orders read",
    "fills read",
    "market data read",
    "adapter instantiation",
    "adapter runtime wiring",
    "runtime loop",
    "scheduler",
    "network I/O",
    "DNS lookup",
    "HTTP request",
    "WebSocket connection",
    "credential read",
    "secret read",
    "secure store read",
    "secure store write",
    "config file read",
    "config discovery",
    "YAML parse",
    "JSON parse",
    "environment variable read",
    "TradingController change",
    "DecisionEnvelope change",
    "QML action dispatch",
    "bridge API changes",
    "PyInstaller/EXE packaging",
]

SOURCE_BOUNDARIES: Final[list[str]] = [
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
    "no observability runtime import",
    "no logger/exporter runtime import",
    "no metrics exporter import",
    "no audit writer import",
    "no audit exporter import",
    "no rollback runner import",
    "no soak runner import",
    "no filesystem I/O",
    "no audit file read",
    "no audit file write",
    "no log file read",
    "no log file write",
    "no audit write",
    "no audit export",
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
    "no real market data read",
    "no private endpoint access",
    "no account read",
    "no balance read",
    "no positions read",
    "no orders read",
    "no fills read",
    "no order generation",
    "no order submission",
    "no order cancel",
    "no order replace",
    "no position mutation",
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


def _safe_subset(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: source[key] for key in keys}


def _build_scope() -> dict[str, Any]:
    scope: dict[str, Any] = {
        "scope_name": "audit_envelope_read_model",
        "read_model_only": True,
        "derived_from_block_k_contract_13_0": True,
        "derived_from_observability_read_model_13_1": True,
    }
    scope.update({key: False for key in _FALSE_SCOPE_FLAGS})
    scope["exe_direction_preserved"] = True
    return scope


def _build_entries() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for definition in _ENTRY_DEFINITIONS:
        source_id = definition["source_audit_event_id"]
        entries.append(
            {
                "audit_envelope_read_model_id": f"audit_envelope_read_model_{source_id}",
                "source_audit_event_id": source_id,
                "display_name": definition["display_name"],
                "read_model_classification": "static_audit_envelope_read_model_only",
                "audit_domain": definition["audit_domain"],
                "planned_event_source": definition["planned_event_source"],
                "planned_event_type": definition["planned_event_type"],
                "required_envelope_profile": definition["required_envelope_profile"],
                "planned_retention_class": definition["planned_retention_class"],
                "operator_visibility": "future_read_only_audit_envelope",
                "eligible_for_13_5_gate_matrix": True,
                "audit_writer_allowed_now": False,
                "audit_export_allowed_now": False,
                "audit_file_read_allowed_now": False,
                "audit_file_write_allowed_now": False,
                "filesystem_io_allowed_now": False,
                "network_io_allowed_now": False,
                "order_flow_allowed_now": False,
                "private_endpoint_access_allowed_now": False,
                "safe_for_offline_tests": True,
                "notes": "Static planned audit envelope category; no writer, export, or I/O is enabled in 13.2.",
            }
        )
    return entries


def _build_required_fields() -> list[dict[str, Any]]:
    return [
        {
            "field_name": field_name,
            "field_classification": "planned_audit_envelope_field_only",
            "required_for_future_writer": True,
            "contains_secret_material": False,
            "contains_private_key_material": False,
            "allowed_to_collect_now": False,
            "notes": "Planned envelope field only; no runtime collection is enabled in 13.2.",
        }
        for field_name in _REQUIRED_FIELD_NAMES
    ]


def _build_summary(
    entries: list[dict[str, Any]], required_fields: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "entry_count": len(entries),
        "required_field_count": len(required_fields),
        "default_selection_id": "audit_envelope_read_model_runtime_lifecycle_event",
        "audit_writer_enabled_entry_count": 0,
        "audit_export_enabled_entry_count": 0,
        "audit_file_read_enabled_entry_count": 0,
        "audit_file_write_enabled_entry_count": 0,
        "filesystem_io_enabled_entry_count": 0,
        "network_enabled_entry_count": 0,
        "order_flow_enabled_entry_count": 0,
        "private_endpoint_enabled_entry_count": 0,
        "offline_safe_entry_count": len(entries),
        "entries_eligible_for_13_5_gate_matrix": len(entries),
        "runtime_domain_entry_count": 1,
        "decision_domain_entry_count": 1,
        "risk_domain_entry_count": 1,
        "order_flow_domain_entry_count": 1,
        "private_endpoint_domain_entry_count": 1,
        "network_domain_entry_count": 1,
        "rollback_domain_entry_count": 1,
        "soak_domain_entry_count": 1,
        "safe_to_render_in_future_ui_as_read_only": True,
        "safe_for_audit_writer_now": False,
        "safe_for_export_now": False,
        "safe_for_filesystem_io_now": False,
    }


def _build_matrix(entries: list[dict[str, Any]]) -> dict[str, Any]:
    ids = [entry["audit_envelope_read_model_id"] for entry in entries]
    return {
        "audit_envelope_read_model_ids": ids,
        "runtime_domain_ids": ["audit_envelope_read_model_runtime_lifecycle_event"],
        "decision_domain_ids": ["audit_envelope_read_model_decision_event"],
        "risk_domain_ids": ["audit_envelope_read_model_risk_gate_event"],
        "order_flow_domain_ids": ["audit_envelope_read_model_order_flow_event"],
        "private_endpoint_domain_ids": ["audit_envelope_read_model_private_endpoint_gate_event"],
        "network_domain_ids": ["audit_envelope_read_model_network_gate_event"],
        "rollback_domain_ids": ["audit_envelope_read_model_rollback_event"],
        "soak_domain_ids": ["audit_envelope_read_model_soak_event"],
        "entries_requiring_13_5_gate_matrix": ids,
        "entries_never_written_in_13_2": ids,
        "planned_event_sources_by_id": {
            entry["audit_envelope_read_model_id"]: entry["planned_event_source"]
            for entry in entries
        },
        "required_envelope_profiles_by_id": {
            entry["audit_envelope_read_model_id"]: entry["required_envelope_profile"]
            for entry in entries
        },
    }


def build_preview_audit_envelope_read_model() -> dict[str, Any]:
    """Build the static Block K 13.2 audit envelope read model."""
    block_k_contract = build_preview_observability_audit_rollback_soak_contract()
    observability_read_model = build_preview_observability_read_model()
    entries = _build_entries()
    required_fields = _build_required_fields()
    ids = [entry["audit_envelope_read_model_id"] for entry in entries]
    return {
        "schema_version": PREVIEW_AUDIT_ENVELOPE_READ_MODEL_SCHEMA_VERSION,
        "audit_envelope_read_model_kind": PREVIEW_AUDIT_ENVELOPE_READ_MODEL_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "audit_envelope_read_model_status": AUDIT_ENVELOPE_READ_MODEL_STATUS,
        "audit_envelope_read_model_decision": AUDIT_ENVELOPE_READ_MODEL_DECISION,
        "ready_for_block_k_3": READY_FOR_BLOCK_K_3,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_k_contract_reference": _safe_subset(block_k_contract, _BLOCK_K_CONTRACT_SAFE_KEYS),
        "observability_read_model_reference": _safe_subset(
            observability_read_model, _OBSERVABILITY_READ_MODEL_SAFE_KEYS
        ),
        "audit_envelope_read_model_scope": _build_scope(),
        "audit_envelope_read_model_entries": entries,
        "default_audit_envelope_read_model_selection": {
            "audit_envelope_read_model_id": "audit_envelope_read_model_runtime_lifecycle_event",
            "source_audit_event_id": "runtime_lifecycle_event",
            "reason": "first audit envelope read-model event; static only, no writer, no exports, no filesystem I/O",
            "audit_writer_allowed_now": False,
            "audit_export_allowed_now": False,
        },
        "audit_envelope_required_fields": required_fields,
        "audit_envelope_read_model_summary": _build_summary(entries, required_fields),
        "audit_envelope_read_model_matrix": _build_matrix(entries),
        "audit_envelope_surface_contract": {
            "surface_contract_id": "block_k_audit_envelope_read_model_surface_contract",
            "read_model_is_static": True,
            "events_are_planned_only": True,
            "events_are_not_written_now": True,
            "events_are_not_exported_now": True,
            "audit_files_are_not_read_now": True,
            "audit_files_are_not_written_now": True,
            "filesystem_io_forbidden_now": True,
            "network_io_forbidden_now": True,
            "writer_requires_future_gate": True,
            "export_requires_future_gate": True,
            "ui_surface_requires_future_qml_gate": True,
        },
        "blocked_audit_envelope_read_model_capabilities": BLOCKED_AUDIT_ENVELOPE_READ_MODEL_CAPABILITIES,
        "audit_envelope_read_model_boundaries": {
            "audit_envelope_read_model_is_static": True,
            "audit_envelope_read_model_is_derived_from_13_0_contract": True,
            "audit_envelope_read_model_is_derived_from_13_1_observability": True,
            "audit_envelope_read_model_can_feed_13_5_gate_matrix": True,
            "audit_envelope_read_model_cannot_feed_runtime_directly": True,
            "audit_envelope_read_model_cannot_write_audit": True,
            "audit_envelope_read_model_cannot_export_audit": True,
            "audit_envelope_read_model_cannot_read_audit_files": True,
            "audit_envelope_read_model_cannot_write_audit_files": True,
            "audit_envelope_read_model_cannot_read_logs": True,
            "audit_envelope_read_model_cannot_write_logs": True,
            "audit_envelope_read_model_cannot_touch_filesystem": True,
            "audit_envelope_read_model_cannot_collect_metrics": True,
            "audit_envelope_read_model_cannot_export_metrics": True,
            "audit_envelope_read_model_cannot_execute_rollback": True,
            "audit_envelope_read_model_cannot_run_soak": True,
            "audit_envelope_read_model_cannot_enable_runtime": True,
            "audit_envelope_read_model_cannot_enforce_limits": True,
            "audit_envelope_read_model_cannot_trigger_kill_switch": True,
            "audit_envelope_read_model_cannot_mutate_kill_switch_state": True,
            "audit_envelope_read_model_cannot_generate_orders": True,
            "audit_envelope_read_model_cannot_submit_orders": True,
            "audit_envelope_read_model_cannot_cancel_orders": True,
            "audit_envelope_read_model_cannot_replace_orders": True,
            "audit_envelope_read_model_cannot_access_private_endpoints": True,
            "audit_envelope_read_model_cannot_read_account": True,
            "audit_envelope_read_model_balance_read_blocked": True,
            "audit_envelope_read_model_cannot_read_positions": True,
            "audit_envelope_read_model_cannot_read_orders": True,
            "audit_envelope_read_model_cannot_read_fills": True,
            "audit_envelope_read_model_cannot_read_market_data": True,
            "audit_envelope_read_model_cannot_open_network_connection": True,
            "audit_envelope_read_model_cannot_perform_dns_lookup": True,
            "audit_envelope_read_model_cannot_perform_http_request": True,
            "audit_envelope_read_model_cannot_open_websocket": True,
            "audit_envelope_read_model_cannot_read_credentials": True,
            "audit_envelope_read_model_cannot_read_secrets": True,
            "audit_envelope_read_model_cannot_read_secure_store": True,
            "audit_envelope_read_model_cannot_change_qml_or_bridge": True,
        },
        "non_activation_evidence": {
            "block_k_contract_13_0_read": True,
            "observability_read_model_13_1_read": True,
            "audit_envelope_read_model_built": True,
            **{
                key: False
                for key in [
                    "audit_writer_started",
                    "audit_exported",
                    "audit_file_read",
                    "audit_file_written",
                    "log_file_read",
                    "log_file_written",
                    "observability_runtime_started",
                    "metrics_collected",
                    "metrics_exported",
                    "rollback_executed",
                    "runtime_shutdown_executed",
                    "soak_runtime_started",
                    "soak_scheduler_started",
                    "filesystem_io_performed",
                    "risk_runtime_enforcement_started",
                    "limit_runtime_enforcement_started",
                    "kill_switch_runtime_trigger_enabled",
                    "manual_kill_switch_trigger_enabled",
                    "automatic_kill_switch_trigger_enabled",
                    "kill_switch_state_mutated",
                    "order_generated",
                    "order_submitted",
                    "order_cancelled",
                    "order_replaced",
                    "position_mutated",
                    "private_endpoint_accessed",
                    "account_read_performed",
                    "balance_read_performed",
                    "positions_read_performed",
                    "orders_read_performed",
                    "fills_read_performed",
                    "market_data_read_performed",
                    "adapter_instantiated",
                    "adapter_wired_to_runtime",
                    "runtime_started",
                    "scheduler_started",
                    "network_io_performed",
                    "dns_lookup_performed",
                    "http_request_performed",
                    "websocket_opened",
                    "credentials_read",
                    "secrets_read",
                    "secure_store_read",
                    "secure_store_write",
                    "config_files_read",
                    "config_discovery_performed",
                    "yaml_parsed",
                    "json_parsed",
                    "environment_variables_read",
                    "trading_controller_touched",
                    "decision_envelope_touched",
                    "qml_changed",
                    "bridge_api_changed",
                ]
            },
        },
        "source_boundaries": SOURCE_BOUNDARIES,
        "future_steps": [
            "functional_preview_13_3_rollback_read_model",
            "functional_preview_13_4_soak_read_model",
            "functional_preview_13_5_observability_audit_rollback_soak_gate_matrix",
            "functional_preview_13_6_block_k_closure_audit",
        ],
        "status": STATUS,
    }
