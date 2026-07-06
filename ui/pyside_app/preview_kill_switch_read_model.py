"""FUNCTIONAL-PREVIEW-12.3 Block J kill switch read model.

Pure-data read model derived from the 12.2 risk limits static fixture. This
module is static only: it does not trigger a kill switch, mutate state, enforce
limits, access endpoints, start loops, or create order flow.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_risk_limits_static_fixture import (
    build_preview_risk_limits_static_fixture,
)

PREVIEW_KILL_SWITCH_READ_MODEL_SCHEMA_VERSION: Final[str] = "preview_kill_switch_read_model.v1"
PREVIEW_KILL_SWITCH_READ_MODEL_KIND: Final[str] = (
    "functional_preview_block_j_kill_switch_read_model"
)
BLOCK_ID: Final[str] = "J"
STEP_ID: Final[str] = "12.3"
KILL_SWITCH_READ_MODEL_STATUS: Final[str] = "kill_switch_read_model_ready_no_runtime_trigger"
KILL_SWITCH_READ_MODEL_DECISION: Final[str] = (
    "BUILD_KILL_SWITCH_READ_MODEL_ONLY_NO_RUNTIME_TRIGGER_NO_ORDER_FLOW"
)
READY_FOR_BLOCK_J_4: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-12.4"
NEXT_STEP_TITLE: Final[str] = "RISK GOVERNOR GATE MATRIX"
STATUS: Final[str] = "ready_for_functional_preview_12_4_risk_governor_gate_matrix"

_RISK_LIMITS_STATIC_FIXTURE_SAFE_KEYS: Final[list[str]] = [
    "schema_version",
    "risk_limits_static_fixture_kind",
    "risk_limits_static_fixture_status",
    "risk_limits_static_fixture_decision",
    "ready_for_block_j_3",
    "next_step",
    "next_step_title",
    "status",
]

_FALSE_SCOPE_FLAGS: Final[list[str]] = [
    "runtime_trigger_allowed_now",
    "manual_trigger_allowed_now",
    "automatic_trigger_allowed_now",
    "kill_switch_state_mutation_allowed_now",
    "runtime_enforcement_allowed_now",
    "risk_decision_runtime_allowed_now",
    "limit_enforcement_runtime_allowed_now",
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

_TRIGGER_SPECS: Final[list[tuple[str, str, str, str, str, str, str]]] = [
    (
        "operator_manual_stop",
        "Operator manual stop",
        "operator_manual",
        "future_operator_control",
        "risk_limit_static_fixture_max_order_notional",
        "critical",
        "Static future operator stop row; no manual trigger is enabled now.",
    ),
    (
        "loss_limit_breach",
        "Loss limit breach",
        "risk_limit",
        "risk_limit_static_fixture_max_daily_loss",
        "risk_limit_static_fixture_max_daily_loss",
        "critical",
        "Static future loss breach row; no automatic trigger is enabled now.",
    ),
    (
        "drawdown_limit_breach",
        "Drawdown limit breach",
        "risk_limit",
        "risk_limit_static_fixture_max_drawdown",
        "risk_limit_static_fixture_max_drawdown",
        "critical",
        "Static future drawdown breach row; no automatic trigger is enabled now.",
    ),
    (
        "order_rate_limit_breach",
        "Order rate limit breach",
        "risk_limit",
        "risk_limit_static_fixture_max_order_rate",
        "risk_limit_static_fixture_max_order_rate",
        "high",
        "Static future order-rate breach row; no runtime loop is enabled now.",
    ),
    (
        "private_endpoint_error_spike",
        "Private endpoint error spike",
        "private_endpoint_health",
        "future_private_endpoint_health_monitor",
        "risk_limit_static_fixture_allowed_modes",
        "high",
        "Static future endpoint-health row; private endpoints remain blocked now.",
    ),
    (
        "runtime_health_failure",
        "Runtime health failure",
        "runtime_health",
        "future_runtime_health_monitor",
        "risk_limit_static_fixture_allowed_modes",
        "critical",
        "Static future runtime-health row; runtime loops remain blocked now.",
    ),
]

BLOCKED_KILL_SWITCH_READ_MODEL_CAPABILITIES: Final[list[str]] = [
    "kill switch runtime trigger",
    "manual kill switch trigger",
    "automatic kill switch trigger",
    "kill switch state mutation",
    "risk runtime enforcement",
    "limit runtime enforcement",
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

FUTURE_STEPS: Final[list[str]] = [
    "functional_preview_12_4_risk_governor_gate_matrix",
    "functional_preview_12_5_block_j_closure_audit",
]


def _risk_limits_static_fixture_reference() -> dict[str, Any]:
    fixture = build_preview_risk_limits_static_fixture()
    return {key: fixture[key] for key in _RISK_LIMITS_STATIC_FIXTURE_SAFE_KEYS}


def _kill_switch_read_model_entries() -> list[dict[str, Any]]:
    return [
        {
            "kill_switch_read_model_id": f"kill_switch_read_model_{trigger_id}",
            "source_trigger_id": trigger_id,
            "display_name": display_name,
            "read_model_classification": "static_kill_switch_read_model_only",
            "trigger_source_type": trigger_type,
            "planned_input_source": planned_input_source,
            "required_prior_fixture_id": required_fixture_id,
            "planned_severity": severity,
            "operator_visibility": "future_read_only_kill_switch",
            "eligible_for_12_4_gate_matrix": True,
            "runtime_trigger_allowed_now": False,
            "manual_trigger_allowed_now": False,
            "automatic_trigger_allowed_now": False,
            "kill_switch_state_mutation_allowed_now": False,
            "order_generation_allowed_now": False,
            "order_submission_allowed_now": False,
            "runtime_loop_allowed_now": False,
            "private_endpoint_access_allowed_now": False,
            "network_io_allowed_now": False,
            "config_file_read_allowed_now": False,
            "credential_secret_read_allowed_now": False,
            "safe_for_offline_tests": True,
            "notes": notes,
        }
        for (
            trigger_id,
            display_name,
            trigger_type,
            planned_input_source,
            required_fixture_id,
            severity,
            notes,
        ) in _TRIGGER_SPECS
    ]


def build_preview_kill_switch_read_model() -> dict[str, Any]:
    """Build the Block J kill switch read model without runtime triggers."""
    entries = _kill_switch_read_model_entries()
    ids = [entry["kill_switch_read_model_id"] for entry in entries]
    required_by_id = {
        entry["kill_switch_read_model_id"]: entry["required_prior_fixture_id"] for entry in entries
    }
    inputs_by_id = {
        entry["kill_switch_read_model_id"]: entry["planned_input_source"] for entry in entries
    }
    return {
        "schema_version": PREVIEW_KILL_SWITCH_READ_MODEL_SCHEMA_VERSION,
        "kill_switch_read_model_kind": PREVIEW_KILL_SWITCH_READ_MODEL_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "kill_switch_read_model_status": KILL_SWITCH_READ_MODEL_STATUS,
        "kill_switch_read_model_decision": KILL_SWITCH_READ_MODEL_DECISION,
        "ready_for_block_j_4": READY_FOR_BLOCK_J_4,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "risk_limits_static_fixture_reference": _risk_limits_static_fixture_reference(),
        "kill_switch_read_model_scope": {
            "scope_name": "kill_switch_read_model",
            "read_model_only": True,
            "derived_from_risk_limits_static_fixture_12_2": True,
            **{key: False for key in _FALSE_SCOPE_FLAGS},
            "exe_direction_preserved": True,
        },
        "kill_switch_read_model_entries": entries,
        "default_kill_switch_read_model_selection": {
            "kill_switch_read_model_id": "kill_switch_read_model_operator_manual_stop",
            "source_trigger_id": "operator_manual_stop",
            "reason": "lowest-risk first read-model trigger; static only, no runtime trigger, no order flow",
            "runtime_trigger_allowed_now": False,
            "order_submission_allowed_now": False,
        },
        "kill_switch_read_model_summary": {
            "entry_count": 6,
            "default_selection_id": "kill_switch_read_model_operator_manual_stop",
            "runtime_trigger_enabled_entry_count": 0,
            "manual_trigger_enabled_entry_count": 0,
            "automatic_trigger_enabled_entry_count": 0,
            "kill_switch_state_mutation_enabled_entry_count": 0,
            "order_generation_enabled_entry_count": 0,
            "order_submission_enabled_entry_count": 0,
            "runtime_loop_enabled_entry_count": 0,
            "private_endpoint_enabled_entry_count": 0,
            "network_enabled_entry_count": 0,
            "config_file_read_enabled_entry_count": 0,
            "credential_secret_read_enabled_entry_count": 0,
            "offline_safe_entry_count": 6,
            "entries_eligible_for_12_4_gate_matrix": 6,
            "critical_severity_entry_count": 4,
            "high_severity_entry_count": 2,
            "operator_manual_entry_count": 1,
            "risk_limit_entry_count": 3,
            "health_entry_count": 2,
            "safe_to_render_in_future_ui_as_read_only": True,
            "safe_for_runtime_execution_now": False,
            "safe_for_order_execution_now": False,
        },
        "kill_switch_read_model_matrix": {
            "kill_switch_read_model_ids": ids,
            "critical_trigger_ids": [
                "kill_switch_read_model_operator_manual_stop",
                "kill_switch_read_model_loss_limit_breach",
                "kill_switch_read_model_drawdown_limit_breach",
                "kill_switch_read_model_runtime_health_failure",
            ],
            "high_trigger_ids": [
                "kill_switch_read_model_order_rate_limit_breach",
                "kill_switch_read_model_private_endpoint_error_spike",
            ],
            "operator_manual_trigger_ids": ["kill_switch_read_model_operator_manual_stop"],
            "risk_limit_trigger_ids": [
                "kill_switch_read_model_loss_limit_breach",
                "kill_switch_read_model_drawdown_limit_breach",
                "kill_switch_read_model_order_rate_limit_breach",
            ],
            "health_trigger_ids": [
                "kill_switch_read_model_private_endpoint_error_spike",
                "kill_switch_read_model_runtime_health_failure",
            ],
            "entries_requiring_12_4_gate_matrix": ids,
            "entries_never_runtime_triggered_in_12_3": ids,
            "required_fixture_ids_by_trigger_id": required_by_id,
            "planned_input_sources_by_trigger_id": inputs_by_id,
        },
        "kill_switch_trigger_contract": {
            "trigger_contract_id": "hard_kill_switch_read_model_contract",
            "trigger_values_are_static": True,
            "trigger_values_are_examples_only": True,
            "trigger_values_are_not_runtime_triggers": True,
            "manual_trigger_cannot_fire_now": True,
            "automatic_trigger_cannot_fire_now": True,
            "kill_switch_state_cannot_mutate_now": True,
            "order_flow_remains_blocked": True,
            "runtime_loop_remains_blocked": True,
            "private_endpoint_access_remains_blocked": True,
            "trigger_values_require_12_4_gate_before_any_runtime_use": True,
            "live_production_trading_forbidden": True,
        },
        "blocked_kill_switch_read_model_capabilities": BLOCKED_KILL_SWITCH_READ_MODEL_CAPABILITIES,
        "kill_switch_read_model_boundaries": {
            "kill_switch_read_model_is_static": True,
            "kill_switch_read_model_is_derived_from_12_2": True,
            "kill_switch_read_model_can_feed_12_4_gate_matrix": True,
            "kill_switch_read_model_cannot_feed_runtime_directly": True,
            "kill_switch_read_model_cannot_trigger_kill_switch": True,
            "kill_switch_read_model_cannot_mutate_kill_switch_state": True,
            "kill_switch_read_model_cannot_enforce_limits": True,
            "kill_switch_read_model_cannot_generate_orders": True,
            "kill_switch_read_model_cannot_submit_orders": True,
            "kill_switch_read_model_cannot_cancel_orders": True,
            "kill_switch_read_model_cannot_replace_orders": True,
            "kill_switch_read_model_cannot_access_private_endpoints": True,
            "kill_switch_read_model_cannot_read_account": True,
            "kill_switch_read_model_balance_read_blocked": True,
            "kill_switch_read_model_cannot_read_positions": True,
            "kill_switch_read_model_cannot_read_orders": True,
            "kill_switch_read_model_cannot_read_fills": True,
            "kill_switch_read_model_cannot_read_market_data": True,
            "kill_switch_read_model_cannot_open_network_connection": True,
            "kill_switch_read_model_cannot_perform_dns_lookup": True,
            "kill_switch_read_model_cannot_perform_http_request": True,
            "kill_switch_read_model_cannot_open_websocket": True,
            "kill_switch_read_model_cannot_read_credentials": True,
            "kill_switch_read_model_cannot_read_secrets": True,
            "kill_switch_read_model_cannot_read_secure_store": True,
            "kill_switch_read_model_cannot_change_qml_or_bridge": True,
        },
        "non_activation_evidence": {
            "risk_limits_static_fixture_12_2_read": True,
            "kill_switch_read_model_built": True,
            **{
                key: False
                for key in [
                    "kill_switch_runtime_trigger_enabled",
                    "manual_kill_switch_trigger_enabled",
                    "automatic_kill_switch_trigger_enabled",
                    "kill_switch_state_mutated",
                    "risk_runtime_enforcement_started",
                    "limit_runtime_enforcement_started",
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
        "future_steps": FUTURE_STEPS,
        "status": STATUS,
    }


__all__: Final[list[str]] = [
    "BLOCK_ID",
    "KILL_SWITCH_READ_MODEL_DECISION",
    "KILL_SWITCH_READ_MODEL_STATUS",
    "NEXT_STEP",
    "NEXT_STEP_TITLE",
    "PREVIEW_KILL_SWITCH_READ_MODEL_KIND",
    "PREVIEW_KILL_SWITCH_READ_MODEL_SCHEMA_VERSION",
    "READY_FOR_BLOCK_J_4",
    "STATUS",
    "STEP_ID",
    "build_preview_kill_switch_read_model",
]
