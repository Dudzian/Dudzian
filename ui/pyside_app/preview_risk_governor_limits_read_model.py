"""FUNCTIONAL-PREVIEW-12.1 Block J risk limits read model.

Pure-data read model for future risk limits derived from the 12.0 contract.
This module is static only: it does not enforce limits, trigger kill switches,
access endpoints, start loops, or create order flow.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_risk_governor_limits_kill_switch_contract import (
    build_preview_risk_governor_limits_kill_switch_contract,
)

PREVIEW_RISK_GOVERNOR_LIMITS_READ_MODEL_SCHEMA_VERSION: Final[str] = (
    "preview_risk_governor_limits_read_model.v1"
)
PREVIEW_RISK_GOVERNOR_LIMITS_READ_MODEL_KIND: Final[str] = (
    "functional_preview_block_j_risk_governor_limits_read_model"
)
BLOCK_ID: Final[str] = "J"
STEP_ID: Final[str] = "12.1"
RISK_GOVERNOR_LIMITS_READ_MODEL_STATUS: Final[str] = (
    "risk_governor_limits_read_model_ready_no_runtime_enforcement"
)
RISK_GOVERNOR_LIMITS_READ_MODEL_DECISION: Final[str] = (
    "BUILD_RISK_GOVERNOR_LIMITS_READ_MODEL_ONLY_NO_RUNTIME_NO_ORDER_FLOW"
)
READY_FOR_BLOCK_J_2: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-12.2"
NEXT_STEP_TITLE: Final[str] = "RISK LIMITS STATIC FIXTURE"
STATUS: Final[str] = "ready_for_functional_preview_12_2_risk_limits_static_fixture"

_RISK_CONTRACT_SAFE_KEYS: Final[list[str]] = [
    "schema_version",
    "risk_governor_limits_kill_switch_contract_kind",
    "risk_governor_limits_kill_switch_contract_status",
    "risk_governor_limits_kill_switch_contract_decision",
    "ready_for_block_j_1",
    "next_step",
    "next_step_title",
    "status",
]

_FALSE_SCOPE_FLAGS: Final[list[str]] = [
    "runtime_enforcement_allowed_now",
    "risk_decision_runtime_allowed_now",
    "limit_enforcement_runtime_allowed_now",
    "kill_switch_runtime_allowed_now",
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

_LIMIT_SPECS: Final[list[tuple[str, str, str, str, str, str]]] = [
    (
        "max_order_notional",
        "Max order notional",
        "single_order",
        "planned static notional ceiling per order",
        "money",
        "Defines a future per-order notional cap for static fixtures only.",
    ),
    (
        "max_daily_notional",
        "Max daily notional",
        "trading_day",
        "planned cumulative daily notional ceiling",
        "money",
        "Defines a future day-level notional cap for static fixtures only.",
    ),
    (
        "max_position_notional",
        "Max position notional",
        "position",
        "planned position notional ceiling",
        "money",
        "Defines a future maximum exposure per position for static fixtures only.",
    ),
    (
        "max_open_positions",
        "Max open positions",
        "portfolio",
        "planned open position count ceiling",
        "integer",
        "Defines a future portfolio breadth cap for static fixtures only.",
    ),
    (
        "max_daily_loss",
        "Max daily loss",
        "trading_day",
        "planned realized and unrealized loss ceiling",
        "money",
        "Defines a future daily loss boundary for static fixtures only.",
    ),
    (
        "max_drawdown",
        "Max drawdown",
        "equity_curve",
        "planned peak-to-trough drawdown ceiling",
        "percent",
        "Defines a future drawdown boundary for static fixtures only.",
    ),
    (
        "max_order_rate",
        "Max order rate",
        "runtime_window",
        "planned order attempts per time window",
        "rate",
        "Defines a future order-attempt throttle for static fixtures only.",
    ),
    (
        "allowed_symbols",
        "Allowed symbols",
        "instrument_universe",
        "planned explicit symbol allowlist",
        "symbol_allowlist",
        "Defines a future symbol allowlist before any order flow.",
    ),
    (
        "allowed_modes",
        "Allowed modes",
        "execution_mode",
        "planned explicit mode allowlist",
        "mode_allowlist",
        "Live production remains blocked; only later gated modes may be considered.",
    ),
]

BLOCKED_RISK_LIMITS_READ_MODEL_CAPABILITIES: Final[list[str]] = [
    "risk runtime enforcement",
    "limit runtime enforcement",
    "kill switch runtime trigger",
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
    "functional_preview_12_2_risk_limits_static_fixture",
    "functional_preview_12_3_kill_switch_read_model",
    "functional_preview_12_4_risk_governor_gate_matrix",
    "functional_preview_12_5_block_j_closure_audit",
]


def _risk_contract_reference() -> dict[str, Any]:
    contract = build_preview_risk_governor_limits_kill_switch_contract()
    return {key: contract[key] for key in _RISK_CONTRACT_SAFE_KEYS}


def _risk_limits_read_model_entries() -> list[dict[str, Any]]:
    return [
        {
            "risk_limit_read_model_id": f"risk_limit_read_model_{category_id}",
            "source_limit_category_id": category_id,
            "display_name": display_name,
            "read_model_classification": "static_limit_read_model_only",
            "limit_scope": limit_scope,
            "planned_measurement": measurement,
            "planned_value_type": value_type,
            "planned_fixture_key": f"fixture_{category_id}",
            "required_before_order_flow": True,
            "runtime_enforced_now": False,
            "operator_visibility": "future_read_only_contract",
            "eligible_for_12_2_static_fixture": True,
            "eligible_for_12_4_gate_matrix": True,
            "limit_enforcement_allowed_now": False,
            "order_generation_allowed_now": False,
            "order_submission_allowed_now": False,
            "private_endpoint_access_allowed_now": False,
            "network_io_allowed_now": False,
            "config_file_read_allowed_now": False,
            "credential_secret_read_allowed_now": False,
            "safe_for_offline_tests": True,
            "notes": notes,
        }
        for category_id, display_name, limit_scope, measurement, value_type, notes in _LIMIT_SPECS
    ]


def build_preview_risk_governor_limits_read_model() -> dict[str, Any]:
    """Build the Block J static risk limits read model without runtime enforcement."""
    entries = _risk_limits_read_model_entries()
    ids = [entry["risk_limit_read_model_id"] for entry in entries]
    return {
        "schema_version": PREVIEW_RISK_GOVERNOR_LIMITS_READ_MODEL_SCHEMA_VERSION,
        "risk_governor_limits_read_model_kind": PREVIEW_RISK_GOVERNOR_LIMITS_READ_MODEL_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "risk_governor_limits_read_model_status": RISK_GOVERNOR_LIMITS_READ_MODEL_STATUS,
        "risk_governor_limits_read_model_decision": RISK_GOVERNOR_LIMITS_READ_MODEL_DECISION,
        "ready_for_block_j_2": READY_FOR_BLOCK_J_2,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "risk_contract_reference": _risk_contract_reference(),
        "risk_limits_read_model_scope": {
            "scope_name": "risk_governor_limits_read_model",
            "read_model_only": True,
            "derived_from_risk_contract_12_0": True,
            **{key: False for key in _FALSE_SCOPE_FLAGS},
            "exe_direction_preserved": True,
        },
        "risk_limits_read_model_entries": entries,
        "default_risk_limit_read_model_selection": {
            "risk_limit_read_model_id": "risk_limit_read_model_max_order_notional",
            "source_limit_category_id": "max_order_notional",
            "reason": "lowest-risk first read-model entry; static only, no runtime enforcement, no order flow",
            "runtime_enforced_now": False,
            "order_submission_allowed_now": False,
        },
        "risk_limits_read_model_summary": {
            "entry_count": 9,
            "default_selection_id": "risk_limit_read_model_max_order_notional",
            "runtime_enforced_entry_count": 0,
            "limit_enforcement_enabled_entry_count": 0,
            "order_generation_enabled_entry_count": 0,
            "order_submission_enabled_entry_count": 0,
            "private_endpoint_enabled_entry_count": 0,
            "network_enabled_entry_count": 0,
            "config_file_read_enabled_entry_count": 0,
            "credential_secret_read_enabled_entry_count": 0,
            "offline_safe_entry_count": 9,
            "entries_eligible_for_12_2_static_fixture": 9,
            "entries_eligible_for_12_4_gate_matrix": 9,
            "safe_to_render_in_future_ui_as_read_only": True,
            "safe_for_runtime_execution_now": False,
            "safe_for_order_execution_now": False,
        },
        "risk_limits_read_model_matrix": {
            "risk_limit_read_model_ids": ids,
            "money_limit_ids": [
                "risk_limit_read_model_max_order_notional",
                "risk_limit_read_model_max_daily_notional",
                "risk_limit_read_model_max_position_notional",
                "risk_limit_read_model_max_daily_loss",
            ],
            "count_or_rate_limit_ids": [
                "risk_limit_read_model_max_open_positions",
                "risk_limit_read_model_max_order_rate",
            ],
            "percent_limit_ids": ["risk_limit_read_model_max_drawdown"],
            "allowlist_limit_ids": [
                "risk_limit_read_model_allowed_symbols",
                "risk_limit_read_model_allowed_modes",
            ],
            "entries_requiring_12_2_static_fixture": ids,
            "entries_requiring_12_4_gate_matrix": ids,
            "entries_never_runtime_enabled_in_12_1": ids,
        },
        "blocked_risk_limits_read_model_capabilities": BLOCKED_RISK_LIMITS_READ_MODEL_CAPABILITIES,
        "risk_limits_read_model_boundaries": {
            "risk_limits_read_model_is_static": True,
            "risk_limits_read_model_is_derived_from_12_0": True,
            "risk_limits_read_model_can_feed_12_2_static_fixture": True,
            "risk_limits_read_model_can_feed_12_4_gate_matrix": True,
            "risk_limits_read_model_cannot_feed_runtime_directly": True,
            "risk_limits_read_model_cannot_enforce_limits": True,
            "risk_limits_read_model_cannot_trigger_kill_switch": True,
            "risk_limits_read_model_cannot_generate_orders": True,
            "risk_limits_read_model_cannot_submit_orders": True,
            "risk_limits_read_model_cannot_cancel_orders": True,
            "risk_limits_read_model_cannot_replace_orders": True,
            "risk_limits_read_model_cannot_access_private_endpoints": True,
            "risk_limits_read_model_cannot_read_account": True,
            "risk_limits_read_model_balance_read_blocked": True,
            "risk_limits_read_model_cannot_read_positions": True,
            "risk_limits_read_model_cannot_read_orders": True,
            "risk_limits_read_model_cannot_read_fills": True,
            "risk_limits_read_model_cannot_read_market_data": True,
            "risk_limits_read_model_cannot_open_network_connection": True,
            "risk_limits_read_model_cannot_perform_dns_lookup": True,
            "risk_limits_read_model_cannot_perform_http_request": True,
            "risk_limits_read_model_cannot_open_websocket": True,
            "risk_limits_read_model_cannot_read_credentials": True,
            "risk_limits_read_model_cannot_read_secrets": True,
            "risk_limits_read_model_cannot_read_secure_store": True,
            "risk_limits_read_model_cannot_change_qml_or_bridge": True,
        },
        "non_activation_evidence": {
            "risk_contract_12_0_read": True,
            "risk_limits_read_model_built": True,
            **{
                key: False
                for key in [
                    "risk_runtime_enforcement_started",
                    "limit_runtime_enforcement_started",
                    "kill_switch_runtime_trigger_enabled",
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
    "NEXT_STEP",
    "NEXT_STEP_TITLE",
    "PREVIEW_RISK_GOVERNOR_LIMITS_READ_MODEL_KIND",
    "PREVIEW_RISK_GOVERNOR_LIMITS_READ_MODEL_SCHEMA_VERSION",
    "READY_FOR_BLOCK_J_2",
    "RISK_GOVERNOR_LIMITS_READ_MODEL_DECISION",
    "RISK_GOVERNOR_LIMITS_READ_MODEL_STATUS",
    "STATUS",
    "STEP_ID",
    "build_preview_risk_governor_limits_read_model",
]
