"""FUNCTIONAL-PREVIEW-12.2 Block J risk limits static fixture.

Pure-data fixture derived from the 12.1 risk limits read model. This module is
static only: it does not enforce limits, trigger kill switches, access
endpoints, start loops, or create order flow.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_risk_governor_limits_read_model import (
    build_preview_risk_governor_limits_read_model,
)

PREVIEW_RISK_LIMITS_STATIC_FIXTURE_SCHEMA_VERSION: Final[str] = (
    "preview_risk_limits_static_fixture.v1"
)
PREVIEW_RISK_LIMITS_STATIC_FIXTURE_KIND: Final[str] = (
    "functional_preview_block_j_risk_limits_static_fixture"
)
BLOCK_ID: Final[str] = "J"
STEP_ID: Final[str] = "12.2"
RISK_LIMITS_STATIC_FIXTURE_STATUS: Final[str] = (
    "risk_limits_static_fixture_ready_no_runtime_enforcement"
)
RISK_LIMITS_STATIC_FIXTURE_DECISION: Final[str] = (
    "BUILD_RISK_LIMITS_STATIC_FIXTURE_ONLY_NO_RUNTIME_NO_ORDER_FLOW"
)
READY_FOR_BLOCK_J_3: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-12.3"
NEXT_STEP_TITLE: Final[str] = "KILL SWITCH READ MODEL"
STATUS: Final[str] = "ready_for_functional_preview_12_3_kill_switch_read_model"

_RISK_LIMITS_READ_MODEL_SAFE_KEYS: Final[list[str]] = [
    "schema_version",
    "risk_governor_limits_read_model_kind",
    "risk_governor_limits_read_model_status",
    "risk_governor_limits_read_model_decision",
    "ready_for_block_j_2",
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

_LIMIT_SPECS: Final[list[tuple[str, str, str, str, str, Any, str, str]]] = [
    (
        "max_order_notional",
        "Max order notional",
        "single_order",
        "planned static notional ceiling per order",
        "money",
        100.0,
        "USDT",
        "Static conservative per-order notional example; no runtime enforcement.",
    ),
    (
        "max_daily_notional",
        "Max daily notional",
        "trading_day",
        "planned cumulative daily notional ceiling",
        "money",
        500.0,
        "USDT",
        "Static conservative daily notional example; no runtime enforcement.",
    ),
    (
        "max_position_notional",
        "Max position notional",
        "position",
        "planned position notional ceiling",
        "money",
        250.0,
        "USDT",
        "Static conservative position notional example; no runtime enforcement.",
    ),
    (
        "max_open_positions",
        "Max open positions",
        "portfolio",
        "planned open position count ceiling",
        "integer",
        3,
        "positions",
        "Static conservative open-position count example; no runtime enforcement.",
    ),
    (
        "max_daily_loss",
        "Max daily loss",
        "trading_day",
        "planned realized and unrealized loss ceiling",
        "money",
        50.0,
        "USDT",
        "Static conservative daily loss example; no runtime enforcement.",
    ),
    (
        "max_drawdown",
        "Max drawdown",
        "equity_curve",
        "planned peak-to-trough drawdown ceiling",
        "percent",
        5.0,
        "percent",
        "Static conservative drawdown example; no runtime enforcement.",
    ),
    (
        "max_order_rate",
        "Max order rate",
        "runtime_window",
        "planned order attempts per time window",
        "rate",
        5,
        "orders_per_minute",
        "Static conservative order-rate example; no runtime enforcement.",
    ),
    (
        "allowed_symbols",
        "Allowed symbols",
        "instrument_universe",
        "planned explicit symbol allowlist",
        "symbol_allowlist",
        ["BTC/USDT", "ETH/USDT"],
        "symbols",
        "Static conservative symbol allowlist example; no runtime enforcement.",
    ),
    (
        "allowed_modes",
        "Allowed modes",
        "execution_mode",
        "planned explicit mode allowlist",
        "mode_allowlist",
        [
            "local_mock",
            "recorded_fixture_replay",
            "paper",
            "testnet_contract_only",
            "sandbox_contract_only",
        ],
        "modes",
        "Live production remains blocked; static mode allowlist is not runtime wiring.",
    ),
]

BLOCKED_RISK_LIMITS_STATIC_FIXTURE_CAPABILITIES: Final[list[str]] = [
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
    "functional_preview_12_3_kill_switch_read_model",
    "functional_preview_12_4_risk_governor_gate_matrix",
    "functional_preview_12_5_block_j_closure_audit",
]


def _risk_limits_read_model_reference() -> dict[str, Any]:
    read_model = build_preview_risk_governor_limits_read_model()
    return {key: read_model[key] for key in _RISK_LIMITS_READ_MODEL_SAFE_KEYS}


def _risk_limits_static_fixture_entries() -> list[dict[str, Any]]:
    return [
        {
            "risk_limit_static_fixture_id": f"risk_limit_static_fixture_{category_id}",
            "source_risk_limit_read_model_id": f"risk_limit_read_model_{category_id}",
            "source_limit_category_id": category_id,
            "display_name": display_name,
            "fixture_classification": "static_limit_fixture_only",
            "limit_scope": limit_scope,
            "planned_measurement": measurement,
            "planned_value_type": value_type,
            "fixture_value": fixture_value,
            "fixture_unit": fixture_unit,
            "fixture_profile": "conservative_preview",
            "required_before_order_flow": True,
            "runtime_enforced_now": False,
            "operator_visibility": "future_read_only_fixture",
            "eligible_for_12_3_kill_switch_read_model": True,
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
        for (
            category_id,
            display_name,
            limit_scope,
            measurement,
            value_type,
            fixture_value,
            fixture_unit,
            notes,
        ) in _LIMIT_SPECS
    ]


def build_preview_risk_limits_static_fixture() -> dict[str, Any]:
    """Build the Block J static risk limits fixture without runtime enforcement."""
    entries = _risk_limits_static_fixture_entries()
    ids = [entry["risk_limit_static_fixture_id"] for entry in entries]
    values_by_id = {
        entry["risk_limit_static_fixture_id"]: entry["fixture_value"] for entry in entries
    }
    units_by_id = {
        entry["risk_limit_static_fixture_id"]: entry["fixture_unit"] for entry in entries
    }
    return {
        "schema_version": PREVIEW_RISK_LIMITS_STATIC_FIXTURE_SCHEMA_VERSION,
        "risk_limits_static_fixture_kind": PREVIEW_RISK_LIMITS_STATIC_FIXTURE_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "risk_limits_static_fixture_status": RISK_LIMITS_STATIC_FIXTURE_STATUS,
        "risk_limits_static_fixture_decision": RISK_LIMITS_STATIC_FIXTURE_DECISION,
        "ready_for_block_j_3": READY_FOR_BLOCK_J_3,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "risk_limits_read_model_reference": _risk_limits_read_model_reference(),
        "risk_limits_static_fixture_scope": {
            "scope_name": "risk_limits_static_fixture",
            "fixture_only": True,
            "derived_from_risk_limits_read_model_12_1": True,
            **{key: False for key in _FALSE_SCOPE_FLAGS},
            "exe_direction_preserved": True,
        },
        "risk_limits_static_fixture_entries": entries,
        "default_risk_limits_static_fixture_selection": {
            "risk_limit_static_fixture_id": "risk_limit_static_fixture_max_order_notional",
            "source_limit_category_id": "max_order_notional",
            "reason": "lowest-risk first static fixture entry; no runtime enforcement, no order flow",
            "runtime_enforced_now": False,
            "order_submission_allowed_now": False,
        },
        "risk_limits_static_fixture_summary": {
            "entry_count": 9,
            "default_selection_id": "risk_limit_static_fixture_max_order_notional",
            "runtime_enforced_entry_count": 0,
            "limit_enforcement_enabled_entry_count": 0,
            "order_generation_enabled_entry_count": 0,
            "order_submission_enabled_entry_count": 0,
            "private_endpoint_enabled_entry_count": 0,
            "network_enabled_entry_count": 0,
            "config_file_read_enabled_entry_count": 0,
            "credential_secret_read_enabled_entry_count": 0,
            "offline_safe_entry_count": 9,
            "entries_eligible_for_12_3_kill_switch_read_model": 9,
            "entries_eligible_for_12_4_gate_matrix": 9,
            "money_fixture_entry_count": 4,
            "integer_fixture_entry_count": 1,
            "percent_fixture_entry_count": 1,
            "rate_fixture_entry_count": 1,
            "allowlist_fixture_entry_count": 2,
            "safe_to_render_in_future_ui_as_read_only": True,
            "safe_for_runtime_execution_now": False,
            "safe_for_order_execution_now": False,
        },
        "risk_limits_static_fixture_matrix": {
            "risk_limit_static_fixture_ids": ids,
            "money_fixture_ids": [
                "risk_limit_static_fixture_max_order_notional",
                "risk_limit_static_fixture_max_daily_notional",
                "risk_limit_static_fixture_max_position_notional",
                "risk_limit_static_fixture_max_daily_loss",
            ],
            "count_or_rate_fixture_ids": [
                "risk_limit_static_fixture_max_open_positions",
                "risk_limit_static_fixture_max_order_rate",
            ],
            "percent_fixture_ids": ["risk_limit_static_fixture_max_drawdown"],
            "allowlist_fixture_ids": [
                "risk_limit_static_fixture_allowed_symbols",
                "risk_limit_static_fixture_allowed_modes",
            ],
            "entries_requiring_12_3_kill_switch_read_model": ids,
            "entries_requiring_12_4_gate_matrix": ids,
            "entries_never_runtime_enabled_in_12_2": ids,
            "fixture_values_by_id": values_by_id,
            "fixture_units_by_id": units_by_id,
        },
        "risk_limits_fixture_value_contract": {
            "fixture_profile": "conservative_preview",
            "fixture_values_are_static": True,
            "fixture_values_are_examples_only": True,
            "fixture_values_are_not_runtime_limits": True,
            "fixture_values_cannot_be_loaded_from_config": True,
            "fixture_values_cannot_be_loaded_from_env": True,
            "fixture_values_cannot_be_loaded_from_credentials": True,
            "fixture_values_cannot_be_overridden_now": True,
            "fixture_values_require_12_4_gate_before_any_enforcement": True,
            "live_production_mode_forbidden": True,
            "allowed_modes_fixture_excludes_live_production": True,
        },
        "blocked_risk_limits_static_fixture_capabilities": BLOCKED_RISK_LIMITS_STATIC_FIXTURE_CAPABILITIES,
        "risk_limits_static_fixture_boundaries": {
            "risk_limits_static_fixture_is_static": True,
            "risk_limits_static_fixture_is_derived_from_12_1": True,
            "risk_limits_static_fixture_can_feed_12_3_kill_switch_read_model": True,
            "risk_limits_static_fixture_can_feed_12_4_gate_matrix": True,
            "risk_limits_static_fixture_cannot_feed_runtime_directly": True,
            "risk_limits_static_fixture_cannot_enforce_limits": True,
            "risk_limits_static_fixture_cannot_trigger_kill_switch": True,
            "risk_limits_static_fixture_cannot_generate_orders": True,
            "risk_limits_static_fixture_cannot_submit_orders": True,
            "risk_limits_static_fixture_cannot_cancel_orders": True,
            "risk_limits_static_fixture_cannot_replace_orders": True,
            "risk_limits_static_fixture_cannot_access_private_endpoints": True,
            "risk_limits_static_fixture_cannot_read_account": True,
            "risk_limits_static_fixture_balance_read_blocked": True,
            "risk_limits_static_fixture_cannot_read_positions": True,
            "risk_limits_static_fixture_cannot_read_orders": True,
            "risk_limits_static_fixture_cannot_read_fills": True,
            "risk_limits_static_fixture_cannot_read_market_data": True,
            "risk_limits_static_fixture_cannot_open_network_connection": True,
            "risk_limits_static_fixture_cannot_perform_dns_lookup": True,
            "risk_limits_static_fixture_cannot_perform_http_request": True,
            "risk_limits_static_fixture_cannot_open_websocket": True,
            "risk_limits_static_fixture_cannot_read_credentials": True,
            "risk_limits_static_fixture_cannot_read_secrets": True,
            "risk_limits_static_fixture_cannot_read_secure_store": True,
            "risk_limits_static_fixture_cannot_change_qml_or_bridge": True,
        },
        "non_activation_evidence": {
            "risk_limits_read_model_12_1_read": True,
            "risk_limits_static_fixture_built": True,
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
    "PREVIEW_RISK_LIMITS_STATIC_FIXTURE_KIND",
    "PREVIEW_RISK_LIMITS_STATIC_FIXTURE_SCHEMA_VERSION",
    "READY_FOR_BLOCK_J_3",
    "RISK_LIMITS_STATIC_FIXTURE_DECISION",
    "RISK_LIMITS_STATIC_FIXTURE_STATUS",
    "STATUS",
    "STEP_ID",
    "build_preview_risk_limits_static_fixture",
]
