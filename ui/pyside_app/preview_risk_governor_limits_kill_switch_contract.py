"""FUNCTIONAL-PREVIEW-12.0 Block J risk contract.

Pure-data contract for the future risk governor, limits, and hard kill switch.
This module is static only: it does not enforce limits, trigger a kill switch,
instantiate adapters, access endpoints, start loops, or create order flow.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_testnet_sandbox_adapter_closure_audit import (
    build_preview_testnet_sandbox_adapter_closure_audit,
)

PREVIEW_RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "preview_risk_governor_limits_kill_switch_contract.v1"
)
PREVIEW_RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_KIND: Final[str] = (
    "functional_preview_block_j_risk_governor_limits_kill_switch_contract"
)
BLOCK_ID: Final[str] = "J"
STEP_ID: Final[str] = "12.0"
RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_STATUS: Final[str] = (
    "risk_governor_limits_kill_switch_contract_ready_no_runtime_enforcement"
)
RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_DECISION: Final[str] = (
    "START_BLOCK_J_WITH_CONTRACT_ONLY_NO_ORDER_FLOW_NO_RUNTIME_ENFORCEMENT"
)
READY_FOR_BLOCK_J_1: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-12.1"
NEXT_STEP_TITLE: Final[str] = "RISK GOVERNOR LIMITS READ MODEL"
STATUS: Final[str] = "ready_for_functional_preview_12_1_risk_governor_limits_read_model"

_BLOCK_I_CLOSURE_SAFE_KEYS: Final[list[str]] = [
    "schema_version",
    "testnet_sandbox_adapter_closure_audit_kind",
    "testnet_sandbox_adapter_closure_audit_status",
    "testnet_sandbox_adapter_closure_audit_decision",
    "ready_for_block_j",
    "next_block",
    "next_step",
    "next_step_title",
    "closure_line",
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
    "account_fetch_allowed_now",
    "balance_fetch_allowed_now",
    "positions_fetch_allowed_now",
    "orders_fetch_allowed_now",
    "fills_fetch_allowed_now",
    "market_data_fetch_allowed_now",
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

_PRINCIPLE_SPECS: Final[list[tuple[str, str, str]]] = [
    (
        "fail_closed_by_default",
        "Fail closed by default",
        "Future risk decisions must deny unsafe or incomplete inputs by default.",
    ),
    (
        "explicit_allowlist_required",
        "Explicit allowlist required",
        "Future tradable modes and symbols require explicit static allowlists before order flow.",
    ),
    (
        "hard_kill_switch_overrides_all",
        "Hard kill switch overrides all",
        "Future hard kill switch state must supersede every runtime trading path.",
    ),
    (
        "position_and_notional_limits_required",
        "Position and notional limits required",
        "Future order flow requires position and notional ceilings before any submission path.",
    ),
    (
        "loss_limits_required",
        "Loss limits required",
        "Future runtime soak and order flow require daily loss and drawdown limits.",
    ),
    (
        "private_endpoint_requires_read_only_gate",
        "Private endpoint requires read-only gate",
        "Future private endpoint access requires a read-only gate before any account-side lookup.",
    ),
    (
        "order_flow_requires_block_j_completion",
        "Order flow requires Block J completion",
        "No order path may be enabled until the complete Block J risk contract sequence closes.",
    ),
    (
        "live_trading_requires_later_live_canary_gate",
        "Live trading requires later live canary gate",
        "Live production trading remains blocked until a later live canary gate explicitly allows it.",
    ),
]

_LIMIT_SPECS: Final[list[tuple[str, str, str, str, str]]] = [
    (
        "max_order_notional",
        "Max order notional",
        "single_order",
        "planned static notional ceiling per order",
        "Defines a future per-order notional cap without enforcing it now.",
    ),
    (
        "max_daily_notional",
        "Max daily notional",
        "trading_day",
        "planned cumulative daily notional ceiling",
        "Defines a future day-level notional cap without runtime counting now.",
    ),
    (
        "max_position_notional",
        "Max position notional",
        "position",
        "planned position notional ceiling",
        "Defines a future maximum exposure per position.",
    ),
    (
        "max_open_positions",
        "Max open positions",
        "portfolio",
        "planned open position count ceiling",
        "Defines a future portfolio breadth cap.",
    ),
    (
        "max_daily_loss",
        "Max daily loss",
        "trading_day",
        "planned realized and unrealized loss ceiling",
        "Defines a future daily loss stop boundary.",
    ),
    (
        "max_drawdown",
        "Max drawdown",
        "equity_curve",
        "planned peak-to-trough drawdown ceiling",
        "Defines a future drawdown stop boundary.",
    ),
    (
        "max_order_rate",
        "Max order rate",
        "runtime_window",
        "planned order attempts per time window",
        "Defines a future throttle for order attempts.",
    ),
    (
        "allowed_symbols",
        "Allowed symbols",
        "instrument_universe",
        "planned explicit symbol allowlist",
        "Defines a future symbol allowlist before order flow.",
    ),
    (
        "allowed_modes",
        "Allowed modes",
        "execution_mode",
        "planned explicit mode allowlist",
        "Live production remains blocked; only later gated modes may be considered.",
    ),
]

BLOCKED_RISK_CONTRACT_CAPABILITIES: Final[list[str]] = [
    "risk runtime enforcement",
    "limit runtime enforcement",
    "kill switch runtime trigger",
    "manual kill switch trigger",
    "automatic kill switch trigger",
    "order generation",
    "order submission",
    "order cancel",
    "order replace",
    "position mutation",
    "private endpoint access",
    "account fetch",
    "balance fetch",
    "positions fetch",
    "orders fetch",
    "fills fetch",
    "market data fetch",
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
    "no real market data fetch",
    "no private endpoint access",
    "no account fetch",
    "no balance read",
    "no positions fetch",
    "no orders fetch",
    "no fills fetch",
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
    "functional_preview_12_1_risk_governor_limits_read_model",
    "functional_preview_12_2_risk_limits_static_fixture",
    "functional_preview_12_3_kill_switch_read_model",
    "functional_preview_12_4_risk_governor_gate_matrix",
    "functional_preview_12_5_block_j_closure_audit",
]


def _block_i_closure_reference() -> dict[str, Any]:
    closure = build_preview_testnet_sandbox_adapter_closure_audit()
    return {key: closure[key] for key in _BLOCK_I_CLOSURE_SAFE_KEYS}


def _risk_governor_contract_principles() -> list[dict[str, Any]]:
    return [
        {
            "principle_id": pid,
            "display_name": name,
            "description": desc,
            "runtime_enforced_now": False,
            "required_before_order_flow": True,
        }
        for pid, name, desc in _PRINCIPLE_SPECS
    ]


def _risk_limit_categories() -> list[dict[str, Any]]:
    return [
        {
            "limit_category_id": lid,
            "display_name": name,
            "limit_scope": scope,
            "planned_measurement": measurement,
            "required_before_order_flow": True,
            "runtime_enforced_now": False,
            "operator_visibility": "future_read_only_contract",
            "notes": notes,
        }
        for lid, name, scope, measurement, notes in _LIMIT_SPECS
    ]


def build_preview_risk_governor_limits_kill_switch_contract() -> dict[str, Any]:
    """Build the Block J static risk governor contract without runtime enforcement."""
    return {
        "schema_version": PREVIEW_RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_SCHEMA_VERSION,
        "risk_governor_limits_kill_switch_contract_kind": PREVIEW_RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "risk_governor_limits_kill_switch_contract_status": RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_STATUS,
        "risk_governor_limits_kill_switch_contract_decision": RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_DECISION,
        "ready_for_block_j_1": READY_FOR_BLOCK_J_1,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_i_closure_reference": _block_i_closure_reference(),
        "risk_contract_scope": {
            "scope_name": "risk_governor_limits_kill_switch_contract",
            "contract_only": True,
            "derived_from_block_i_closure_11_8": True,
            "starts_block_j": True,
            **{key: False for key in _FALSE_SCOPE_FLAGS},
            "exe_direction_preserved": True,
        },
        "risk_governor_contract_principles": _risk_governor_contract_principles(),
        "risk_limit_categories": _risk_limit_categories(),
        "kill_switch_contract": {
            "kill_switch_contract_id": "hard_kill_switch_contract",
            "display_name": "Hard kill switch contract",
            "contract_only": True,
            "runtime_trigger_allowed_now": False,
            "manual_operator_trigger_allowed_now": False,
            "automatic_trigger_allowed_now": False,
            "must_override_order_generation": True,
            "must_override_order_submission": True,
            "must_override_runtime_loop": True,
            "must_override_private_endpoint_access": True,
            "required_before_any_order_flow": True,
            "required_before_any_runtime_soak": True,
            "required_before_any_live_canary": True,
            "blocked_actions_now": [
                "order_generation",
                "order_submission",
                "order_cancel",
                "order_replace",
                "runtime_loop",
                "private_endpoint_access",
                "live_trading",
            ],
            "future_trigger_sources": [
                "operator_manual_stop",
                "loss_limit_breach",
                "drawdown_limit_breach",
                "order_rate_limit_breach",
                "private_endpoint_error_spike",
                "runtime_health_failure",
            ],
            "notes": "Static hard kill switch contract only; no manual or automatic trigger is active now.",
        },
        "risk_governor_dependency_matrix": {
            "requires_block_i_closure": True,
            "block_i_closure_ready": True,
            "requires_limits_read_model_next": True,
            "requires_static_limit_fixture_later": True,
            "requires_kill_switch_read_model_later": True,
            "requires_order_flow_to_remain_blocked": True,
            "requires_private_endpoint_to_remain_blocked": True,
            "requires_network_io_to_remain_blocked": True,
            "requires_live_trading_to_remain_blocked": True,
            "risk_contract_ready_for_12_1": True,
            "runtime_enforcement_enabled_now": False,
            "order_flow_enabled_now": False,
            "private_endpoint_enabled_now": False,
            "network_io_enabled_now": False,
            "live_trading_enabled_now": False,
        },
        "blocked_risk_contract_capabilities": BLOCKED_RISK_CONTRACT_CAPABILITIES,
        "risk_contract_boundaries": {
            "risk_contract_is_static": True,
            "risk_contract_is_derived_from_block_i_closure": True,
            "risk_contract_starts_block_j": True,
            "risk_contract_can_feed_12_1_read_model": True,
            "risk_contract_cannot_feed_runtime_directly": True,
            "risk_contract_cannot_generate_orders": True,
            "risk_contract_cannot_submit_orders": True,
            "risk_contract_cannot_cancel_orders": True,
            "risk_contract_cannot_replace_orders": True,
            "risk_contract_cannot_access_private_endpoints": True,
            "risk_contract_cannot_fetch_account": True,
            "risk_contract_balance_read_blocked": True,
            "risk_contract_cannot_fetch_positions": True,
            "risk_contract_cannot_fetch_orders": True,
            "risk_contract_cannot_fetch_fills": True,
            "risk_contract_cannot_fetch_market_data": True,
            "risk_contract_cannot_open_network_connection": True,
            "risk_contract_cannot_perform_dns_lookup": True,
            "risk_contract_cannot_perform_http_request": True,
            "risk_contract_cannot_open_websocket": True,
            "risk_contract_cannot_read_credentials": True,
            "risk_contract_cannot_read_secrets": True,
            "risk_contract_cannot_read_secure_store": True,
            "risk_contract_cannot_change_qml_or_bridge": True,
        },
        "non_activation_evidence": {
            "block_i_closure_11_8_read": True,
            "risk_governor_contract_built": True,
            **{
                key: False
                for key in [
                    "risk_runtime_enforcement_started",
                    "limit_runtime_enforcement_started",
                    "kill_switch_runtime_trigger_enabled",
                    "manual_kill_switch_trigger_enabled",
                    "automatic_kill_switch_trigger_enabled",
                    "order_generated",
                    "order_submitted",
                    "order_cancelled",
                    "order_replaced",
                    "position_mutated",
                    "private_endpoint_accessed",
                    "account_fetch_performed",
                    "balance_fetch_performed",
                    "positions_fetch_performed",
                    "orders_fetch_performed",
                    "fills_fetch_performed",
                    "market_data_fetch_performed",
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
    "PREVIEW_RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_KIND",
    "PREVIEW_RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_SCHEMA_VERSION",
    "READY_FOR_BLOCK_J_1",
    "RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_DECISION",
    "RISK_GOVERNOR_LIMITS_KILL_SWITCH_CONTRACT_STATUS",
    "STATUS",
    "STEP_ID",
    "build_preview_risk_governor_limits_kill_switch_contract",
]
