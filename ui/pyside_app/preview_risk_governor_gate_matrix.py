"""FUNCTIONAL-PREVIEW-12.4 Block J risk governor gate matrix.

Pure-data gate matrix derived from the 12.0-12.3 Block J artifacts. This
module is static only: it does not enforce limits, trigger kill switches,
access endpoints, start loops, or create order flow.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_kill_switch_read_model import build_preview_kill_switch_read_model
from ui.pyside_app.preview_risk_governor_limits_kill_switch_contract import (
    build_preview_risk_governor_limits_kill_switch_contract,
)
from ui.pyside_app.preview_risk_governor_limits_read_model import (
    build_preview_risk_governor_limits_read_model,
)
from ui.pyside_app.preview_risk_limits_static_fixture import (
    build_preview_risk_limits_static_fixture,
)

PREVIEW_RISK_GOVERNOR_GATE_MATRIX_SCHEMA_VERSION: Final[str] = (
    "preview_risk_governor_gate_matrix.v1"
)
PREVIEW_RISK_GOVERNOR_GATE_MATRIX_KIND: Final[str] = (
    "functional_preview_block_j_risk_governor_gate_matrix"
)
BLOCK_ID: Final[str] = "J"
STEP_ID: Final[str] = "12.4"
RISK_GOVERNOR_GATE_MATRIX_STATUS: Final[str] = (
    "risk_governor_gate_matrix_ready_no_runtime_enforcement"
)
RISK_GOVERNOR_GATE_MATRIX_DECISION: Final[str] = (
    "BUILD_RISK_GOVERNOR_GATE_MATRIX_ONLY_NO_RUNTIME_NO_ORDER_FLOW"
)
READY_FOR_BLOCK_J_5: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-12.5"
NEXT_STEP_TITLE: Final[str] = "BLOCK J CLOSURE AUDIT"
STATUS: Final[str] = "ready_for_functional_preview_12_5_block_j_closure_audit"

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
_KILL_SWITCH_READ_MODEL_SAFE_KEYS: Final[list[str]] = [
    "schema_version",
    "kill_switch_read_model_kind",
    "kill_switch_read_model_status",
    "kill_switch_read_model_decision",
    "ready_for_block_j_4",
    "next_step",
    "next_step_title",
    "status",
]

_GATE_SPECS: Final[list[tuple[str, str, str, str, str]]] = [
    (
        "contract_present_gate",
        "Risk contract present gate",
        "FUNCTIONAL-PREVIEW-12.0",
        "preview_risk_governor_limits_kill_switch_contract",
        "risk contract exists and remains contract-only",
    ),
    (
        "limits_read_model_present_gate",
        "Risk limits read model present gate",
        "FUNCTIONAL-PREVIEW-12.1",
        "preview_risk_governor_limits_read_model",
        "limits read model exists and remains read-only",
    ),
    (
        "static_fixture_present_gate",
        "Risk limits static fixture present gate",
        "FUNCTIONAL-PREVIEW-12.2",
        "preview_risk_limits_static_fixture",
        "static fixture exists and remains example-only",
    ),
    (
        "kill_switch_read_model_present_gate",
        "Kill switch read model present gate",
        "FUNCTIONAL-PREVIEW-12.3",
        "preview_kill_switch_read_model",
        "kill switch read model exists and cannot trigger runtime",
    ),
    (
        "order_flow_blocked_gate",
        "Order flow blocked gate",
        "FUNCTIONAL-PREVIEW-12.4",
        "risk_governor_gate_matrix",
        "order generation and submission remain blocked",
    ),
    (
        "private_endpoint_blocked_gate",
        "Private endpoint blocked gate",
        "FUNCTIONAL-PREVIEW-12.4",
        "risk_governor_gate_matrix",
        "private endpoint access remains blocked",
    ),
    (
        "network_blocked_gate",
        "Network blocked gate",
        "FUNCTIONAL-PREVIEW-12.4",
        "risk_governor_gate_matrix",
        "network, DNS, HTTP and WebSocket access remain blocked",
    ),
    (
        "live_trading_blocked_gate",
        "Live trading blocked gate",
        "FUNCTIONAL-PREVIEW-12.4",
        "risk_governor_gate_matrix",
        "live production trading remains blocked",
    ),
]

_FALSE_SCOPE_FLAGS: Final[list[str]] = [
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

BLOCKED_RISK_GOVERNOR_GATE_MATRIX_CAPABILITIES: Final[list[str]] = [
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
FUTURE_STEPS: Final[list[str]] = ["functional_preview_12_5_block_j_closure_audit"]


def _safe_subset(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: source[key] for key in keys}


def _build_entries() -> list[dict[str, Any]]:
    return [
        {
            "gate_matrix_entry_id": f"risk_governor_gate_matrix_{source_gate_id}",
            "source_gate_id": source_gate_id,
            "display_name": display_name,
            "gate_classification": "static_gate_matrix_entry_only",
            "required_prior_step": required_prior_step,
            "required_prior_artifact": required_prior_artifact,
            "gate_condition": gate_condition,
            "gate_state_now": "static_contract_verified",
            "gate_result_now": "blocked_until_future_gate",
            "blocks_runtime_until_future_gate": True,
            "blocks_order_flow_now": True,
            "blocks_private_endpoint_now": True,
            "blocks_network_now": True,
            "blocks_live_trading_now": True,
            "eligible_for_12_5_closure_audit": True,
            "operator_visibility": "future_read_only_gate_matrix",
            "safe_for_offline_tests": True,
            "notes": "Static gate matrix row only; no runtime, private endpoint, network, live trading, or order flow is enabled.",
        }
        for source_gate_id, display_name, required_prior_step, required_prior_artifact, gate_condition in _GATE_SPECS
    ]


def build_preview_risk_governor_gate_matrix() -> dict[str, Any]:
    entries = _build_entries()
    entry_ids = [entry["gate_matrix_entry_id"] for entry in entries]
    return {
        "schema_version": PREVIEW_RISK_GOVERNOR_GATE_MATRIX_SCHEMA_VERSION,
        "risk_governor_gate_matrix_kind": PREVIEW_RISK_GOVERNOR_GATE_MATRIX_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "risk_governor_gate_matrix_status": RISK_GOVERNOR_GATE_MATRIX_STATUS,
        "risk_governor_gate_matrix_decision": RISK_GOVERNOR_GATE_MATRIX_DECISION,
        "ready_for_block_j_5": READY_FOR_BLOCK_J_5,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "risk_contract_reference": _safe_subset(
            build_preview_risk_governor_limits_kill_switch_contract(), _RISK_CONTRACT_SAFE_KEYS
        ),
        "risk_limits_read_model_reference": _safe_subset(
            build_preview_risk_governor_limits_read_model(), _RISK_LIMITS_READ_MODEL_SAFE_KEYS
        ),
        "risk_limits_static_fixture_reference": _safe_subset(
            build_preview_risk_limits_static_fixture(), _RISK_LIMITS_STATIC_FIXTURE_SAFE_KEYS
        ),
        "kill_switch_read_model_reference": _safe_subset(
            build_preview_kill_switch_read_model(), _KILL_SWITCH_READ_MODEL_SAFE_KEYS
        ),
        "risk_governor_gate_matrix_scope": {
            "scope_name": "risk_governor_gate_matrix",
            "gate_matrix_only": True,
            "derived_from_risk_contract_12_0": True,
            "derived_from_limits_read_model_12_1": True,
            "derived_from_static_fixture_12_2": True,
            "derived_from_kill_switch_read_model_12_3": True,
            **{key: False for key in _FALSE_SCOPE_FLAGS},
            "exe_direction_preserved": True,
        },
        "risk_governor_gate_matrix_entries": entries,
        "default_risk_governor_gate_matrix_selection": {
            "gate_matrix_entry_id": "risk_governor_gate_matrix_contract_present_gate",
            "source_gate_id": "contract_present_gate",
            "reason": "first gate verifies Block J contract chain; static only, no runtime, no order flow",
            "gate_result_now": "blocked_until_future_gate",
            "order_submission_allowed_now": False,
        },
        "risk_governor_gate_matrix_summary": {
            "entry_count": 8,
            "default_selection_id": "risk_governor_gate_matrix_contract_present_gate",
            "runtime_enabled_entry_count": 0,
            "order_flow_enabled_entry_count": 0,
            "private_endpoint_enabled_entry_count": 0,
            "network_enabled_entry_count": 0,
            "live_trading_enabled_entry_count": 0,
            "entries_blocking_runtime_until_future_gate": 8,
            "entries_blocking_order_flow_now": 8,
            "entries_blocking_private_endpoint_now": 8,
            "entries_blocking_network_now": 8,
            "entries_blocking_live_trading_now": 8,
            "entries_eligible_for_12_5_closure_audit": 8,
            "offline_safe_entry_count": 8,
            "safe_to_render_in_future_ui_as_read_only": True,
            "safe_for_runtime_execution_now": False,
            "safe_for_order_execution_now": False,
            "ready_for_12_5_closure_audit": True,
        },
        "risk_governor_gate_matrix_cross_reference": {
            "referenced_steps": [
                "FUNCTIONAL-PREVIEW-12.0",
                "FUNCTIONAL-PREVIEW-12.1",
                "FUNCTIONAL-PREVIEW-12.2",
                "FUNCTIONAL-PREVIEW-12.3",
            ],
            "referenced_artifacts": [
                "preview_risk_governor_limits_kill_switch_contract",
                "preview_risk_governor_limits_read_model",
                "preview_risk_limits_static_fixture",
                "preview_kill_switch_read_model",
            ],
            "gate_matrix_entry_ids": entry_ids,
            "runtime_blocking_gate_ids": entry_ids,
            "order_flow_blocking_gate_ids": entry_ids,
            "private_endpoint_blocking_gate_ids": entry_ids,
            "network_blocking_gate_ids": entry_ids,
            "live_trading_blocking_gate_ids": entry_ids,
            "entries_requiring_12_5_closure_audit": entry_ids,
        },
        "gate_matrix_contract": {
            "gate_matrix_contract_id": "block_j_risk_governor_gate_matrix_contract",
            "gate_matrix_is_static": True,
            "gate_matrix_is_read_only": True,
            "gate_matrix_values_are_not_runtime_gates": True,
            "gate_matrix_cannot_enable_runtime": True,
            "gate_matrix_cannot_enable_order_flow": True,
            "gate_matrix_cannot_enable_private_endpoints": True,
            "gate_matrix_cannot_enable_network": True,
            "gate_matrix_cannot_enable_live_trading": True,
            "gate_matrix_requires_12_5_closure_before_next_block": True,
            "live_production_trading_forbidden": True,
        },
        "blocked_risk_governor_gate_matrix_capabilities": BLOCKED_RISK_GOVERNOR_GATE_MATRIX_CAPABILITIES,
        "risk_governor_gate_matrix_boundaries": {
            "risk_governor_gate_matrix_is_static": True,
            "risk_governor_gate_matrix_is_derived_from_12_0": True,
            "risk_governor_gate_matrix_is_derived_from_12_1": True,
            "risk_governor_gate_matrix_is_derived_from_12_2": True,
            "risk_governor_gate_matrix_is_derived_from_12_3": True,
            "risk_governor_gate_matrix_can_feed_12_5_closure_audit": True,
            "risk_governor_gate_matrix_cannot_feed_runtime_directly": True,
            "risk_governor_gate_matrix_cannot_enable_runtime": True,
            "risk_governor_gate_matrix_cannot_enforce_limits": True,
            "risk_governor_gate_matrix_cannot_trigger_kill_switch": True,
            "risk_governor_gate_matrix_cannot_mutate_kill_switch_state": True,
            "risk_governor_gate_matrix_cannot_generate_orders": True,
            "risk_governor_gate_matrix_cannot_submit_orders": True,
            "risk_governor_gate_matrix_cannot_cancel_orders": True,
            "risk_governor_gate_matrix_cannot_replace_orders": True,
            "risk_governor_gate_matrix_cannot_access_private_endpoints": True,
            "risk_governor_gate_matrix_cannot_read_account": True,
            "risk_governor_gate_matrix_balance_read_blocked": True,
            "risk_governor_gate_matrix_cannot_read_positions": True,
            "risk_governor_gate_matrix_cannot_read_orders": True,
            "risk_governor_gate_matrix_cannot_read_fills": True,
            "risk_governor_gate_matrix_cannot_read_market_data": True,
            "risk_governor_gate_matrix_cannot_open_network_connection": True,
            "risk_governor_gate_matrix_cannot_perform_dns_lookup": True,
            "risk_governor_gate_matrix_cannot_perform_http_request": True,
            "risk_governor_gate_matrix_cannot_open_websocket": True,
            "risk_governor_gate_matrix_cannot_read_credentials": True,
            "risk_governor_gate_matrix_cannot_read_secrets": True,
            "risk_governor_gate_matrix_cannot_read_secure_store": True,
            "risk_governor_gate_matrix_cannot_change_qml_or_bridge": True,
        },
        "non_activation_evidence": {
            "risk_contract_12_0_read": True,
            "risk_limits_read_model_12_1_read": True,
            "risk_limits_static_fixture_12_2_read": True,
            "kill_switch_read_model_12_3_read": True,
            "risk_governor_gate_matrix_built": True,
            **{
                key: False
                for key in [
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
        "future_steps": FUTURE_STEPS,
        "status": STATUS,
    }
