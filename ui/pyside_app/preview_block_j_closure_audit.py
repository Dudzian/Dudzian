"""FUNCTIONAL-PREVIEW-12.5 Block J closure audit.

Pure-data closure audit for the accepted Block J preview artifacts. This module
only assembles static dictionaries from safe prior preview helpers; it does not
activate runtime enforcement, endpoint access, networking, or order flow.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_kill_switch_read_model import build_preview_kill_switch_read_model
from ui.pyside_app.preview_risk_governor_gate_matrix import (
    build_preview_risk_governor_gate_matrix,
)
from ui.pyside_app.preview_risk_governor_limits_kill_switch_contract import (
    build_preview_risk_governor_limits_kill_switch_contract,
)
from ui.pyside_app.preview_risk_governor_limits_read_model import (
    build_preview_risk_governor_limits_read_model,
)
from ui.pyside_app.preview_risk_limits_static_fixture import (
    build_preview_risk_limits_static_fixture,
)

PREVIEW_BLOCK_J_CLOSURE_AUDIT_SCHEMA_VERSION: Final[str] = "preview_block_j_closure_audit.v1"
PREVIEW_BLOCK_J_CLOSURE_AUDIT_KIND: Final[str] = "functional_preview_block_j_closure_audit"
BLOCK_ID: Final[str] = "J"
STEP_ID: Final[str] = "12.5"
BLOCK_J_CLOSURE_AUDIT_STATUS: Final[str] = (
    "block_j_risk_governor_limits_kill_switch_complete_ready_for_next_block"
)
BLOCK_J_CLOSURE_AUDIT_DECISION: Final[str] = (
    "CLOSE_BLOCK_J_RISK_GOVERNOR_LIMITS_KILL_SWITCH_NO_RUNTIME_NO_ORDER_FLOW"
)
READY_FOR_NEXT_BLOCK: Final[bool] = True
NEXT_BLOCK: Final[str] = "BLOK K — OBSERVABILITY / AUDIT / ROLLBACK / SOAK"
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-13.0"
NEXT_STEP_TITLE: Final[str] = "BLOK K — OBSERVABILITY / AUDIT / ROLLBACK / SOAK CONTRACT"
CLOSURE_LINE: Final[str] = "BLOK GOTOWY — PRZECHODZIMY DO KOLEJNEGO BLOKU"
STATUS: Final[str] = "ready_for_functional_preview_13_0_observability_audit_rollback_soak_contract"

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
_GATE_MATRIX_SAFE_KEYS: Final[list[str]] = [
    "schema_version",
    "risk_governor_gate_matrix_kind",
    "risk_governor_gate_matrix_status",
    "risk_governor_gate_matrix_decision",
    "ready_for_block_j_5",
    "next_step",
    "next_step_title",
    "status",
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

BLOCKED_BLOCK_J_CAPABILITIES: Final[list[str]] = [
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

BLOCK_J_SOURCE_BOUNDARIES: Final[list[str]] = [
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


def _subset(source: dict[str, Any], keys: list[str]) -> dict[str, Any]:
    return {key: source[key] for key in keys}


def _completed_step(step: str, title: str, artifact: str) -> dict[str, Any]:
    return {
        "step": step,
        "title": title,
        "artifact": artifact,
        "status": "complete",
        "runtime_enabled": False,
        "order_flow_enabled": False,
        "private_endpoint_enabled": False,
        "network_enabled": False,
        "ready_for_next_step": True,
    }


def build_preview_block_j_closure_audit() -> dict[str, Any]:
    """Build the static Block J closure audit payload."""
    risk_contract = build_preview_risk_governor_limits_kill_switch_contract()
    limits_read_model = build_preview_risk_governor_limits_read_model()
    static_fixture = build_preview_risk_limits_static_fixture()
    kill_switch = build_preview_kill_switch_read_model()
    gate_matrix = build_preview_risk_governor_gate_matrix()

    return {
        "schema_version": PREVIEW_BLOCK_J_CLOSURE_AUDIT_SCHEMA_VERSION,
        "block_j_closure_audit_kind": PREVIEW_BLOCK_J_CLOSURE_AUDIT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_j_closure_audit_status": BLOCK_J_CLOSURE_AUDIT_STATUS,
        "block_j_closure_audit_decision": BLOCK_J_CLOSURE_AUDIT_DECISION,
        "ready_for_next_block": READY_FOR_NEXT_BLOCK,
        "next_block": NEXT_BLOCK,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "closure_line": CLOSURE_LINE,
        "risk_contract_reference": _subset(risk_contract, _RISK_CONTRACT_SAFE_KEYS),
        "risk_limits_read_model_reference": _subset(
            limits_read_model, _RISK_LIMITS_READ_MODEL_SAFE_KEYS
        ),
        "risk_limits_static_fixture_reference": _subset(
            static_fixture, _RISK_LIMITS_STATIC_FIXTURE_SAFE_KEYS
        ),
        "kill_switch_read_model_reference": _subset(kill_switch, _KILL_SWITCH_READ_MODEL_SAFE_KEYS),
        "risk_governor_gate_matrix_reference": _subset(gate_matrix, _GATE_MATRIX_SAFE_KEYS),
        "block_j_closure_scope": {
            "scope_name": "block_j_closure_audit",
            "closure_audit_only": True,
            "closes_block_j": True,
            "derived_from_risk_contract_12_0": True,
            "derived_from_limits_read_model_12_1": True,
            "derived_from_static_fixture_12_2": True,
            "derived_from_kill_switch_read_model_12_3": True,
            "derived_from_gate_matrix_12_4": True,
            **{key: False for key in _FALSE_SCOPE_FLAGS},
            "exe_direction_preserved": True,
        },
        "block_j_completed_steps": [
            _completed_step(
                "FUNCTIONAL-PREVIEW-12.0",
                "Risk Governor Limits Kill Switch Contract",
                "preview_risk_governor_limits_kill_switch_contract",
            ),
            _completed_step(
                "FUNCTIONAL-PREVIEW-12.1",
                "Risk Governor Limits Read Model",
                "preview_risk_governor_limits_read_model",
            ),
            _completed_step(
                "FUNCTIONAL-PREVIEW-12.2",
                "Risk Limits Static Fixture",
                "preview_risk_limits_static_fixture",
            ),
            _completed_step(
                "FUNCTIONAL-PREVIEW-12.3",
                "Kill Switch Read Model",
                "preview_kill_switch_read_model",
            ),
            _completed_step(
                "FUNCTIONAL-PREVIEW-12.4",
                "Risk Governor Gate Matrix",
                "preview_risk_governor_gate_matrix",
            ),
            _completed_step(
                "FUNCTIONAL-PREVIEW-12.5", "Block J Closure Audit", "preview_block_j_closure_audit"
            ),
        ],
        "block_j_completion_matrix": {
            "completed_step_count": 6,
            "expected_step_count": 6,
            "all_steps_complete": True,
            "all_references_ready": True,
            "all_runtime_paths_blocked": True,
            "all_order_flow_paths_blocked": True,
            "all_private_endpoint_paths_blocked": True,
            "all_network_paths_blocked": True,
            "all_live_trading_paths_blocked": True,
            "ready_for_next_block": True,
            "closure_line": CLOSURE_LINE,
        },
        "block_j_safety_summary": {
            "risk_contract_complete": True,
            "limits_read_model_complete": True,
            "static_fixture_complete": True,
            "kill_switch_read_model_complete": True,
            "gate_matrix_complete": True,
            "closure_audit_complete": True,
            "runtime_enforcement_enabled": False,
            "limit_enforcement_enabled": False,
            "kill_switch_runtime_trigger_enabled": False,
            "manual_trigger_enabled": False,
            "automatic_trigger_enabled": False,
            "order_flow_enabled": False,
            "private_endpoint_enabled": False,
            "network_io_enabled": False,
            "live_trading_enabled": False,
            "safe_to_enter_next_block": True,
        },
        "risk_governor_capability_closure": {
            "risk_governor_contract_defined": True,
            "limits_read_model_defined": True,
            "static_limit_fixture_defined": True,
            "kill_switch_read_model_defined": True,
            "gate_matrix_defined": True,
            "closure_audit_defined": True,
            "risk_governor_runtime_enabled": False,
            "limit_runtime_enforcement_enabled": False,
            "kill_switch_runtime_enabled": False,
            "gate_matrix_runtime_enabled": False,
            "runtime_activation_requires_future_block": True,
            "order_flow_activation_requires_future_block": True,
            "private_endpoint_activation_requires_future_block": True,
            "network_activation_requires_future_block": True,
            "live_trading_activation_requires_later_live_canary": True,
        },
        "runtime_order_private_network_closure": {
            key: False
            for key in [
                "runtime_loop_started",
                "scheduler_started",
                "adapter_instantiated",
                "adapter_wired_to_runtime",
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
        "blocked_block_j_capabilities": list(BLOCKED_BLOCK_J_CAPABILITIES),
        "block_j_source_boundaries": list(BLOCK_J_SOURCE_BOUNDARIES),
        "non_activation_evidence": {
            "risk_contract_12_0_read": True,
            "risk_limits_read_model_12_1_read": True,
            "risk_limits_static_fixture_12_2_read": True,
            "kill_switch_read_model_12_3_read": True,
            "risk_governor_gate_matrix_12_4_read": True,
            "block_j_closure_audit_built": True,
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
        "next_block_entry_requirements": {
            "requires_observability_contract": True,
            "requires_audit_envelope_contract": True,
            "requires_rollback_contract": True,
            "requires_soak_contract": True,
            "requires_no_runtime_activation_at_entry": True,
            "requires_order_flow_to_remain_blocked": True,
            "requires_private_endpoint_to_remain_blocked": True,
            "requires_network_io_to_remain_blocked_until_explicit_gate": True,
            "requires_live_trading_to_remain_blocked_until_live_canary": True,
            "requires_exe_direction_to_remain_preserved": True,
        },
        "future_blocks": [
            "BLOK K — OBSERVABILITY / AUDIT / ROLLBACK / SOAK",
            "BLOK L — LIVE CANARY / LIVE TRANSITION GATES",
            "RELEASE — EXE PACKAGING / INSTALLER / SIGNING",
        ],
        "status": STATUS,
    }
