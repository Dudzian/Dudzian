"""FUNCTIONAL-PREVIEW-14.0 Block L runtime activation contract.

Pure-data contract that starts Block L from the accepted Block K closure audit.
It declares future gate and mode candidates without starting runtime, live canary,
orders, private endpoints, network access, filesystem access, QML, or bridge
surfaces.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_block_k_closure_audit import build_preview_block_k_closure_audit

PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "preview_block_l_runtime_activation_contract.v1"
)
PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_CONTRACT_KIND: Final[str] = (
    "functional_preview_block_l_runtime_activation_contract"
)
BLOCK_ID: Final[str] = "L"
STEP_ID: Final[str] = "14.0"
BLOCK_L_RUNTIME_ACTIVATION_CONTRACT_STATUS: Final[str] = (
    "block_l_runtime_activation_contract_ready_no_runtime_activation"
)
BLOCK_L_RUNTIME_ACTIVATION_CONTRACT_DECISION: Final[str] = (
    "START_BLOCK_L_WITH_CONTRACT_ONLY_NO_RUNTIME_ACTIVATION_NO_LIVE_CANARY_NO_IO"
)
READY_FOR_BLOCK_L_1: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-14.1"
NEXT_STEP_TITLE: Final[str] = "RUNTIME ACTIVATION READ MODEL"
STATUS: Final[str] = "ready_for_functional_preview_14_1_runtime_activation_read_model"

_BLOCK_K_REFERENCE_KEYS: Final[list[str]] = [
    "schema_version",
    "block_k_closure_audit_kind",
    "block_k_closure_audit_status",
    "block_k_closure_audit_decision",
    "ready_for_next_block",
    "next_block",
    "next_step",
    "next_step_title",
    "closure_line",
    "status",
]

_PRINCIPLES: Final[list[str]] = [
    "contract_before_activation",
    "read_model_before_gate_execution",
    "gate_matrix_before_runtime_start",
    "paper_before_testnet",
    "testnet_before_live_canary",
    "live_canary_before_live_scale",
    "observability_before_runtime_expansion",
    "rollback_before_runtime_expansion",
    "kill_switch_before_any_activation",
    "fail_closed_on_missing_gate",
]

_GATE_SPECS: Final[list[tuple[str, str, str, str, str]]] = [
    (
        "block_k_closure_verified_gate",
        "Block K closure verified gate",
        "block_transition",
        "source_closure",
        "FUNCTIONAL-PREVIEW-13.6",
    ),
    (
        "runtime_activation_read_model_gate",
        "Runtime activation read model gate",
        "runtime_activation",
        "read_model_presence",
        "FUNCTIONAL-PREVIEW-14.1",
    ),
    (
        "operator_explicit_activation_gate",
        "Operator explicit activation gate",
        "operator_control",
        "operator_confirmation",
        "FUNCTIONAL-PREVIEW-14.x",
    ),
    (
        "kill_switch_ready_gate",
        "Kill switch ready gate",
        "safety",
        "kill_switch_readiness",
        "FUNCTIONAL-PREVIEW-14.x",
    ),
    (
        "observability_ready_gate",
        "Observability ready gate",
        "observability",
        "observability_readiness",
        "FUNCTIONAL-PREVIEW-14.x",
    ),
    (
        "rollback_ready_gate",
        "Rollback ready gate",
        "rollback",
        "rollback_readiness",
        "FUNCTIONAL-PREVIEW-14.x",
    ),
    (
        "soak_ready_gate",
        "Soak ready gate",
        "soak",
        "soak_readiness",
        "FUNCTIONAL-PREVIEW-14.x",
    ),
    (
        "order_private_network_block_gate",
        "Order private network block gate",
        "execution_safety",
        "blocked_capability",
        "FUNCTIONAL-PREVIEW-14.x",
    ),
]

_MODE_SPECS: Final[list[tuple[str, str, str, str]]] = [
    (
        "local_mock_runtime_candidate",
        "Local mock runtime candidate",
        "offline_local_mock",
        "future_local_offline_activation",
    ),
    (
        "recorded_fixture_runtime_candidate",
        "Recorded fixture runtime candidate",
        "offline_recorded_fixture",
        "future_local_replay_activation",
    ),
    (
        "paper_runtime_candidate",
        "Paper runtime candidate",
        "paper_runtime",
        "future_paper_activation",
    ),
    (
        "testnet_sandbox_runtime_candidate",
        "Testnet sandbox runtime candidate",
        "testnet_sandbox_runtime",
        "future_testnet_activation",
    ),
    (
        "live_canary_runtime_candidate",
        "Live canary runtime candidate",
        "live_canary",
        "future_live_canary_activation",
    ),
    (
        "live_scaled_runtime_candidate",
        "Live scaled runtime candidate",
        "live_scaled",
        "future_live_scale_activation",
    ),
]

_BLOCKED_CAPABILITIES: Final[list[str]] = [
    "runtime activation",
    "runtime contract execution",
    "runtime gate execution",
    "gate state mutation",
    "live canary",
    "testnet runtime",
    "paper runtime activation",
    "local mock runtime activation",
    "recorded fixture runtime activation",
    "live scaled runtime",
    "observability runtime collection",
    "metrics collection",
    "metrics export",
    "audit writer",
    "audit export",
    "audit file read",
    "audit file write",
    "log file read",
    "log file write",
    "rollback execution",
    "runtime shutdown",
    "soak runtime",
    "soak scheduler",
    "runtime loop",
    "wall-clock runtime measurement",
    "stability probe",
    "state mutation",
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
    "scheduler",
    "filesystem I/O",
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
    "no observability runtime import",
    "no logger/exporter runtime import",
    "no metrics exporter import",
    "no audit writer import",
    "no audit exporter import",
    "no rollback runner import",
    "no rollback executor import",
    "no soak runner import",
    "no soak scheduler import",
    "no filesystem I/O",
    "no audit file read",
    "no audit file write",
    "no log file read",
    "no log file write",
    "no audit write",
    "no audit export",
    "no runtime shutdown",
    "no state mutation",
    "no wall-clock runtime measurement",
    "no stability probe",
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
    "no runtime contract execution",
    "no gate execution",
    "no gate state mutation",
    "no runtime activation",
    "no live canary",
    "no testnet runtime",
    "no paper runtime activation",
    "no local mock runtime activation",
    "no recorded fixture runtime activation",
    "no live scaled runtime",
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


def _build_block_k_reference() -> dict[str, Any]:
    block_k = build_preview_block_k_closure_audit()
    return {key: block_k[key] for key in _BLOCK_K_REFERENCE_KEYS}


def _build_candidate_gates() -> list[dict[str, Any]]:
    return [
        {
            "runtime_activation_gate_id": f"runtime_activation_gate_{source_gate_id}",
            "source_gate_id": source_gate_id,
            "display_name": display_name,
            "gate_domain": gate_domain,
            "gate_type": gate_type,
            "planned_source_step": planned_source_step,
            "required_for_future_activation": True,
            "eligible_for_14_1_read_model": True,
            "eligible_for_future_gate_matrix": True,
            "runtime_activation_allowed_now": False,
            "runtime_gate_execution_allowed_now": False,
            "gate_state_mutation_allowed_now": False,
            "order_flow_allowed_now": False,
            "private_endpoint_access_allowed_now": False,
            "network_io_allowed_now": False,
            "filesystem_io_allowed_now": False,
            "safe_for_offline_tests": True,
            "notes": f"Static candidate only for {planned_source_step}; no activation in 14.0.",
        }
        for source_gate_id, display_name, gate_domain, gate_type, planned_source_step in _GATE_SPECS
    ]


def _build_candidate_modes() -> list[dict[str, Any]]:
    return [
        {
            "runtime_activation_mode_id": f"runtime_activation_mode_{source_mode_id}",
            "source_mode_id": source_mode_id,
            "display_name": display_name,
            "mode_classification": mode_classification,
            "activation_stage": activation_stage,
            "requires_future_gate": True,
            "allowed_in_14_0": False,
            "runtime_activation_allowed_now": False,
            "order_flow_allowed_now": False,
            "private_endpoint_access_allowed_now": False,
            "network_io_allowed_now": False,
            "credential_read_allowed_now": False,
            "live_trading_allowed_now": False,
            "safe_for_offline_tests": True,
            "notes": f"Static candidate only for {activation_stage}; no activation in 14.0.",
        }
        for source_mode_id, display_name, mode_classification, activation_stage in _MODE_SPECS
    ]


def build_preview_block_l_runtime_activation_contract() -> dict[str, Any]:
    """Build the static Block L runtime activation contract."""
    block_k_reference = _build_block_k_reference()
    gates = _build_candidate_gates()
    modes = _build_candidate_modes()
    gate_ids = [gate["source_gate_id"] for gate in gates]
    mode_ids = [mode["source_mode_id"] for mode in modes]

    return {
        "schema_version": PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_CONTRACT_SCHEMA_VERSION,
        "block_l_runtime_activation_contract_kind": PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_CONTRACT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "block_l_runtime_activation_contract_status": BLOCK_L_RUNTIME_ACTIVATION_CONTRACT_STATUS,
        "block_l_runtime_activation_contract_decision": BLOCK_L_RUNTIME_ACTIVATION_CONTRACT_DECISION,
        "ready_for_block_l_1": READY_FOR_BLOCK_L_1,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "block_k_closure_reference": block_k_reference,
        "runtime_activation_contract_scope": _build_scope(),
        "runtime_activation_contract_principles": list(_PRINCIPLES),
        "runtime_activation_candidate_gates": gates,
        "runtime_activation_candidate_modes": modes,
        "runtime_activation_contract_summary": _build_summary(),
        "runtime_activation_contract_matrix": {
            "candidate_gate_ids": gate_ids,
            "candidate_mode_ids": mode_ids,
            "principles_in_order": list(_PRINCIPLES),
            "gates_requiring_future_execution_gate": gate_ids,
            "modes_requiring_future_activation_gate": mode_ids,
            "blocked_in_14_0_gate_ids": gate_ids,
            "blocked_in_14_0_mode_ids": mode_ids,
            "candidate_gate_domains_by_id": {
                gate["source_gate_id"]: gate["gate_domain"] for gate in gates
            },
            "candidate_mode_classifications_by_id": {
                mode["source_mode_id"]: mode["mode_classification"] for mode in modes
            },
            "candidate_mode_activation_stages_by_id": {
                mode["source_mode_id"]: mode["activation_stage"] for mode in modes
            },
        },
        "runtime_activation_surface_contract": _build_surface_contract(),
        "blocked_runtime_activation_contract_capabilities": list(_BLOCKED_CAPABILITIES),
        "runtime_activation_contract_boundaries": _build_boundaries(),
        "non_activation_evidence": _build_non_activation_evidence(),
        "source_boundaries": list(_SOURCE_BOUNDARIES),
        "future_steps": [
            "functional_preview_14_1_runtime_activation_read_model",
            "functional_preview_14_2_runtime_activation_gate_matrix",
            "functional_preview_14_3_paper_runtime_activation_gate",
            "functional_preview_14_4_testnet_runtime_activation_gate",
            "functional_preview_14_5_live_canary_gate_contract",
            "functional_preview_14_6_block_l_closure_audit",
        ],
        "status": STATUS,
    }


def _build_scope() -> dict[str, bool | str]:
    scope: dict[str, bool | str] = {
        "scope_name": "block_l_runtime_activation_contract",
        "contract_only": True,
        "starts_block_l": True,
        "derived_from_block_k_closure_13_6": True,
        "block_k_closure_required": True,
        "block_k_closure_verified": True,
    }
    false_flags = [
        "runtime_activation_allowed_now",
        "runtime_contract_execution_allowed_now",
        "runtime_gate_execution_allowed_now",
        "gate_state_mutation_allowed_now",
        "live_canary_allowed_now",
        "testnet_runtime_allowed_now",
        "paper_runtime_activation_allowed_now",
        "local_mock_runtime_activation_allowed_now",
        "recorded_fixture_runtime_activation_allowed_now",
        "observability_runtime_allowed_now",
        "audit_writer_allowed_now",
        "audit_export_allowed_now",
        "rollback_execution_allowed_now",
        "runtime_shutdown_allowed_now",
        "soak_runtime_allowed_now",
        "soak_scheduler_allowed_now",
        "runtime_loop_allowed_now",
        "metrics_collection_allowed_now",
        "metrics_export_allowed_now",
        "log_file_read_allowed_now",
        "log_file_write_allowed_now",
        "audit_file_read_allowed_now",
        "audit_file_write_allowed_now",
        "filesystem_io_allowed_now",
        "state_mutation_allowed_now",
        "order_cancel_allowed_now",
        "order_replace_allowed_now",
        "order_generation_allowed_now",
        "order_submission_allowed_now",
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
    for flag in false_flags:
        scope[flag] = False
    scope["exe_direction_preserved"] = True
    return scope


def _build_summary() -> dict[str, bool | int]:
    return {
        "candidate_gate_count": 8,
        "candidate_mode_count": 6,
        "principle_count": 10,
        "runtime_activation_enabled_gate_count": 0,
        "runtime_gate_execution_enabled_gate_count": 0,
        "gate_state_mutation_enabled_gate_count": 0,
        "order_flow_enabled_gate_count": 0,
        "private_endpoint_enabled_gate_count": 0,
        "network_io_enabled_gate_count": 0,
        "filesystem_io_enabled_gate_count": 0,
        "runtime_activation_enabled_mode_count": 0,
        "order_flow_enabled_mode_count": 0,
        "private_endpoint_enabled_mode_count": 0,
        "network_io_enabled_mode_count": 0,
        "credential_read_enabled_mode_count": 0,
        "live_trading_enabled_mode_count": 0,
        "offline_safe_gate_count": 8,
        "offline_safe_mode_count": 6,
        "safe_to_enter_14_1_read_model": True,
        "safe_to_activate_runtime_now": False,
        "safe_to_enter_live_canary_now": False,
        "safe_for_order_flow_now": False,
        "safe_for_private_endpoint_now": False,
        "safe_for_network_io_now": False,
    }


def _build_surface_contract() -> dict[str, bool | str]:
    return {
        "surface_contract_id": "block_l_runtime_activation_contract_surface",
        "contract_is_static": True,
        "block_l_started_by_contract_only": True,
        "runtime_activation_is_not_started_now": True,
        "live_canary_is_not_started_now": True,
        "testnet_runtime_is_not_started_now": True,
        "paper_runtime_is_not_started_now": True,
        "gate_execution_is_not_started_now": True,
        "order_flow_is_not_enabled_now": True,
        "private_endpoint_access_is_not_enabled_now": True,
        "network_io_is_forbidden_now": True,
        "filesystem_io_is_forbidden_now": True,
        "qml_bridge_changes_forbidden_now": True,
        "runtime_activation_requires_future_gate": True,
        "live_canary_requires_future_gate": True,
        "block_l_next_step_is_read_model": True,
    }


def _build_boundaries() -> dict[str, bool]:
    return dict.fromkeys(
        [
            "runtime_activation_contract_is_static",
            "runtime_activation_contract_starts_block_l",
            "runtime_activation_contract_is_derived_from_block_k_closure",
            "runtime_activation_contract_can_feed_14_1_read_model",
            "runtime_activation_contract_cannot_activate_runtime",
            "runtime_activation_contract_cannot_execute_runtime_contract",
            "runtime_activation_contract_cannot_execute_gates",
            "runtime_activation_contract_cannot_mutate_gate_state",
            "runtime_activation_contract_cannot_start_live_canary",
            "runtime_activation_contract_cannot_start_testnet_runtime",
            "runtime_activation_contract_cannot_activate_paper_runtime",
            "runtime_activation_contract_cannot_activate_local_mock_runtime",
            "runtime_activation_contract_cannot_activate_recorded_fixture_runtime",
            "runtime_activation_contract_cannot_start_live_scaled_runtime",
            "runtime_activation_contract_cannot_start_observability_runtime",
            "runtime_activation_contract_cannot_collect_metrics",
            "runtime_activation_contract_cannot_export_metrics",
            "runtime_activation_contract_cannot_write_audit",
            "runtime_activation_contract_cannot_export_audit",
            "runtime_activation_contract_cannot_read_audit_files",
            "runtime_activation_contract_cannot_write_audit_files",
            "runtime_activation_contract_cannot_read_logs",
            "runtime_activation_contract_cannot_write_logs",
            "runtime_activation_contract_cannot_execute_rollback",
            "runtime_activation_contract_cannot_execute_runtime_shutdown",
            "runtime_activation_contract_cannot_run_soak",
            "runtime_activation_contract_cannot_start_scheduler",
            "runtime_activation_contract_cannot_start_runtime_loop",
            "runtime_activation_contract_cannot_measure_wall_clock_runtime",
            "runtime_activation_contract_cannot_run_stability_probe",
            "runtime_activation_contract_cannot_mutate_state",
            "runtime_activation_contract_cannot_generate_orders",
            "runtime_activation_contract_cannot_submit_orders",
            "runtime_activation_contract_cannot_cancel_orders",
            "runtime_activation_contract_cannot_replace_orders",
            "runtime_activation_contract_cannot_mutate_positions",
            "runtime_activation_contract_cannot_access_private_endpoints",
            "runtime_activation_contract_cannot_read_account",
            "runtime_activation_contract_balance_read_blocked",
            "runtime_activation_contract_cannot_read_positions",
            "runtime_activation_contract_cannot_read_orders",
            "runtime_activation_contract_cannot_read_fills",
            "runtime_activation_contract_cannot_read_market_data",
            "runtime_activation_contract_cannot_open_network_connection",
            "runtime_activation_contract_cannot_perform_dns_lookup",
            "runtime_activation_contract_cannot_perform_http_request",
            "runtime_activation_contract_cannot_open_websocket",
            "runtime_activation_contract_cannot_read_credentials",
            "runtime_activation_contract_cannot_read_secrets",
            "runtime_activation_contract_cannot_read_secure_store",
            "runtime_activation_contract_cannot_change_qml_or_bridge",
        ],
        True,
    )


def _build_non_activation_evidence() -> dict[str, bool]:
    evidence = dict.fromkeys(
        [
            "runtime_activation_started",
            "runtime_contract_executed",
            "runtime_gate_executed",
            "gate_state_mutated",
            "live_canary_started",
            "testnet_runtime_started",
            "paper_runtime_activated",
            "local_mock_runtime_activated",
            "recorded_fixture_runtime_activated",
            "live_scaled_runtime_started",
            "observability_runtime_started",
            "metrics_collected",
            "metrics_exported",
            "audit_writer_started",
            "audit_exported",
            "audit_file_read",
            "audit_file_written",
            "log_file_read",
            "log_file_written",
            "rollback_executed",
            "runtime_shutdown_executed",
            "soak_runtime_started",
            "soak_scheduler_started",
            "runtime_loop_started",
            "wall_clock_runtime_measured",
            "stability_probe_started",
            "state_mutated",
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
            "scheduler_started",
            "filesystem_io_performed",
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
        ],
        False,
    )
    evidence["block_k_closure_13_6_read"] = True
    evidence["runtime_activation_contract_built"] = True
    return evidence
