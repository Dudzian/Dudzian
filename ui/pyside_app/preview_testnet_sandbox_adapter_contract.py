"""FUNCTIONAL-PREVIEW-11.0 Block I testnet/sandbox adapter contract.

Pure-data contract for entering Block I. This module intentionally defines only
JSON-serializable boundaries and gates. It performs no I/O, imports no UI,
runtime, exchange, account, credentials, secrets, order, live, testnet, sandbox,
or adapter runtime modules, and implements no adapter behavior.
"""

from __future__ import annotations

from typing import Any, Final

PREVIEW_TESTNET_SANDBOX_ADAPTER_CONTRACT_SCHEMA_VERSION: Final[str] = (
    "preview_testnet_sandbox_adapter_contract.v1"
)
PREVIEW_TESTNET_SANDBOX_ADAPTER_CONTRACT_KIND: Final[str] = (
    "functional_preview_block_i_testnet_sandbox_adapter_contract"
)
BLOCK_ID: Final[str] = "I"
STEP_ID: Final[str] = "11.0"
TESTNET_SANDBOX_ADAPTER_CONTRACT_STATUS: Final[str] = (
    "testnet_sandbox_adapter_contract_ready_no_adapter_implementation"
)
TESTNET_SANDBOX_ADAPTER_CONTRACT_DECISION: Final[str] = (
    "START_BLOCK_I_WITH_CONTRACT_ONLY_NO_TESTNET_RUNTIME_NO_NETWORK_IO"
)
READY_FOR_BLOCK_I_1: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-11.1"
NEXT_STEP_TITLE: Final[str] = "TESTNET/SANDBOX BACKEND CAPABILITY HANDOFF"
STATUS: Final[str] = "ready_for_functional_preview_11_1_testnet_sandbox_backend_capability_handoff"


def _allowed_modes() -> list[dict[str, Any]]:
    modes = [
        "local_mock",
        "recorded_fixture_replay",
        "paper",
        "testnet_contract_only",
        "sandbox_contract_only",
    ]
    return [
        {
            "mode": mode,
            "allowed_now": True,
            "runtime_execution_allowed_now": False,
            "network_io_allowed_now": False,
            "credentials_allowed_now": False,
            "order_submission_allowed_now": False,
            "notes": (
                "Contract eligibility only for Block I 11.0; no adapter, "
                "runtime execution, network I/O, credentials, or order submission."
            ),
        }
        for mode in modes
    ]


def _blocked_modes() -> list[dict[str, Any]]:
    return [
        {"mode": mode, "blocked_now": True, "reason": reason}
        for mode, reason in [
            ("live_production", "Live production trading is outside Block I 11.0."),
            ("live_credentials", "Live credentials remain blocked."),
            ("live_private_account", "Private live account access remains blocked."),
            ("live_order_submission", "Live order submission remains blocked."),
            ("unbounded_testnet_runtime", "Testnet runtime needs later gates."),
            ("unbounded_sandbox_runtime", "Sandbox runtime needs later gates."),
            ("testnet_without_risk_gate", "Risk governor is required before execution."),
            ("sandbox_without_risk_gate", "Risk governor is required before execution."),
            ("testnet_without_observability", "Observability is required before soak."),
            ("sandbox_without_observability", "Observability is required before soak."),
        ]
    ]


def build_preview_testnet_sandbox_adapter_contract() -> dict[str, Any]:
    """Build the pure-data Block I contract-first adapter boundary."""

    return {
        "schema_version": PREVIEW_TESTNET_SANDBOX_ADAPTER_CONTRACT_SCHEMA_VERSION,
        "testnet_sandbox_adapter_contract_kind": PREVIEW_TESTNET_SANDBOX_ADAPTER_CONTRACT_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "testnet_sandbox_adapter_contract_status": TESTNET_SANDBOX_ADAPTER_CONTRACT_STATUS,
        "testnet_sandbox_adapter_contract_decision": TESTNET_SANDBOX_ADAPTER_CONTRACT_DECISION,
        "ready_for_block_i_1": READY_FOR_BLOCK_I_1,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "contract_scope": {
            "scope_name": "testnet_sandbox_adapter_contract",
            "contract_only": True,
            "adapter_implementation_allowed_now": False,
            "testnet_runtime_allowed_now": False,
            "sandbox_runtime_allowed_now": False,
            "live_runtime_allowed_now": False,
            "network_io_allowed_now": False,
            "credentials_allowed_now": False,
            "secrets_allowed_now": False,
            "private_endpoint_access_allowed_now": False,
            "account_fetch_allowed_now": False,
            "balance_fetch_allowed_now": False,
            "positions_fetch_allowed_now": False,
            "orders_fetch_allowed_now": False,
            "fills_fetch_allowed_now": False,
            "order_submission_allowed_now": False,
            "runtime_loop_allowed_now": False,
            "scheduler_allowed_now": False,
            "qml_changes_allowed": False,
            "new_qml_method_calls_allowed": False,
            "bridge_api_changes_allowed": False,
            "exe_packaging_in_scope": False,
            "bat_productization_allowed": False,
            "exe_direction_preserved": True,
        },
        "allowed_modes": _allowed_modes(),
        "blocked_modes": _blocked_modes(),
        "adapter_lifecycle_gates": {
            "contract_required_before_adapter": True,
            "backend_capability_handoff_required_before_wiring": True,
            "adapter_read_model_required_before_runtime": True,
            "credential_gate_required_before_private_endpoint": True,
            "risk_governor_required_before_order_submission": True,
            "kill_switch_required_before_order_submission": True,
            "observability_required_before_soak": True,
            "soak_required_before_live_transition": True,
            "live_transition_gate_required_before_live": True,
            "adapter_implementation_allowed_in_11_0": False,
            "runtime_execution_allowed_in_11_0": False,
        },
        "credential_and_secret_gates": {
            "credential_discovery_allowed_now": False,
            "credential_read_allowed_now": False,
            "secret_read_allowed_now": False,
            "api_key_material_allowed_in_report": False,
            "testnet_credentials_required_future_gate": True,
            "sandbox_credentials_required_future_gate": True,
            "live_credentials_blocked": True,
            "credentials_must_not_be_logged": True,
            "credentials_must_not_enter_audit_payloads": True,
            "secrets_gate_future_step_required": True,
        },
        "private_endpoint_gates": {
            "private_endpoint_access_allowed_now": False,
            "account_fetch_allowed_now": False,
            "balance_fetch_allowed_now": False,
            "positions_fetch_allowed_now": False,
            "orders_fetch_allowed_now": False,
            "fills_fetch_allowed_now": False,
            "private_endpoint_gate_required_future_step": True,
            "read_only_private_account_allowed_without_gate": False,
            "order_endpoint_allowed_without_gate": False,
        },
        "market_data_gates": {
            "public_market_data_contract_allowed_now": True,
            "public_market_data_fetch_allowed_now": False,
            "recorded_replay_allowed_now": True,
            "static_fixture_allowed_now": True,
            "testnet_market_data_fetch_requires_adapter_gate": True,
            "sandbox_market_data_fetch_requires_adapter_gate": True,
            "live_market_data_fetch_blocked": True,
            "network_market_data_allowed_in_11_0": False,
        },
        "order_execution_gates": {
            "paper_order_path_exists_from_block_g": True,
            "testnet_order_submission_allowed_now": False,
            "sandbox_order_submission_allowed_now": False,
            "live_order_submission_allowed_now": False,
            "risk_governor_required": True,
            "kill_switch_required": True,
            "idempotency_required": True,
            "order_lifecycle_audit_required": True,
            "rollback_or_cancel_policy_required": True,
            "operator_confirmation_required_until_later_gate": True,
        },
        "risk_governor_dependency": {
            "block_j_required_before_unattended_order_execution": True,
            "position_limits_required": True,
            "notional_limits_required": True,
            "daily_loss_limits_required": True,
            "symbol_allowlist_required": True,
            "max_order_rate_required": True,
            "kill_switch_required": True,
            "risk_override_policy_required": True,
            "risk_governor_not_implemented_in_11_0": True,
        },
        "observability_dependency": {
            "block_k_required_before_soak": True,
            "health_metrics_required": True,
            "adapter_metrics_required": True,
            "order_lifecycle_metrics_required": True,
            "audit_log_required": True,
            "operator_event_log_required": True,
            "rollback_recovery_required": True,
            "soak_tests_required": True,
            "observability_not_implemented_in_11_0": True,
        },
        "backend_capability_handoff_requirements": {
            "backend_capability_inventory_required": True,
            "premium_saas_modular_audit_can_feed_this": True,
            "adapter_candidate_modules_must_be_mapped_before_wiring": True,
            "existing_backend_modules_must_not_be_reimplemented_blindly": True,
            "existing_backend_modules_must_be_classified_before_use": True,
            "classification_required": [
                "implemented",
                "implemented_not_wired",
                "contract_only",
                "fixture_only",
                "mock_only",
                "exists_but_blocked",
                "high_risk_requires_gate",
                "ready_for_contract_gate",
                "missing",
            ],
            "next_step_expected_output": "testnet_sandbox_backend_capability_handoff",
            "no_backend_module_activation_in_11_0": True,
        },
        "safety_invariants": {
            "no_network_io_performed": True,
            "no_testnet_connection_opened": True,
            "no_sandbox_connection_opened": True,
            "no_live_connection_opened": True,
            "no_credentials_read": True,
            "no_secrets_read": True,
            "no_account_fetch_performed": True,
            "no_balance_fetch_performed": True,
            "no_positions_fetch_performed": True,
            "no_orders_fetch_performed": True,
            "no_fills_fetch_performed": True,
            "no_order_submitted": True,
            "no_order_generated": True,
            "no_fill_simulated": True,
            "no_runtime_loop_started": True,
            "no_scheduler_started": True,
            "no_qml_changes_performed": True,
            "no_bridge_api_changes_performed": True,
            "no_app_py_changes_performed": True,
            "no_bat_changes_performed": True,
            "no_workflow_changes_performed": True,
            "no_dependency_changes_performed": True,
            "exe_direction_preserved": True,
        },
        "blocked_capabilities": [
            "testnet adapter implementation in 11.0",
            "sandbox adapter implementation in 11.0",
            "network I/O in 11.0",
            "exchange API connection in 11.0",
            "credentials/secrets access in 11.0",
            "private account endpoint access",
            "account balance fetch",
            "positions fetch",
            "orders fetch",
            "fills fetch",
            "testnet order submission",
            "sandbox order submission",
            "live order submission",
            "runtime loop",
            "scheduler",
            "unattended execution",
            "risk governor implementation",
            "observability/soak implementation",
            "bridge API changes",
            "QML changes / new QML calls",
            "EXE packaging",
            "live production trading",
        ],
        "source_boundaries": [
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
        ],
        "future_steps": [
            "functional_preview_11_1_testnet_sandbox_backend_capability_handoff",
            "functional_preview_11_2_testnet_sandbox_adapter_read_model",
            "functional_preview_11_3_testnet_sandbox_static_connectivity_fixture",
            "functional_preview_11_4_testnet_sandbox_adapter_config_gate",
            "functional_preview_11_5_testnet_sandbox_credentials_gate_contract",
            "functional_preview_11_6_testnet_sandbox_public_market_data_probe_preview",
            "functional_preview_11_7_testnet_sandbox_private_endpoint_gate",
            "functional_preview_11_8_block_i_closure_audit",
        ],
        "status": STATUS,
    }
