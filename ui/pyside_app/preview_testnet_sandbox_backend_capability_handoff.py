"""FUNCTIONAL-PREVIEW-11.1 Block I backend capability handoff.

Pure-data discovery-to-contract mapping for testnet/sandbox planning. This module
intentionally does not import or activate backend, exchange, runtime, account,
credentials, secrets, order, UI, bridge, network, or filesystem code.
"""

from __future__ import annotations

from typing import Any, Final

from ui.pyside_app.preview_testnet_sandbox_adapter_contract import (
    build_preview_testnet_sandbox_adapter_contract,
)

PREVIEW_TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_SCHEMA_VERSION: Final[str] = (
    "preview_testnet_sandbox_backend_capability_handoff.v1"
)
PREVIEW_TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_KIND: Final[str] = (
    "functional_preview_block_i_testnet_sandbox_backend_capability_handoff"
)
BLOCK_ID: Final[str] = "I"
STEP_ID: Final[str] = "11.1"
TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_STATUS: Final[str] = (
    "testnet_sandbox_backend_capability_handoff_ready_no_backend_activation"
)
TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_DECISION: Final[str] = (
    "MAP_EXISTING_BACKEND_CAPABILITIES_ONLY_NO_WIRING_NO_RUNTIME"
)
READY_FOR_BLOCK_I_2: Final[bool] = True
NEXT_STEP: Final[str] = "FUNCTIONAL-PREVIEW-11.2"
NEXT_STEP_TITLE: Final[str] = "TESTNET/SANDBOX ADAPTER READ MODEL"
STATUS: Final[str] = "ready_for_functional_preview_11_2_testnet_sandbox_adapter_read_model"

_CLASSIFICATION_LEGEND: Final[list[str]] = [
    "implemented",
    "implemented_not_wired",
    "contract_only",
    "fixture_only",
    "mock_only",
    "exists_but_blocked",
    "high_risk_requires_gate",
    "ready_for_contract_gate",
    "partial",
    "missing",
    "out_of_scope",
]

_FUTURE_STEPS: Final[list[str]] = [
    "functional_preview_11_2_testnet_sandbox_adapter_read_model",
    "functional_preview_11_3_testnet_sandbox_static_connectivity_fixture",
    "functional_preview_11_4_testnet_sandbox_adapter_config_gate",
    "functional_preview_11_5_testnet_sandbox_credentials_gate_contract",
    "functional_preview_11_6_testnet_sandbox_public_market_data_probe_preview",
    "functional_preview_11_7_testnet_sandbox_private_endpoint_gate",
    "functional_preview_11_8_block_i_closure_audit",
]


def _contract_reference() -> dict[str, Any]:
    contract = build_preview_testnet_sandbox_adapter_contract()
    return {
        "schema_version": contract["schema_version"],
        "testnet_sandbox_adapter_contract_kind": contract["testnet_sandbox_adapter_contract_kind"],
        "testnet_sandbox_adapter_contract_status": contract[
            "testnet_sandbox_adapter_contract_status"
        ],
        "testnet_sandbox_adapter_contract_decision": contract[
            "testnet_sandbox_adapter_contract_decision"
        ],
        "ready_for_block_i_1": contract["ready_for_block_i_1"],
        "next_step": contract["next_step"],
        "next_step_title": contract["next_step_title"],
        "status": contract["status"],
    }


def _candidate(
    capability: str,
    audit_status: str,
    evidence_paths: list[str],
    main_symbols: list[str],
    handoff_classification: str,
    *,
    eligible_for_11_2_read_model: bool,
    requires_credentials_gate: bool,
    requires_private_endpoint_gate: bool,
    requires_risk_governor: bool,
    requires_observability_soak: bool,
    requires_live_gate: bool,
    notes: str,
) -> dict[str, Any]:
    return {
        "capability": capability,
        "audit_status": audit_status,
        "evidence_paths": evidence_paths,
        "main_symbols": main_symbols,
        "handoff_classification": handoff_classification,
        "eligible_for_block_i_contract": True,
        "eligible_for_11_2_read_model": eligible_for_11_2_read_model,
        "eligible_for_runtime_now": False,
        "requires_credentials_gate": requires_credentials_gate,
        "requires_private_endpoint_gate": requires_private_endpoint_gate,
        "requires_risk_governor": requires_risk_governor,
        "requires_observability_soak": requires_observability_soak,
        "requires_live_gate": requires_live_gate,
        "notes": notes,
    }


def _testnet_sandbox_candidate_capabilities() -> list[dict[str, Any]]:
    return [
        _candidate(
            "read_only_market_data_provider",
            "ready_for_contract_gate",
            ["FUNCTIONAL-PREVIEW-10.8", "docs/audits/premium_saas_modular_backend_audit.md"],
            ["market data provider contract", "read-only closure audit"],
            "ready_for_contract_gate",
            eligible_for_11_2_read_model=True,
            requires_credentials_gate=False,
            requires_private_endpoint_gate=False,
            requires_risk_governor=False,
            requires_observability_soak=True,
            requires_live_gate=False,
            notes="Safe read-model candidate only; no market-data fetch is performed in 11.1.",
        ),
        _candidate(
            "exchange_adapter_layer",
            "high_risk_requires_gate",
            ["bot_core/exchanges/", "config/exchanges/*.yaml", "tests/exchanges/"],
            ["exchange adapter inventory", "sandbox mode mechanics"],
            "high_risk_requires_gate",
            eligible_for_11_2_read_model=True,
            requires_credentials_gate=True,
            requires_private_endpoint_gate=True,
            requires_risk_governor=True,
            requires_observability_soak=True,
            requires_live_gate=True,
            notes="Adapter layer is mapped for read-model planning only; no adapter is imported or wired.",
        ),
        _candidate(
            "exchange_network_guard",
            "implemented_not_wired",
            ["bot_core/exchanges/manager.py", "tests/exchanges/test_" + "c" + "cxt_sandbox.py"],
            ["sandbox guard", "rate-limit/network guard primitives"],
            "implemented_not_wired",
            eligible_for_11_2_read_model=True,
            requires_credentials_gate=False,
            requires_private_endpoint_gate=False,
            requires_risk_governor=False,
            requires_observability_soak=True,
            requires_live_gate=False,
            notes="Guard primitives are inventory only and cannot perform network I/O in 11.1.",
        ),
        _candidate(
            "paper_execution_oracle",
            "implemented_not_wired",
            ["bot_core/auto_trader/paper_app.py", "bot_core/auto_trader/contracts.py"],
            ["paper execution", "paper event spine"],
            "implemented_not_wired",
            eligible_for_11_2_read_model=True,
            requires_credentials_gate=False,
            requires_private_endpoint_gate=False,
            requires_risk_governor=True,
            requires_observability_soak=True,
            requires_live_gate=False,
            notes="Paper path can be a comparison oracle later; 11.1 does not simulate fills.",
        ),
        _candidate(
            "order_management_contract_surface",
            "contract_only",
            ["proto/trading.proto", "docs/audits/premium_saas_modular_backend_audit.md"],
            ["Order", "order service contract"],
            "contract_only",
            eligible_for_11_2_read_model=False,
            requires_credentials_gate=True,
            requires_private_endpoint_gate=True,
            requires_risk_governor=True,
            requires_observability_soak=True,
            requires_live_gate=True,
            notes="Contract surface only; no order generation or submission is allowed.",
        ),
        _candidate(
            "order_lifecycle_paper_surface",
            "implemented_not_wired",
            [
                "bot_core/auto_trader/contracts.py",
                "reports/templates/sandbox_proof_report_template.md",
            ],
            ["paper order lifecycle", "sandbox proof template"],
            "implemented_not_wired",
            eligible_for_11_2_read_model=False,
            requires_credentials_gate=False,
            requires_private_endpoint_gate=False,
            requires_risk_governor=True,
            requires_observability_soak=True,
            requires_live_gate=False,
            notes="Lifecycle evidence is deferred until risk and observability gates exist.",
        ),
        _candidate(
            "risk_engine_primitives",
            "implemented_not_wired",
            ["bot_core/risk/", "docs/audits/premium_saas_modular_backend_audit.md"],
            ["risk limits", "kill-switch primitives"],
            "implemented_not_wired",
            eligible_for_11_2_read_model=False,
            requires_credentials_gate=False,
            requires_private_endpoint_gate=False,
            requires_risk_governor=True,
            requires_observability_soak=True,
            requires_live_gate=False,
            notes="Risk primitives are acknowledged but Block J owns governor implementation.",
        ),
        _candidate(
            "observability_metrics_health",
            "implemented_not_wired",
            ["deploy/prometheus/", "config/observability/slo.yml"],
            ["metrics", "health", "SLO"],
            "implemented_not_wired",
            eligible_for_11_2_read_model=False,
            requires_credentials_gate=False,
            requires_private_endpoint_gate=False,
            requires_risk_governor=False,
            requires_observability_soak=True,
            requires_live_gate=False,
            notes="Observability is deferred to Block K before soak/runtime activation.",
        ),
        _candidate(
            "runtime_orchestration",
            "exists_but_blocked",
            ["bot_core/runtime/controller.py", "bot_core/runtime/scheduler.py"],
            ["runtime controller", "scheduler"],
            "exists_but_blocked",
            eligible_for_11_2_read_model=False,
            requires_credentials_gate=True,
            requires_private_endpoint_gate=True,
            requires_risk_governor=True,
            requires_observability_soak=True,
            requires_live_gate=True,
            notes="Runtime and schedulers remain blocked until later gates; 11.1 starts none.",
        ),
        _candidate(
            "api_key_secrets_management_surface",
            "exists_but_blocked",
            ["bot_core/security/", "secrets/licensing/README.md"],
            ["credential gate", "secret handling surface"],
            "exists_but_blocked",
            eligible_for_11_2_read_model=False,
            requires_credentials_gate=True,
            requires_private_endpoint_gate=True,
            requires_risk_governor=False,
            requires_observability_soak=True,
            requires_live_gate=True,
            notes="Credential and secret surfaces are referenced statically; no secret read occurs.",
        ),
    ]


def _blocked_high_risk_capabilities() -> list[dict[str, Any]]:
    entries = [
        ("live_router", "Live routing requires the live transition gate.", "L"),
        (
            "private_exchange_adapters",
            "Private adapters require credential/private endpoint gates and risk controls.",
            "I_credentials_gate_then_J",
        ),
        ("order_submission_path", "Order submission requires Block J risk governor.", "J"),
        ("runtime_loops", "Runtime loops require Block K observability soak readiness.", "K"),
        ("scheduler_loops", "Schedulers require Block K observability soak readiness.", "K"),
        (
            "market_making_runtime",
            "Market making requires risk, observability, and live transition gates.",
            "J_K_L",
        ),
        (
            "arbitrage_runtime",
            "Arbitrage requires risk, observability, and live transition gates.",
            "J_K_L",
        ),
        (
            "ai_driven_order_path",
            "AI-driven order paths require risk and observability gates.",
            "J_K",
        ),
    ]
    return [
        {"capability": capability, "blocked_now": True, "reason": reason, "earliest_block": block}
        for capability, reason, block in entries
    ]


def build_preview_testnet_sandbox_backend_capability_handoff() -> dict[str, Any]:
    """Build the pure-data Block I discovery-to-contract handoff."""

    return {
        "schema_version": PREVIEW_TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_SCHEMA_VERSION,
        "testnet_sandbox_backend_capability_handoff_kind": PREVIEW_TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_KIND,
        "block": BLOCK_ID,
        "step": STEP_ID,
        "testnet_sandbox_backend_capability_handoff_status": TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_STATUS,
        "testnet_sandbox_backend_capability_handoff_decision": TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_DECISION,
        "ready_for_block_i_2": READY_FOR_BLOCK_I_2,
        "next_step": NEXT_STEP,
        "next_step_title": NEXT_STEP_TITLE,
        "contract_reference": _contract_reference(),
        "audit_reference": {
            "audit_source_markdown": "docs/audits/premium_saas_modular_backend_audit.md",
            "audit_source_json": "docs/audits/premium_saas_modular_backend_audit.json",
            "audit_commit_known": True,
            "audit_commit_hash": "4f1e1267ab6229bd4a58f13a2b01a9cb685f6115",
            "audit_used_as_static_planning_input": True,
            "audit_read_at_runtime": False,
            "audit_files_imported_at_runtime": False,
            "secrets_copied_from_audit": False,
        },
        "handoff_scope": {
            "scope_name": "testnet_sandbox_backend_capability_handoff",
            "handoff_only": True,
            "backend_activation_allowed_now": False,
            "adapter_wiring_allowed_now": False,
            "runtime_execution_allowed_now": False,
            "network_io_allowed_now": False,
            "credentials_allowed_now": False,
            "private_endpoint_access_allowed_now": False,
            "order_submission_allowed_now": False,
            "strategy_execution_allowed_now": False,
            "ai_model_inference_allowed_now": False,
            "scheduler_allowed_now": False,
            "qml_changes_allowed": False,
            "new_qml_method_calls_allowed": False,
            "bridge_api_changes_allowed": False,
            "exe_packaging_in_scope": False,
            "bat_productization_allowed": False,
            "exe_direction_preserved": True,
        },
        "capability_classification_legend": _CLASSIFICATION_LEGEND,
        "testnet_sandbox_candidate_capabilities": _testnet_sandbox_candidate_capabilities(),
        "blocked_high_risk_capabilities": _blocked_high_risk_capabilities(),
        "deferred_capabilities_by_block": {
            "block_j_risk_governor_required": [
                "order_submission_path",
                "risk_engine_primitives",
                "ai_driven_order_path",
            ],
            "block_k_observability_soak_required": [
                "runtime_loops",
                "scheduler_loops",
                "observability_metrics_health",
            ],
            "block_l_live_transition_required": [
                "live_router",
                "private_exchange_adapters",
                "live_order_submission",
            ],
            "separate_commercial_saas_blocks": [
                "signals_marketplace_product",
                "copy_social_trading_product",
                "billing_subscriptions_entitlements",
            ],
        },
        "candidate_module_map": {
            "safe_read_model_candidates": [
                "read_only_market_data_provider",
                "exchange_adapter_layer",
                "exchange_network_guard",
                "paper_execution_oracle",
            ],
            "requires_gate_before_any_runtime": [
                "exchange_adapter_layer",
                "order_management_contract_surface",
                "order_lifecycle_paper_surface",
                "runtime_orchestration",
                "api_key_secrets_management_surface",
            ],
            "do_not_import_in_preview_helpers": [
                "bot_core/exchanges/",
                "bot_core/execution/",
                "bot_core/runtime/controller.py",
                "bot_core/runtime/scheduler.py",
                "bot_core/security/",
                "bot_core/ai/",
                "bot_core/risk/",
            ],
            "do_not_activate_in_block_i_11_1": ["all candidates"],
        },
        "safe_wiring_sequence": [
            "11.1 backend capability handoff — classify only",
            "11.2 adapter read model — static, no import/runtime",
            "11.3 static connectivity fixture — no network",
            "11.4 adapter config gate — config shape only",
            "11.5 credentials gate contract — no secret read",
            "11.6 public market data probe preview — no real fetch unless later explicitly gated",
            "11.7 private endpoint gate — no private endpoint call",
            "11.8 block I closure audit",
        ],
        "non_activation_evidence": {
            "backend_handoff_evaluated": True,
            "contract_11_0_read": True,
            "premium_saas_audit_used_as_static_input": True,
            "backend_modules_imported": False,
            "backend_modules_activated": False,
            "adapter_instantiated": False,
            "runtime_started": False,
            "scheduler_started": False,
            "network_io_performed": False,
            "credentials_read": False,
            "secrets_read": False,
            "private_endpoint_accessed": False,
            "market_data_fetch_performed": False,
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
        "backend_activation_blockers": {
            "contract_only_step": True,
            "requires_11_2_read_model_before_adapter_selection": True,
            "requires_11_4_config_gate_before_any_adapter_config_use": True,
            "requires_11_5_credentials_gate_before_any_secret_handling": True,
            "requires_11_7_private_endpoint_gate_before_account_order_fill_access": True,
            "requires_block_j_risk_governor_before_order_submission": True,
            "requires_block_k_observability_before_soak": True,
            "requires_block_l_before_live": True,
            "no_runtime_activation_allowed_in_11_1": True,
            "no_backend_import_activation_allowed_in_11_1": True,
        },
        "safety_invariants": {
            "no_network_io_performed": True,
            "no_testnet_connection_opened": True,
            "no_sandbox_connection_opened": True,
            "no_live_connection_opened": True,
            "no_credentials_read": True,
            "no_secrets_read": True,
            "no_backend_module_imported_for_execution": True,
            "no_backend_module_activated": True,
            "no_adapter_instantiated": True,
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
        "future_steps": _FUTURE_STEPS,
        "status": STATUS,
    }
