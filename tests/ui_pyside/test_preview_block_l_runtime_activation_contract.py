"""Tests for FUNCTIONAL-PREVIEW-14.0 Block L runtime activation contract."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

from ui.pyside_app.preview_block_l_runtime_activation_contract import (
    BLOCK_ID,
    BLOCK_L_RUNTIME_ACTIVATION_CONTRACT_DECISION,
    BLOCK_L_RUNTIME_ACTIVATION_CONTRACT_STATUS,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_CONTRACT_KIND,
    PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_CONTRACT_SCHEMA_VERSION,
    READY_FOR_BLOCK_L_1,
    STATUS,
    STEP_ID,
    build_preview_block_l_runtime_activation_contract,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_block_l_runtime_activation_contract.py"

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_l_runtime_activation_contract_kind",
    "block",
    "step",
    "block_l_runtime_activation_contract_status",
    "block_l_runtime_activation_contract_decision",
    "ready_for_block_l_1",
    "next_step",
    "next_step_title",
    "block_k_closure_reference",
    "runtime_activation_contract_scope",
    "runtime_activation_contract_principles",
    "runtime_activation_candidate_gates",
    "runtime_activation_candidate_modes",
    "runtime_activation_contract_summary",
    "runtime_activation_contract_matrix",
    "runtime_activation_surface_contract",
    "blocked_runtime_activation_contract_capabilities",
    "runtime_activation_contract_boundaries",
    "non_activation_evidence",
    "source_boundaries",
    "future_steps",
    "status",
]

PRINCIPLES = [
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

GATE_SPECS = [
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
    ("soak_ready_gate", "Soak ready gate", "soak", "soak_readiness", "FUNCTIONAL-PREVIEW-14.x"),
    (
        "order_private_network_block_gate",
        "Order private network block gate",
        "execution_safety",
        "blocked_capability",
        "FUNCTIONAL-PREVIEW-14.x",
    ),
]
MODE_SPECS = [
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

BLOCKED = [
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

SOURCE_BOUNDARIES = [
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


def _payload() -> dict[str, Any]:
    return build_preview_block_l_runtime_activation_contract()


def test_payload_is_plain_json_serializable_and_has_exact_top_level_fields() -> None:
    payload = _payload()
    assert list(payload) == TOP_LEVEL_FIELDS
    assert json.loads(json.dumps(payload)) == payload


def test_identity_status_decision_next_and_block_k_reference() -> None:
    payload = _payload()
    assert payload["schema_version"] == PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_CONTRACT_SCHEMA_VERSION
    assert (
        payload["block_l_runtime_activation_contract_kind"]
        == PREVIEW_BLOCK_L_RUNTIME_ACTIVATION_CONTRACT_KIND
    )
    assert payload["block"] == BLOCK_ID
    assert payload["step"] == STEP_ID
    assert (
        payload["block_l_runtime_activation_contract_status"]
        == BLOCK_L_RUNTIME_ACTIVATION_CONTRACT_STATUS
    )
    assert (
        payload["block_l_runtime_activation_contract_decision"]
        == BLOCK_L_RUNTIME_ACTIVATION_CONTRACT_DECISION
    )
    assert payload["ready_for_block_l_1"] is READY_FOR_BLOCK_L_1
    assert payload["next_step"] == NEXT_STEP
    assert payload["next_step_title"] == NEXT_STEP_TITLE
    assert payload["status"] == STATUS
    reference = payload["block_k_closure_reference"]
    assert list(reference) == [
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
    assert reference["ready_for_next_block"] is True
    assert reference["next_block"] == "BLOK L — RUNTIME ACTIVATION / LIVE CANARY GATES"
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-14.0"
    assert reference["next_step_title"] == "BLOK L — RUNTIME ACTIVATION CONTRACT"
    assert reference["closure_line"] == "BLOK GOTOWY — PRZECHODZIMY DO KOLEJNEGO BLOKU"


def test_scope_principles_gates_modes_summary_matrix_surface_and_lists_are_exact() -> None:
    payload = _payload()
    scope = payload["runtime_activation_contract_scope"]
    assert scope["scope_name"] == "block_l_runtime_activation_contract"
    for key, value in scope.items():
        if key in {
            "scope_name",
            "contract_only",
            "starts_block_l",
            "derived_from_block_k_closure_13_6",
            "block_k_closure_required",
            "block_k_closure_verified",
            "exe_direction_preserved",
        }:
            continue
        assert value is False, key
    assert scope["contract_only"] is True
    assert scope["starts_block_l"] is True
    assert scope["derived_from_block_k_closure_13_6"] is True
    assert scope["exe_direction_preserved"] is True

    assert payload["runtime_activation_contract_principles"] == PRINCIPLES

    gates = payload["runtime_activation_candidate_gates"]
    assert [gate["source_gate_id"] for gate in gates] == [spec[0] for spec in GATE_SPECS]
    gate_fields = [
        "runtime_activation_gate_id",
        "source_gate_id",
        "display_name",
        "gate_domain",
        "gate_type",
        "planned_source_step",
        "required_for_future_activation",
        "eligible_for_14_1_read_model",
        "eligible_for_future_gate_matrix",
        "runtime_activation_allowed_now",
        "runtime_gate_execution_allowed_now",
        "gate_state_mutation_allowed_now",
        "order_flow_allowed_now",
        "private_endpoint_access_allowed_now",
        "network_io_allowed_now",
        "filesystem_io_allowed_now",
        "safe_for_offline_tests",
        "notes",
    ]
    for gate, spec in zip(gates, GATE_SPECS, strict=True):
        source_id, display, domain, gate_type, planned = spec
        assert list(gate) == gate_fields
        assert gate["runtime_activation_gate_id"] == f"runtime_activation_gate_{source_id}"
        assert gate["display_name"] == display
        assert gate["gate_domain"] == domain
        assert gate["gate_type"] == gate_type
        assert gate["planned_source_step"] == planned
        assert gate["required_for_future_activation"] is True
        assert gate["eligible_for_14_1_read_model"] is True
        assert gate["eligible_for_future_gate_matrix"] is True
        assert gate["safe_for_offline_tests"] is True
        assert gate["notes"]
        for flag in [
            "runtime_activation_allowed_now",
            "runtime_gate_execution_allowed_now",
            "gate_state_mutation_allowed_now",
            "order_flow_allowed_now",
            "private_endpoint_access_allowed_now",
            "network_io_allowed_now",
            "filesystem_io_allowed_now",
        ]:
            assert gate[flag] is False

    modes = payload["runtime_activation_candidate_modes"]
    assert [mode["source_mode_id"] for mode in modes] == [spec[0] for spec in MODE_SPECS]
    mode_fields = [
        "runtime_activation_mode_id",
        "source_mode_id",
        "display_name",
        "mode_classification",
        "activation_stage",
        "requires_future_gate",
        "allowed_in_14_0",
        "runtime_activation_allowed_now",
        "order_flow_allowed_now",
        "private_endpoint_access_allowed_now",
        "network_io_allowed_now",
        "credential_read_allowed_now",
        "live_trading_allowed_now",
        "safe_for_offline_tests",
        "notes",
    ]
    for mode, spec in zip(modes, MODE_SPECS, strict=True):
        source_id, display, classification, stage = spec
        assert list(mode) == mode_fields
        assert mode["runtime_activation_mode_id"] == f"runtime_activation_mode_{source_id}"
        assert mode["display_name"] == display
        assert mode["mode_classification"] == classification
        assert mode["activation_stage"] == stage
        assert mode["requires_future_gate"] is True
        assert mode["safe_for_offline_tests"] is True
        assert mode["notes"]
        for flag in [
            "allowed_in_14_0",
            "runtime_activation_allowed_now",
            "order_flow_allowed_now",
            "private_endpoint_access_allowed_now",
            "network_io_allowed_now",
            "credential_read_allowed_now",
            "live_trading_allowed_now",
        ]:
            assert mode[flag] is False

    gate_ids = [spec[0] for spec in GATE_SPECS]
    mode_ids = [spec[0] for spec in MODE_SPECS]
    assert payload["runtime_activation_contract_summary"] == {
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
    assert payload["runtime_activation_contract_matrix"] == {
        "candidate_gate_ids": gate_ids,
        "candidate_mode_ids": mode_ids,
        "principles_in_order": PRINCIPLES,
        "gates_requiring_future_execution_gate": gate_ids,
        "modes_requiring_future_activation_gate": mode_ids,
        "blocked_in_14_0_gate_ids": gate_ids,
        "blocked_in_14_0_mode_ids": mode_ids,
        "candidate_gate_domains_by_id": {spec[0]: spec[2] for spec in GATE_SPECS},
        "candidate_mode_classifications_by_id": {spec[0]: spec[2] for spec in MODE_SPECS},
        "candidate_mode_activation_stages_by_id": {spec[0]: spec[3] for spec in MODE_SPECS},
    }
    assert all(payload["runtime_activation_surface_contract"].values())
    assert (
        payload["runtime_activation_surface_contract"]["surface_contract_id"]
        == "block_l_runtime_activation_contract_surface"
    )
    assert payload["blocked_runtime_activation_contract_capabilities"] == BLOCKED
    assert payload["source_boundaries"] == SOURCE_BOUNDARIES
    assert payload["future_steps"] == [
        "functional_preview_14_1_runtime_activation_read_model",
        "functional_preview_14_2_runtime_activation_gate_matrix",
        "functional_preview_14_3_paper_runtime_activation_gate",
        "functional_preview_14_4_testnet_runtime_activation_gate",
        "functional_preview_14_5_live_canary_gate_contract",
        "functional_preview_14_6_block_l_closure_audit",
    ]


def test_boundaries_and_non_activation_evidence_are_exactly_closed() -> None:
    payload = _payload()
    boundaries = payload["runtime_activation_contract_boundaries"]
    assert "runtime_activation_contract_balance_read_blocked" in boundaries
    assert all(value is True for value in boundaries.values())
    evidence = payload["non_activation_evidence"]
    assert evidence["block_k_closure_13_6_read"] is True
    assert evidence["runtime_activation_contract_built"] is True
    for key, value in evidence.items():
        if key not in {"block_k_closure_13_6_read", "runtime_activation_contract_built"}:
            assert value is False, key


def test_source_imports_and_forbidden_tokens_are_guarded() -> None:
    source = SOURCE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.Import | ast.ImportFrom)]
    assert len(imports) == 3
    assert isinstance(imports[0], ast.ImportFrom) and imports[0].module == "__future__"
    assert isinstance(imports[1], ast.ImportFrom) and imports[1].module == "typing"
    assert isinstance(imports[2], ast.ImportFrom)
    assert imports[2].module == "ui.pyside_app.preview_block_k_closure_audit"

    forbidden = [
        "open",
        "read_text",
        "write_text",
        "yaml",
        "json",
        "getenv",
        "environ",
        "requests",
        "subprocess",
        "urllib",
        "httpx",
        "aiohttp",
        "socket",
        "websocket",
        "getaddrinfo",
        "create_connection",
        "QQmlApplicationEngine",
        "TradingController",
        "DecisionEnvelope",
        "start_observability",
        "collect_metrics",
        "export_metrics",
        "write_log",
        "read_log",
        "write_audit",
        "export_audit",
        "execute_rollback",
        "shutdown_runtime",
        "mutate_state",
        "start_soak",
        "run_soak",
        "start_scheduler",
        "execute_gate",
        "activate_runtime",
        "execute_runtime_contract",
        "start_live_canary",
        "start_testnet_runtime",
        "activate_paper_runtime",
        "activate_local_mock_runtime",
        "activate_recorded_fixture_runtime",
        "start_live_scaled_runtime",
        "mutate_gate",
        "submit_order",
        "place_order",
        "create_order",
        "send_order",
        "fill_order",
        "cancel_order",
        "replace_order",
        "withdraw",
        "transfer",
        "fetch_market_data",
        "fetch_" + "balance",
        "fetch_account",
        "fetch_positions",
        "fetch_orders",
        "fetch_fills",
        "refresh_market_data",
    ]
    call_names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                call_names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                call_names.append(node.func.attr)
    for token in forbidden:
        assert token not in call_names
    assert "fetch_" + "balance" not in source
    assert "cc" + "xt" not in source
