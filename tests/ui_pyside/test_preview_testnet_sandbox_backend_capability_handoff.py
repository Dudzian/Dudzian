"""Tests for FUNCTIONAL-PREVIEW-11.1 Block I backend capability handoff."""

from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_testnet_sandbox_backend_capability_handoff import (
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_KIND,
    PREVIEW_TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_SCHEMA_VERSION,
    READY_FOR_BLOCK_I_2,
    STATUS,
    TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_DECISION,
    TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_STATUS,
    build_preview_testnet_sandbox_backend_capability_handoff,
)

SIMPLE_TYPES = (dict, list, str, bool, int, float, type(None))
EXPECTED_TOP_LEVEL_FIELDS = {
    "schema_version",
    "testnet_sandbox_backend_capability_handoff_kind",
    "block",
    "step",
    "testnet_sandbox_backend_capability_handoff_status",
    "testnet_sandbox_backend_capability_handoff_decision",
    "ready_for_block_i_2",
    "next_step",
    "next_step_title",
    "contract_reference",
    "audit_reference",
    "handoff_scope",
    "capability_classification_legend",
    "testnet_sandbox_candidate_capabilities",
    "blocked_high_risk_capabilities",
    "deferred_capabilities_by_block",
    "candidate_module_map",
    "safe_wiring_sequence",
    "non_activation_evidence",
    "backend_activation_blockers",
    "safety_invariants",
    "source_boundaries",
    "future_steps",
    "status",
}
EXPECTED_CLASSIFICATION_LEGEND = [
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
EXPECTED_CANDIDATE_NAMES = [
    "read_only_market_data_provider",
    "exchange_adapter_layer",
    "exchange_network_guard",
    "paper_execution_oracle",
    "order_management_contract_surface",
    "order_lifecycle_paper_surface",
    "risk_engine_primitives",
    "observability_metrics_health",
    "runtime_orchestration",
    "api_key_secrets_management_surface",
]
EXPECTED_CANDIDATE_FIELDS = {
    "capability",
    "audit_status",
    "evidence_paths",
    "main_symbols",
    "handoff_classification",
    "eligible_for_block_i_contract",
    "eligible_for_11_2_read_model",
    "eligible_for_runtime_now",
    "requires_credentials_gate",
    "requires_private_endpoint_gate",
    "requires_risk_governor",
    "requires_observability_soak",
    "requires_live_gate",
    "notes",
}
EXPECTED_HIGH_RISK_BLOCKS = {
    "live_router": "L",
    "private_exchange_adapters": "I_credentials_gate_then_J",
    "order_submission_path": "J",
    "runtime_loops": "K",
    "scheduler_loops": "K",
    "market_making_runtime": "J_K_L",
    "arbitrage_runtime": "J_K_L",
    "ai_driven_order_path": "J_K",
}
EXPECTED_SAFE_WIRING_SEQUENCE = [
    "11.1 backend capability handoff — classify only",
    "11.2 adapter read model — static, no import/runtime",
    "11.3 static connectivity fixture — no network",
    "11.4 adapter config gate — config shape only",
    "11.5 credentials gate contract — no secret read",
    "11.6 public market data probe preview — no real fetch unless later explicitly gated",
    "11.7 private endpoint gate — no private endpoint call",
    "11.8 block I closure audit",
]
EXPECTED_SOURCE_BOUNDARIES = [
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
]
EXPECTED_FUTURE_STEPS = [
    "functional_preview_11_2_testnet_sandbox_adapter_read_model",
    "functional_preview_11_3_testnet_sandbox_static_connectivity_fixture",
    "functional_preview_11_4_testnet_sandbox_adapter_config_gate",
    "functional_preview_11_5_testnet_sandbox_credentials_gate_contract",
    "functional_preview_11_6_testnet_sandbox_public_market_data_probe_preview",
    "functional_preview_11_7_testnet_sandbox_private_endpoint_gate",
    "functional_preview_11_8_block_i_closure_audit",
]
FORBIDDEN_IMPORT_ROOTS = {
    "PySide6",
    "qml",
    "runtime",
    "scheduler",
    "order",
    "live",
    "testnet",
    "sandbox",
    "exchange",
    "account",
    "secrets",
    "network",
    "requests",
    "urllib",
    "httpx",
    "aiohttp",
    "socket",
    "websocket",
    "pathlib",
    "subprocess",
}
FORBIDDEN_CALLS = {
    "open",
    "read_text",
    "write_text",
    "requests",
    "subprocess",
    "urllib",
    "httpx",
    "aiohttp",
    "socket",
    "websocket",
    "start_runtime",
    "start_loop",
    "submit_order",
    "place_order",
    "create_order",
    "send_order",
    "fill_order",
    "fetch_market_data",
    "fetch_balance",
    "fetch_account",
    "refresh_market_data",
    "export",
}
ALLOWED_IMPORTS = [
    "__future__",
    "typing",
    "ui.pyside_app.preview_testnet_sandbox_adapter_contract",
]


def _assert_simple_types_only(value: object) -> None:
    assert isinstance(value, SIMPLE_TYPES)
    if isinstance(value, dict):
        for key, nested in value.items():
            assert isinstance(key, str)
            _assert_simple_types_only(nested)
    if isinstance(value, list):
        for nested in value:
            _assert_simple_types_only(nested)


def test_handoff_is_plain_json_serializable_dict_with_exact_top_level_fields() -> None:
    handoff = build_preview_testnet_sandbox_backend_capability_handoff()

    assert isinstance(handoff, dict)
    _assert_simple_types_only(handoff)
    assert json.loads(json.dumps(handoff, sort_keys=True)) == handoff
    assert set(handoff) == EXPECTED_TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step() -> None:
    handoff = build_preview_testnet_sandbox_backend_capability_handoff()

    assert (
        handoff["schema_version"]
        == PREVIEW_TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_SCHEMA_VERSION
    )
    assert (
        handoff["testnet_sandbox_backend_capability_handoff_kind"]
        == PREVIEW_TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_KIND
    )
    assert handoff["block"] == "I"
    assert handoff["step"] == "11.1"
    assert (
        handoff["testnet_sandbox_backend_capability_handoff_status"]
        == TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_STATUS
    )
    assert (
        handoff["testnet_sandbox_backend_capability_handoff_decision"]
        == TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_DECISION
    )
    assert handoff["ready_for_block_i_2"] is READY_FOR_BLOCK_I_2 is True
    assert handoff["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-11.2"
    assert handoff["next_step_title"] == NEXT_STEP_TITLE
    assert handoff["status"] == STATUS
    assert TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_STATUS == (
        "testnet_sandbox_backend_capability_handoff_ready_no_backend_activation"
    )
    assert TESTNET_SANDBOX_BACKEND_CAPABILITY_HANDOFF_DECISION == (
        "MAP_EXISTING_BACKEND_CAPABILITIES_ONLY_NO_WIRING_NO_RUNTIME"
    )


def test_contract_reference_is_lifted_from_functional_preview_11_0() -> None:
    reference = build_preview_testnet_sandbox_backend_capability_handoff()["contract_reference"]

    assert set(reference) == {
        "schema_version",
        "testnet_sandbox_adapter_contract_kind",
        "testnet_sandbox_adapter_contract_status",
        "testnet_sandbox_adapter_contract_decision",
        "ready_for_block_i_1",
        "next_step",
        "next_step_title",
        "status",
    }
    assert reference["ready_for_block_i_1"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-11.1"
    assert reference["next_step_title"] == "TESTNET/SANDBOX BACKEND CAPABILITY HANDOFF"
    assert reference["status"] == (
        "ready_for_functional_preview_11_1_testnet_sandbox_backend_capability_handoff"
    )


def test_audit_reference_is_static_and_not_runtime_io() -> None:
    audit = build_preview_testnet_sandbox_backend_capability_handoff()["audit_reference"]

    assert audit == {
        "audit_source_markdown": "docs/audits/premium_saas_modular_backend_audit.md",
        "audit_source_json": "docs/audits/premium_saas_modular_backend_audit.json",
        "audit_commit_known": True,
        "audit_commit_hash": "4f1e1267ab6229bd4a58f13a2b01a9cb685f6115",
        "audit_used_as_static_planning_input": True,
        "audit_read_at_runtime": False,
        "audit_files_imported_at_runtime": False,
        "secrets_copied_from_audit": False,
    }


def test_handoff_scope_matches_full_non_activation_contract() -> None:
    scope = build_preview_testnet_sandbox_backend_capability_handoff()["handoff_scope"]

    assert scope["scope_name"] == "testnet_sandbox_backend_capability_handoff"
    assert scope["handoff_only"] is True
    assert scope["exe_direction_preserved"] is True
    for key, value in scope.items():
        if key in {"scope_name", "handoff_only", "exe_direction_preserved"}:
            continue
        assert value is False, key


def test_classification_legend_and_ten_candidate_capabilities() -> None:
    handoff = build_preview_testnet_sandbox_backend_capability_handoff()
    candidates = handoff["testnet_sandbox_candidate_capabilities"]

    assert handoff["capability_classification_legend"] == EXPECTED_CLASSIFICATION_LEGEND
    assert len(candidates) == 10
    assert [entry["capability"] for entry in candidates] == EXPECTED_CANDIDATE_NAMES

    for entry in candidates:
        assert set(entry) == EXPECTED_CANDIDATE_FIELDS
        assert entry["audit_status"] in EXPECTED_CLASSIFICATION_LEGEND
        assert entry["handoff_classification"] in EXPECTED_CLASSIFICATION_LEGEND
        assert entry["evidence_paths"]
        assert entry["main_symbols"]
        assert entry["eligible_for_block_i_contract"] is True
        assert entry["eligible_for_runtime_now"] is False
        assert entry["notes"]


def test_candidate_minimum_gate_requirements() -> None:
    candidates = {
        entry["capability"]: entry
        for entry in build_preview_testnet_sandbox_backend_capability_handoff()[
            "testnet_sandbox_candidate_capabilities"
        ]
    }

    market = candidates["read_only_market_data_provider"]
    assert market["audit_status"] == "ready_for_contract_gate"
    assert market["handoff_classification"] == "ready_for_contract_gate"
    assert market["eligible_for_11_2_read_model"] is True
    assert market["requires_credentials_gate"] is False
    assert market["requires_private_endpoint_gate"] is False
    assert market["requires_risk_governor"] is False

    adapter = candidates["exchange_adapter_layer"]
    assert adapter["audit_status"] == "high_risk_requires_gate"
    assert adapter["handoff_classification"] == "high_risk_requires_gate"
    assert adapter["eligible_for_11_2_read_model"] is True
    assert adapter["requires_credentials_gate"] is True
    assert adapter["requires_private_endpoint_gate"] is True
    assert adapter["requires_risk_governor"] is True

    guard = candidates["exchange_network_guard"]
    assert guard["audit_status"] == "implemented_not_wired"
    assert guard["handoff_classification"] == "implemented_not_wired"
    assert guard["eligible_for_11_2_read_model"] is True

    paper = candidates["paper_execution_oracle"]
    assert paper["audit_status"] == "implemented_not_wired"
    assert paper["handoff_classification"] == "implemented_not_wired"
    assert paper["eligible_for_11_2_read_model"] is True

    for name in EXPECTED_CANDIDATE_NAMES[4:]:
        assert candidates[name]["eligible_for_block_i_contract"] is True
        assert candidates[name]["eligible_for_runtime_now"] is False
        assert candidates[name]["eligible_for_11_2_read_model"] is False


def test_blocked_high_risk_capabilities_are_exactly_eight() -> None:
    blocked = build_preview_testnet_sandbox_backend_capability_handoff()[
        "blocked_high_risk_capabilities"
    ]

    assert len(blocked) == 8
    assert [entry["capability"] for entry in blocked] == list(EXPECTED_HIGH_RISK_BLOCKS)
    for entry in blocked:
        assert set(entry) == {"capability", "blocked_now", "reason", "earliest_block"}
        assert entry["blocked_now"] is True
        assert entry["reason"]
        assert entry["earliest_block"] == EXPECTED_HIGH_RISK_BLOCKS[entry["capability"]]


def test_deferred_capabilities_by_block_have_required_sections_and_non_empty_lists() -> None:
    deferred = build_preview_testnet_sandbox_backend_capability_handoff()[
        "deferred_capabilities_by_block"
    ]

    assert set(deferred) == {
        "block_j_risk_governor_required",
        "block_k_observability_soak_required",
        "block_l_live_transition_required",
        "separate_commercial_saas_blocks",
    }
    assert all(isinstance(values, list) and values for values in deferred.values())


def test_candidate_module_map_matches_allowed_read_model_and_forbidden_imports() -> None:
    module_map = build_preview_testnet_sandbox_backend_capability_handoff()["candidate_module_map"]

    assert module_map["safe_read_model_candidates"] == [
        "read_only_market_data_provider",
        "exchange_adapter_layer",
        "exchange_network_guard",
        "paper_execution_oracle",
    ]
    for required in [
        "exchange_adapter_layer",
        "order_management_contract_surface",
        "order_lifecycle_paper_surface",
        "runtime_orchestration",
        "api_key_secrets_management_surface",
    ]:
        assert required in module_map["requires_gate_before_any_runtime"]
    for forbidden in [
        "bot_core/exchanges/",
        "bot_core/execution/",
        "bot_core/runtime/controller.py",
        "bot_core/runtime/scheduler.py",
        "bot_core/security/",
        "bot_core/ai/",
        "bot_core/risk/",
    ]:
        assert forbidden in module_map["do_not_import_in_preview_helpers"]
    assert module_map["do_not_activate_in_block_i_11_1"] == ["all candidates"]


def test_safe_wiring_sequence_non_activation_and_backend_blockers() -> None:
    handoff = build_preview_testnet_sandbox_backend_capability_handoff()

    assert handoff["safe_wiring_sequence"] == EXPECTED_SAFE_WIRING_SEQUENCE
    evidence = handoff["non_activation_evidence"]
    assert evidence["backend_handoff_evaluated"] is True
    assert evidence["contract_11_0_read"] is True
    assert evidence["premium_saas_audit_used_as_static_input"] is True
    for key, value in evidence.items():
        if key in {
            "backend_handoff_evaluated",
            "contract_11_0_read",
            "premium_saas_audit_used_as_static_input",
        }:
            continue
        assert value is False, key
    assert all(value is True for value in handoff["backend_activation_blockers"].values())


def test_safety_invariants_source_boundaries_and_future_steps_are_complete() -> None:
    handoff = build_preview_testnet_sandbox_backend_capability_handoff()

    assert all(value is True for value in handoff["safety_invariants"].values())
    assert set(handoff["safety_invariants"]) == {
        "no_network_io_performed",
        "no_testnet_connection_opened",
        "no_sandbox_connection_opened",
        "no_live_connection_opened",
        "no_credentials_read",
        "no_secrets_read",
        "no_backend_module_imported_for_execution",
        "no_backend_module_activated",
        "no_adapter_instantiated",
        "no_account_fetch_performed",
        "no_balance_fetch_performed",
        "no_positions_fetch_performed",
        "no_orders_fetch_performed",
        "no_fills_fetch_performed",
        "no_order_submitted",
        "no_order_generated",
        "no_fill_simulated",
        "no_runtime_loop_started",
        "no_scheduler_started",
        "no_qml_changes_performed",
        "no_bridge_api_changes_performed",
        "no_app_py_changes_performed",
        "no_bat_changes_performed",
        "no_workflow_changes_performed",
        "no_dependency_changes_performed",
        "exe_direction_preserved",
    }
    assert handoff["source_boundaries"] == EXPECTED_SOURCE_BOUNDARIES
    assert handoff["future_steps"] == EXPECTED_FUTURE_STEPS


def test_source_imports_only_allowed_modules_and_has_no_runtime_calls() -> None:
    source_path = Path("ui/pyside_app/preview_testnet_sandbox_backend_capability_handoff.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    imports: list[str] = []
    calls: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imports.append(module)
            if module == "typing":
                assert {alias.name for alias in node.names} <= {"Any", "Final"}
            elif module == "ui.pyside_app.preview_testnet_sandbox_adapter_contract":
                assert {alias.name for alias in node.names} == {
                    "build_preview_testnet_sandbox_adapter_contract"
                }
            else:
                assert module == "__future__"
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                calls.append(func.attr)

    assert imports == ALLOWED_IMPORTS
    assert not ({item.split(".")[0] for item in imports} & FORBIDDEN_IMPORT_ROOTS)
    assert not (set(calls) & FORBIDDEN_CALLS)
