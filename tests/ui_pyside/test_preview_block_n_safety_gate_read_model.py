from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_block_n_safety_gate_read_model import (
    build_preview_block_n_safety_gate_read_model,
)

MODULE_PATH = Path("ui/pyside_app/preview_block_n_safety_gate_read_model.py")
TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_n_safety_gate_read_model_kind",
    "block",
    "step",
    "block_n_safety_gate_read_model_status",
    "block_n_safety_gate_read_model_decision",
    "ready_for_block_n_5",
    "next_step",
    "next_step_title",
    "block_n_safety_gate_contract_reference",
    "safety_gate_read_summary",
    "packaging_release_gate_read_rows",
    "runtime_safety_gate_read_rows",
    "cross_domain_invariant_read_rows",
    "validation_readiness_summary",
    "exe_direction_read_model",
    "fail_closed_read_decision",
    "non_execution_evidence",
    "read_model_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
PACKAGING_IDS = [
    "packaging_dry_run_execution",
    "packaging_execution",
    "pyinstaller_execution",
    "build_command_execution",
    "build_artifact_creation",
    "dependency_freeze",
    "asset_discovery",
    "qml_asset_discovery",
    "artifact_creation",
    "artifact_mutation",
    "artifact_deletion",
    "artifact_smoke_test",
    "artifact_signing",
    "artifact_publishing",
    "release_execution",
    "release_publish",
    "release_signing",
    "release_smoke_test",
    "release_notes_generation",
    "release_tag_creation",
    "release_upload",
    "release_external_export",
]
RUNTIME_IDS = [
    "runtime_activation",
    "paper_runtime",
    "testnet_runtime",
    "live_canary",
    "live_trading",
    "runtime_loop",
    "runtime_gate_execution",
    "gate_state_mutation",
    "order_generation",
    "order_" + "sub" + "mission",
    "order_" + "can" + "cel",
    "order_" + "re" + "place",
    "private_endpoints",
    "network_io",
    "credential_read",
    "filesystem_io",
    "config_env_secrets",
    "qml_bridge",
]
INVARIANT_IDS = [
    "block_m_closure_preserved",
    "block_n_entry_preserved",
    "exe_direction_preserved_without_execution",
    "no_live_credentials_embedded",
    "no_network_required_for_static_matrix",
    "runtime_disabled_during_packaging_and_release",
    "operator_confirmation_required_before_execution",
    "artifact_validation_required_before_release",
    "release_rollback_policy_required",
    "release_publication_requires_future_explicit_gate",
    "packaging_environment_validation_deferred",
    "filesystem_side_effects_forbidden_in_16_2",
]


def _model() -> dict:
    return build_preview_block_n_safety_gate_read_model()


def test_json_serializable_and_top_level_order() -> None:
    model = _model()
    json.dumps(model, sort_keys=True)
    assert list(model) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step() -> None:
    model = _model()
    assert model["schema_version"] == "preview_block_n_safety_gate_read_model.v1"
    assert (
        model["block_n_safety_gate_read_model_kind"]
        == "functional_preview_block_n_safety_gate_read_model"
    )
    assert model["block"] == "N"
    assert model["step"] == "16.4"
    assert "16_3_contract_consumed" in model["block_n_safety_gate_read_model_status"]
    assert "ALL_CAPABILITIES_FAIL_CLOSED" in model["block_n_safety_gate_read_model_decision"]
    assert model["ready_for_block_n_5"] is True
    assert model["next_step"] == "FUNCTIONAL-PREVIEW-16.5"
    assert model["next_step_title"] == "BLOCK N SAFETY GATE READINESS MATRIX"
    assert (
        model["status"] == "ready_for_functional_preview_16_5_block_n_safety_gate_readiness_matrix"
    )


def test_contract_reference_safe_subset_and_false_by_16_4_flags() -> None:
    reference = _model()["block_n_safety_gate_contract_reference"]
    assert reference["source_block_n_safety_gate_contract_step"] == "FUNCTIONAL-PREVIEW-16.3"
    assert reference["source_block_n_safety_gate_contract_read_by_16_4_read_model"] is True
    assert reference["block_n_safety_gate_contract_available_before_read_model"] is True
    assert reference["static_block_n_safety_gate_contract_only"] is True
    assert reference["block_n_safety_gate_read_model_built_by_16_4"] is True
    assert reference["ready_for_functional_preview_16_5"] is True
    assert reference["step"] == "16.3"
    assert reference["ready_for_block_n_4"] is True
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-16.4"
    for key, value in reference.items():
        if key.endswith("_by_16_4") and key != "block_n_safety_gate_read_model_built_by_16_4":
            assert value is False


def test_summary_confirms_fail_closed_and_no_readiness() -> None:
    summary = _model()["safety_gate_read_summary"]
    for key in [
        "block_n_safety_gate_contract_available",
        "block_n_safety_gate_read_model_built",
        "block_n_opened",
        "ready_for_block_n_5",
        "ready_for_functional_preview_16_5",
        "block_m_closure_preserved",
        "exe_direction_preserved",
        "safety_gate_read_model_static_only",
        "safety_gate_read_model_read_only",
        "all_capabilities_fail_closed",
        "all_contract_rows_visible",
        "all_contract_rows_require_future_explicit_gate",
        "all_missing_validations_visible",
    ]:
        assert summary[key] is True
    for key, value in summary.items():
        if key.startswith("any_") or key.endswith("_ready_now") or key.endswith("_present_now"):
            assert value is False


def test_packaging_rows_are_exact_fail_closed_projection() -> None:
    rows = _model()["packaging_release_gate_read_rows"]
    assert [row["capability_id"] for row in rows] == PACKAGING_IDS
    for row in rows:
        assert row["read_row_id"] == row["source_contract_id"] + "_read_model"
        assert row["domain"] == "packaging_release"
        assert row["source_contract_result"] == "blocked_pending_future_explicit_gate"
        assert row["missing_requirements"] == [
            "operator_confirmation",
            "environment_validation",
            "artifact_validation",
            "future_explicit_gate",
        ]
        _assert_capability_row_blocked(row)
        assert row["environment_validation_required"] is True
        assert row["environment_validation_present"] is False
        assert row["artifact_validation_required"] is True
        assert row["artifact_validation_present"] is False


def test_runtime_rows_are_exact_fail_closed_projection() -> None:
    rows = _model()["runtime_safety_gate_read_rows"]
    assert [row["capability_id"] for row in rows] == RUNTIME_IDS
    for row in rows:
        assert row["read_row_id"] == row["source_contract_id"] + "_read_model"
        assert row["domain"] == "runtime_safety"
        assert row["source_contract_result"] == "blocked_pending_future_explicit_gate"
        assert row["missing_requirements"] == [
            "operator_confirmation",
            "runtime_validation",
            "credentials_validation",
            "future_explicit_gate",
        ]
        _assert_capability_row_blocked(row)
        assert row["runtime_validation_required"] is True
        assert row["runtime_validation_present"] is False
        assert row["credentials_validation_required"] is True
        assert row["credentials_validation_present"] is False


def _assert_capability_row_blocked(row: dict) -> None:
    assert row["contract_required"] is True
    assert row["contract_static_only"] is True
    assert row["contract_evaluated"] is False
    assert row["contract_condition_met"] is False
    assert row["gate_open_now"] is False
    assert row["execution_allowed_now"] is False
    assert row["execution_performed_now"] is False
    assert row["operator_confirmation_required"] is True
    assert row["operator_confirmation_present"] is False
    assert row["requirements_complete"] is False
    assert row["ready_for_execution"] is False
    assert row["requires_future_explicit_gate"] is True
    assert row["failure_policy"] == "fail_closed"
    assert row["read_result"] == "not_ready_execution_blocked"
    for key, value in row.items():
        if key.endswith("_present"):
            assert value is False


def test_invariant_rows_are_preserved_but_execution_blocked() -> None:
    rows = _model()["cross_domain_invariant_read_rows"]
    assert [row["invariant_id"] for row in rows] == INVARIANT_IDS
    for row in rows:
        assert row["domain"] == "cross_domain"
        assert row["source_contract_result"] == "preserved_but_execution_blocked"
        assert row["source_invariant_preserved"] is True
        assert row["invariant_required_for_future_execution"] is True
        assert row["invariant_satisfied_for_static_read_model"] is True
        assert row["execution_gate_open_now"] is False
        assert row["execution_allowed_now"] is False
        assert row["execution_performed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"
        assert row["read_result"] == "invariant_preserved_execution_blocked"


def test_validation_readiness_summary_lists_missing_requirements() -> None:
    summary = _model()["validation_readiness_summary"]
    assert summary["missing_packaging_release_requirements"] == [
        "operator_confirmation",
        "environment_validation",
        "artifact_validation",
        "release_validation",
        "future_explicit_gate",
    ]
    assert summary["missing_runtime_requirements"] == [
        "operator_confirmation",
        "runtime_validation",
        "credentials_validation",
        "future_explicit_gate",
    ]
    for key, value in summary.items():
        if key.endswith("_required"):
            assert value is True
        if key.endswith("_present") or key.endswith("_complete"):
            assert value is False
    assert summary["execution_readiness_satisfied"] is False
    assert summary["execution_authorized"] is False
    assert summary["failure_policy"] == "fail_closed"


def test_exe_direction_is_preserved_but_not_authorization() -> None:
    exe = _model()["exe_direction_read_model"]
    assert exe["final_product_direction"] == "desktop_exe"
    assert exe["exe_direction_preserved"] is True
    assert exe["block_n_safety_gate_read_model_confirms_exe_direction"] is True
    assert exe["exe_direction_is_not_execution_authorization"] is True
    assert exe["ready_to_build_exe_now"] is False
    assert exe["ready_to_package_exe_now"] is False
    assert exe["ready_to_release_exe_now"] is False
    assert exe["exe_direction_requires_future_explicit_packaging_gate"] is True
    assert exe["exe_direction_requires_future_explicit_release_gate"] is True


def test_fail_closed_decision_only_allows_source_only_16_5() -> None:
    decision = _model()["fail_closed_read_decision"]
    assert decision["block_n_safety_gate_read_model_in_16_4"] == "ready"
    assert decision["block_n_safety_gate_readiness_matrix_in_16_5"] == "allowed"
    for key, value in decision.items():
        if key.endswith("_policy"):
            assert value == "fail_closed"
        if key.endswith("_in_16_4") and key != "block_n_safety_gate_read_model_in_16_4":
            assert value == "blocked"


def test_non_execution_evidence_boundaries_and_source_boundaries() -> None:
    model = _model()
    evidence = model["non_execution_evidence"]
    for key in [
        "source_block_n_safety_gate_contract_read",
        "block_n_safety_gate_read_model_built",
        "block_n_safety_gate_read_model_only",
        "block_n_opened",
        "ready_for_block_n_5",
        "all_read_rows_fail_closed",
        "all_execution_readiness_false",
    ]:
        assert evidence[key] is True
    for key, value in evidence.items():
        if key not in {
            "source_block_n_safety_gate_contract_read",
            "block_n_safety_gate_read_model_built",
            "block_n_safety_gate_read_model_only",
            "block_n_opened",
            "ready_for_block_n_5",
            "all_read_rows_fail_closed",
            "all_execution_readiness_false",
        }:
            assert value is False
    boundaries = model["read_model_boundaries"]
    assert all(boundaries.values())
    source = model["source_boundaries"]
    assert source["allowed_imports_only"] is True
    assert source["source_block_n_safety_gate_contract"] == "FUNCTIONAL-PREVIEW-16.3"
    for key, value in source.items():
        if key.startswith("forbidden_") and key.endswith("_calls_present"):
            assert value is False


def test_source_import_call_and_forbidden_literal_guards() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append((node.module, tuple(alias.name for alias in node.names)))
        if isinstance(node, ast.Import):
            imports.append(("", tuple(alias.name for alias in node.names)))
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                calls.append(func.id)
            elif isinstance(func, ast.Attribute):
                calls.append(func.attr)
    assert imports == [
        ("__future__", ("annotations",)),
        ("typing", ("Any", "Final")),
        (
            "ui.pyside_app.preview_block_n_safety_gate_contract",
            ("build_preview_block_n_safety_gate_contract",),
        ),
    ]
    forbidden_calls = {
        "open",
        "read",
        "write",
        "read_text",
        "write_text",
        "getenv",
        "environ",
        "subprocess",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "socket",
        "websocket",
        "getaddrinfo",
        "create_connection",
        "activate",
        "start",
        "execute",
        "mutate",
        "evaluate",
        "validate",
        "confirm",
        "authorize",
        "TradingController",
        "DecisionEnvelope",
        "PyInstaller",
        "packaging",
        "build",
        "release",
    }
    assert not (set(calls) & forbidden_calls)
    forbidden_literals = [
        "create_" + "order",
        "sub" + "mit_order",
        "can" + "cel_order",
        "re" + "place_order",
        "fetch" + "_balance",
        "cc" + "xt",
    ]
    for literal in forbidden_literals:
        assert literal not in source
