from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_block_n_safety_gate_contract import (
    build_preview_block_n_safety_gate_contract,
)

TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_n_safety_gate_contract_kind",
    "block",
    "step",
    "block_n_safety_gate_contract_status",
    "block_n_safety_gate_contract_decision",
    "ready_for_block_n_4",
    "next_step",
    "next_step_title",
    "block_n_safety_gate_matrix_reference",
    "safety_gate_contract_summary",
    "packaging_release_gate_contract_rows",
    "runtime_safety_gate_contract_rows",
    "cross_domain_invariant_contract_rows",
    "exe_direction_gate_contract",
    "fail_closed_contract_decision",
    "non_execution_evidence",
    "contract_boundaries",
    "source_boundaries",
    "future_steps",
    "status",
]
PACKAGING_CAPABILITIES = [
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
RUNTIME_CAPABILITIES = [
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
INVARIANTS = [
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
HELPER = Path("ui/pyside_app/preview_block_n_safety_gate_contract.py")


def payload() -> dict[str, object]:
    return build_preview_block_n_safety_gate_contract()


def all_capability_rows(data: dict[str, object]) -> list[dict[str, object]]:
    return [
        *data["packaging_release_gate_contract_rows"],
        *data["runtime_safety_gate_contract_rows"],
    ]


def test_payload_is_json_serializable_and_top_level_order_is_stable() -> None:
    data = payload()
    json.dumps(data, sort_keys=True)
    assert list(data) == TOP_LEVEL_FIELDS


def test_identity_status_decision_and_next_step() -> None:
    data = payload()
    assert data["schema_version"] == "preview_block_n_safety_gate_contract.v1"
    assert (
        data["block_n_safety_gate_contract_kind"]
        == "functional_preview_block_n_safety_gate_contract"
    )
    assert data["block"] == "N"
    assert data["step"] == "16.3"
    assert data["ready_for_block_n_4"] is True
    assert data["next_step"] == "FUNCTIONAL-PREVIEW-16.4"
    assert data["next_step_title"] == "BLOCK N SAFETY GATE READ MODEL"
    assert "16_2_matrix_consumed" in data["block_n_safety_gate_contract_status"]
    assert "NO_GATE_EVALUATION" in data["block_n_safety_gate_contract_decision"]
    assert data["status"] == "ready_for_functional_preview_16_4_block_n_safety_gate_read_model"


def test_matrix_reference_is_safe_subset_with_non_execution_fields() -> None:
    reference = payload()["block_n_safety_gate_matrix_reference"]
    expected_prefix = [
        "schema_version",
        "block_n_safety_gate_matrix_kind",
        "block",
        "step",
        "block_n_safety_gate_matrix_status",
        "block_n_safety_gate_matrix_decision",
        "ready_for_block_n_3",
        "next_step",
        "next_step_title",
        "source_block_n_safety_gate_matrix_step",
        "source_block_n_safety_gate_matrix_read_by_16_3_contract",
        "block_n_safety_gate_matrix_available_before_contract",
        "static_block_n_safety_gate_matrix_only",
        "block_n_safety_gate_contract_built_by_16_3",
        "ready_for_functional_preview_16_4",
    ]
    assert list(reference)[: len(expected_prefix)] == expected_prefix
    assert reference["step"] == "16.2"
    assert reference["source_block_n_safety_gate_matrix_step"] == "FUNCTIONAL-PREVIEW-16.2"
    assert reference["source_block_n_safety_gate_matrix_read_by_16_3_contract"] is True
    for key, value in reference.items():
        if key.endswith("_by_16_3") and key != "block_n_safety_gate_contract_built_by_16_3":
            assert value is False


def test_summary_confirms_fail_closed_and_no_current_gate_state() -> None:
    summary = payload()["safety_gate_contract_summary"]
    for key in [
        "block_n_safety_gate_matrix_available",
        "block_n_safety_gate_contract_built",
        "block_n_opened",
        "ready_for_block_n_4",
        "ready_for_functional_preview_16_4",
        "block_m_closure_preserved",
        "exe_direction_preserved",
        "safety_gate_contract_static_only",
        "safety_gate_contract_read_only",
        "all_capabilities_fail_closed",
        "all_contract_rows_require_future_explicit_gate",
        "missing_evidence_blocks_execution",
        "missing_operator_confirmation_blocks_execution",
        "missing_environment_validation_blocks_execution",
        "missing_runtime_validation_blocks_execution",
    ]:
        assert summary[key] is True
    for key in [
        "any_gate_evaluated_now",
        "any_gate_condition_met_now",
        "any_gate_open_now",
        "any_execution_allowed_now",
        "any_execution_performed_now",
        "any_gate_state_mutated_now",
        "operator_confirmation_present_now",
        "environment_validation_present_now",
        "artifact_validation_present_now",
        "release_validation_present_now",
        "runtime_validation_present_now",
        "packaging_execution_allowed_now",
        "release_execution_allowed_now",
        "artifact_work_allowed_now",
        "runtime_activation_allowed_now",
        "order_activity_allowed_now",
        "private_endpoint_access_allowed_now",
        "network_io_allowed_now",
        "credential_read_allowed_now",
        "filesystem_io_allowed_now",
        "qml_bridge_change_allowed_now",
    ]:
        assert summary[key] is False


def test_packaging_release_rows_are_exact_and_blocked_with_required_validations() -> None:
    rows = payload()["packaging_release_gate_contract_rows"]
    assert [row["capability_id"] for row in rows] == PACKAGING_CAPABILITIES
    for row in rows:
        assert row["contract_id"] == row["source_gate_id"] + "_contract"
        assert row["domain"] == "packaging_release"
        assert row["source_gate_result"] == "blocked"
        assert row["source_gate_open_now"] is False
        assert row["source_execution_allowed_now"] is False
        assert row["source_execution_performed_now"] is False
        assert row["operator_confirmation_required"] is True
        assert row["operator_confirmation_present"] is False
        assert row["environment_validation_required"] is True
        assert row["environment_validation_present"] is False
        assert row["artifact_validation_required"] is True
        assert row["artifact_validation_present"] is False


def test_runtime_rows_are_exact_and_blocked_with_required_validations() -> None:
    rows = payload()["runtime_safety_gate_contract_rows"]
    assert [row["capability_id"] for row in rows] == RUNTIME_CAPABILITIES
    for row in rows:
        assert row["contract_id"] == row["source_gate_id"] + "_contract"
        assert row["domain"] == "runtime_safety"
        assert row["source_gate_result"] == "blocked"
        assert row["source_gate_open_now"] is False
        assert row["source_execution_allowed_now"] is False
        assert row["source_execution_performed_now"] is False
        assert row["operator_confirmation_required"] is True
        assert row["operator_confirmation_present"] is False
        assert row["runtime_validation_required"] is True
        assert row["runtime_validation_present"] is False
        assert row["credentials_validation_required"] is True
        assert row["credentials_validation_present"] is False


def test_invariants_are_exact_preserved_and_execution_blocked() -> None:
    rows = payload()["cross_domain_invariant_contract_rows"]
    assert [row["source_invariant_id"] for row in rows] == INVARIANTS
    for row in rows:
        assert row["contract_id"] == "block_n_" + row["source_invariant_id"] + "_contract"
        assert row["domain"] == "cross_domain"
        assert row["source_invariant_preserved"] is True
        assert row["invariant_satisfied_for_static_contract"] is True
        assert row["execution_gate_open_now"] is False
        assert row["execution_allowed_now"] is False
        assert row["execution_performed_now"] is False
        assert row["failure_policy"] == "fail_closed"
        assert row["contract_result"] == "preserved_but_execution_blocked"


def test_exe_direction_is_not_execution_authorization() -> None:
    contract = payload()["exe_direction_gate_contract"]
    assert contract["final_product_direction"] == "desktop_exe"
    assert contract["exe_direction_preserved"] is True
    assert contract["block_n_safety_gate_contract_confirms_exe_direction"] is True
    assert contract["exe_direction_is_not_execution_authorization"] is True
    assert contract["exe_direction_requires_future_explicit_packaging_gate"] is True
    assert contract["exe_direction_requires_future_explicit_release_gate"] is True
    for key, value in contract.items():
        if (
            key.endswith("_open_now")
            or key.endswith("_started_now")
            or key.endswith("_created_now")
        ):
            assert value is False


def test_fail_closed_decision_blocks_real_capabilities_and_allows_only_16_4_source_model() -> None:
    decision = payload()["fail_closed_contract_decision"]
    assert decision["block_n_safety_gate_contract_in_16_3"] == "ready"
    assert decision["block_n_safety_gate_read_model_in_16_4"] == "allowed"
    for key in [
        "missing_block_n_safety_gate_matrix_policy",
        "missing_gate_contract_row_policy",
        "missing_operator_confirmation_policy",
        "missing_environment_validation_policy",
        "missing_artifact_validation_policy",
        "missing_release_validation_policy",
        "missing_runtime_validation_policy",
        "missing_credentials_validation_policy",
        "failed_contract_policy",
    ]:
        assert decision[key] == "fail_closed"
    blocked = {key for key, value in decision.items() if value == "blocked"}
    assert "release_execution_in_16_3" in blocked
    assert "packaging_dry_run_in_16_3" in blocked
    assert "runtime_activation_in_16_3" in blocked
    assert "order_" + "sub" + "mission_in_16_3" in blocked
    assert "order_" + "can" + "cellation_in_16_3" in blocked
    assert "order_" + "re" + "placement_in_16_3" in blocked
    assert "network_io_in_16_3" in blocked
    assert "filesystem_io_in_16_3" in blocked


def test_non_execution_evidence() -> None:
    evidence = payload()["non_execution_evidence"]
    for key in [
        "source_block_n_safety_gate_matrix_read",
        "block_n_safety_gate_contract_built",
        "block_n_safety_gate_contract_only",
        "block_n_opened",
        "ready_for_block_n_4",
        "all_contract_rows_fail_closed",
    ]:
        assert evidence[key] is True
    for key, value in evidence.items():
        if key not in {
            "source_block_n_safety_gate_matrix_read",
            "block_n_safety_gate_contract_built",
            "block_n_safety_gate_contract_only",
            "block_n_opened",
            "ready_for_block_n_4",
            "all_contract_rows_fail_closed",
        }:
            assert value is False


def test_contract_and_source_boundaries() -> None:
    data = payload()
    boundaries = data["contract_boundaries"]
    for key, value in boundaries.items():
        assert value is True
    source = data["source_boundaries"]
    assert source["allowed_imports_only"] is True
    assert source["source_block_n_safety_gate_matrix"] == "FUNCTIONAL-PREVIEW-16.2"
    for key, value in source.items():
        if key.startswith("forbidden_"):
            assert value is False
    nested = source["source_block_n_safety_gate_matrix_boundaries"]
    assert nested["allowed_imports_only"] is True
    assert nested["source_block_n_read_model"] == "FUNCTIONAL-PREVIEW-16.1"
    assert (
        nested["plain_data_source_only_subset"]["block_n_safety_gate_matrix_is_plain_data_only"]
        is True
    )
    assert nested["static_and_non_evaluating_boundary"] is True
    assert nested["can_feed_16_3_boundary"] is True


def test_no_contract_row_opens_allows_performs_or_has_present_validation() -> None:
    for row in all_capability_rows(payload()):
        assert row["contract_required"] is True
        assert row["contract_static_only"] is True
        assert row["contract_evaluated_by_16_3"] is False
        assert row["contract_condition_met"] is False
        assert row["contract_gate_open_now"] is False
        assert row["execution_allowed_now"] is False
        assert row["execution_performed_now"] is False
        assert row["operator_confirmation_present"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["failure_policy"] == "fail_closed"
        assert row["contract_result"] == "blocked_pending_future_explicit_gate"
        for key, value in row.items():
            if key.endswith("_present"):
                assert value is False


def test_source_import_call_and_forbidden_literal_guards() -> None:
    source = HELPER.read_text()
    tree = ast.parse(source)
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            imports.append((node.module, [alias.name for alias in node.names]))
        elif isinstance(node, ast.Import):
            imports.append(("", [alias.name for alias in node.names]))
    assert imports == [
        ("__future__", ["annotations"]),
        ("typing", ["Any", "Final"]),
        (
            "ui.pyside_app.preview_block_n_safety_gate_matrix",
            ["build_preview_block_n_safety_gate_matrix"],
        ),
    ]
    forbidden_calls = {
        "open",
        "read",
        "write",
        "read_text",
        "write_text",
        "getenv",
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
        "PyInstaller",
        "packaging",
        "build",
        "release",
    }
    calls = {
        node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    attrs = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert not (calls | attrs) & forbidden_calls
    for token in [
        "create_order",
        "sub" + "mit_order",
        "can" + "cel_order",
        "re" + "place_order",
        "fetch" + "_balance",
        "cc" + "xt",
    ]:
        assert token not in source
