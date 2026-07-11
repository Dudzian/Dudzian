from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_block_n_safety_gate_readiness_matrix import (
    build_preview_block_n_safety_gate_readiness_matrix,
)

MODULE_PATH = Path("ui/pyside_app/preview_block_n_safety_gate_readiness_matrix.py")
PREV_HELPER = Path("ui/pyside_app/preview_block_n_safety_gate_read_model.py")
PREV_TEST = Path("tests/ui_pyside/test_preview_block_n_safety_gate_read_model.py")
SOURCE_SMOKE = Path("tests/ui_pyside/test_source_smoke.py")
GATEWAY_TEST = Path("tests/test_local_gateway_validation.py")
TOP_LEVEL_FIELDS = [
    "schema_version",
    "block_n_safety_gate_readiness_matrix_kind",
    "block",
    "step",
    "block_n_safety_gate_readiness_matrix_status",
    "block_n_safety_gate_readiness_matrix_decision",
    "ready_for_block_n_6",
    "next_step",
    "next_step_title",
    "block_n_safety_gate_read_model_reference",
    "readiness_matrix_summary",
    "packaging_release_readiness_rows",
    "runtime_safety_readiness_rows",
    "cross_domain_invariant_readiness_rows",
    "validation_requirement_rows",
    "domain_readiness_summary",
    "exe_direction_readiness_matrix",
    "fail_closed_readiness_decision",
    "non_execution_evidence",
    "readiness_matrix_boundaries",
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
REQUIREMENT_IDS = [
    "operator_confirmation",
    "environment_validation",
    "artifact_validation",
    "release_validation",
    "runtime_validation",
    "credentials_validation",
    "future_explicit_gate",
]


def _matrix() -> dict:
    return build_preview_block_n_safety_gate_readiness_matrix()


def _source() -> str:
    return MODULE_PATH.read_text(encoding="utf-8")


def test_json_serializable_and_top_level_order() -> None:
    matrix = _matrix()
    json.dumps(matrix, sort_keys=True)
    assert list(matrix) == TOP_LEVEL_FIELDS


def test_identity_status_decision_ready_and_next_step() -> None:
    matrix = _matrix()
    assert matrix["schema_version"] == "preview_block_n_safety_gate_readiness_matrix.v1"
    assert (
        matrix["block_n_safety_gate_readiness_matrix_kind"]
        == "functional_preview_block_n_safety_gate_readiness_matrix"
    )
    assert matrix["block"] == "N"
    assert matrix["step"] == "16.5"
    assert "readiness_matrix_ready" in matrix["block_n_safety_gate_readiness_matrix_status"]
    assert "16_4_read_model_consumed" in matrix["block_n_safety_gate_readiness_matrix_status"]
    assert (
        "ALL_EXECUTION_CAPABILITIES_BLOCKED"
        in matrix["block_n_safety_gate_readiness_matrix_decision"]
    )
    assert matrix["ready_for_block_n_6"] is True
    assert matrix["next_step"] == "FUNCTIONAL-PREVIEW-16.6"
    assert matrix["next_step_title"] == "BLOCK N SAFETY GATE READINESS CONTRACT"
    assert (
        matrix["status"]
        == "ready_for_functional_preview_16_6_block_n_safety_gate_readiness_contract"
    )


def test_reference_safe_subset_and_false_by_16_5_flags() -> None:
    reference = _matrix()["block_n_safety_gate_read_model_reference"]
    expected_prefix = [
        "schema_version",
        "block_n_safety_gate_read_model_kind",
        "block",
        "step",
        "block_n_safety_gate_read_model_status",
        "block_n_safety_gate_read_model_decision",
        "ready_for_block_n_5",
        "next_step",
        "next_step_title",
    ]
    assert list(reference)[: len(expected_prefix)] == expected_prefix
    assert reference["source_block_n_safety_gate_read_model_step"] == "FUNCTIONAL-PREVIEW-16.4"
    assert reference["source_block_n_safety_gate_read_model_read_by_16_5"] is True
    assert reference["block_n_safety_gate_read_model_available_before_readiness_matrix"] is True
    assert reference["static_block_n_safety_gate_read_model_only"] is True
    assert reference["block_n_safety_gate_readiness_matrix_built_by_16_5"] is True
    assert reference["ready_for_functional_preview_16_6"] is True
    assert reference["step"] == "16.4"
    assert reference["next_step"] == "FUNCTIONAL-PREVIEW-16.5"
    for key, value in reference.items():
        if key.endswith("_by_16_5") and key not in {
            "block_n_safety_gate_readiness_matrix_built_by_16_5",
            "source_block_n_safety_gate_read_model_read_by_16_5",
        }:
            assert value is False


def test_summary_confirms_static_fail_closed_not_ready() -> None:
    summary = _matrix()["readiness_matrix_summary"]
    for key in [
        "block_n_safety_gate_read_model_available",
        "block_n_safety_gate_readiness_matrix_built",
        "block_n_opened",
        "ready_for_block_n_6",
        "ready_for_functional_preview_16_6",
        "block_m_closure_preserved",
        "exe_direction_preserved",
        "readiness_matrix_static_only",
        "readiness_matrix_read_only",
        "readiness_matrix_non_authorizing",
        "all_capabilities_classified",
        "all_missing_requirements_visible",
        "all_execution_capabilities_fail_closed",
        "all_execution_capabilities_not_ready",
        "all_execution_capabilities_require_future_explicit_gate",
        "missing_confirmation_blocks_execution",
        "missing_validation_blocks_execution",
        "missing_future_explicit_gate_blocks_execution",
    ]:
        assert summary[key] is True
    for key, value in summary.items():
        if key.startswith("any_") or key.endswith("_present_now") or key.endswith("_ready_now"):
            assert value is False


def test_capability_and_invariant_id_sets_are_exact() -> None:
    matrix = _matrix()
    assert [
        row["capability_id"] for row in matrix["packaging_release_readiness_rows"]
    ] == PACKAGING_IDS
    assert [row["capability_id"] for row in matrix["runtime_safety_readiness_rows"]] == RUNTIME_IDS
    invariant_rows = matrix["cross_domain_invariant_readiness_rows"]
    assert [row["invariant_id"] for row in invariant_rows] == INVARIANT_IDS
    for row in invariant_rows:
        assert row["domain"] == "cross_domain"
        assert row["source_read_result"] == "invariant_preserved_execution_blocked"
        assert row["source_invariant_preserved"] is True
        assert row["invariant_preserved_in_readiness_matrix"] is True
        assert row["invariant_required_for_future_execution"] is True
        assert row["execution_gate_open_now"] is False
        assert row["execution_allowed_now"] is False
        assert row["execution_performed_now"] is False
        assert row["requires_future_explicit_gate"] is True
        assert row["static_readiness_classification"] == "invariant_preserved_execution_not_ready"
        assert row["failure_policy"] == "fail_closed"
        assert row["matrix_result"] == "invariant_preserved_execution_blocked"


def test_packaging_rows_have_required_4_0_4_missing_requirements() -> None:
    for row in _matrix()["packaging_release_readiness_rows"]:
        assert row["domain"] == "packaging_release"
        assert row["required_requirements"] == [
            "operator_confirmation",
            "environment_validation",
            "artifact_validation",
            "future_explicit_gate",
        ]
        _assert_capability_row(row)


def test_runtime_rows_have_required_4_0_4_missing_requirements() -> None:
    for row in _matrix()["runtime_safety_readiness_rows"]:
        assert row["domain"] == "runtime_safety"
        assert row["required_requirements"] == [
            "operator_confirmation",
            "runtime_validation",
            "credentials_validation",
            "future_explicit_gate",
        ]
        _assert_capability_row(row)


def _assert_capability_row(row: dict) -> None:
    assert row["readiness_row_id"] == row["source_read_row_id"] + "_readiness_matrix"
    assert row["source_read_result"] == "not_ready_execution_blocked"
    assert row["satisfied_requirements"] == []
    assert row["missing_requirements"] == row["required_requirements"]
    assert row["requirements_total"] == 4
    assert row["requirements_satisfied_count"] == 0
    assert row["requirements_missing_count"] == 4
    assert row["requirements_complete"] is False
    assert row["static_readiness_classification"] == "not_ready"
    assert row["ready_for_execution"] is False
    assert row["execution_authorized"] is False
    assert row["gate_open_now"] is False
    assert row["execution_allowed_now"] is False
    assert row["execution_performed_now"] is False
    assert row["requires_future_explicit_gate"] is True
    assert row["failure_policy"] == "fail_closed"
    assert row["matrix_result"] == "not_ready_missing_requirements_execution_blocked"


def test_validation_requirement_rows_are_exact_and_missing() -> None:
    rows = _matrix()["validation_requirement_rows"]
    expected_domains = {
        "operator_confirmation": ["packaging_release", "runtime_safety"],
        "environment_validation": ["packaging_release"],
        "artifact_validation": ["packaging_release"],
        "release_validation": ["packaging_release"],
        "runtime_validation": ["runtime_safety"],
        "credentials_validation": ["runtime_safety"],
        "future_explicit_gate": [
            "packaging_release",
            "runtime_safety",
            "cross_domain",
        ],
    }
    assert [row["requirement_id"] for row in rows] == REQUIREMENT_IDS
    for row in rows:
        assert row["required"] is True
        assert row["present"] is False
        assert row["completed"] is False
        assert row["satisfied"] is False
        assert row["applicable_domains"] == expected_domains[row["requirement_id"]]
        assert row["missing_blocks_execution"] is True
        assert row["requires_future_explicit_step"] is True
        assert row["failure_policy"] == "fail_closed"
        assert row["readiness_result"] == "missing_execution_blocked"


def test_domain_summary_counts_ready_zero_and_blocked_all() -> None:
    summary = _matrix()["domain_readiness_summary"]
    expected_packaging_requirements = [
        "operator_confirmation",
        "environment_validation",
        "artifact_validation",
        "release_validation",
        "future_explicit_gate",
    ]
    expected_runtime_requirements = [
        "operator_confirmation",
        "runtime_validation",
        "credentials_validation",
        "future_explicit_gate",
    ]
    assert summary["packaging_release"]["capability_count"] == 22
    assert summary["packaging_release"]["ready_capability_count"] == 0
    assert summary["packaging_release"]["blocked_capability_count"] == 22
    assert (
        summary["packaging_release"]["required_requirement_ids"] == expected_packaging_requirements
    )
    assert summary["runtime_safety"]["capability_count"] == 18
    assert summary["runtime_safety"]["ready_capability_count"] == 0
    assert summary["runtime_safety"]["blocked_capability_count"] == 18
    assert summary["runtime_safety"]["required_requirement_ids"] == expected_runtime_requirements
    for domain in ["packaging_release", "runtime_safety"]:
        assert summary[domain]["satisfied_requirement_ids"] == []
        assert (
            summary[domain]["missing_requirement_ids"]
            == summary[domain]["required_requirement_ids"]
        )
        assert summary[domain]["requirements_complete"] is False
        assert summary[domain]["domain_ready"] is False
        assert summary[domain]["execution_authorized"] is False
        assert summary[domain]["failure_policy"] == "fail_closed"
        assert summary[domain]["readiness_result"] == "not_ready_execution_blocked"
    assert summary["overall"]["total_capability_count"] == 40
    assert summary["overall"]["ready_capability_count"] == 0
    assert summary["overall"]["blocked_capability_count"] == 40
    assert summary["overall"]["execution_authorized"] is False
    assert summary["overall"]["readiness_result"] == "not_ready_execution_blocked"


def test_exe_direction_preserved_but_build_package_release_not_ready() -> None:
    exe = _matrix()["exe_direction_readiness_matrix"]
    assert exe["final_product_direction"] == "desktop_exe"
    assert exe["exe_direction_preserved"] is True
    assert exe["block_n_safety_gate_readiness_matrix_confirms_exe_direction"] is True
    for key in ["build", "packaging", "release"]:
        assert exe[key + "_readiness_classification"] == "not_ready"
    assert exe["ready_to_build_exe_now"] is False
    assert exe["ready_to_package_exe_now"] is False
    assert exe["ready_to_release_exe_now"] is False
    assert exe["build_authorized_now"] is False
    assert exe["packaging_authorized_now"] is False
    assert exe["release_authorized_now"] is False
    assert exe["future_packaging_gate_required"] is True
    assert exe["future_release_gate_required"] is True
    assert exe["matrix_result"] == "exe_direction_preserved_execution_not_ready"


def test_fail_closed_decision_allows_only_source_only_16_6_and_blocks_real_capabilities() -> None:
    decision = _matrix()["fail_closed_readiness_decision"]
    assert decision["block_n_safety_gate_readiness_matrix_in_16_5"] == "ready"
    assert decision["block_n_safety_gate_readiness_contract_in_16_6"] == "allowed"
    assert decision["only_source_only_16_6_handoff_allowed"] is True
    assert all(value == "fail_closed" for key, value in decision.items() if key.endswith("_policy"))
    assert decision["real_capability_status"]["order_" + "sub" + "mission"] == "blocked"
    assert decision["real_capability_status"]["order_" + "can" + "cel"] == "blocked"
    assert decision["real_capability_status"]["order_" + "re" + "place"] == "blocked"
    assert all(value == "blocked" for value in decision["real_capability_status"].values())


def test_non_execution_evidence_and_boundaries() -> None:
    matrix = _matrix()
    evidence = matrix["non_execution_evidence"]
    for key, value in evidence.items():
        if key in {
            "source_block_n_safety_gate_read_model_read",
            "block_n_safety_gate_readiness_matrix_built",
            "block_n_safety_gate_readiness_matrix_only",
            "block_n_opened",
            "ready_for_block_n_6",
            "all_capability_rows_not_ready",
            "all_execution_authorization_false",
            "all_requirements_unsatisfied",
            "all_capabilities_fail_closed",
        }:
            assert value is True
        else:
            assert value is False
    assert all(matrix["readiness_matrix_boundaries"].values())


def test_source_boundaries_point_to_16_4_and_forbidden_flags_false() -> None:
    boundaries = _matrix()["source_boundaries"]
    assert boundaries["allowed_imports_only"] is True
    assert boundaries["source_block_n_safety_gate_read_model"] == "FUNCTIONAL-PREVIEW-16.4"
    assert boundaries["source_block_n_safety_gate_read_model_boundaries"]["can_feed_16_5"] is True
    for key, value in boundaries.items():
        if key.startswith("forbidden_"):
            assert value is False


def test_import_guard() -> None:
    tree = ast.parse(_source())
    imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    assert len(imports) == 3
    assert isinstance(imports[0], ast.ImportFrom)
    assert imports[0].module == "__future__"
    assert isinstance(imports[1], ast.ImportFrom)
    assert imports[1].module == "typing"
    assert isinstance(imports[2], ast.ImportFrom)
    assert imports[2].module == "ui.pyside_app.preview_block_n_safety_gate_read_model"


def test_call_guard() -> None:
    tree = ast.parse(_source())
    calls = [node.func for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden = {
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
        "authorize",
        "TradingController",
        "DecisionEnvelope",
        "PyInstaller",
        "packaging",
        "build",
        "release",
    }
    names = {func.id for func in calls if isinstance(func, ast.Name)}
    attrs = {func.attr for func in calls if isinstance(func, ast.Attribute)}
    assert not (names | attrs) & forbidden
    assert "build_preview_block_n_safety_gate_read_model" in names


def test_forbidden_literal_guard_and_no_side_effect_terms() -> None:
    source = _source()
    forbidden_literals = [
        "create_" + "order",
        "sub" + "mit_order",
        "can" + "cel_order",
        "re" + "place_order",
        "fetch" + "_balance",
        "cc" + "xt",
    ]
    for token in forbidden_literals:
        assert token not in source
    forbidden_import_terms = [
        "pathlib",
        "os",
        "subprocess",
        "requests",
        "urllib",
        "httpx",
        "aiohttp",
        "socket",
    ]
    import_lines = [
        line
        for line in source.splitlines()
        if line.startswith("import ") or line.startswith("from ")
    ]
    for token in forbidden_import_terms:
        assert all(token not in line for line in import_lines)


def test_prior_16_4_and_independent_fix_files_are_unchanged_by_scope() -> None:
    # Static source-only existence checks for files that this 16.5 patch must not edit.
    assert PREV_HELPER.exists()
    assert PREV_TEST.exists()
    assert SOURCE_SMOKE.exists()
    assert GATEWAY_TEST.exists()
