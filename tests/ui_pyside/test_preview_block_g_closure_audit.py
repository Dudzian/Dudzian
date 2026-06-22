from __future__ import annotations

import ast
import json
from pathlib import Path

from ui.pyside_app.preview_block_g_closure_audit import (
    BLOCK_G_CLOSURE_DECISION,
    NEXT_STEP,
    NEXT_STEP_TITLE,
    PREVIEW_BLOCK_G_CLOSURE_AUDIT_SCHEMA_VERSION,
    READY_FOR_BLOCK_H,
    build_preview_block_g_closure_audit,
)


def test_block_g_closure_audit_is_plain_json_serializable_dict() -> None:
    audit = build_preview_block_g_closure_audit()

    assert isinstance(audit, dict)
    assert audit["schema_version"] == PREVIEW_BLOCK_G_CLOSURE_AUDIT_SCHEMA_VERSION
    assert json.loads(json.dumps(audit)) == audit


def test_status_decision_next_step_and_handoff_are_correct() -> None:
    audit = build_preview_block_g_closure_audit()

    assert audit["block_g_closure_decision"] == BLOCK_G_CLOSURE_DECISION
    assert audit["ready_for_block_h"] is READY_FOR_BLOCK_H
    assert audit["next_step"] == NEXT_STEP == "FUNCTIONAL-PREVIEW-10.0"
    assert (
        audit["next_step_title"]
        == NEXT_STEP_TITLE
        == "BLOK H — READ-ONLY MARKET DATA ADAPTER CONTRACT"
    )
    assert (
        audit["status"]
        == "ready_for_functional_preview_10_0_block_h_read_only_market_data_contract"
    )

    handoff = audit["handoff_to_block_h"]
    assert handoff["handoff_ready"] is True
    assert handoff["handoff_target_block"] == "H"
    assert handoff["handoff_target_step"] == "FUNCTIONAL-PREVIEW-10.0"
    assert handoff["handoff_scope"] == "read_only_market_data_contract_only"
    assert handoff["handoff_requires_new_block_contract"] is True
    assert handoff["handoff_runtime_execution_allowed"] is False
    assert handoff["handoff_live_trading_allowed"] is False
    assert handoff["handoff_account_fetch_allowed"] is False
    assert handoff["handoff_order_submission_allowed"] is False


def test_block_g_references_cover_steps_9_0_through_9_8_without_execution() -> None:
    audit = build_preview_block_g_closure_audit()
    references = audit["block_g_step_references"]

    assert [reference["step"] for reference in references] == [
        "9.0",
        "9.1",
        "9.2",
        "9.3",
        "9.4",
        "9.5",
        "9.6",
        "9.7",
        "9.8",
    ]
    for reference in references:
        assert reference["ready_flag"] is True
        assert reference["execution_allowed"] is False
        assert reference["runtime_allowed"] is False
        assert reference["live_or_testnet_allowed"] is False
        assert "schema_version" in reference or "surface_name" in reference
        assert "kind" in reference or "surface_kind" in reference


def test_completion_matrix_and_artifact_summary_close_block_g() -> None:
    audit = build_preview_block_g_closure_audit()
    matrix = audit["block_g_completion_matrix"]

    for key in (
        "contract_9_0_complete",
        "read_model_9_1_complete",
        "static_fixture_9_2_complete",
        "audit_envelope_9_3_complete",
        "ui_read_only_surface_9_4_complete",
        "selection_gate_9_5_complete",
        "controlled_intent_preview_9_6_complete",
        "fill_simulator_contract_9_7_complete",
        "lifecycle_audit_9_8_complete",
        "block_g_complete",
        "ready_for_block_h",
        "paper_only_scope_preserved",
    ):
        assert matrix[key] is True

    for key in (
        "runtime_execution_added",
        "live_or_testnet_added",
        "order_submission_added",
        "fill_simulation_added",
        "lifecycle_mutation_added",
        "market_data_fetch_added",
        "audit_export_added",
        "qml_expansion_added",
    ):
        assert matrix[key] is False

    summary = audit["block_g_static_artifact_summary"]
    assert all(summary.values())


def test_case_coverage_has_four_known_cases_and_no_execution_paths() -> None:
    audit = build_preview_block_g_closure_audit()
    cases = audit["block_g_case_coverage"]

    assert [case["case_id"] for case in cases] == [
        "baseline_btc_no_intent_no_order",
        "eth_size_preview_no_intent_no_order",
        "sol_risk_blocked_preview_no_intent_no_order",
        "unknown_input_keys_reported_no_execution",
    ]
    for case in cases:
        assert case["selection_gate_available"] is True
        assert case["controlled_intent_preview_available"] is True
        assert case["fill_simulator_contract_available"] is True
        assert case["lifecycle_audit_available"] is True
        for key in (
            "order_generation_allowed",
            "submission_allowed",
            "fill_simulation_allowed",
            "fill_event_generation_allowed",
            "lifecycle_mutation_allowed",
            "runtime_execution_allowed",
            "live_or_testnet_allowed",
            "account_or_secrets_allowed",
            "export_allowed",
        ):
            assert case[key] is False


def test_no_execution_evidence_boundary_checks_and_closure_summary() -> None:
    audit = build_preview_block_g_closure_audit()
    evidence = audit["block_g_no_execution_evidence"]

    assert evidence["closure_audit_evaluated"] is True
    for key, value in evidence.items():
        if key != "closure_audit_evaluated":
            assert value is False

    boundary = audit["block_g_boundary_checks"]
    true_keys = {
        "local_only",
        "paper_only",
        "block_g_closure_audit_only",
        "block_g_complete",
        "ready_for_block_h",
        "read_only_market_data_next_block_only",
        "paper_decision_to_order_path_static_complete",
        "paper_order_intent_preview_allowed_now",
        "paper_fill_simulation_contract_allowed_now",
        "order_lifecycle_audit_allowed_now",
        "exe_direction_preserved",
    }
    for key, value in boundary.items():
        assert value is (key in true_keys)

    closure = audit["closure_summary"]
    assert closure["block_g_closed"] is True
    assert closure["completed_steps"] == [
        "9.0",
        "9.1",
        "9.2",
        "9.3",
        "9.4",
        "9.5",
        "9.6",
        "9.7",
        "9.8",
    ]
    assert closure["ready_for_block_h"] is True
    assert closure["paper_only_path_complete"] is True
    assert closure["runtime_execution_present"] is False
    assert closure["live_or_testnet_present"] is False
    assert closure["orders_or_fills_present"] is False
    assert closure["qml_expansion_present"] is False


def test_blocked_capabilities_and_source_boundaries_are_complete() -> None:
    audit = build_preview_block_g_closure_audit()

    assert set(audit["blocked_capabilities"]) == {
        "paper order generation",
        "paper order submission",
        "paper fill simulation execution",
        "paper fill event generation",
        "paper order lifecycle mutation",
        "paper order lifecycle transition execution",
        "paper runtime execution",
        "risk governor execution now",
        "market data adapter implementation",
        "market data fetch",
        "account fetch",
        "audit export",
        "live/testnet/account/secrets/export/cloud",
        "TradingController / DecisionEnvelope",
        "QML changes / new QML calls",
        "EXE packaging",
    }
    assert set(audit["source_boundaries"]) == {
        "no PySide import",
        "no QML import",
        "no runtime loop import",
        "no TradingController import",
        "no DecisionEnvelope import",
        "no strategy/AI/scoring/recommendation import",
        "no order module import",
        "no live adapter import",
        "no testnet adapter import",
        "no market data adapter import",
        "no account module import",
        "no secrets module import",
        "no filesystem I/O",
        "no network I/O",
        "no QML changes",
        "no .bat changes",
        "no app.py changes",
        "no dependency declarations changes",
        "no workflow changes",
    }


def test_helper_imports_only_allowed_static_block_g_helpers() -> None:
    source_path = Path("ui/pyside_app/preview_block_g_closure_audit.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")

    assert set(imports) <= {
        "__future__",
        "copy",
        "typing",
        "ui.pyside_app.preview_paper_decision_to_order_contract",
        "ui.pyside_app.preview_paper_order_intent_read_model",
        "ui.pyside_app.preview_paper_order_static_fixture",
        "ui.pyside_app.preview_paper_order_audit_envelope",
        "ui.pyside_app.preview_paper_order_intent_selection_gate",
        "ui.pyside_app.preview_controlled_paper_order_intent",
        "ui.pyside_app.preview_paper_fill_simulator_contract",
        "ui.pyside_app.preview_paper_order_lifecycle_audit",
    }

    forbidden_import_fragments = (
        "PySide",
        "QQml",
        "qml",
        "runtime",
        "TradingController",
        "DecisionEnvelope",
        "strategy",
        "scoring",
        "recommendation",
        "order_adapter",
        "live",
        "testnet",
        "market_data",
        "account",
        "secrets",
        "requests",
        "subprocess",
        "pathlib",
    )
    for module in imports:
        assert not any(fragment in module for fragment in forbidden_import_fragments)


def test_helper_source_has_no_forbidden_runtime_calls_or_io() -> None:
    source = Path("ui/pyside_app/preview_block_g_closure_audit.py").read_text(encoding="utf-8")

    for forbidden in (
        "open(",
        "read_text",
        "write_text",
        "requests",
        "subprocess",
        "QQmlApplicationEngine",
        "TradingController(",
        "DecisionEnvelope(",
        "start_runtime",
        "start_loop",
        "submit_order",
        "place_order",
        "create_order",
        "send_order",
        "fill_order",
    ):
        assert forbidden not in source


def test_qml_preview_selection_surface_remains_single_allowed_call() -> None:
    qml_files = [
        Path("ui/pyside_app/qml/views/OperatorDashboard.qml"),
        Path("ui/pyside_app/qml/MainWindow.qml"),
    ]
    qml_source = "\n".join(path.read_text(encoding="utf-8") for path in qml_files)

    allowed_literal = 'paperRuntimeActionDispatchBridge.previewSelectAction("paper_runtime_snapshot_refresh_requested")'
    assert qml_source.count("previewSelectAction(") == 1
    assert qml_source.count(allowed_literal) == 1
    assert "previewSelectSourceControl(" not in qml_source
    assert "resetPreviewSelection(" not in qml_source
    for forbidden in (
        "startRuntime",
        "start_runtime",
        "startLoop",
        "start_loop",
        "stopRuntime",
        "pauseRuntime",
        "resumeRuntime",
        "submitOrder",
        "placeOrder",
        "sendOrder",
        "fillOrder",
        "lifecycleTransition",
    ):
        assert forbidden not in qml_source
