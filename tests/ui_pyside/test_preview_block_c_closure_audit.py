from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from ui.pyside_app.preview_read_only_binding import (
    PreviewReadOnlyBindingError,
    build_default_preview_read_only_binding_ui_state,
    build_preview_block_c_closure_audit,
    build_preview_read_only_binding_bridge_preflight,
    build_preview_read_only_binding_bridge_refusal_report,
    build_preview_read_only_binding_ui_state_boundary_matrix,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
QML_SOURCE = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"
BRIDGE_SOURCE = REPO_ROOT / "ui" / "pyside_app" / "preview_state_bridge.py"
SMOKE_SOURCE = REPO_ROOT / "ui" / "pyside_app" / "smoke.py"
PYSIDE_SMOKE_TEST = (
    REPO_ROOT / "tests" / "ui_pyside" / "test_preview_read_only_binding_bridge_refusal.py"
)


def _state_pair() -> tuple[dict[str, object], dict[str, object]]:
    state = build_default_preview_read_only_binding_ui_state()
    return state, dict(state)


def test_block_c_closure_audit_happy_path_exact_contract() -> None:
    state, reread = _state_pair()

    report = build_preview_block_c_closure_audit(state, reread_state=reread)

    assert report["block_name"] == "BLOK C — UI READ-ONLY BINDING"
    assert report["closure_kind"] == "block_c_ui_read_only_binding_closure_audit"
    assert report["block_status"] == "contract_complete_read_only"
    assert report["previous_block"] == "BLOK B — LOCAL RUNTIME SERVICE WRAPPER"
    assert report["next_block"] == "BLOK D — UI ACTION DISPATCH DO PAPER RUNTIME"
    assert report["closure_score"] == 100
    assert report["ready_for_block_d"] is True
    for flag in (
        "ready_for_ui_runtime_integration",
        "ready_for_runtime_loop",
        "ready_for_decision_engine",
        "ready_for_order_generation",
        "ready_for_order_submission",
        "ready_for_export",
        "ready_for_live",
        "external_export",
    ):
        assert report[flag] is False
    for flag in (
        "no_action_surface",
        "no_command_dispatch",
        "no_lifecycle_execution",
        "no_runtime_loop",
        "no_scheduler_worker_thread_timer",
        "no_trading_controller",
        "no_decision_envelope",
        "no_order_generation",
        "no_order_submission",
        "no_export_path",
        "no_cloud_external_path",
        "no_live_testnet_path",
        "no_account_secret_path",
    ):
        assert report[flag] is True
    assert report["integration_gate_status"] == "blocked"
    assert report["generated_order_count"] == 0
    assert report["generated_decision_count"] == 0
    assert report["export_sink"] == "none"
    assert report["cloud_sink"] == "none"


def test_block_c_closure_audit_aggregates_existing_block_c_artifacts() -> None:
    state, reread = _state_pair()
    matrix = build_preview_read_only_binding_ui_state_boundary_matrix(state)
    preflight = build_preview_read_only_binding_bridge_preflight(state, reread_state=reread)
    refusal = build_preview_read_only_binding_bridge_refusal_report(
        state, reread_state=reread, boundary_matrix=matrix
    )

    report = build_preview_block_c_closure_audit(state, reread_state=reread, boundary_matrix=matrix)

    assert report["controlled_ui_state_present"] is True
    assert report["boundary_matrix_all_refused"] is matrix.all_boundaries_refused is True
    assert report["bridge_preflight_passed"] is preflight["all_boundaries_refused"] is True
    assert report["bridge_refusal_passed"] is refusal["boundary_matrix_all_refused"] is True
    assert report["source_smoke_dynamic_static_contract_present"] is True
    assert report["source_smoke_safe_fallbacks_present"] is True
    assert report["pyside_bridge_smoke_present"] is True

    qml = QML_SOURCE.read_text(encoding="utf-8")
    bridge = BRIDGE_SOURCE.read_text(encoding="utf-8")
    smoke = SMOKE_SOURCE.read_text(encoding="utf-8")
    pyside_smoke = PYSIDE_SMOKE_TEST.read_text(encoding="utf-8")
    assert "function blockCReadOnlyBindingValue(key, fallback)" in qml
    assert 'blockCReadOnlyBindingValue("runtimeLoopStarted", false)' in qml
    assert '@Property("QVariantMap", constant=True)' in bridge
    assert "return dict(self._block_c_read_only_binding_state)" in bridge
    assert "block_c_read_only_binding_visible_source" in smoke
    assert "safe_fallbacks_present" in smoke
    assert "LocalPreviewStateBridge" in pyside_smoke


@pytest.mark.parametrize("key", ["runtimeLoopStarted", "runtimeBacked", "uiBound"])
def test_block_c_closure_audit_rejects_unsafe_runtime_flags(key: str) -> None:
    state, _ = _state_pair()
    state[key] = True

    with pytest.raises(PreviewReadOnlyBindingError, match=key):
        build_preview_block_c_closure_audit(state, reread_state=dict(state))


@pytest.mark.parametrize(
    ("key", "value"),
    [("exportSink", "file"), ("cloudSink", "s3"), ("externalExport", True)],
)
def test_block_c_closure_audit_rejects_export_cloud_external(key: str, value: object) -> None:
    state, _ = _state_pair()
    state[key] = value

    with pytest.raises(PreviewReadOnlyBindingError, match=key):
        build_preview_block_c_closure_audit(state, reread_state=dict(state))


@pytest.mark.parametrize(
    ("key", "value"), [("generatedOrderCount", 1), ("generatedDecisionCount", 1)]
)
def test_block_c_closure_audit_rejects_generated_counts(key: str, value: int) -> None:
    state, _ = _state_pair()
    state[key] = value

    with pytest.raises(PreviewReadOnlyBindingError, match=key):
        build_preview_block_c_closure_audit(state, reread_state=dict(state))


def test_block_c_closure_audit_rejects_unsafe_extra_key_and_callable_value() -> None:
    state, _ = _state_pair()
    unsafe_extra = dict(state)
    unsafe_extra["submitOrderHandler"] = "unsafe"
    with pytest.raises(PreviewReadOnlyBindingError, match="unsafe_ui_state_key"):
        build_preview_block_c_closure_audit(unsafe_extra, reread_state=dict(unsafe_extra))

    callable_state = dict(state)
    callable_state["scenarioName"] = lambda: "unsafe"
    with pytest.raises(PreviewReadOnlyBindingError, match="callable_ui_state_value"):
        build_preview_block_c_closure_audit(callable_state, reread_state=dict(callable_state))


def test_block_c_closure_audit_rejects_allowed_boundary_row_and_same_mapping() -> None:
    state, _ = _state_pair()
    matrix = build_preview_read_only_binding_ui_state_boundary_matrix(state)
    unsafe_row = replace(matrix.rows[0], allowed=True, refused=False)
    unsafe_matrix = replace(
        matrix,
        rows=(unsafe_row, *matrix.rows[1:]),
        refused_count=matrix.refused_count - 1,
        allowed_count=1,
        all_boundaries_refused=False,
    )

    with pytest.raises(
        PreviewReadOnlyBindingError, match="refused_count|allowed_count|all_boundaries_refused"
    ):
        build_preview_block_c_closure_audit(
            state, reread_state=dict(state), boundary_matrix=unsafe_matrix
        )
    with pytest.raises(PreviewReadOnlyBindingError, match="copy_on_read"):
        build_preview_block_c_closure_audit(state, reread_state=state)
