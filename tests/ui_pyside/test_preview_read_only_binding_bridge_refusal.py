from __future__ import annotations

import pytest

from ui.pyside_app.preview_read_only_binding import (
    PreviewReadOnlyBindingError,
    build_preview_read_only_binding_bridge_preflight,
    build_preview_read_only_binding_bridge_refusal_report,
    build_preview_read_only_binding_ui_state_boundary_matrix,
)

pytest.importorskip(
    "PySide6",
    reason="PySide6 unavailable: controlled skip; real bridge smoke runs in PySide-enabled CI.",
)

from ui.pyside_app.preview_state_bridge import LocalPreviewStateBridge  # noqa: E402


def test_block_c_bridge_copy_on_read_mutation_refusal_with_pyside() -> None:
    bridge = LocalPreviewStateBridge()

    first = bridge.blockCReadOnlyBindingState
    second = bridge.blockCReadOnlyBindingState

    assert isinstance(first, dict)
    assert isinstance(second, dict)
    assert first == second
    assert first is not second

    assert second["bindingKind"] == "static_local_block_b_closure_ui_read_only_binding"
    assert second["blockStatus"] == "contract_complete_static_local"
    assert second["integrationGateStatus"] == "blocked"
    assert second["readyForUiRuntimeIntegration"] is False
    assert second["readyForDecisionEngine"] is False
    assert second["readyForExport"] is False
    assert second["readyForLive"] is False
    assert second["runtimeLoopStarted"] is False
    assert second["runtimeBacked"] is False
    assert second["uiBound"] is False
    assert second["generatedOrderCount"] == 0
    assert second["generatedDecisionCount"] == 0
    assert second["exportSink"] == "none"
    assert second["cloudSink"] == "none"
    assert second["externalExport"] is False
    assert second["readOnly"] is True
    assert second["paperOnly"] is True
    assert not any(callable(value) for value in second.values())

    first["integrationGateStatus"] = "open"
    first["submitOrderHandler"] = "unsafe"
    first["runtimeLoopStarted"] = True
    third = bridge.blockCReadOnlyBindingState

    assert third["integrationGateStatus"] == "blocked"
    assert "submitOrderHandler" not in third
    assert third["runtimeLoopStarted"] is False
    assert third is not first

    matrix = build_preview_read_only_binding_ui_state_boundary_matrix(third)
    assert matrix.all_boundaries_refused is True

    preflight = build_preview_read_only_binding_bridge_preflight(
        third, reread_state=bridge.blockCReadOnlyBindingState
    )
    assert preflight["copy_on_read"] is True
    assert preflight["all_boundaries_refused"] is True

    report = build_preview_read_only_binding_bridge_refusal_report(
        third,
        reread_state=bridge.blockCReadOnlyBindingState,
        boundary_matrix=matrix,
    )

    assert report["copy_on_read_required"] is True
    assert report["boundary_matrix_all_refused"] is True
    assert report["integration_gate_status"] == "blocked"
    assert report["ready_for_ui_runtime_integration"] is False
    assert report["runtime_backed"] is False
    assert report["runtime_loop_started"] is False
    assert report["ui_bound"] is False
    assert report["generated_order_count"] == 0
    assert report["generated_decision_count"] == 0
    assert report["export_sink"] == "none"
    assert report["cloud_sink"] == "none"
    assert report["external_export"] is False
    assert report["actions_refused"] is True
    assert report["commands_refused"] is True
    assert report["lifecycle_refused"] is True
    assert report["export_refused"] is True
    assert report["cloud_external_refused"] is True
    assert report["live_testnet_refused"] is True
    assert report["account_secret_refused"] is True

    with pytest.raises(PreviewReadOnlyBindingError, match="copy_on_read"):
        build_preview_read_only_binding_bridge_refusal_report(third, reread_state=third)

    unsafe_state = dict(third)
    safe_matrix = build_preview_read_only_binding_ui_state_boundary_matrix(third)
    unsafe_state["runtimeLoopStarted"] = True
    with pytest.raises(PreviewReadOnlyBindingError, match="runtimeLoopStarted"):
        build_preview_read_only_binding_bridge_refusal_report(
            unsafe_state, reread_state=dict(unsafe_state), boundary_matrix=safe_matrix
        )

    unsafe_surface = dict(third)
    unsafe_surface["submitOrderHandler"] = "unsafe"
    with pytest.raises(PreviewReadOnlyBindingError, match="unsafe_ui_state_key"):
        build_preview_read_only_binding_bridge_refusal_report(
            unsafe_surface, reread_state=dict(unsafe_surface), boundary_matrix=safe_matrix
        )
