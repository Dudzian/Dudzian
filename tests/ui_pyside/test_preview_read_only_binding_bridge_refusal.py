from __future__ import annotations

import pytest

from ui.pyside_app.preview_read_only_binding import (
    build_preview_read_only_binding_bridge_refusal_report,
    build_preview_read_only_binding_ui_state_boundary_matrix,
)

pytest.importorskip("PySide6")

from ui.pyside_app.preview_state_bridge import LocalPreviewStateBridge  # noqa: E402


def test_block_c_bridge_copy_on_read_mutation_refusal_with_pyside() -> None:
    bridge = LocalPreviewStateBridge()

    first = bridge.blockCReadOnlyBindingState
    first["integrationGateStatus"] = "open"
    first["submitOrderHandler"] = "unsafe"
    second = bridge.blockCReadOnlyBindingState

    assert second["integrationGateStatus"] == "blocked"
    assert "submitOrderHandler" not in second
    assert first is not second

    matrix = build_preview_read_only_binding_ui_state_boundary_matrix(second)
    report = build_preview_read_only_binding_bridge_refusal_report(
        second,
        reread_state=bridge.blockCReadOnlyBindingState,
        boundary_matrix=matrix,
    )

    assert matrix.all_boundaries_refused is True
    assert report["copy_on_read_required"] is True
    assert report["boundary_matrix_all_refused"] is True
    assert report["integration_gate_status"] == "blocked"
    assert report["runtime_backed"] is False
