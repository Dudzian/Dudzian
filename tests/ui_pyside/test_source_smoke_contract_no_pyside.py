"""Source-only UI smoke contracts that must not require PySide6."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SOURCE = REPO_ROOT / "ui" / "pyside_app" / "smoke.py"
MAIN_WINDOW_QML = REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml"


def _smoke_source() -> str:
    return SMOKE_SOURCE.read_text(encoding="utf-8")


def _main_window_qml() -> str:
    return MAIN_WINDOW_QML.read_text(encoding="utf-8")


def test_operator_workflow_qml_select_scanner_pair_source_contract() -> None:
    main_window = _main_window_qml()

    assert "function selectScannerPair(pair)" in main_window
    assert "selectedPairs = [scannerSelectedPair].concat(selectedCopy)" in main_window
    assert "whitelistPairs = selectedPairs.slice()" in main_window
    assert 'setTerminalPairFromSource(scannerSelectedPair, "selectScannerPair")' in main_window
    assert "selectedTerminalPair = scannerSelectedPair" not in main_window


def test_operator_workflow_selected_candidate_does_not_use_terminal_root_fallback() -> None:
    smoke_source = _smoke_source()

    assert "before_terminal_pair" in smoke_source
    assert (
        'selected_candidate_pair = before_scanner_pair or scanner_first_row_pair or "BTC/USDT"'
        in smoke_source
    )
    assert "selected_candidate_pair = before_terminal_pair" not in smoke_source
    assert "operator_selected_candidate_source_diagnostic" in smoke_source
    assert '"scannerSelectedPair"' in smoke_source
    assert '"scannerRows[0].pair"' in smoke_source


def test_operator_workflow_timeline_diagnostics_source_contract() -> None:
    smoke_source = _smoke_source()

    for diagnostic in (
        "operator_pair_before_select_selected_terminal_pair",
        "operator_pair_after_select_selected_terminal_pair",
        "operator_pair_after_terminal_open_selected_terminal_pair",
        "operator_selected_pairs_diagnostic",
        "operator_terminal_panel_active_pair_diagnostic",
    ):
        assert diagnostic in smoke_source


def test_operator_workflow_fail_closed_helper_source_evidence() -> None:
    smoke_source = _smoke_source()

    for token in (
        "def _operator_pair_state_matches",
        "operator_selected_candidate_updates_shared_state",
        "operator_terminal_pair_matches_selected_candidate",
        "operator_pair_match_diagnostic",
    ):
        assert token in smoke_source


def test_operator_workflow_required_section_source_contract() -> None:
    smoke_source = _smoke_source()

    assert '"operator_workflow": ("operator_workflow_smoke_complete",)' in smoke_source
    assert "FRONTEND_LIVE_PARITY_REQUIRED_SECTIONS" in smoke_source
    assert "operator_workflow_smoke_complete" in smoke_source
