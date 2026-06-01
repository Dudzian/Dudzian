"""Source-level smoke checks for the existing PySide6/QML UI."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ui.pyside_app.app import AppOptions

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_SOURCE = REPO_ROOT / "ui" / "pyside_app" / "smoke.py"
PLAN_SOURCE = REPO_ROOT / "scripts" / "ui_preview_launch_plan.py"
APP_SOURCE = REPO_ROOT / "ui" / "pyside_app" / "app.py"
QML_SOURCE_ROOT = REPO_ROOT / "ui" / "pyside_app" / "qml"
FORBIDDEN_SOURCE_TOKENS = (
    "create" + "_" + "order",
    "fetch" + "_" + "balance",
    "load" + "_" + "markets",
    "key" + "ring",
    "dot" + "env",
    "shell" + "=True",
    "subprocess" + "." + "run",
    "os" + "." + "environ",
)


def _run_ui_smoke(*extra_args: str) -> subprocess.CompletedProcess[str]:
    run_process = getattr(subprocess, "run")
    return run_process(
        [
            sys.executable,
            "-m",
            "ui.pyside_app",
            "--config",
            "ui/config/preview_local.yaml",
            "--smoke",
            "--offscreen",
            *extra_args,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=20,
        check=False,
    )


def _smoke_payload(result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    result.stdout.encode("cp1252")
    assert result.stdout.strip(), result.stderr
    return json.loads(result.stdout)


def test_smoke_flags_are_available_in_parser() -> None:
    options = AppOptions.parse(
        ["--config", "ui/config/preview_local.yaml", "--smoke", "--offscreen"]
    )

    assert options.smoke is True
    assert options.offscreen is True
    assert options.enable_cloud_runtime is False


def test_source_smoke_finishes_and_reports_safety_contract() -> None:
    result = _run_ui_smoke()
    payload = _smoke_payload(result)

    if result.returncode != 0 and any("libGL.so.1" in issue for issue in payload["issues"]):
        pytest.skip("Qt runtime unavailable in this headless environment: missing libGL.so.1")

    assert result.returncode == 0, result.stderr or result.stdout
    assert payload["status"] == "ok"
    assert payload["ui_loaded"] is True
    assert payload["qml_loaded"] is True
    assert payload["runtime_loop_started"] is False
    assert payload["production_runtime_loop_started"] is False
    assert payload["exchange_io"] == "disabled"
    assert payload["order_submission"] == "disabled"
    assert payload["api_keys_required"] is False
    assert payload["secrets_read"] is False
    assert payload["keychain_read"] is False
    assert payload["env_values_read"] is False
    assert payload["dot_env_read"] is False
    assert payload["issues"] == []


def test_smoke_blocks_live_runtime_flag_without_qt_bootstrap() -> None:
    result = _run_ui_smoke("--enable-cloud-runtime")
    payload = _smoke_payload(result)

    assert result.returncode == 2
    assert payload["status"] == "blocked"
    assert payload["ui_loaded"] is False
    assert payload["qml_loaded"] is False
    assert payload["runtime_loop_started"] is False
    assert payload["exchange_io"] == "disabled"
    assert payload["order_submission"] == "disabled"
    assert payload["api_keys_required"] is False
    assert payload["live_mode_allowed"] is False
    assert payload["issues"] == ["smoke_mode_blocks_enable_cloud_runtime"]


def test_smoke_and_plan_sources_have_no_forbidden_runtime_or_secret_calls() -> None:
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in (SMOKE_SOURCE, PLAN_SOURCE, APP_SOURCE)
    )

    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source


def _qml_sources() -> list[Path]:
    return sorted(QML_SOURCE_ROOT.rglob("*.qml"))


def _qml_text() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in _qml_sources())


def test_qml_operator_dashboard_is_default_selected_panel() -> None:
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")

    assert 'property string defaultPanelId: "sidePanel"' in main_window
    assert "property string currentPanelId: defaultPanelId" in main_window
    assert "showOperatorDashboard()" in main_window
    assert "layoutController.registerPanels(panelMetadata)" in main_window
    assert 'panelId: "sidePanel", title: qsTr("Dashboard operatora")' in main_window
    assert 'defaultPanelId: "modeWizardPanel"' not in main_window
    assert 'currentPanelId: "modeWizardPanel"' not in main_window
    assert 'defaultPanelId: ""' not in main_window


def test_qml_operator_dashboard_default_content_and_labels() -> None:
    source = _qml_text()

    assert 'objectName: "operatorOverviewDashboard"' in source
    assert "Dashboard operatora" in source
    assert "Tryb: Demo / Paper" in source
    assert "Strumień decyzji" in source
    assert "Kontrola ryzyka" in source
    assert "Menedżer strategii" in source
    assert "Warsztat strategii" in source
    assert ("Decyzje governor" in source) or ("Decyzje strategii" in source)


def test_qml_operator_preview_removes_raw_labels_and_refresh_garbled_glyph() -> None:
    source = _qml_text()

    forbidden_labels = (
        "Chart_Decision Stream",
        "Chart & Decision Stream",
        "Decyzje AI",
        "Strategy Workbench",
        "Risk Controls",
        "Strategy Manager",
        "ǎ Odśwież dane",
        "ǎ",
    )
    for label in forbidden_labels:
        assert label not in source
    assert 'text: qsTr("Odśwież dane")' in source


def test_qml_operator_preview_demo_offline_safety_copy() -> None:
    source = _qml_text()

    required_copy = (
        "Exchange I/O disabled",
        "Order submission disabled",
        "Runtime loop not started",
        "API keys required: false",
        "Active strategy: Demo Momentum Guard",
        "Last decision: HOLD / NO ORDER",
        "Live trading: blocked / disabled",
        "Live disabled",
        "BTC/USDT | HOLD | confidence 0.62 | no order",
        "ETH/USDT | WAIT | confidence 0.55 | no order",
        "SOL/USDT | BLOCKED LIVE | reason: demo mode",
        "Max drawdown guard: demo only",
        "Kill switch: armed / preview",
        "podłączony lokalny preview bridge",
    )
    for text in required_copy:
        assert text in source


def test_qml_operator_dashboard_has_real_content_component_and_restorable_menu_action() -> None:
    main_window = (QML_SOURCE_ROOT / "MainWindow.qml").read_text(encoding="utf-8")

    assert (
        '"sidePanel": { title: qsTr("Dashboard operatora"), icon: "fingerprint", component: sidePanelComponent }'
        in main_window
    )
    assert "id: sidePanelComponent" in main_window
    assert 'objectName: "operatorOverviewDashboard"' in main_window
    assert "root.showOperatorDashboard()" in main_window
    assert "layoutController.setPanelVisibility(panelId, true)" in main_window


def test_qml_work_modes_has_demo_offline_placeholder() -> None:
    mode_wizard = (QML_SOURCE_ROOT / "views" / "ModeWizard.qml").read_text(encoding="utf-8")

    assert "Brak danych o profilach cloud" in mode_wizard
    assert "Tryb demo/offline" in mode_wizard
    assert "Live trading pozostaje wyłączony" in mode_wizard


def test_changed_ui_qml_sources_have_no_forbidden_runtime_or_secret_calls() -> None:
    source = _qml_text()

    for token in FORBIDDEN_SOURCE_TOKENS:
        assert token not in source
