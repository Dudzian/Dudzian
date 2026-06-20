"""Tests for BLOK E controlled QmlContextBridge action dispatch wiring."""

from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Any

from ui.pyside_app.config import UiAppConfig
from ui.pyside_app.preview_action_dispatch_audit import ACCEPTED_INTENT_NOT_EXECUTED
from ui.pyside_app.preview_action_dispatch_bridge_snapshot import NO_SELECTION_STATUS
from ui.pyside_app.preview_action_dispatch_qt_bridge import PaperRuntimeActionDispatchQtBridge
from ui.pyside_app.preview_action_dispatch_qt_bridge_registration import (
    PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY,
)
from ui.pyside_app.qml_bridge import QmlContextBridge

REPO_ROOT = Path(__file__).resolve().parents[2]
QML_BRIDGE = REPO_ROOT / "ui" / "pyside_app" / "qml_bridge.py"
APP = REPO_ROOT / "ui" / "pyside_app" / "app.py"
QML_FILES = (
    REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml",
    REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml",
    REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "PaperTerminal.qml",
)
BAT_LAUNCHERS = (
    REPO_ROOT / "run_ui_preview_visible_doubleclick.bat",
    REPO_ROOT / "scripts" / "windows" / "run_ui_preview_visible.bat",
)
EXPECTED_CONTEXT_PROPERTIES = (
    "uiConfig",
    "cloudRuntimeEnabled",
    "grpcBridge",
    "runtimeState",
    "licensingController",
    "diagnosticsController",
    "layoutController",
    "modeWizardController",
    "strategyManagementController",
    "theme",
    "typedPreviewBridge",
    "paperRuntimeActionDispatchBridge",
)
FORBIDDEN_EXECUTION_TOKENS = (
    "dispatch_command",
    "execute_command",
    "start_runtime",
    "start_loop",
    "submit_order",
    "create_order",
    "place_order",
    "send_order",
    "fill_order",
    "TradingController",
    "DecisionEnvelope",
    "live_adapter",
    "testnet_adapter",
    "account_fetch",
    "secrets",
    "export_to_cloud",
)


class FakeContext:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def setContextProperty(self, name: str, value: object) -> None:  # noqa: N802 - Qt API name.
        self.calls.append((name, value))


class FakeEngine:
    def __init__(self) -> None:
        self.context = FakeContext()

    def rootContext(self) -> FakeContext:  # noqa: N802 - Qt API name.
        return self.context


def _config() -> UiAppConfig:
    return UiAppConfig(
        source_path=REPO_ROOT / "ui_config.yaml",
        profile="test",
        payload={"history_limit": 5, "theme": {"palette": "dark"}},
        qml_entrypoint=REPO_ROOT / "ui" / "pyside_app" / "qml" / "MainWindow.qml",
        decision_limit=5,
        theme_palette="dark",
    )


def _installed_bridge() -> tuple[QmlContextBridge, FakeContext, dict[str, object]]:
    engine = FakeEngine()
    bridge = QmlContextBridge(engine, _config(), enable_cloud_runtime=False)  # type: ignore[arg-type]
    original_typed = bridge.typed_preview_bridge
    original_grpc = bridge.ui_grpc_bridge
    original_runtime = bridge.ui_runtime_state

    bridge.install()

    values = dict(engine.context.calls)
    assert values["typedPreviewBridge"] is original_typed
    assert values["grpcBridge"] is original_grpc
    assert values["runtimeState"] is original_runtime
    return bridge, engine.context, values


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_qml_context_bridge_registers_paper_runtime_action_dispatch_bridge_once() -> None:
    bridge, context, values = _installed_bridge()

    names = [name for name, _value in context.calls]
    assert names == list(EXPECTED_CONTEXT_PROPERTIES)
    assert names.count(PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY) == 1
    registered = values[PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY]
    assert isinstance(registered, PaperRuntimeActionDispatchQtBridge)
    assert (
        bridge.paper_runtime_action_dispatch_bridge_registration_evidence["context_property_name"]
        == PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY
    )
    assert bridge.paper_runtime_action_dispatch_bridge_registration_evidence["registered"] is True


def test_qml_context_bridge_preserves_existing_context_properties_and_names() -> None:
    _bridge, context, values = _installed_bridge()

    assert [name for name, _value in context.calls] == list(EXPECTED_CONTEXT_PROPERTIES)
    for existing_name in EXPECTED_CONTEXT_PROPERTIES[:-1]:
        assert existing_name in values
    assert values["typedPreviewBridge"] is not values["grpcBridge"]
    assert values["runtimeState"] is not values["typedPreviewBridge"]


def test_registered_bridge_default_and_action_preview_remain_no_execution() -> None:
    _bridge, _context, values = _installed_bridge()
    registered = values[PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY]
    assert isinstance(registered, PaperRuntimeActionDispatchQtBridge)

    snapshot = registered.snapshot
    assert snapshot["status"] == NO_SELECTION_STATUS
    for key in (
        "execution_allowed",
        "execution_performed",
        "provider_execution_allowed",
        "provider_execution_performed",
        "qt_bridge_execution_allowed",
        "qt_bridge_execution_performed",
    ):
        assert snapshot[key] is False

    selected = registered.previewSelectAction("paper_runtime_start_requested")
    assert selected["selected_result"]["result_status"] == ACCEPTED_INTENT_NOT_EXECUTED
    for key in (
        "execution_allowed",
        "execution_performed",
        "provider_execution_allowed",
        "provider_execution_performed",
        "qt_bridge_execution_allowed",
        "qt_bridge_execution_performed",
    ):
        assert selected[key] is False
    assert selected["selected_result"]["execution_allowed"] is False
    assert selected["selected_result"]["execution_performed"] is False


def test_qml_context_bridge_uses_registration_helper_only_in_install() -> None:
    source = _source(QML_BRIDGE)
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "register_paper_runtime_action_dispatch_qt_bridge"
    ]

    assert len(calls) == 1
    assert "PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY" in source
    install_source = source[source.index("def install") :]
    assert "register_paper_runtime_action_dispatch_qt_bridge" in install_source


def test_registered_property_consumption_limited_to_operator_dashboard_read_only_snapshot() -> None:
    qml_consumers = [
        path.relative_to(REPO_ROOT).as_posix()
        for path in QML_FILES
        if PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY in _source(path)
    ]

    assert qml_consumers == ["ui/pyside_app/qml/views/OperatorDashboard.qml"]
    operator_source = _source(
        REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"
    )
    assert f"{PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY}.snapshot" in operator_source
    for method in ("previewSelectAction", "previewSelectSourceControl", "resetPreviewSelection"):
        assert f".{method}(" not in operator_source
        assert f"paperRuntimeActionDispatchBridge.{method}" not in operator_source
    for path in (*BAT_LAUNCHERS, APP):
        assert PAPER_RUNTIME_ACTION_DISPATCH_QT_BRIDGE_CONTEXT_PROPERTY not in _source(path)
    assert "register_paper_runtime_action_dispatch_qt_bridge" not in _source(APP)


def test_launchers_remain_unchanged_in_worktree() -> None:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-only",
            "--",
            *(p.relative_to(REPO_ROOT).as_posix() for p in BAT_LAUNCHERS),
        ],
        cwd=REPO_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip() == ""


def test_no_second_engine_qml_handler_or_execution_path_added_to_qml_bridge() -> None:
    source = _source(QML_BRIDGE)

    tree = ast.parse(source)
    engine_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and (
            (isinstance(node.func, ast.Name) and node.func.id == "QQmlApplicationEngine")
            or (isinstance(node.func, ast.Attribute) and node.func.attr == "QQmlApplicationEngine")
        )
    ]
    assert engine_calls == []
    assert "Button.onClicked" not in source
    assert "onClicked:" not in source
    assert "MouseArea" not in source
    assert "Connections" not in source
    offenders = [token for token in FORBIDDEN_EXECUTION_TOKENS if token in source]
    assert offenders == []
