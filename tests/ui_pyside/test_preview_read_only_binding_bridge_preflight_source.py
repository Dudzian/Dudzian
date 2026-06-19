"""Non-PySide source proof for BLOK C read-only bridge preflight."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
BRIDGE_SOURCE = REPO_ROOT / "ui" / "pyside_app" / "preview_state_bridge.py"
QML_SOURCE = REPO_ROOT / "ui" / "pyside_app" / "qml" / "views" / "OperatorDashboard.qml"


def _local_preview_state_bridge_class() -> ast.ClassDef:
    module = ast.parse(BRIDGE_SOURCE.read_text(encoding="utf-8"))
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "LocalPreviewStateBridge":
            return node
    raise AssertionError("LocalPreviewStateBridge not found")


def _method(name: str) -> ast.FunctionDef:
    for node in _local_preview_state_bridge_class().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method not found: {name}")


def _decorator_call(function: ast.FunctionDef, name: str) -> ast.Call:
    for decorator in function.decorator_list:
        if isinstance(decorator, ast.Call) and isinstance(decorator.func, ast.Name):
            if decorator.func.id == name:
                return decorator
    raise AssertionError(f"decorator not found: {name}")


def _block_c_panel_source(source: str) -> str:
    marker = 'objectName: "operatorDashboardBlockCReadOnlyBindingSummary"'
    start = source.find(marker)
    assert start >= 0
    end = source.find("Components.PreviewCard", start + len(marker))
    assert end > start
    return source[start:end]


def test_block_c_bridge_property_is_qvariantmap_constant_and_copy_on_read() -> None:
    getter = _method("blockCReadOnlyBindingState")
    property_call = _decorator_call(getter, "Property")

    assert isinstance(property_call.args[0], ast.Constant)
    assert property_call.args[0].value == "QVariantMap"
    assert any(
        keyword.arg == "constant"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in property_call.keywords
    )
    assert not any(keyword.arg in {"notify", "fset"} for keyword in property_call.keywords)

    returns = [node for node in ast.walk(getter) if isinstance(node, ast.Return)]
    assert len(returns) == 1
    returned = returns[0].value
    assert isinstance(returned, ast.Call)
    assert isinstance(returned.func, ast.Name)
    assert returned.func.id == "dict"
    assert ast.unparse(returned.args[0]) == "self._block_c_read_only_binding_state"


def test_block_c_bridge_has_no_setter_slot_or_action_surface() -> None:
    bridge_class = _local_preview_state_bridge_class()
    source = BRIDGE_SOURCE.read_text(encoding="utf-8")
    getter_source = ast.unparse(_method("blockCReadOnlyBindingState"))

    assert "@blockCReadOnlyBindingState.setter" not in source
    assert "def setBlockCReadOnlyBindingState" not in source
    for node in bridge_class.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if "blockC" in node.name or "BlockC" in node.name:
            assert node.name == "blockCReadOnlyBindingState"
            assert not any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "Slot"
                for decorator in node.decorator_list
            )

    forbidden_tokens = (
        "submit",
        "execute",
        "export",
        "startRuntime",
        "stopRuntime",
        "runRuntime",
        "lifecycle",
        "command",
        "callback",
    )
    assert all(token not in getter_source for token in forbidden_tokens)


def test_block_c_qml_consumer_uses_bridge_value_helper_and_safe_fallbacks() -> None:
    source = QML_SOURCE.read_text(encoding="utf-8")
    panel = _block_c_panel_source(source)

    assert "function blockCReadOnlyBindingValue(key, fallback)" in source
    assert 'typedBridgeValue("blockCReadOnlyBindingState", null)' in source
    assert 'blockCReadOnlyBindingValue("integrationGateStatus", "blocked")' in panel
    assert 'blockCReadOnlyBindingValue("runtimeLoopStarted", false)' in panel
    assert 'blockCReadOnlyBindingValue("exportSink", "none")' in panel
    assert 'blockCReadOnlyBindingValue("generatedOrderCount", 0)' in panel

    forbidden_action_tokens = (
        "Button",
        "onClicked",
        "MouseArea",
        "submit",
        "execute",
        "exportButton",
        "exportHandler",
        "startRuntime",
        "stopRuntime",
        "runRuntime",
        "command",
        "lifecycle",
    )
    offenders = [token for token in forbidden_action_tokens if token in panel]
    assert offenders == []
