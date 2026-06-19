"""Source-only guards for the BLOK D preview action dispatch contract."""

from __future__ import annotations

import ast
import builtins
import importlib
import sys
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = REPO_ROOT / "ui" / "pyside_app" / "preview_action_dispatch_contract.py"
MODULE_NAME = "ui.pyside_app.preview_action_dispatch_contract"

FORBIDDEN_IMPORT_ROOTS = {
    "PySide6",
    "QtQuick",
    "QtCore",
    "qml",
    "subprocess",
    "threading",
    "asyncio",
    "socket",
    "requests",
    "httpx",
    "TradingController",
    "DecisionEnvelope",
    "os",
    "dotenv",
    "keyring",
}

FORBIDDEN_IMPORTED_NAMES = {
    "PySide6",
    "QtQuick",
    "QtCore",
    "qml",
    "TradingController",
    "DecisionEnvelope",
    "load_dotenv",
    "keyring",
}

FORBIDDEN_QML_HANDLER_TOKENS = (
    "Button.onClicked",
    "onClicked:",
    "MouseArea",
    "Connections {",
)

FORBIDDEN_CALL_NAMES = {
    "dispatch_command",
    "execute_command",
    "start_runtime",
    "start_loop",
    "submit_order",
    "create_order",
    "place_order",
    "send_order",
    "fill_order",
    "load_dotenv",
}

FORBIDDEN_ATTRIBUTE_CALLS = {
    ("os", "getenv"),
    ("os", "environ"),
    ("socket", "socket"),
    ("socket", "create_connection"),
    ("requests", "get"),
    ("requests", "post"),
    ("httpx", "get"),
    ("httpx", "post"),
}

FORBIDDEN_EXECUTION_NAMES = {
    "live_adapter",
    "testnet_adapter",
    "export_to_cloud",
}

SAFE_REJECTION_LITERAL_TERMS = {
    "secret",
    "secrets",
    "credential",
    "credentials",
    "api_key",
    "account_balance_fetch",
    "secrets_blocked",
}


class _BlockedImport(RuntimeError):
    pass


def _source() -> str:
    return CONTRACT_PATH.read_text(encoding="utf-8")


def _tree() -> ast.Module:
    return ast.parse(_source())


def _call_name(node: ast.Call) -> str | None:
    if isinstance(node.func, ast.Name):
        return node.func.id
    if isinstance(node.func, ast.Attribute):
        return node.func.attr
    return None


def _attribute_owner_name(node: ast.Attribute) -> str | None:
    if isinstance(node.value, ast.Name):
        return node.value.id
    return None


def _literal_strings() -> list[str]:
    values: list[str] = []
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            values.append(node.value)
    return values


def test_contract_source_does_not_import_forbidden_modules_or_symbols() -> None:
    offenders: list[str] = []
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".", 1)[0]
                if root in FORBIDDEN_IMPORT_ROOTS or alias.name in FORBIDDEN_IMPORTED_NAMES:
                    offenders.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            root = module.split(".", 1)[0]
            imported_names = {alias.name for alias in node.names}
            if root in FORBIDDEN_IMPORT_ROOTS or module in FORBIDDEN_IMPORTED_NAMES:
                offenders.append(module)
            offenders.extend(sorted(imported_names & FORBIDDEN_IMPORTED_NAMES))

    assert offenders == []


def test_contract_source_does_not_add_qml_handlers_or_ui_wiring_tokens() -> None:
    source = _source()

    offenders = [token for token in FORBIDDEN_QML_HANDLER_TOKENS if token in source]
    assert offenders == []


def test_contract_source_does_not_read_env_or_secret_stores() -> None:
    offenders: list[str] = []
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call) and _call_name(node) == "load_dotenv":
            offenders.append("load_dotenv")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            owner = _attribute_owner_name(node.func)
            if (owner, node.func.attr) in {("os", "getenv"), ("keyring", "get_password")}:
                offenders.append(f"{owner}.{node.func.attr}")
        elif isinstance(node, ast.Attribute):
            owner = _attribute_owner_name(node)
            if (owner, node.attr) == ("os", "environ"):
                offenders.append("os.environ")
        elif isinstance(node, ast.Name) and node.id == "keyring":
            offenders.append("keyring")

    unsafe_literals = []
    for literal in _literal_strings():
        lowered = literal.lower()
        if any(term in lowered for term in SAFE_REJECTION_LITERAL_TERMS):
            continue
        if any(term in lowered for term in ("secret", "credential", "api_key")):
            unsafe_literals.append(literal)

    assert offenders == []
    assert unsafe_literals == []


def test_contract_source_does_not_contain_runtime_or_order_execution_paths() -> None:
    offenders: list[str] = []
    for node in ast.walk(_tree()):
        if isinstance(node, ast.Call):
            call_name = _call_name(node)
            if call_name in FORBIDDEN_CALL_NAMES:
                offenders.append(f"{call_name}(")
            if isinstance(node.func, ast.Attribute):
                owner = _attribute_owner_name(node.func)
                if (owner, node.func.attr) in FORBIDDEN_ATTRIBUTE_CALLS:
                    offenders.append(f"{owner}.{node.func.attr}(")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_EXECUTION_NAMES:
            offenders.append(node.id)

    assert offenders == []


def test_contract_can_import_without_pyside_io_network_env_or_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def guarded_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "PySide6" or name.startswith("PySide6."):
            raise _BlockedImport(name)
        return real_import(name, *args, **kwargs)

    def forbidden_side_effect(*args: object, **kwargs: object) -> None:
        raise AssertionError("contract import attempted a forbidden side effect")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(builtins, "open", forbidden_side_effect)
    monkeypatch.setattr("os.getenv", forbidden_side_effect)
    monkeypatch.setattr("socket.socket", forbidden_side_effect)
    monkeypatch.setattr("socket.create_connection", forbidden_side_effect)
    monkeypatch.delitem(sys.modules, MODULE_NAME, raising=False)

    module = importlib.import_module(MODULE_NAME)

    assert module.RUNTIME_MODE == "paper"
    assert module.ALLOWED_PAPER_RUNTIME_ACTIONS


def test_contract_rejection_literals_are_limited_to_safe_refusal_categories() -> None:
    literals = _literal_strings()
    sensitive_literals = [
        literal
        for literal in literals
        if any(term in literal.lower() for term in ("secret", "credential", "api_key"))
    ]

    assert sensitive_literals
    assert all(
        literal in SAFE_REJECTION_LITERAL_TERMS
        or literal == "export_cloud_secrets"
        or "does not import PySide/QML" in literal
        for literal in sensitive_literals
    )


def test_evidence_nested_mappings_are_immutable() -> None:
    from ui.pyside_app.preview_action_dispatch_contract import (
        build_paper_runtime_action_dispatch_contract,
    )

    evidence = build_paper_runtime_action_dispatch_contract("unexpected_action")

    assert not isinstance(evidence.boundary_checks, dict)
    assert not isinstance(evidence.rejected_actions, dict)
    assert not isinstance(evidence.boundary_checks, MappingProxyType)
    assert not isinstance(evidence.rejected_actions, MappingProxyType)

    with pytest.raises(TypeError):
        evidence.boundary_checks["fail_closed"] = False  # type: ignore[index]
    with pytest.raises(TypeError):
        evidence.rejected_actions["live_mode"] = ()  # type: ignore[index]

    reread = build_paper_runtime_action_dispatch_contract("unexpected_action")
    assert reread.boundary_checks["fail_closed"] is True
    assert reread.rejected_actions["live_mode"] == ("live", "prod", "production", "real_trading")
