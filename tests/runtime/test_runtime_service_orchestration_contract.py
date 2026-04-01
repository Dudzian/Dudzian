from __future__ import annotations

import ast
from pathlib import Path


RUNTIME_SERVICE_PATH = Path("ui/backend/runtime_service.py")


def _runtime_service_class() -> ast.ClassDef:
    tree = ast.parse(RUNTIME_SERVICE_PATH.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "RuntimeService":
            return node
    raise AssertionError("RuntimeService class not found")


def _method_node(class_node: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in class_node.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"{name} method not found")


def _attr_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def test_finalize_grpc_activation_contract_order() -> None:
    runtime_service = _runtime_service_class()
    method = _method_node(runtime_service, "_finalize_grpc_activation")

    ops: list[str] = []
    for statement in method.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
            call = statement.value
            target = _attr_name(call.func)
            if target == "_activate_source_state":
                ops.append("activate_source_state")
            elif target == "emit":
                owner = (
                    _attr_name(call.func.value) if isinstance(call.func, ast.Attribute) else None
                )
                if owner == "errorMessageChanged":
                    ops.append("error_message_emit")
                elif owner == "liveSourceChanged":
                    ops.append("live_source_emit")
            elif target == "_mark_feed_disconnected":
                ops.append("mark_feed_disconnected")
            elif target == "_update_feed_health":
                ops.append("update_feed_health")
        elif len(statement.targets) == 1 and isinstance(statement, ast.Assign):
            target = _attr_name(statement.targets[0])
            if target == "_error_message":
                ops.append("reset_error_message")
            elif target == "_feed_reconnects":
                ops.append("reset_reconnects")
            elif target == "_feed_last_error":
                ops.append("reset_last_error")

    assert ops == [
        "activate_source_state",
        "reset_error_message",
        "error_message_emit",
        "live_source_emit",
        "reset_reconnects",
        "reset_last_error",
        "mark_feed_disconnected",
        "update_feed_health",
    ]


def test_grpc_paths_delegate_to_finalize_activation() -> None:
    runtime_service = _runtime_service_class()
    attach = _method_node(runtime_service, "attachToLiveDecisionLog")
    auto_connect = _method_node(runtime_service, "_auto_connect_grpc")

    attach_calls_finalize = any(
        isinstance(node, ast.Call) and _attr_name(node.func) == "_finalize_grpc_activation"
        for node in ast.walk(attach)
    )
    auto_connect_calls_finalize = any(
        isinstance(node, ast.Call) and _attr_name(node.func) == "_finalize_grpc_activation"
        for node in ast.walk(auto_connect)
    )

    assert attach_calls_finalize is True
    assert auto_connect_calls_finalize is True
