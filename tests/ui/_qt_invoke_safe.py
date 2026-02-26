from __future__ import annotations

import json
import sys
from typing import Any


class _InvokeSafeVariantWrapper:
    def __init__(self, payload: Any, *, wrap_key: str | None = None) -> None:
        self._payload = payload
        self._wrap_key = wrap_key

    def toVariant(self) -> Any:
        if isinstance(self._payload, dict):
            payload: Any = dict(self._payload)
        else:
            payload = self._payload
        if self._wrap_key:
            return {self._wrap_key: payload}
        return payload


def _is_qjsvalue_instance(value: Any) -> bool:
    try:
        from PySide6.QtQml import QJSValue  # type: ignore[attr-defined]

        return isinstance(value, QJSValue)
    except Exception:
        pass
    return value.__class__.__name__ == "QJSValue"


def invoke_safe_variant(arg: Any, *, wrap_dict_in_key: str | None = None) -> Any:
    """Return a Windows-safe QVariant payload for Python slot invocation.

    Rule of thumb:
    - QML method invocation on win32 should prefer QJSValue via invoke_safe_qml_variant().
    - Python slot invocation may pass dict through this wrapper (toVariant()).
    - Never pass raw dict into Q_ARG("QVariant"|"QVariantMap", ...) on win32.
    """
    if sys.platform != "win32":
        return arg
    if _is_qjsvalue_instance(arg):
        return arg
    if isinstance(arg, dict):
        return _InvokeSafeVariantWrapper(arg, wrap_key=wrap_dict_in_key)
    return arg


def invoke_safe_qml_variant(engine: Any, arg: Any) -> Any:
    """Return a Windows-safe QVariant payload for QML method invocation.

    On win32, dict payloads are converted to QJSValue via QQml/QJS engine methods
    so QML receives native JS objects instead of opaque Python wrappers.
    """
    if sys.platform != "win32":
        return arg
    if _is_qjsvalue_instance(arg):
        return arg

    qml_engine = engine
    get_engine = getattr(engine, "engine", None)
    if callable(get_engine):
        try:
            qml_engine = get_engine()
        except Exception:
            qml_engine = engine

    if isinstance(arg, dict):
        to_script_value = getattr(qml_engine, "toScriptValue", None)
        if callable(to_script_value):
            try:
                return to_script_value(dict(arg))
            except Exception:
                pass

        evaluate = getattr(qml_engine, "evaluate", None)
        if callable(evaluate):
            try:
                evaluated = evaluate(f"({json.dumps(arg)})")
                if _is_qjsvalue_instance(evaluated):
                    return evaluated
            except Exception:
                pass

    return invoke_safe_variant(arg)


def assert_has_overload(qobj: Any, signature: str) -> None:
    meta = qobj.metaObject()
    idx = meta.indexOfMethod(signature.encode())
    assert idx != -1, f"Brak overloadu {signature} w metaObject()"
