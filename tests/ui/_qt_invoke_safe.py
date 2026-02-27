from __future__ import annotations

import contextlib
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


def _normalize_method_signature(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).decode(errors="ignore")
    with_data = getattr(value, "data", None)
    if callable(with_data):
        with_data_value = with_data()
        if isinstance(with_data_value, (bytes, bytearray)):
            return bytes(with_data_value).decode(errors="ignore")
    with contextlib.suppress(Exception):
        return bytes(value).decode(errors="ignore")
    return str(value)


def _binding_version() -> str:
    with contextlib.suppress(Exception):
        import PySide6  # type: ignore

        return getattr(PySide6, "__version__", "unknown")
    return "unavailable"


def _collect_method_candidates(qobj: Any, method_name: str) -> tuple[list[str], int]:
    candidates: list[str] = []
    scan_errors = 0
    meta = qobj.metaObject()
    for i in range(meta.methodCount()):
        try:
            sig = _normalize_method_signature(meta.method(i).methodSignature())
        except Exception:
            scan_errors += 1
            continue
        if sig.startswith(method_name + "("):
            candidates.append(sig)
    return candidates, scan_errors


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
    - For Q_ARG("QVariantMap", ...) on win32 prefer dispatch-only payloads
      (e.g. ``None``) in tests; avoid structured marshalling through invokeMethod.
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


def invoke_safe_qvariantmap(arg: Any) -> Any:
    """Return safest payload for Q_ARG("QVariantMap", ...) tests.

    Test-only helper for invokeMethod arguments; do not use in production paths.

    On win32, structured QVariantMap marshalling in invokeMethod may crash before
    entering Python slot logic. For dispatch/meta-call tests use ``None``.
    On non-win32 this function is pass-through.
    """
    if sys.platform == "win32":
        return None
    return arg


def assert_has_overload(qobj: Any, signature: str) -> None:
    """Assert that QObject exposes the exact method overload with robust diagnostics.

    Rule of thumb (Windows/PySide6):
    - validate overloads through metaObject() only,
    - never probe availability by calling invokeMethod with test payloads.
    """

    meta = qobj.metaObject()
    idx = -1
    try:
        # PySide6: indexOfMethod expects str
        idx = meta.indexOfMethod(str(signature))
    except TypeError:
        idx = -1

    if idx == -1:
        method_name = str(signature).split("(", 1)[0]
        candidates, _ = _collect_method_candidates(qobj, method_name)
        assert str(signature) in candidates, (
            "Brak overloadu w metaObject(); "
            f"target={signature!r}; "
            f"method={method_name!r}; "
            f"candidates={candidates!r}; "
            f"platform={sys.platform}; "
            f"binding=PySide6/{_binding_version()}"
        )


def assert_has_any_overload(qobj: Any, *signatures: str) -> None:
    """Assert that QObject exposes at least one of the provided overload signatures."""
    if not signatures:
        raise ValueError("Brak sygnatur do sprawdzenia; signatures=()")

    method_names = {str(signature).split("(", 1)[0] for signature in signatures}
    if len(method_names) != 1:
        raise ValueError(
            "Mieszane nazwy metod w signatures; "
            f"signatures={signatures!r}; methods={sorted(method_names)!r}"
        )

    errors: list[str] = []
    for signature in signatures:
        try:
            assert_has_overload(qobj, signature)
            return
        except AssertionError as exc:
            errors.append(str(exc))

    method_name = next(iter(method_names))
    candidates, scan_errors = _collect_method_candidates(qobj, method_name)

    details_preview = errors[:2]
    if len(errors) > 2:
        details_preview.append(f"... +{len(errors) - 2} more")

    raise AssertionError(
        "Brak oczekiwanego overloadu (żaden wariant nie pasuje); "
        f"method={method_name!r}; signatures={signatures!r}; "
        f"candidates={candidates!r}; scan_errors={scan_errors}; "
        f"platform={sys.platform}; binding=PySide6/{_binding_version()}; "
        f"details={details_preview!r}"
    )
