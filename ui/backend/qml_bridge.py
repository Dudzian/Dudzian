from __future__ import annotations

import dataclasses
from collections import deque
from collections.abc import Mapping, Sequence
from decimal import Decimal
from numbers import Number
from typing import Any

try:  # pragma: no cover - PySide6 może nie być dostępne w środowiskach light
    from PySide6.QtQml import QJSValue  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - defensywne obejście braku Qt
    QJSValue = None  # type: ignore[assignment]

_PRIMITIVES = (str, type(None))


def _from_qjsvalue(value: object) -> object:
    """Konwertuje QJSValue na wariant Pythona, gdy to możliwe."""

    if QJSValue is None or not isinstance(value, QJSValue):
        return value
    try:
        if value.isNull() or value.isUndefined():  # type: ignore[attr-defined]
            return None
        variant = value.toVariant()  # type: ignore[attr-defined]
    except Exception:
        try:
            variant = value.toString()  # type: ignore[attr-defined]
        except Exception:
            return None
    return variant


def to_plain_value(value: object) -> object:
    """Rekurencyjnie sprowadza obiekt do prymitywów Pythona/JSON."""

    value = _from_qjsvalue(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, Number):
        coerced = _coerce_number(value)
        if coerced is not None:
            return coerced
    if isinstance(value, _PRIMITIVES):
        return value
    if isinstance(value, Mapping):
        return {str(key): to_plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset, deque)):
        return [to_plain_value(item) for item in value]
    if dataclasses.is_dataclass(value):
        return {str(key): to_plain_value(item) for key, item in dataclasses.asdict(value).items()}
    if hasattr(value, "model_dump"):
        try:
            return to_plain_value(value.model_dump())  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return to_plain_value(value.dict())  # type: ignore[attr-defined]
        except Exception:
            pass
    if hasattr(value, "_asdict"):
        try:
            return to_plain_value(value._asdict())  # type: ignore[attr-defined]
        except Exception:
            pass
    return str(value)


def _coerce_number(value: Number) -> int | float | None:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return float(value)
    if isinstance(value, Decimal):
        try:
            if value == value.to_integral_value():
                return int(value)
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return None
    if hasattr(value, "is_integer"):
        try:
            if value.is_integer():  # type: ignore[call-arg]
                return int(value)
        except Exception:
            pass
    try:
        if value == int(value):
            return int(value)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return None


def to_plain_dict(value: object) -> dict[str, object]:
    """Zapewnia, że wynik jest dictem gotowym do QML."""

    plain = to_plain_value(value)
    if isinstance(plain, dict):
        return plain
    if isinstance(plain, Mapping):
        return {str(key): to_plain_value(item) for key, item in plain.items()}
    return {}


def to_plain_list(value: object) -> list[object]:
    """Zapewnia, że wynik jest listą gotową do QML."""

    plain = to_plain_value(value)
    if isinstance(plain, list):
        return plain
    if isinstance(plain, Sequence) and not isinstance(plain, (str, bytes, bytearray)):
        return [to_plain_value(item) for item in plain]
    return []


def to_plain_text(value: object) -> str:
    """Konwertuje dowolną wartość do tekstu, zwracając pusty string dla None."""

    plain = to_plain_value(value)
    if plain is None:
        return ""
    if isinstance(plain, str):
        return plain
    return str(plain)


__all__ = ["to_plain_value", "to_plain_dict", "to_plain_list", "to_plain_text"]
