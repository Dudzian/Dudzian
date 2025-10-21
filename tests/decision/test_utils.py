"""Tests for decision engine utility helpers."""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path
from types import ModuleType

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = PROJECT_ROOT / "bot_core" / "decision" / "utils.py"


def _load_utils_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("bot_core.decision.utils", MODULE_PATH)
    assert spec and spec.loader, "Unable to create module spec for utils"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


coerce_float = _load_utils_module().coerce_float


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, 0.0),
        (1.5, 1.5),
        (" 2.5 ", 2.5),
        (None, None),
        ("", None),
        ("not-a-number", None),
    ],
)
def test_coerce_float(value: object, expected: float | None) -> None:
    result = coerce_float(value)
    if expected is None:
        assert result is None
    else:
        assert result is not None
        assert math.isclose(result, expected)
