"""Testy sanity-check dla środowiska bundla desktopowego."""
from __future__ import annotations

import importlib

import pytest

DESKTOP_MODULES = [
    "bot_core.ai.manager",
    "bot_core.strategies.cross_exchange_arbitrage",
    "bot_core.reporting.ui_bridge",
]


@pytest.mark.parametrize("module_name", DESKTOP_MODULES)
def test_desktop_modules_import(module_name: str) -> None:
    module = importlib.import_module(module_name)
    assert module is not None, f"Nie udało się zaimportować modułu {module_name}"


def test_numeric_stack_is_available() -> None:
    numpy = importlib.import_module("numpy")
    pandas = importlib.import_module("pandas")
    joblib = importlib.import_module("joblib")

    assert hasattr(numpy, "__version__"), "Brak atrybutu wersji w numpy"
    assert hasattr(pandas, "__version__"), "Brak atrybutu wersji w pandas"
    assert hasattr(joblib, "__version__"), "Brak atrybutu wersji w joblib"
