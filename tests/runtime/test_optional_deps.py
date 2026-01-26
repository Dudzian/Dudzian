from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest


class _DummyModule(types.ModuleType):
    def __getattr__(self, name: str):
        if name and name[0].isupper():
            return type(name, (), {})
        raise AttributeError(name)


def _reload_module(name: str):
    sys.modules.pop(name, None)
    return importlib.import_module(name)


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_exchanges_missing_dependency_is_scoped(monkeypatch) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, *args, **kwargs):
        if name == "bot_core.exchanges.binance.spot":
            raise ModuleNotFoundError(name="ccxt")
        if name == "bot_core.exchanges.kraken.spot":
            return _DummyModule(name)
        if name.startswith("bot_core.exchanges."):
            return _DummyModule(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    ex = _reload_module("bot_core.exchanges")
    assert isinstance(ex.KrakenSpotAdapter, type)

    with pytest.raises(RuntimeError) as excinfo:
        ex.BinanceSpotAdapter()

    assert "ccxt" in str(excinfo.value)
    assert "bot_core.exchanges.binance.spot:BinanceSpotAdapter" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)


def test_exchanges_does_not_mask_non_import_errors(monkeypatch) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, *args, **kwargs):
        if name == "bot_core.exchanges.binance.spot":
            raise AttributeError("bug inside adapter")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    sys.modules.pop("bot_core.exchanges", None)

    with pytest.raises(AttributeError):
        importlib.import_module("bot_core.exchanges")


def test_bootstrap_import_tolerates_missing_adapter_dependency(monkeypatch) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, *args, **kwargs):
        if name == "bot_core.exchanges.binance.spot":
            raise ModuleNotFoundError(name="ccxt")
        if name.startswith("bot_core.exchanges."):
            return _DummyModule(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    sys.modules.pop("bot_core.runtime.bootstrap", None)
    importlib.import_module("bot_core.runtime.bootstrap")


@pytest.mark.parametrize(
    "module_name",
    [
        "bot_core.runtime.bootstrap",
        "bot_core.runtime.metadata",
        "bot_core.observability.io",
    ],
)
def test_require_yaml_preserves_cause(monkeypatch, module_name: str) -> None:
    real_import = importlib.import_module

    def fake_import(name: str, *args, **kwargs):
        if name == "yaml":
            raise ModuleNotFoundError(name="yaml")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", fake_import)

    if module_name == "bot_core.observability.io":
        module_path = Path(__file__).resolve().parents[2] / "bot_core" / "observability" / "io.py"
        module = _load_module_from_path(module_name, module_path)
    else:
        module = _reload_module(module_name)

    with pytest.raises(RuntimeError) as excinfo:
        module._require_yaml()

    assert "PyYAML" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)
