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


def _load_runtime_module_with_blocked_imports(
    module_name: str,
    path: Path,
    *,
    blocked_prefixes: tuple[str, ...],
    blocked_exc_factory=None,  # type: ignore[no-untyped-def]
):
    import builtins

    real_import = builtins.__import__
    factory = blocked_exc_factory or (lambda name: ModuleNotFoundError(name=name))

    def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):  # type: ignore[no-untyped-def]
        if any(name.startswith(prefix) for prefix in blocked_prefixes):
            raise factory(name)
        return real_import(name, globals, locals, fromlist, level)

    builtins.__import__ = blocked_import
    try:
        return _load_module_from_path(module_name, path)
    finally:
        builtins.__import__ = real_import


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
    pytest.importorskip("httpx")
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
    ("module_name", "relative_path"),
    [
        ("bot_core.runtime.metrics_alerts_import_probe", "bot_core/runtime/metrics_alerts.py"),
        ("bot_core.runtime.observability_import_probe", "bot_core/runtime/observability.py"),
    ],
)
def test_runtime_observability_modules_import_without_security_dependency(
    module_name: str,
    relative_path: str,
) -> None:
    module_path = Path(__file__).resolve().parents[2] / relative_path
    module = _load_runtime_module_with_blocked_imports(
        module_name,
        module_path,
        blocked_prefixes=("bot_core.security",),
    )
    assert module is not None


def test_runtime_metadata_import_without_risk_security_chain() -> None:
    module_path = Path(__file__).resolve().parents[2] / "bot_core" / "runtime" / "metadata.py"
    module = _load_runtime_module_with_blocked_imports(
        "bot_core.runtime.metadata_import_probe",
        module_path,
        blocked_prefixes=("bot_core.risk.settings", "bot_core.security"),
    )
    assert module is not None


def test_runtime_bootstrap_import_without_optional_adapter_security_chain() -> None:
    module_path = Path(__file__).resolve().parents[2] / "bot_core" / "runtime" / "bootstrap.py"
    module = _load_runtime_module_with_blocked_imports(
        "bot_core.runtime.bootstrap_import_probe",
        module_path,
        blocked_prefixes=(
            "bot_core.exchanges.nowa_gielda",
            "bot_core.exchanges.testing.loopback",
            "bot_core.security",
        ),
    )
    assert module is not None


def test_runtime_bootstrap_expected_missing_dependency_matches_nested_prefixes() -> None:
    module_path = Path(__file__).resolve().parents[2] / "bot_core" / "runtime" / "bootstrap.py"
    module = _load_module_from_path("bot_core.runtime.bootstrap_dependency_probe", module_path)

    nested = ModuleNotFoundError(name="cryptography.hazmat.primitives")
    assert module._is_expected_missing_dependency(  # type: ignore[attr-defined]
        nested,
        allowed_prefixes=("cryptography",),
    )
    assert not module._is_expected_missing_dependency(  # type: ignore[attr-defined]
        nested,
        allowed_prefixes=("bot_core.security",),
    )


def test_runtime_bootstrap_fallbacks_raise_diagnostic_runtime_errors() -> None:
    module_path = Path(__file__).resolve().parents[2] / "bot_core" / "runtime" / "bootstrap.py"
    module = _load_runtime_module_with_blocked_imports(
        "bot_core.runtime.bootstrap_fallback_probe",
        module_path,
        blocked_prefixes=(
            "bot_core.exchanges.nowa_gielda",
            "bot_core.exchanges.testing.loopback",
            "bot_core.security",
        ),
    )

    with pytest.raises(RuntimeError, match="NowaGieldaSpotAdapter") as nowa_exc:
        module.NowaGieldaSpotAdapter()
    message = str(nowa_exc.value)
    assert "optional dependency" in message
    assert "bot_core.exchanges.nowa_gielda -> httpx" in message

    with pytest.raises(RuntimeError, match="build_service_token_validator") as validator_exc:
        module.build_service_token_validator()
    validator_message = str(validator_exc.value)
    assert "optional dependency" in validator_message
    assert "bot_core.security" in validator_message


def test_runtime_bootstrap_does_not_mask_non_optional_import_errors() -> None:
    module_path = Path(__file__).resolve().parents[2] / "bot_core" / "runtime" / "bootstrap.py"

    def _runtime_error(_name: str) -> Exception:
        return RuntimeError("synthetic import bug")

    with pytest.raises(RuntimeError, match="synthetic import bug"):
        _load_runtime_module_with_blocked_imports(
            "bot_core.runtime.bootstrap_non_optional_bug_probe",
            module_path,
            blocked_prefixes=("bot_core.exchanges.nowa_gielda",),
            blocked_exc_factory=_runtime_error,
        )


@pytest.mark.parametrize(
    "module_name",
    [
        "bot_core.runtime.bootstrap",
        "bot_core.runtime.metadata",
        "bot_core.observability.io",
    ],
)
def test_require_yaml_preserves_cause(monkeypatch, module_name: str) -> None:
    if module_name == "bot_core.runtime.bootstrap":
        pytest.importorskip("httpx")
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
