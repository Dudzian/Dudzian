import importlib
from pathlib import Path

import pytest

from bot_core.ai import backends


@pytest.fixture(autouse=True)
def _clear_backend_caches():
    backends.clear_backend_caches()
    yield
    backends.clear_backend_caches()


def test_is_backend_available_returns_false_when_module_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def _missing(name: str, package: str | None = None):  # pragma: no cover - monkeypatched
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", _missing)
    assert backends.is_backend_available("lightgbm") is False
    with pytest.raises(backends.BackendUnavailableError) as exc:
        backends.require_backend("lightgbm")
    assert "lightgbm" in str(exc.value)


def test_is_backend_available_uses_cached_import(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    original_import = importlib.import_module

    def _stub(name: str, package: str | None = None):
        if name == "lightgbm":
            return sentinel
        return original_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _stub)
    assert backends.is_backend_available("lightgbm") is True
    assert backends.require_backend("lightgbm") is sentinel


def test_get_backend_priority_from_custom_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = tmp_path / "backends.yml"
    config.write_text(
        """
priority:
  - linear
  - lightgbm
backends:
  linear:
    module: null
    optional: false
  lightgbm:
    module: lightgbm
    optional: true
""".strip()
    )
    monkeypatch.setattr(backends, "_DEFAULT_CONFIG_PATH", config)
    assert backends.get_backend_priority(config_path=config) == ("linear", "lightgbm")
    assert backends.is_backend_available("linear", config_path=config) is True
