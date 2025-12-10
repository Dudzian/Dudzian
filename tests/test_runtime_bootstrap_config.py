"""Testy hermetyzujące przygotowanie konfiguracji bootstrapu runtime."""

from pathlib import Path

import pytest

from bot_core.runtime.bootstrap import prepare_runtime_bootstrap_config
from bot_core.security import SecretManager


class _DummySecretManager(SecretManager):
    def __init__(self) -> None:
        # W testach nie korzystamy z magazynu, więc inicjalizujemy minimalnie.
        self._storage = None  # type: ignore[assignment]
        self._namespace = "tests"


def test_prepare_runtime_bootstrap_config_uses_environ(tmp_path: Path) -> None:
    config_path = tmp_path / "core.yaml"
    environ = {"DUDZIAN_CORE_CONFIG": str(config_path)}

    cfg = prepare_runtime_bootstrap_config(
        secret_manager=_DummySecretManager(),
        environ=environ,
    )

    assert cfg.config_path == config_path


def test_prepare_runtime_bootstrap_config_requires_secret_manager() -> None:
    with pytest.raises(ValueError):
        prepare_runtime_bootstrap_config(secret_manager=None)


def test_prepare_runtime_bootstrap_config_preserves_explicit_path(tmp_path: Path) -> None:
    explicit = tmp_path / "custom.yaml"

    cfg = prepare_runtime_bootstrap_config(
        config_path=explicit,
        secret_manager=_DummySecretManager(),
    )

    assert cfg.config_path == explicit
