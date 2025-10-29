from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pytest
import yaml

from bot_core.exchanges import manager as manager_module
from bot_core.exchanges.manager import ExchangeManager, Mode


_ENV_PATTERN = re.compile(r"\$\{([^}]+)\}")


def _collect_env_names(payload: object) -> Iterable[str]:
    if isinstance(payload, str):
        yield from _ENV_PATTERN.findall(payload)
    elif isinstance(payload, dict):
        for value in payload.values():
            yield from _collect_env_names(value)
    elif isinstance(payload, (list, tuple, set)):
        for item in payload:
            yield from _collect_env_names(item)


def test_exchange_profiles_require_all_modes(tmp_path: Path) -> None:
    manager_module._EXCHANGE_PROFILE_CACHE.clear()
    config = {
        "paper": {
            "exchange_manager": {"mode": "paper"},
            "credentials": {"api_key": None, "secret": None},
        },
        "live": {
            "exchange_manager": {"mode": "spot", "testnet": False},
            "credentials": {"api_key": "a", "secret": "b"},
        },
    }
    path = tmp_path / "foo.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="testnet"):
        manager_module._load_exchange_profiles("foo", config_dir=tmp_path)


def test_exchange_profiles_require_boolean_testnet(tmp_path: Path) -> None:
    manager_module._EXCHANGE_PROFILE_CACHE.clear()
    config = {
        "paper": {
            "exchange_manager": {"mode": "paper"},
            "credentials": {"api_key": None, "secret": None},
        },
        "testnet": {
            "exchange_manager": {"mode": "spot", "testnet": "yes"},
            "credentials": {"api_key": "x", "secret": "y"},
        },
        "live": {
            "exchange_manager": {"mode": "spot", "testnet": False},
            "credentials": {"api_key": "a", "secret": "b"},
        },
    }
    path = tmp_path / "foo.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(ValueError, match="boolowskiego pola 'testnet'"):
        manager_module._load_exchange_profiles("foo", config_dir=tmp_path)


@pytest.mark.parametrize("config_path", sorted(Path("config/exchanges").glob("*.yaml")))
def test_exchange_manager_applies_all_profiles(
    config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manager_module._EXCHANGE_PROFILE_CACHE.clear()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    for env_name in _collect_env_names(payload):
        monkeypatch.setenv(env_name, f"stub-{env_name.lower()}")

    exchange_id = config_path.stem
    manager = ExchangeManager(exchange_id)

    manager.apply_environment_profile("paper", config_dir=config_path.parent)
    assert manager.mode is Mode.PAPER

    manager.apply_environment_profile("testnet", config_dir=config_path.parent)
    assert manager.mode in {Mode.SPOT, Mode.MARGIN, Mode.FUTURES}
    assert manager._testnet is True

    manager.apply_environment_profile("live", config_dir=config_path.parent)
    assert manager.mode in {Mode.SPOT, Mode.MARGIN, Mode.FUTURES}
    assert manager._testnet is False

    profile = manager.describe_environment_profile()
    assert profile is not None
    assert profile.get("name") == "live"
