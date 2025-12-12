"""Testy strategii wyboru trybu egzekucji."""

from __future__ import annotations

import pytest

from bot_core.config.models import RuntimeExecutionLiveSettings, RuntimeExecutionSettings
from bot_core.execution.mode_policy import DEFAULT_MODE_SELECTOR, ExecutionModeSelector
from bot_core.exchanges.base import Environment as ExchangeEnvironment


class _Env:
    def __init__(self, environment: str | ExchangeEnvironment = "paper", offline: bool = False):
        self.environment = environment
        self.offline_mode = offline


def _settings(
    *,
    mode: str | None = "paper",
    live_enabled: bool | None = None,
    force_paper_when_offline: bool = False,
) -> RuntimeExecutionSettings:
    live_cfg = None
    if live_enabled is not None:
        live_cfg = RuntimeExecutionLiveSettings(enabled=live_enabled)
    return RuntimeExecutionSettings(
        default_mode=mode,
        live=live_cfg,
        force_paper_when_offline=force_paper_when_offline,
    )


def test_default_selector_keeps_existing_paper_fallback():
    selector = DEFAULT_MODE_SELECTOR
    env = _Env(environment="paper")

    assert selector.resolve(_settings(mode=None), env) == "paper"
    assert selector.resolve(_settings(mode="unknown"), env) == "paper"


def test_live_mode_requires_enabled_live_config():
    selector = ExecutionModeSelector()
    env = _Env(environment=ExchangeEnvironment.LIVE)

    with pytest.raises(ValueError):
        selector.resolve(_settings(mode="live", live_enabled=None), env)

    assert selector.resolve(_settings(mode="live", live_enabled=True), env) == "live"


def test_auto_mode_uses_environment_and_config():
    selector = ExecutionModeSelector()
    live_env = _Env(environment=ExchangeEnvironment.LIVE)
    paper_env = _Env(environment=ExchangeEnvironment.PAPER)

    assert selector.resolve(_settings(mode="auto", live_enabled=True), live_env) == "live"
    assert selector.resolve(_settings(mode="auto", live_enabled=True), paper_env) == "paper"


def test_offline_mode_forces_paper_when_enabled():
    selector = ExecutionModeSelector()
    env = _Env(environment=ExchangeEnvironment.LIVE, offline=True)

    assert selector.resolve(
        _settings(mode="live", live_enabled=True, force_paper_when_offline=True), env
    ) == "paper"
