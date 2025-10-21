"""Regression tests ensuring legacy namespaces reuse canonical implementations."""

from __future__ import annotations

import importlib
from importlib import util


def test_ai_manager_reexports_canonical_symbols() -> None:
    legacy = importlib.import_module("KryptoLowca.ai_manager")
    modern = importlib.import_module("bot_core.ai.manager")

    assert legacy.AIManager is modern.AIManager
    assert legacy.__all__ == list(getattr(modern, "__all__", ()))


def test_event_emitter_adapter_uses_modern_bus() -> None:
    legacy = importlib.import_module("KryptoLowca.event_emitter_adapter")
    modern = importlib.import_module("bot_core.events.emitter")

    assert legacy.EventBus is modern.EventBus
    assert legacy.Event is modern.Event


def test_strategy_data_provider_aliases_protocol() -> None:
    legacy_engine = importlib.import_module("KryptoLowca.strategies.base.engine")
    modern_engine = importlib.import_module("bot_core.backtest.engine")

    assert issubclass(legacy_engine.DataProvider, modern_engine.DataProviderProtocol)


def test_legacy_archive_package_removed() -> None:
    assert util.find_spec("archive.legacy_bot") is None
