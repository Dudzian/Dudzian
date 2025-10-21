"""Regression tests ensuring legacy namespaces reuse canonical implementations."""

from __future__ import annotations

import importlib


def test_ai_manager_reexports_canonical_symbols() -> None:
    legacy = importlib.import_module("KryptoLowca.ai_manager")
    modern = importlib.import_module("bot_core.ai.manager")

    assert legacy.AIManager is modern.AIManager
    assert legacy.__all__ == list(getattr(modern, "__all__", ()))


def test_event_emitter_adapter_uses_modern_bus() -> None:
    legacy = importlib.import_module("archive.legacy_bot.event_emitter_adapter")
    modern = importlib.import_module("bot_core.events.emitter")

    assert legacy.EventBus is modern.EventBus
    assert legacy.Event is modern.Event


def test_exchange_core_wrappers_delegate_to_bot_core() -> None:
    legacy = importlib.import_module("archive.legacy_bot.managers.exchange_core")
    modern = importlib.import_module("bot_core.exchanges.core")

    assert legacy.BaseBackend is modern.BaseBackend
    assert legacy.OrderDTO is modern.OrderDTO


def test_exchange_manager_wrapper_shares_class() -> None:
    legacy = importlib.import_module("archive.legacy_bot.managers.exchange_manager")
    modern = importlib.import_module("bot_core.exchanges.manager")

    assert legacy.ExchangeManager is modern.ExchangeManager


def test_strategy_data_provider_aliases_protocol() -> None:
    legacy_engine = importlib.import_module("KryptoLowca.strategies.base.engine")
    modern_engine = importlib.import_module("bot_core.backtest.engine")

    assert issubclass(legacy_engine.DataProvider, modern_engine.DataProviderProtocol)
