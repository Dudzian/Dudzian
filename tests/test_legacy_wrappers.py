"""Regression tests ensuring legacy namespaces reuse canonical implementations."""

from __future__ import annotations

import importlib

import pytest


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
    modern = importlib.import_module("KryptoLowca.exchange_manager")

    assert legacy.ExchangeManager is modern.ExchangeManager


def test_strategy_data_provider_aliases_protocol() -> None:
    legacy_engine = importlib.import_module("KryptoLowca.strategies.base.engine")
    modern_engine = importlib.import_module("bot_core.backtest.engine")

    assert issubclass(legacy_engine.DataProvider, modern_engine.DataProviderProtocol)


@pytest.mark.parametrize(
    ("legacy_module", "canonical_module", "symbol"),
    [
        ("archive.legacy_bot.auto_trader", "bot_core.auto_trader.app", "AutoTrader"),
        ("archive.legacy_bot.managers.ai_manager", "KryptoLowca.ai_manager", "AIManager"),
        ("archive.legacy_bot.managers.config_manager", "KryptoLowca.config_manager", "ConfigManager"),
        ("archive.legacy_bot.managers.database_manager", "KryptoLowca.database_manager", "DatabaseManager"),
        ("archive.legacy_bot.managers.exchange_manager", "KryptoLowca.exchange_manager", "ExchangeManager"),
        ("archive.legacy_bot.managers.security_manager", "KryptoLowca.security_manager", "SecurityManager"),
        (
            "archive.legacy_bot.trading_strategies",
            "KryptoLowca.trading_strategies.engine",
            "TradingEngine",
        ),
        (
            "archive.legacy_bot.managers.live_exchange_ccxt",
            "bot_core.exchanges.ccxt_adapter",
            "CCXTSpotAdapter",
        ),
    ],
)
def test_archive_legacy_proxies_surface_canonical_symbols(
    legacy_module: str, canonical_module: str, symbol: str
) -> None:
    legacy = importlib.import_module(legacy_module)
    canonical = importlib.import_module(canonical_module)

    assert getattr(legacy, symbol) is getattr(canonical, symbol)
