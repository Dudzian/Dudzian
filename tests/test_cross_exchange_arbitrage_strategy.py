from bot_core.strategies.cross_exchange_arbitrage import (
    CrossExchangeArbitrageSettings,
    CrossExchangeArbitrageStrategy,
)
from tests.fixtures import build_cross_exchange_fixture


def test_cross_exchange_arbitrage_entry_and_exit() -> None:
    settings = CrossExchangeArbitrageSettings(
        primary_exchange="binance_spot",
        secondary_exchange="kraken_spot",
        spread_entry=0.001,
        spread_exit=0.0005,
        max_notional=10_000,
        max_open_seconds=60,
    )
    strategy = CrossExchangeArbitrageStrategy(settings)

    fixtures = build_cross_exchange_fixture()

    signals = strategy.on_data(fixtures.entry)
    assert signals and signals[0].side == "long_primary_short_secondary"

    exit_signals = strategy.on_data(fixtures.exit)
    assert exit_signals and exit_signals[0].side.startswith("close_")


def test_cross_exchange_arbitrage_time_stop() -> None:
    settings = CrossExchangeArbitrageSettings(
        primary_exchange="binance_spot",
        secondary_exchange="kraken_spot",
        spread_entry=0.001,
        spread_exit=0.0005,
        max_notional=10_000,
        max_open_seconds=10,
    )
    strategy = CrossExchangeArbitrageStrategy(settings)

    fixtures = build_cross_exchange_fixture()

    strategy.on_data(fixtures.entry)

    # brak zamknięcia spreadu, ale przekroczono maksymalny czas
    late_signals = strategy.on_data(fixtures.timeout)
    assert late_signals and late_signals[0].side.startswith("close_")
