from bot_core.strategies.base import MarketSnapshot
from bot_core.strategies.cross_exchange_arbitrage import (
    CrossExchangeArbitrageSettings,
    CrossExchangeArbitrageStrategy,
)


def _snapshot(
    *,
    primary_bid: float,
    primary_ask: float,
    secondary_bid: float,
    secondary_ask: float,
    timestamp: int,
) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="BTC_USDT_CROSS",
        timestamp=timestamp,
        open=primary_bid,
        high=primary_ask,
        low=primary_bid,
        close=primary_ask,
        volume=1_000_000.0,
        indicators={
            "primary_bid": primary_bid,
            "primary_ask": primary_ask,
            "secondary_bid": secondary_bid,
            "secondary_ask": secondary_ask,
            "secondary_timestamp": timestamp,
        },
    )


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

    entry_snapshot = _snapshot(
        primary_bid=100.0,
        primary_ask=100.2,
        secondary_bid=101.0,
        secondary_ask=101.2,
        timestamp=1_000,
    )
    signals = strategy.on_data(entry_snapshot)
    assert signals and signals[0].side == "long_primary_short_secondary"

    exit_snapshot = _snapshot(
        primary_bid=100.3,
        primary_ask=100.35,
        secondary_bid=100.37,
        secondary_ask=100.4,
        timestamp=1_040,
    )
    exit_signals = strategy.on_data(exit_snapshot)
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

    entry_snapshot = _snapshot(
        primary_bid=100.0,
        primary_ask=100.2,
        secondary_bid=101.0,
        secondary_ask=101.2,
        timestamp=1_000,
    )
    strategy.on_data(entry_snapshot)

    # brak zamknięcia spreadu, ale przekroczono maksymalny czas
    late_snapshot = _snapshot(
        primary_bid=100.1,
        primary_ask=100.3,
        secondary_bid=101.0,
        secondary_ask=101.2,
        timestamp=1_020,
    )
    late_signals = strategy.on_data(late_snapshot)
    assert late_signals and late_signals[0].side.startswith("close_")
