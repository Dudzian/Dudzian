from bot_core.strategies.base import MarketSnapshot
from bot_core.strategies.options import OptionsIncomeSettings, OptionsIncomeStrategy


def _snapshot(iv: float, delta: float, days: int, price: float, ts: int) -> MarketSnapshot:
    return MarketSnapshot(
        symbol="AAPL",
        timestamp=ts,
        open=price,
        high=price,
        low=price,
        close=price,
        volume=1_000.0,
        indicators={"option_iv": iv, "option_delta": delta, "days_to_expiry": days},
    )


def test_options_income_strategy_enter_and_exit() -> None:
    strategy = OptionsIncomeStrategy(
        OptionsIncomeSettings(min_iv=0.3, max_delta=0.4, min_days_to_expiry=10, roll_threshold_iv=0.2)
    )

    enter = strategy.on_data(_snapshot(iv=0.35, delta=0.25, days=14, price=100.0, ts=1))
    assert enter and enter[0].side == "sell_call"
    assert enter[0].metadata["strategy"]["risk_label"] == "income"

    exit_signal = strategy.on_data(_snapshot(iv=0.15, delta=0.2, days=5, price=101.0, ts=2))
    assert exit_signal and exit_signal[0].side == "buy_to_close"
