import pandas as pd

from bot_core.backtest.engine import BacktestEngine
from bot_core.backtest.simulation import MatchingConfig


class _HoldSignal:
    action = "HOLD"
    size = None
    stop_loss = None
    take_profit = None


class _HoldStrategy:
    async def prepare(self, context, data_provider):
        return None

    async def handle_market_data(self, context, market_payload):
        return _HoldSignal()

    async def notify_fill(self, context, fill):
        return None

    async def shutdown(self):
        return None


def _build_context(payload):
    return type(
        "Context",
        (),
        {
            "symbol": payload["symbol"],
            "timeframe": payload["timeframe"],
            "portfolio_value": payload.get("portfolio_value", 0.0),
            "position": payload.get("position", 0.0),
            "timestamp": payload.get("timestamp"),
            "metadata": payload.get("metadata"),
            "extra": payload.get("extra", {}),
        },
    )()


def test_missing_required_data_emits_warning_and_metadata():
    data = pd.DataFrame({"close": [100.0, 101.0, 99.5], "volume": [1, 1, 1]})
    engine = BacktestEngine(
        strategy_factory=_HoldStrategy,
        context_builder=_build_context,
        data=data,
        symbol="BTC/USDT",
        timeframe="1h",
        initial_balance=1_000.0,
        matching=MatchingConfig(),
        metadata={"required_data": ["close", "open"]},
    )

    report = engine.run()

    assert any("Strategia wymaga danych" in warning for warning in report.warnings)
    assert report.strategy_metadata["required_data"] == ("close", "open")
    assert report.strategy_metadata["required_data_missing"] == ("open",)
    assert report.strategy_metadata["available_data"] == ("close", "volume")
