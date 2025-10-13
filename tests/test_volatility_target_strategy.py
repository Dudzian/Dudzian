from bot_core.strategies.volatility_target import (
    VolatilityTargetSettings,
    VolatilityTargetStrategy,
)
from tests.fixtures import build_volatility_series_fixture


def test_volatility_target_rebalances_when_threshold_exceeded() -> None:
    settings = VolatilityTargetSettings(
        target_volatility=0.05,
        lookback=5,
        rebalance_threshold=0.05,
        min_allocation=0.2,
        max_allocation=1.0,
        floor_volatility=0.01,
    )
    strategy = VolatilityTargetStrategy(settings)

    fixtures = build_volatility_series_fixture()
    strategy.warm_up(fixtures.history)

    signals = strategy.on_data(fixtures.volatile_tick)
    assert signals and signals[0].side == "rebalance"
    metadata = signals[0].metadata
    assert "target_allocation" in metadata
    assert metadata["target_allocation"] <= settings.max_allocation

    # kolejny pomiar bez istotnej zmiany nie powinien generować sygnału
    assert not strategy.on_data(fixtures.follow_up)
