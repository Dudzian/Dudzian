import asyncio
from datetime import datetime, timezone

import pytest

from bot_core.config.models import (
    PortfolioGovernorConfig,
    PortfolioGovernorScoringWeights,
    PortfolioGovernorStrategyConfig,
)
from bot_core.portfolio import PortfolioGovernor
from bot_core.runtime.multi_strategy_scheduler import (
    MultiStrategyScheduler,
    StrategyDataFeed,
    StrategySignalSink,
)
from bot_core.strategies.base import MarketSnapshot, StrategyEngine, StrategySignal


class _StaticStrategy(StrategyEngine):
    def __init__(self, confidences: tuple[float, ...]) -> None:
        self._confidences = confidences

    def warm_up(self, history: tuple[MarketSnapshot, ...]) -> None:  # pragma: no cover - deterministic
        return None

    def on_data(self, snapshot: MarketSnapshot) -> tuple[StrategySignal, ...]:
        return tuple(
            StrategySignal(
                symbol=snapshot.symbol,
                side="buy",
                confidence=value,
                metadata={"price": snapshot.close},
            )
            for value in self._confidences
        )


class _SingleSnapshotFeed(StrategyDataFeed):
    def __init__(self, price: float) -> None:
        self._snapshot = MarketSnapshot(
            symbol="BTCUSDT",
            timestamp=1,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=100.0,
        )

    def load_history(self, strategy_name: str, bars: int) -> tuple[MarketSnapshot, ...]:  # pragma: no cover - deterministic
        return ()

    def fetch_latest(self, strategy_name: str) -> tuple[MarketSnapshot, ...]:
        return (self._snapshot,)


class _CollectSink(StrategySignalSink):
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[StrategySignal, ...]]] = []

    def submit(
        self,
        *,
        strategy_name: str,
        schedule_name: str,
        risk_profile: str,
        timestamp: datetime,
        signals: tuple[StrategySignal, ...],
    ) -> None:
        self.calls.append((schedule_name, signals))


def test_scheduler_applies_portfolio_governor_allocations() -> None:
    governor = PortfolioGovernor(
        PortfolioGovernorConfig(
            enabled=True,
            rebalance_interval_minutes=0.0,
            smoothing=1.0,
            min_score_threshold=0.0,
            default_cost_bps=0.0,
            scoring=PortfolioGovernorScoringWeights(alpha=1.0, cost=0.0, slo=0.0, risk=0.0),
            strategies={
                "trend": PortfolioGovernorStrategyConfig(
                    baseline_weight=0.6,
                    min_weight=0.2,
                    max_weight=0.8,
                    baseline_max_signals=4,
                    max_signal_factor=1.5,
                ),
                "mean_reversion": PortfolioGovernorStrategyConfig(
                    baseline_weight=0.4,
                    min_weight=0.1,
                    max_weight=0.6,
                    baseline_max_signals=2,
                    max_signal_factor=1.2,
                ),
            },
            max_signal_floor=1,
        ),
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
    )

    scheduler = MultiStrategyScheduler(
        environment="demo",
        portfolio="core",
        clock=lambda: datetime(2024, 1, 1, tzinfo=timezone.utc),
        telemetry_emitter=None,
        decision_journal=None,
        portfolio_governor=governor,
    )

    sink_a = _CollectSink()
    sink_b = _CollectSink()

    scheduler.register_schedule(
        name="trend_sched",
        strategy_name="trend",
        strategy=_StaticStrategy((0.9, 0.85, 0.8)),
        feed=_SingleSnapshotFeed(101.0),
        sink=sink_a,
        cadence_seconds=10,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="balanced",
        max_signals=3,
    )

    scheduler.register_schedule(
        name="mr_sched",
        strategy_name="mean_reversion",
        strategy=_StaticStrategy((0.2, 0.1)),
        feed=_SingleSnapshotFeed(99.0),
        sink=sink_b,
        cadence_seconds=10,
        max_drift_seconds=1,
        warmup_bars=0,
        risk_profile="conservative",
        max_signals=2,
    )

    asyncio.run(scheduler.run_once())

    assert sink_a.calls and sink_b.calls
    decision = scheduler._last_portfolio_decision
    assert decision is not None

    trend_schedule = next(item for item in scheduler._schedules if item.strategy_name == "trend")
    mr_schedule = next(item for item in scheduler._schedules if item.strategy_name == "mean_reversion")

    trend_weight = governor.current_weights["trend"]
    mr_weight = governor.current_weights["mean_reversion"]

    assert trend_schedule.metrics["portfolio_weight"] == pytest.approx(trend_weight)
    assert trend_schedule.active_max_signals == 5
    assert mr_schedule.metrics["portfolio_weight"] == pytest.approx(mr_weight)
    assert mr_schedule.active_max_signals >= governor.min_signal_floor
