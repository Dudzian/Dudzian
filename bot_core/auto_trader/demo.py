"""Helpery demowe dla AI Governora wykorzystywane przez UI i skrypty."""

from __future__ import annotations

from typing import Mapping

from bot_core.ai.regime import MarketRegime
from bot_core.auto_trader.ai_governor import AutoTraderAIGovernor, AutoTraderAIGovernorRunner
from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds
from bot_core.decision.orchestrator import DecisionOrchestrator


def _demo_config() -> DecisionEngineConfig:
    thresholds = DecisionOrchestratorThresholds(
        max_cost_bps=18.0,
        min_net_edge_bps=5.0,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.08,
        max_position_ratio=0.4,
        max_open_positions=6,
        max_latency_ms=320.0,
    )
    return DecisionEngineConfig(
        orchestrator=thresholds,
        profile_overrides={},
        stress_tests=None,
        min_probability=0.55,
        require_cost_data=False,
        penalty_cost_bps=0.0,
    )


def _seed_performance(orchestrator: DecisionOrchestrator) -> None:
    orchestrator.record_strategy_performance(
        "scalping_alpha",
        MarketRegime.TREND,
        hit_rate=0.74,
        pnl=14.0,
        sharpe=1.05,
    )
    orchestrator.record_strategy_performance(
        "defensive_grid",
        MarketRegime.DAILY,
        hit_rate=0.62,
        pnl=7.5,
        sharpe=0.65,
    )
    orchestrator.record_strategy_performance(
        "hedge_guardian",
        MarketRegime.MEAN_REVERSION,
        hit_rate=0.58,
        pnl=4.0,
        sharpe=0.25,
    )
    orchestrator.record_strategy_performance(
        "trend_balanced",
        MarketRegime.TREND,
        hit_rate=0.68,
        pnl=9.0,
        sharpe=0.82,
    )


def build_demo_ai_governor_snapshot(history_limit: int = 12) -> Mapping[str, object]:
    """Generuje snapshot demonstracyjny bazujący na nowym runnerze."""

    orchestrator = DecisionOrchestrator(_demo_config())
    _seed_performance(orchestrator)
    runner = AutoTraderAIGovernorRunner(
        orchestrator,
        governor=AutoTraderAIGovernor(history_limit=history_limit),
    )
    for mode in ("scalping", "hedge", "grid", "scalping"):
        runner.run_until(mode=mode, limit=3)
    return runner.snapshot()


__all__ = ["build_demo_ai_governor_snapshot"]
