from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from bot_core.ai.regime import MarketRegime
from bot_core.auto_trader.ai_governor import AIGovernorDecision, AutoTraderAIGovernorRunner
from bot_core.decision.orchestrator import StrategyPerformanceSummary


def _summary(
    *,
    regime: MarketRegime,
    hit_rate: float,
    pnl: float,
    sharpe: float,
) -> StrategyPerformanceSummary:
    return StrategyPerformanceSummary(
        strategy=f"{regime.value}_strategy",
        regime=regime,
        hit_rate=hit_rate,
        pnl=pnl,
        sharpe=sharpe,
        updated_at=datetime.now(timezone.utc),
        observations=5,
    )


def test_run_cycle_exposes_public_entrypoint() -> None:
    snapshot = {
        "scalping_alpha": _summary(regime=MarketRegime.TREND, hit_rate=0.8, pnl=10.0, sharpe=1.5)
    }
    orchestrator = SimpleNamespace(strategy_performance_snapshot=lambda: snapshot)

    runner = AutoTraderAIGovernorRunner(orchestrator)

    decision = runner.run_cycle()

    assert decision.mode == "scalping"
    telemetry = runner.snapshot()["telemetry"]
    assert telemetry["cycleMetrics"]["cycles_total"] == 1.0
    assert telemetry["decisionContract"]["path"] == "policy"
    assert telemetry["decisionContract"]["model"] is None


def test_run_until_stops_on_target_mode() -> None:
    snapshot = {
        "grid_balanced": _summary(regime=MarketRegime.DAILY, hit_rate=0.6, pnl=-10.0, sharpe=0.2),
        "hedge_guardian": _summary(
            regime=MarketRegime.MEAN_REVERSION, hit_rate=0.52, pnl=0.5, sharpe=0.1
        ),
    }
    orchestrator = SimpleNamespace(strategy_performance_snapshot=lambda: snapshot)

    runner = AutoTraderAIGovernorRunner(orchestrator)

    decisions = runner.run_until(mode="grid", limit=3)

    assert len(decisions) == 1
    assert decisions[0].mode == "grid"
    assert runner.snapshot()["telemetry"]["cycleMetrics"]["cycles_total"] == 1.0


def test_run_until_handles_empty_regime_cycle() -> None:
    snapshot = {
        "scalping_alpha": _summary(regime=MarketRegime.TREND, hit_rate=0.8, pnl=10.0, sharpe=1.5)
    }
    orchestrator = SimpleNamespace(strategy_performance_snapshot=lambda: snapshot)

    runner = AutoTraderAIGovernorRunner(orchestrator)

    decisions = runner.run_until(regimes=(), limit=2)

    assert len(decisions) == 1
    assert decisions[0].mode == "scalping"


def test_snapshot_decision_contract_uses_model_metadata_from_last_decision() -> None:
    snapshot = {
        "scalping_alpha": _summary(regime=MarketRegime.TREND, hit_rate=0.8, pnl=10.0, sharpe=1.5)
    }
    orchestrator = SimpleNamespace(strategy_performance_snapshot=lambda: snapshot)
    runner = AutoTraderAIGovernorRunner(orchestrator)
    runner.run_cycle()

    runner.governor._last_decision = AIGovernorDecision(  # noqa: SLF001 - test kontraktu telemetry
        mode="hedge",
        reason="model override",
        confidence=0.9,
        regime="trend",
        risk_score=0.3,
        transaction_cost_bps=10.0,
        decision_source="hybrid",
        inference_model="decision_model",
        inference_model_version="2026.04.02",
    )
    runner._last_decision_source = runner.governor._last_decision.decision_source  # noqa: SLF001
    runner._last_inference_model = runner.governor._last_decision.inference_model  # noqa: SLF001
    runner._last_inference_model_version = runner.governor._last_decision.inference_model_version  # noqa: SLF001

    contract = runner.snapshot()["telemetry"]["decisionContract"]
    assert contract["path"] == "hybrid"
    assert contract["model"] == "decision_model"
    assert contract["modelVersion"] == "2026.04.02"


def test_snapshot_decision_contract_policy_path_has_empty_model_metadata() -> None:
    snapshot = {
        "scalping_alpha": _summary(regime=MarketRegime.TREND, hit_rate=0.8, pnl=10.0, sharpe=1.5)
    }
    orchestrator = SimpleNamespace(strategy_performance_snapshot=lambda: snapshot)
    runner = AutoTraderAIGovernorRunner(orchestrator)

    runner.run_cycle()
    contract = runner.snapshot()["telemetry"]["decisionContract"]

    assert contract["path"] == "policy"
    assert contract["model"] is None
    assert contract["modelVersion"] is None
