from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds
from bot_core.decision.metrics import DecisionMetricSet
from bot_core.decision.models import DecisionCandidate, DecisionEvaluation
from bot_core.decision.orchestrator import DecisionOrchestrator
from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.observability.metrics import MetricsRegistry
from bot_core.risk.engine import InMemoryRiskRepository, ThresholdRiskEngine
from bot_core.risk.guardrails import LossGuardrailConfig, RiskGuardrailMetricSet
from bot_core.risk.profiles.manual import ManualProfile
from bot_core.risk.repository import FileRiskRepository
from scripts.runtime.guardrail_rollback import main as guardrail_rollback_main


def _build_candidate() -> DecisionCandidate:
    return DecisionCandidate(
        strategy="mean_reversion",
        action="buy",
        risk_profile="balanced",
        symbol="BTCUSDT",
        notional=25_000.0,
        expected_return_bps=18.0,
        expected_probability=0.65,
        latency_ms=42.0,
        metadata={"intent": "single"},
    )


def test_alert_smoke_decision_metrics() -> None:
    registry = MetricsRegistry()
    metrics = DecisionMetricSet(registry=registry)
    candidate = _build_candidate()
    evaluation = DecisionEvaluation(
        candidate=candidate,
        accepted=True,
        cost_bps=3.0,
        net_edge_bps=9.5,
        reasons=(),
        risk_flags=("drawdown_watch",),
        stress_failures=("latency",),
        model_success_probability=0.64,
        recommended_risk_score=0.8,
    )

    metrics.observe_evaluation(evaluation)

    snapshot = registry.render_prometheus()
    assert (
        "decision_candidate_evaluations_total{intent=\"single\",profile=\"balanced\",result=\"accepted\",strategy=\"mean_reversion\"}"
        in snapshot
    )
    assert (
        "decision_candidate_risk_flags_total{flag=\"drawdown_watch\",intent=\"single\",profile=\"balanced\"} 1.0"
        in snapshot
    )
    assert (
        "decision_candidate_stress_failures_total{failure=\"latency\",intent=\"single\",profile=\"balanced\"} 1.0"
        in snapshot
    )


def test_alert_smoke_decision_metrics_missing_snapshot() -> None:
    registry = MetricsRegistry()
    metrics = DecisionMetricSet(registry=registry)
    config = DecisionEngineConfig(
        orchestrator=DecisionOrchestratorThresholds(
            max_cost_bps=10.0,
            min_net_edge_bps=0.0,
            max_daily_loss_pct=1.0,
            max_drawdown_pct=2.0,
            max_position_ratio=1.0,
            max_open_positions=10,
        )
    )
    orchestrator = DecisionOrchestrator(config, metrics=metrics)

    evaluations = orchestrator.evaluate_candidates([_build_candidate()], risk_snapshots={})
    assert len(evaluations) == 1
    assert evaluations[0].accepted is False

    snapshot = registry.render_prometheus()
    assert (
        "decision_candidate_evaluations_total{intent=\"single\",profile=\"balanced\",result=\"rejected\",strategy=\"mean_reversion\"}"
        in snapshot
    )
    assert (
        "decision_candidate_rejection_reasons_total{intent=\"single\",profile=\"balanced\",reason=\"missing_risk_snapshot\",strategy=\"mean_reversion\"} 1.0"
        in snapshot
    )


def test_alert_smoke_decision_rejection_reason_categorization() -> None:
    registry = MetricsRegistry()
    metrics = DecisionMetricSet(registry=registry)
    evaluation = DecisionEvaluation(
        candidate=_build_candidate(),
        accepted=False,
        cost_bps=15.0,
        net_edge_bps=-5.0,
        reasons=(
            "koszt 15.00 bps przekracza limit 10.00 bps",
            "net edge -5.00 bps poniżej progu 0.00 bps",
        ),
        risk_flags=("drawdown_watch",),
        stress_failures=(),
    )

    metrics.observe_evaluation(evaluation)

    snapshot = registry.render_prometheus()
    assert (
        "decision_candidate_rejection_reasons_total{intent=\"single\",profile=\"balanced\",reason=\"cost_above_limit\",strategy=\"mean_reversion\"} 1.0"
        in snapshot
    )
    assert (
        "decision_candidate_rejection_reasons_total{intent=\"single\",profile=\"balanced\",reason=\"net_edge_below_threshold\",strategy=\"mean_reversion\"} 1.0"
        in snapshot
    )


def test_alert_chaos_guardrail_switch_to_hedge() -> None:
    repository = InMemoryRiskRepository()
    engine = ThresholdRiskEngine(
        repository=repository,
        guardrail_config=LossGuardrailConfig(daily_loss_pct=0.02),
        guardrail_metrics=RiskGuardrailMetricSet(),
    )
    profile = ManualProfile(
        name="balanced",
        max_positions=10,
        max_leverage=3.0,
        drawdown_limit=1.0,
        daily_loss_limit=1.0,
        max_position_pct=0.5,
        target_volatility=0.2,
        stop_loss_atr_multiple=1.0,
        daily_kill_switch_r_multiple=100.0,
        daily_kill_switch_loss_pct=1.0,
        weekly_kill_switch_loss_pct=1.0,
    )
    engine.register_profile(profile)
    state = engine._states[profile.name]
    state.start_of_day_equity = 100_000.0
    state.daily_realized_pnl = -4_000.0
    state.weekly_realized_pnl = -5_000.0
    state.start_of_week_equity = 110_000.0
    state.peak_equity = 120_000.0
    state.last_equity = 90_000.0

    account = AccountSnapshot(
        balances={},
        total_equity=90_000.0,
        available_margin=50_000.0,
        maintenance_margin=1_000.0,
    )
    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=1.0,
        order_type="market",
        price=20_000.0,
        stop_price=19_000.0,
        atr=1.0,
        metadata={"intent": "single"},
    )

    result = engine.apply_pre_trade_checks(request, account=account, profile_name=profile.name)
    assert result.allowed is False
    assert "trybie hedge" in (result.reason or "")
    assert state.hedge_mode is True

    hedge_request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.5,
        order_type="market",
        price=20_000.0,
        stop_price=19_200.0,
        atr=1.0,
        metadata={"intent": "hedge"},
    )
    allowed_result = engine.apply_pre_trade_checks(hedge_request, account=account, profile_name=profile.name)
    assert allowed_result.allowed is True

    metrics_snapshot = engine._guardrail_metrics.registry.render_prometheus()  # type: ignore[union-attr]
    assert "risk_guardrail_hedge_mode{profile=\"balanced\"} 1.0" in metrics_snapshot


def test_alert_chaos_guardrail_rollback_script(tmp_path: Path) -> None:
    repository_path = tmp_path / "risk"
    repository = FileRiskRepository(repository_path)
    state_payload = {
        "profile": "balanced",
        "current_day": datetime.now(timezone.utc).date().isoformat(),
        "start_of_day_equity": 100_000.0,
        "daily_realized_pnl": -5_000.0,
        "weekly_realized_pnl": -6_000.0,
        "peak_equity": 110_000.0,
        "force_liquidation": True,
        "last_equity": 95_000.0,
        "positions": {},
        "start_of_week_equity": 110_000.0,
        "rolling_profit_30d": 0.0,
        "rolling_costs_30d": 0.0,
        "hedge_mode": True,
        "hedge_reason": "guardrail",
        "hedge_activated_at": datetime.now(timezone.utc).isoformat(),
        "hedge_cooldown_until": None,
    }
    repository.store("balanced", state_payload)

    exit_code = guardrail_rollback_main([
        "--repository",
        str(repository_path),
        "--profile",
        "balanced",
        "--clear-force-liquidation",
    ])
    assert exit_code == 0

    updated = repository.load("balanced")
    assert updated is not None
    assert updated.get("hedge_mode") is False
    assert updated.get("hedge_reason") is None
    assert updated.get("force_liquidation") is False
