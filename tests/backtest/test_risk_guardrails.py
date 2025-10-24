import pytest

from bot_core.backtest.engine import BacktestReport, PerformanceMetrics
from bot_core.risk.base import RiskCheckResult
from bot_core.risk.guardrails import evaluate_backtest_guardrails, summarize_guardrail_results
from bot_core.risk.profiles import BalancedProfile, ConservativeProfile


def _build_report(
    metrics: PerformanceMetrics,
    *,
    strategy_metadata: dict[str, object] | None = None,
) -> BacktestReport:
    metadata = {"risk_profile": "conservative"}
    if strategy_metadata:
        metadata.update(strategy_metadata)
    return BacktestReport(
        trades=[],
        fills=[],
        equity_curve=[10_000.0, 9_800.0],
        equity_timestamps=[],
        starting_balance=10_000.0,
        final_balance=9_800.0,
        metrics=metrics,
        warnings=[],
        parameters={},
        strategy_metadata=metadata,
    )


def test_guardrails_block_on_excessive_drawdown() -> None:
    metrics = PerformanceMetrics(
        total_return_pct=5.0,
        cagr_pct=5.0,
        max_drawdown_pct=35.0,
        sharpe_ratio=1.0,
        sortino_ratio=0.8,
        omega_ratio=1.1,
        hit_ratio_pct=55.0,
        risk_of_ruin_pct=15.0,
        max_exposure_pct=40.0,
        fees_paid=12.0,
        slippage_cost=1.0,
    )
    report = _build_report(metrics)
    profile = ConservativeProfile()
    result = evaluate_backtest_guardrails(report, risk_profile=profile)
    assert result.allowed is False
    assert "drawdown" in (result.reason or "").lower()
    assert result.metadata["thresholds"]["max_drawdown_pct"] == pytest.approx(
        profile.drawdown_limit() * 100.0
    )
    assert result.metadata["threshold_sources"]["max_drawdown_pct"] == "risk_profile:conservative"
    assert result.metadata["risk_profile"] == "conservative"
    assert result.metadata["strategy_metadata"]["risk_profile"] == "conservative"
    violations = result.metadata["violations"]
    assert isinstance(violations, list)
    assert violations and violations[0]["metric"] == "max_drawdown_pct"
    assert violations[0]["source"] == "risk_profile:conservative"
    assert violations[0]["observed"] == pytest.approx(metrics.max_drawdown_pct)
    assert violations[0]["threshold"] == pytest.approx(profile.drawdown_limit() * 100.0)


def test_guardrails_block_on_exposure_limit() -> None:
    metrics = PerformanceMetrics(
        total_return_pct=8.0,
        cagr_pct=8.0,
        max_drawdown_pct=2.0,
        sharpe_ratio=1.5,
        sortino_ratio=1.6,
        omega_ratio=1.8,
        hit_ratio_pct=60.0,
        risk_of_ruin_pct=3.0,
        max_exposure_pct=150.0,
        fees_paid=8.0,
        slippage_cost=0.5,
    )
    report = _build_report(metrics)
    result = evaluate_backtest_guardrails(
        report,
        max_drawdown_pct=10.0,
        max_exposure_pct=50.0,
    )
    assert result.allowed is False
    assert "exposure" in (result.reason or "").lower()
    assert result.metadata["threshold_sources"]["max_exposure_pct"] is None
    second_violations = result.metadata["violations"]
    assert second_violations and second_violations[0]["metric"] == "max_exposure_pct"


def test_guardrails_block_when_required_data_missing() -> None:
    metrics = PerformanceMetrics(
        total_return_pct=12.0,
        cagr_pct=12.0,
        max_drawdown_pct=4.0,
        sharpe_ratio=1.8,
        sortino_ratio=2.1,
        omega_ratio=2.3,
        hit_ratio_pct=62.0,
        risk_of_ruin_pct=1.2,
        max_exposure_pct=35.0,
        fees_paid=5.0,
        slippage_cost=0.3,
    )
    report = _build_report(
        metrics,
        strategy_metadata={
            "required_data": ("open", "volume"),
            "required_data_missing": ("volume",),
        },
    )
    result = evaluate_backtest_guardrails(report)
    assert result.allowed is False
    assert "missing required" in (result.reason or "").lower()
    guardrails = result.metadata
    assert guardrails["observed"]["required_data_missing"] == ("volume",)
    violation = guardrails["violations"][0]
    assert violation["metric"] == "required_data_missing"
    assert violation["observed"] == ("volume",)
    assert violation["source"] == "strategy_metadata"


def test_guardrails_use_profile_sortino_threshold() -> None:
    profile = BalancedProfile()
    metrics = PerformanceMetrics(
        total_return_pct=6.0,
        cagr_pct=6.0,
        max_drawdown_pct=3.5,
        sharpe_ratio=1.1,
        sortino_ratio=0.9,
        omega_ratio=1.4,
        hit_ratio_pct=58.0,
        risk_of_ruin_pct=2.5,
        max_exposure_pct=3.0,
        fees_paid=4.0,
        slippage_cost=0.2,
    )
    report = _build_report(metrics, strategy_metadata={"risk_profile": "balanced"})
    result = evaluate_backtest_guardrails(report, risk_profile=profile)
    assert result.allowed is False
    assert "sortino" in (result.reason or "").lower()
    thresholds = result.metadata["thresholds"]
    assert thresholds["min_sortino_ratio"] == pytest.approx(profile.min_sortino_ratio)
    violation = next(v for v in result.metadata["violations"] if v["metric"] == "min_sortino_ratio")
    assert violation["threshold"] == pytest.approx(profile.min_sortino_ratio)
    assert violation["source"] == "risk_profile:balanced"
    assert violation["observed"] == pytest.approx(metrics.sortino_ratio)


def test_guardrails_honor_explicit_omega_threshold() -> None:
    metrics = PerformanceMetrics(
        total_return_pct=7.0,
        cagr_pct=7.0,
        max_drawdown_pct=3.0,
        sharpe_ratio=1.4,
        sortino_ratio=2.1,
        omega_ratio=0.95,
        hit_ratio_pct=60.0,
        risk_of_ruin_pct=1.5,
        max_exposure_pct=2.5,
        fees_paid=2.0,
        slippage_cost=0.1,
    )
    report = _build_report(metrics)
    result = evaluate_backtest_guardrails(report, min_omega_ratio=1.1)
    assert result.allowed is False
    assert "omega" in (result.reason or "").lower()
    violation = next(v for v in result.metadata["violations"] if v["metric"] == "min_omega_ratio")
    assert violation["threshold"] == pytest.approx(1.1)
    assert violation["source"] is None
    assert violation["observed"] == pytest.approx(metrics.omega_ratio)


def test_guardrails_use_profile_risk_of_ruin_threshold() -> None:
    profile = ConservativeProfile()
    metrics = PerformanceMetrics(
        total_return_pct=4.5,
        cagr_pct=4.5,
        max_drawdown_pct=3.0,
        sharpe_ratio=1.2,
        sortino_ratio=1.9,
        omega_ratio=1.5,
        hit_ratio_pct=60.0,
        risk_of_ruin_pct=6.2,
        max_exposure_pct=2.0,
        fees_paid=1.0,
        slippage_cost=0.1,
    )
    report = _build_report(metrics)
    result = evaluate_backtest_guardrails(report, risk_profile=profile)
    assert result.allowed is False
    assert "risk of ruin" in (result.reason or "").lower()
    thresholds = result.metadata["thresholds"]
    assert thresholds["max_risk_of_ruin_pct"] == pytest.approx(profile.max_risk_of_ruin_pct)
    violation = next(
        v for v in result.metadata["violations"] if v["metric"] == "max_risk_of_ruin_pct"
    )
    assert violation["threshold"] == pytest.approx(profile.max_risk_of_ruin_pct)
    assert violation["source"] == "risk_profile:conservative"
    assert violation["observed"] == pytest.approx(metrics.risk_of_ruin_pct)


def test_guardrails_honor_explicit_hit_ratio_threshold() -> None:
    metrics = PerformanceMetrics(
        total_return_pct=5.5,
        cagr_pct=5.5,
        max_drawdown_pct=2.5,
        sharpe_ratio=1.3,
        sortino_ratio=1.7,
        omega_ratio=1.4,
        hit_ratio_pct=46.0,
        risk_of_ruin_pct=3.0,
        max_exposure_pct=2.0,
        fees_paid=1.2,
        slippage_cost=0.1,
    )
    report = _build_report(metrics)
    result = evaluate_backtest_guardrails(report, min_hit_ratio_pct=52.0)
    assert result.allowed is False
    assert "hit ratio" in (result.reason or "").lower()
    violation = next(v for v in result.metadata["violations"] if v["metric"] == "min_hit_ratio_pct")
    assert violation["threshold"] == pytest.approx(52.0)
    assert violation["source"] is None
    assert violation["observed"] == pytest.approx(metrics.hit_ratio_pct)


def test_guardrail_summary_aggregates_results() -> None:
    allowed = RiskCheckResult(allowed=True, metadata={"warnings": ["low volume"]})
    blocked = RiskCheckResult(
        allowed=False,
        reason="Max drawdown breached",
        metadata={
            "violations": [
                {"metric": "max_drawdown_pct", "observed": 35.0, "threshold": 20.0},
                {"metric": "required_data_missing", "observed": ("volume",)},
            ],
            "warnings": ["missing candle"],
        },
    )

    summary = summarize_guardrail_results([("allowed", allowed), ("blocked", blocked)])
    assert summary.total == 2
    assert summary.allowed == 1
    assert summary.blocked == 1
    assert summary.metrics_violations == {"max_drawdown_pct": 1, "required_data_missing": 1}
    assert summary.warnings == {"low volume": 1, "missing candle": 1}
    assert summary.blocked_scenarios[0]["scenario"] == "blocked"
    assert summary.blocked_scenarios[0]["reason"] == "Max drawdown breached"
