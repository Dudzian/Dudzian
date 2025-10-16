from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from bot_core.config.models import PortfolioAssetConfig
from bot_core.risk.simulation import ProfileSimulationResult, RiskSimulationReport, StressTestResult
from bot_core.risk.stress_lab import (
    StressLabEvaluator,
    write_overrides_csv,
    write_report_csv,
    write_report_json,
    write_report_signature,
)


def _build_report() -> RiskSimulationReport:
    profile = ProfileSimulationResult(
        profile="balanced",
        base_equity=100_000.0,
        final_equity=95_000.0,
        total_return_pct=-0.05,
        max_drawdown_pct=0.13,
        worst_daily_loss_pct=0.06,
        realized_volatility=0.12,
        breaches=("margin_limit",),
        stress_tests=(
            StressTestResult(
                name="flash_crash",
                status="failed",
                metrics={
                    "severity": "critical",
                    "assets": ["BTCUSDT"],
                    "loss_pct": 0.18,
                },
                notes="Utrata kapitału > 15%",
            ),
            StressTestResult(
                name="liquidity_shock",
                status="warning",
                metrics={
                    "tags": ["defi"],
                    "risk_budget": "defi",
                    "liquidity_usd": 150_000.0,
                },
                notes="Płynność poniżej wymaganego minimum",
            ),
            StressTestResult(
                name="latency_spike",
                status="passed",
                metrics={
                    "assets": ["SOLUSDT"],
                    "avg_order_latency_ms": 620.0,
                },
                notes="",
            ),
        ),
        sample_size=480,
    )
    return RiskSimulationReport(
        generated_at="2024-06-01T00:00:00Z",
        base_equity=100_000.0,
        profiles=(profile,),
        synthetic_data=False,
    )


def test_stress_lab_evaluator_generates_overrides() -> None:
    evaluator = StressLabEvaluator(clock=lambda: datetime(2024, 6, 2, tzinfo=timezone.utc))
    assets = {
        "BTCUSDT": PortfolioAssetConfig(symbol="BTCUSDT", target_weight=0.4, tags=("core",)),
        "UNIUSDT": PortfolioAssetConfig(symbol="UNIUSDT", target_weight=0.1, tags=("defi",)),
        "SOLUSDT": PortfolioAssetConfig(symbol="SOLUSDT", target_weight=0.15, tags=("alt", "latency")),
    }
    report = evaluator.evaluate(_build_report(), portfolio=assets)

    assert report.counts["total"] == 4
    assert report.counts["critical"] == 3  # drawdown + flash crash + latency spike
    assert report.counts["warning"] == 1

    critical_override = next(
        override for override in report.overrides if override.symbol == "BTCUSDT"
    )
    assert critical_override.severity == "critical"
    assert critical_override.force_rebalance is True
    assert critical_override.weight_multiplier == 0.0

    warning_override = next(
        override
        for override in report.overrides
        if override.risk_budget == "defi" and override.severity == "warning"
    )
    assert warning_override.severity == "warning"
    assert warning_override.symbol == "UNIUSDT"
    assert warning_override.weight_multiplier == 0.6

    latency_override = next(
        override for override in report.overrides if override.symbol == "SOLUSDT"
    )
    assert latency_override.severity == "critical"
    assert latency_override.force_rebalance is True

    latency_insight = next(
        insight for insight in report.insights if insight.scenario == "latency_spike"
    )
    assert latency_insight.metrics["latency_alert"] == "critical"
    assert "metric:latency" in latency_insight.tags


def test_stress_lab_report_writers(tmp_path: Path) -> None:
    evaluator = StressLabEvaluator(clock=lambda: datetime(2024, 6, 2, 12, 0, tzinfo=timezone.utc))
    assets = {
        "BTCUSDT": PortfolioAssetConfig(symbol="BTCUSDT", target_weight=0.4, tags=("core",)),
    }
    report = evaluator.evaluate(_build_report(), portfolio=assets)

    json_path = tmp_path / "stress_report.json"
    csv_path = tmp_path / "stress_report.csv"
    overrides_path = tmp_path / "stress_overrides.csv"
    sig_path = tmp_path / "stress_report.sig"

    payload = write_report_json(report, json_path)
    write_report_csv(report, csv_path)
    write_overrides_csv(report, overrides_path)
    signature = write_report_signature(payload, sig_path, key=b"secret")

    assert json_path.exists()
    assert payload["schema"] == "stage6.risk.stress_lab.report"
    assert payload["overrides_total"] == len(report.overrides)

    csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines[0].startswith("profile,scenario,severity")
    assert len(csv_lines) == 5  # nagłówek + drawdown + 3 scenariusze

    overrides_lines = overrides_path.read_text(encoding="utf-8").strip().splitlines()
    assert overrides_lines[0].startswith("symbol,risk_budget")
    assert len(overrides_lines) == len(report.overrides) + 1

    assert signature["schema"] == "stage6.risk.stress_lab.report.signature"
    assert signature["signature"]["algorithm"] == "HMAC-SHA256"


def test_stress_lab_liquidity_thresholds_force_critical_override() -> None:
    evaluator = StressLabEvaluator(clock=lambda: datetime(2024, 6, 3, tzinfo=timezone.utc))
    assets = {
        "UNIUSDT": PortfolioAssetConfig(symbol="UNIUSDT", target_weight=0.1, tags=("alts",)),
    }
    report = RiskSimulationReport(
        generated_at="2024-06-02T00:00:00Z",
        base_equity=50_000.0,
        profiles=(
            ProfileSimulationResult(
                profile="aggressive",
                base_equity=50_000.0,
                final_equity=47_000.0,
                total_return_pct=-0.06,
                max_drawdown_pct=0.05,
                worst_daily_loss_pct=0.03,
                realized_volatility=0.18,
                breaches=(),
                stress_tests=(
                    StressTestResult(
                        name="liquidity_drain",
                        status="success",
                        metrics={
                            "assets": ["UNIUSDT"],
                            "risk_budget": "alts",
                            "liquidity_usd": 80_000.0,
                        },
                        notes="",
                    ),
                ),
                sample_size=240,
            ),
        ),
        synthetic_data=False,
    )

    stress_report = evaluator.evaluate(report, portfolio=assets)

    liquidity_insight = next(
        insight for insight in stress_report.insights if insight.scenario == "liquidity_drain"
    )
    assert liquidity_insight.severity == "critical"
    assert liquidity_insight.metrics["liquidity_alert"] == "critical"
    assert "metric:liquidity" in liquidity_insight.tags

    override = next(
        override for override in stress_report.overrides if override.symbol == "UNIUSDT"
    )
    assert override.severity == "critical"
    assert override.force_rebalance is True
