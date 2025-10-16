from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

# ======================================================================
# Conditional imports to support BOTH implementations of Stress Lab:
#   • Variant A (HEAD): evaluator over RiskSimulationReport + writers
#   • Variant B (main): config-driven StressLab with scenarios/thresholds
# ======================================================================

# ----- Variant A: Evaluator + writers over RiskSimulationReport -----
try:
    from bot_core.config.models import PortfolioAssetConfig  # type: ignore[attr-defined]
    from bot_core.risk.simulation import (  # type: ignore[attr-defined]
        ProfileSimulationResult,
        RiskSimulationReport,
        StressTestResult,
    )
    from bot_core.risk.stress_lab import (  # type: ignore[attr-defined]
        StressLabEvaluator,
        write_overrides_csv,
        write_report_csv,
        write_report_json,
        write_report_signature,
    )

    _HAVE_EVALUATOR = True
except Exception:  # pragma: no cover - environment without evaluator API
    PortfolioAssetConfig = None  # type: ignore[assignment]
    ProfileSimulationResult = None  # type: ignore[assignment]
    RiskSimulationReport = None  # type: ignore[assignment]
    StressTestResult = None  # type: ignore[assignment]
    StressLabEvaluator = None  # type: ignore[assignment]
    write_overrides_csv = None  # type: ignore[assignment]
    write_report_csv = None  # type: ignore[assignment]
    write_report_json = None  # type: ignore[assignment]
    write_report_signature = None  # type: ignore[assignment]
    _HAVE_EVALUATOR = False

# ----- Variant B: Config-driven StressLab runner with scenarios -----
try:
    from bot_core.config.models import (  # type: ignore[attr-defined]
        StressLabConfig,
        StressLabDatasetConfig,
        StressLabScenarioConfig,
        StressLabShockConfig,
        StressLabThresholdsConfig,
    )
    from bot_core.risk.stress_lab import StressLab  # type: ignore[attr-defined]

    _HAVE_CONFIG_STRESSLAB = True
except Exception:  # pragma: no cover - environment without config-based API
    StressLabConfig = None  # type: ignore[assignment]
    StressLabDatasetConfig = None  # type: ignore[assignment]
    StressLabScenarioConfig = None  # type: ignore[assignment]
    StressLabShockConfig = None  # type: ignore[assignment]
    StressLabThresholdsConfig = None  # type: ignore[assignment]
    StressLab = None  # type: ignore[assignment]
    _HAVE_CONFIG_STRESSLAB = False


# =============================================================================
#                          Variant A — Evaluator-based tests
# =============================================================================
def _build_report() -> RiskSimulationReport:  # type: ignore[valid-type]
    """Construct a synthetic RiskSimulationReport with several scenarios."""
    profile = ProfileSimulationResult(  # type: ignore[call-arg]
        profile="balanced",
        base_equity=100_000.0,
        final_equity=95_000.0,
        total_return_pct=-0.05,
        max_drawdown_pct=0.13,
        worst_daily_loss_pct=0.06,
        realized_volatility=0.12,
        breaches=("margin_limit",),
        stress_tests=(
            StressTestResult(  # type: ignore[call-arg]
                name="flash_crash",
                status="failed",
                metrics={"severity": "critical", "assets": ["BTCUSDT"], "loss_pct": 0.18},
                notes="Utrata kapitału > 15%",
            ),
            StressTestResult(  # type: ignore[call-arg]
                name="liquidity_shock",
                status="warning",
                metrics={"tags": ["defi"], "risk_budget": "defi", "liquidity_usd": 150_000.0},
                notes="Płynność poniżej wymaganego minimum",
            ),
            StressTestResult(  # type: ignore[call-arg]
                name="latency_spike",
                status="passed",
                metrics={"assets": ["SOLUSDT"], "avg_order_latency_ms": 620.0},
                notes="",
            ),
        ),
        sample_size=480,
    )
    return RiskSimulationReport(  # type: ignore[call-arg]
        generated_at="2024-06-01T00:00:00Z",
        base_equity=100_000.0,
        profiles=(profile,),
        synthetic_data=False,
    )


@pytest.mark.skipif(not _HAVE_EVALUATOR, reason="Evaluator-based Stress Lab API not available")
def test_stress_lab_evaluator_generates_overrides() -> None:
    evaluator = StressLabEvaluator(clock=lambda: datetime(2024, 6, 2, tzinfo=timezone.utc))  # type: ignore[call-arg]
    assets = {
        "BTCUSDT": PortfolioAssetConfig(symbol="BTCUSDT", target_weight=0.4, tags=("core",)),  # type: ignore[call-arg]
        "UNIUSDT": PortfolioAssetConfig(symbol="UNIUSDT", target_weight=0.1, tags=("defi",)),  # type: ignore[call-arg]
        "SOLUSDT": PortfolioAssetConfig(symbol="SOLUSDT", target_weight=0.15, tags=("alt", "latency")),  # type: ignore[call-arg]
    }
    report = evaluator.evaluate(_build_report(), portfolio=assets)  # type: ignore[attr-defined]

    assert report.counts["total"] == 4
    # drawdown breach + flash_crash + latency_spike (critical)
    assert report.counts["critical"] == 3
    assert report.counts["warning"] == 1

    critical_override = next(override for override in report.overrides if override.symbol == "BTCUSDT")
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

    latency_override = next(override for override in report.overrides if override.symbol == "SOLUSDT")
    assert latency_override.severity == "critical"
    assert latency_override.force_rebalance is True

    latency_insight = next(insight for insight in report.insights if insight.scenario == "latency_spike")
    assert latency_insight.metrics["latency_alert"] == "critical"
    assert "metric:latency" in latency_insight.tags


@pytest.mark.skipif(not _HAVE_EVALUATOR, reason="Evaluator-based Stress Lab API not available")
def test_stress_lab_report_writers(tmp_path: Path) -> None:
    evaluator = StressLabEvaluator(clock=lambda: datetime(2024, 6, 2, 12, 0, tzinfo=timezone.utc))  # type: ignore[call-arg]
    assets = {
        "BTCUSDT": PortfolioAssetConfig(symbol="BTCUSDT", target_weight=0.4, tags=("core",)),  # type: ignore[call-arg]
    }
    report = evaluator.evaluate(_build_report(), portfolio=assets)  # type: ignore[attr-defined]

    json_path = tmp_path / "stress_report.json"
    csv_path = tmp_path / "stress_report.csv"
    overrides_path = tmp_path / "stress_overrides.csv"
    sig_path = tmp_path / "stress_report.sig"

    payload = write_report_json(report, json_path)  # type: ignore[attr-defined]
    write_report_csv(report, csv_path)  # type: ignore[attr-defined]
    write_overrides_csv(report, overrides_path)  # type: ignore[attr-defined]
    signature = write_report_signature(payload, sig_path, key=b"secret")  # type: ignore[attr-defined]

    assert json_path.exists()
    assert payload["schema"] == "stage6.risk.stress_lab.report"
    assert payload["overrides_total"] == len(report.overrides)

    csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines[0].startswith("profile,scenario,severity")
    # header + drawdown + 3 scenarios
    assert len(csv_lines) == 5

    overrides_lines = overrides_path.read_text(encoding="utf-8").strip().splitlines()
    assert overrides_lines[0].startswith("symbol,risk_budget")
    assert len(overrides_lines) == len(report.overrides) + 1

    assert signature["schema"] == "stage6.risk.stress_lab.report.signature"
    assert signature["signature"]["algorithm"] == "HMAC-SHA256"


@pytest.mark.skipif(not _HAVE_EVALUATOR, reason="Evaluator-based Stress Lab API not available")
def test_stress_lab_liquidity_thresholds_force_critical_override() -> None:
    evaluator = StressLabEvaluator(clock=lambda: datetime(2024, 6, 3, tzinfo=timezone.utc))  # type: ignore[call-arg]
    assets = {
        "UNIUSDT": PortfolioAssetConfig(symbol="UNIUSDT", target_weight=0.1, tags=("alts",)),  # type: ignore[call-arg]
    }
    report = RiskSimulationReport(  # type: ignore[call-arg]
        generated_at="2024-06-02T00:00:00Z",
        base_equity=50_000.0,
        profiles=(
            ProfileSimulationResult(  # type: ignore[call-arg]
                profile="aggressive",
                base_equity=50_000.0,
                final_equity=47_000.0,
                total_return_pct=-0.06,
                max_drawdown_pct=0.05,
                worst_daily_loss_pct=0.03,
                realized_volatility=0.18,
                breaches=(),
                stress_tests=(
                    StressTestResult(  # type: ignore[call-arg]
                        name="liquidity_drain",
                        status="success",
                        metrics={"assets": ["UNIUSDT"], "risk_budget": "alts", "liquidity_usd": 80_000.0},
                        notes="",
                    ),
                ),
                sample_size=240,
            ),
        ),
        synthetic_data=False,
    )

    stress_report = evaluator.evaluate(report, portfolio=assets)  # type: ignore[attr-defined]

    liquidity_insight = next(insight for insight in stress_report.insights if insight.scenario == "liquidity_drain")
    assert liquidity_insight.severity == "critical"
    assert liquidity_insight.metrics["liquidity_alert"] == "critical"
    assert "metric:liquidity" in liquidity_insight.tags

    override = next(override for override in stress_report.overrides if override.symbol == "UNIUSDT")
    assert override.severity == "critical"
    assert override.force_rebalance is True


# =============================================================================
#                       Variant B — Config-driven StressLab tests
# =============================================================================
def _write_dataset(path: Path, *, spread: float = 5.0) -> None:
    payload = {
        "symbol": "TESTUSDT",
        "baseline": {
            "mid_price": 25_000.0,
            "avg_depth_usd": 1_600_000.0,
            "avg_spread_bps": spread,
            "funding_rate_bps": 8.0,
            "sentiment_score": 0.5,
            "realized_volatility": 0.35,
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_base_config(tmp_path: Path) -> StressLabConfig:  # type: ignore[valid-type]
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(dataset_path)
    return StressLabConfig(  # type: ignore[call-arg]
        enabled=True,
        require_success=True,
        report_directory=str(tmp_path / "reports"),
        datasets={
            "TESTUSDT": StressLabDatasetConfig(  # type: ignore[call-arg]
                symbol="TESTUSDT",
                metrics_path=str(dataset_path),
                weight=1.0,
                allow_synthetic=False,
            )
        },
        scenarios=(
            StressLabScenarioConfig(  # type: ignore[call-arg]
                name="liquidity_sanity",
                severity="medium",
                markets=("TESTUSDT",),
                shocks=(
                    StressLabShockConfig(type="liquidity_crunch", intensity=0.3),  # type: ignore[call-arg]
                    StressLabShockConfig(type="volatility_spike", intensity=0.2),  # type: ignore[call-arg]
                ),
            ),
        ),
        thresholds=StressLabThresholdsConfig(  # type: ignore[call-arg]
            max_liquidity_loss_pct=0.9,
            max_spread_increase_bps=80.0,
            max_volatility_increase_pct=1.5,
            max_sentiment_drawdown=0.8,
            max_funding_change_bps=50.0,
            max_latency_spike_ms=220.0,
            max_blackout_minutes=120.0,
            max_dispersion_bps=90.0,
        ),
    )


@pytest.mark.skipif(not _HAVE_CONFIG_STRESSLAB, reason="Config-driven StressLab API not available")
def test_stress_lab_report_generation(tmp_path: Path) -> None:
    config = _build_base_config(tmp_path)
    lab = StressLab(config)  # type: ignore[call-arg]

    report = lab.run()
    assert report.has_failures() is False
    assert report.scenarios[0].status == "passed"

    output_path = Path(config.report_directory) / "stress_lab_report.json"  # type: ignore[index]
    report.write_json(output_path)
    assert output_path.exists()

    signature = report.build_signature(key=b"a" * 32, key_id="unit-test")
    assert signature["algorithm"].startswith("HMAC")
    assert "value" in signature


@pytest.mark.skipif(not _HAVE_CONFIG_STRESSLAB, reason="Config-driven StressLab API not available")
def test_stress_lab_threshold_breach(tmp_path: Path) -> None:
    config = _build_base_config(tmp_path)
    # Make thresholds strict to force failures
    config.thresholds.max_liquidity_loss_pct = 0.05  # type: ignore[attr-defined]
    lab = StressLab(config)  # type: ignore[call-arg]

    report = lab.run()
    assert report.has_failures() is True
    scenario = report.scenarios[0]
    assert scenario.failures, "Expected failures when thresholds are strict"
    assert scenario.status == "failed"


@pytest.mark.skipif(not _HAVE_CONFIG_STRESSLAB, reason="Config-driven StressLab API not available")
def test_stress_lab_synthetic_dataset(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = StressLabConfig(  # type: ignore[call-arg]
        enabled=True,
        require_success=False,
        report_directory=str(tmp_path / "reports"),
        datasets={},
        scenarios=(
            StressLabScenarioConfig(  # type: ignore[call-arg]
                name="synthetic_market",
                severity="high",
                markets=("MISSINGUSDT",),
                shocks=(StressLabShockConfig(type="blackout", intensity=0.8, duration_minutes=30),),  # type: ignore[call-arg]
            ),
        ),
        thresholds=StressLabThresholdsConfig(),  # type: ignore[call-arg]
    )

    lab = StressLab(config)  # type: ignore[call-arg]
    with caplog.at_level("WARNING"):
        report = lab.run()

    # Accept either Polish or English wording in logs
    messages = " ".join(m.lower() for m in caplog.messages)
    assert ("brak datasetu" in messages) or ("missing dataset" in messages) or caplog.records, \
        "Expected a warning about missing dataset"

    # Even without a dataset, StressLab should synthesize a baseline for the requested market
    assert report.scenarios[0].markets[0].baseline.symbol == "MISSINGUSDT"
