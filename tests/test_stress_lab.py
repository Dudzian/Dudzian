from __future__ import annotations

import json
from pathlib import Path

import pytest

from bot_core.config.models import (
    StressLabConfig,
    StressLabDatasetConfig,
    StressLabScenarioConfig,
    StressLabShockConfig,
    StressLabThresholdsConfig,
)
from bot_core.risk.stress_lab import StressLab


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


def _build_base_config(tmp_path: Path) -> StressLabConfig:
    dataset_path = tmp_path / "dataset.json"
    _write_dataset(dataset_path)
    return StressLabConfig(
        enabled=True,
        require_success=True,
        report_directory=str(tmp_path / "reports"),
        datasets={
            "TESTUSDT": StressLabDatasetConfig(
                symbol="TESTUSDT",
                metrics_path=str(dataset_path),
                weight=1.0,
                allow_synthetic=False,
            )
        },
        scenarios=(
            StressLabScenarioConfig(
                name="liquidity_sanity",
                severity="medium",
                markets=("TESTUSDT",),
                shocks=(
                    StressLabShockConfig(type="liquidity_crunch", intensity=0.3),
                    StressLabShockConfig(type="volatility_spike", intensity=0.2),
                ),
            ),
        ),
        thresholds=StressLabThresholdsConfig(
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


def test_stress_lab_report_generation(tmp_path: Path) -> None:
    config = _build_base_config(tmp_path)
    lab = StressLab(config)

    report = lab.run()
    assert report.has_failures() is False
    assert report.scenarios[0].status == "passed"

    output_path = Path(config.report_directory) / "stress_lab_report.json"
    report.write_json(output_path)
    assert output_path.exists()

    signature = report.build_signature(key=b"a" * 32, key_id="unit-test")
    assert signature["algorithm"].startswith("HMAC")
    assert "value" in signature


def test_stress_lab_threshold_breach(tmp_path: Path) -> None:
    config = _build_base_config(tmp_path)
    config.thresholds.max_liquidity_loss_pct = 0.05
    lab = StressLab(config)

    report = lab.run()
    assert report.has_failures() is True
    scenario = report.scenarios[0]
    assert scenario.failures, "Expected failures when thresholds are strict"
    assert scenario.status == "failed"


def test_stress_lab_synthetic_dataset(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = StressLabConfig(
        enabled=True,
        require_success=False,
        report_directory=str(tmp_path / "reports"),
        datasets={},
        scenarios=(
            StressLabScenarioConfig(
                name="synthetic_market",
                severity="high",
                markets=("MISSINGUSDT",),
                shocks=(StressLabShockConfig(type="blackout", intensity=0.8, duration_minutes=30),),
            ),
        ),
        thresholds=StressLabThresholdsConfig(),
    )

    lab = StressLab(config)
    with caplog.at_level("WARNING"):
        report = lab.run()
    assert "brak datasetu" in " ".join(caplog.messages)
    assert report.scenarios[0].markets[0].baseline.symbol == "MISSINGUSDT"
