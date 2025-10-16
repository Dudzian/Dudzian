from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.market_intel import MarketIntelSnapshot
from bot_core.risk.simulation import ProfileSimulationResult, RiskSimulationReport, StressTestResult
from bot_core.risk.stress_lab_calibration import (
    StressLabCalibrator,
    StressLabCalibrationReport,
    StressLabCalibrationSegment,
    StressLabCalibrationSettings,
    build_volume_segments,
    write_calibration_csv,
    write_calibration_json,
    write_calibration_signature,
)


def _snapshot(symbol: str, liquidity: float) -> MarketIntelSnapshot:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc)
    return MarketIntelSnapshot(
        symbol=symbol,
        interval="1h",
        start=now,
        end=now,
        bar_count=168,
        price_change_pct=2.0,
        volatility_pct=5.0,
        max_drawdown_pct=3.0,
        average_volume=1_000_000.0,
        liquidity_usd=liquidity,
        momentum_score=1.5,
        metadata={},
    )


def _risk_report(*latencies: float) -> RiskSimulationReport:
    stress_tests = (
        StressTestResult(
            name="latency_spike",
            status="warning",
            metrics={"avg_order_latency_ms": value},
        )
        for value in latencies
    )
    profile = ProfileSimulationResult(
        profile="balanced",
        base_equity=100_000.0,
        final_equity=102_000.0,
        total_return_pct=0.02,
        max_drawdown_pct=0.05,
        worst_daily_loss_pct=0.02,
        realized_volatility=0.1,
        breaches=(),
        stress_tests=tuple(stress_tests),
        sample_size=120,
    )
    return RiskSimulationReport(
        generated_at="2024-01-02T00:00:00Z",
        base_equity=100_000.0,
        profiles=(profile,),
        synthetic_data=False,
    )


def test_calibrator_produces_segment_thresholds() -> None:
    snapshots = {
        "BTCUSDT": _snapshot("BTCUSDT", 500_000.0),
        "ETHUSDT": _snapshot("ETHUSDT", 300_000.0),
        "UNIUSDT": _snapshot("UNIUSDT", 120_000.0),
    }
    segments = (
        StressLabCalibrationSegment(name="core", symbols=("BTCUSDT", "ETHUSDT")),
        StressLabCalibrationSegment(name="alts", symbols=("UNIUSDT",)),
    )
    settings = StressLabCalibrationSettings(
        liquidity_warning_percentile=0.5,
        liquidity_critical_percentile=0.25,
        latency_warning_percentile=0.5,
        latency_critical_percentile=0.75,
        min_liquidity_threshold=0.0,
        min_latency_threshold_ms=0.0,
    )
    calibrator = StressLabCalibrator(settings=settings, clock=lambda: datetime(2024, 1, 3, tzinfo=timezone.utc))

    report = calibrator.calibrate(
        market_snapshots=snapshots,
        segments=segments,
        risk_report=_risk_report(180.0, 260.0, 400.0),
    )

    assert isinstance(report, StressLabCalibrationReport)
    assert report.generated_at == datetime(2024, 1, 3, tzinfo=timezone.utc)
    assert len(report.liquidity_segments) == 2

    core_segment = next(seg for seg in report.liquidity_segments if seg.segment == "core")
    assert core_segment.liquidity_warning_threshold_usd == 400_000.0
    assert core_segment.liquidity_critical_threshold_usd == 350_000.0
    assert core_segment.symbol_count == 2

    alts_segment = next(seg for seg in report.liquidity_segments if seg.segment == "alts")
    assert alts_segment.liquidity_warning_threshold_usd == 120_000.0
    assert alts_segment.liquidity_critical_threshold_usd == 120_000.0

    assert report.latency_warning_threshold_ms == 260.0
    assert report.latency_critical_threshold_ms == 330.0
    assert report.metadata["symbols_considered"] == 3


def test_calibrator_handles_missing_metrics() -> None:
    calibrator = StressLabCalibrator(
        settings=StressLabCalibrationSettings(min_liquidity_threshold=0.0, min_latency_threshold_ms=0.0)
    )
    empty_segment = StressLabCalibrationSegment(name="empty", symbols=("UNKNOWN",))
    report = calibrator.calibrate(market_snapshots={}, segments=[empty_segment], risk_report=None)

    assert len(report.liquidity_segments) == 1
    segment_result = report.liquidity_segments[0]
    assert segment_result.liquidity_warning_threshold_usd is None
    assert segment_result.liquidity_critical_threshold_usd is None
    assert report.latency_warning_threshold_ms is None
    assert report.latency_critical_threshold_ms is None


def test_build_volume_segments_distributes_symbols() -> None:
    snapshots = {
        "BTCUSDT": _snapshot("BTCUSDT", 800_000.0),
        "ETHUSDT": _snapshot("ETHUSDT", 500_000.0),
        "XRPUSDT": _snapshot("XRPUSDT", 250_000.0),
        "ADAUSDT": _snapshot("ADAUSDT", 190_000.0),
        "DOGEUSDT": _snapshot("DOGEUSDT", 120_000.0),
    }

    segments = build_volume_segments(
        snapshots,
        buckets=3,
        min_symbols_per_bucket=2,
        name_prefix="liq",
        risk_budget_prefix="rb-",
    )

    assert len(segments) == 3
    assert segments[0].name == "liq_1"
    assert set(segments[0].symbols) == {"BTCUSDT", "ETHUSDT"}
    assert segments[0].risk_budgets == ("rb-1",)
    assert set(segments[1].symbols) == {"XRPUSDT", "ADAUSDT"}
    assert set(segments[2].symbols) == {"DOGEUSDT"}


def test_build_volume_segments_requires_liquidity_metrics() -> None:
    with pytest.raises(ValueError):
        build_volume_segments({}, buckets=2)

def test_calibration_writers_create_signed_payload(tmp_path: Path) -> None:
    settings = StressLabCalibrationSettings(min_liquidity_threshold=0.0, min_latency_threshold_ms=0.0)
    segments = (
        StressLabCalibrationSegment(name="core", symbols=("BTCUSDT",)),
    )
    calibrator = StressLabCalibrator(settings=settings, clock=lambda: datetime(2024, 1, 4, tzinfo=timezone.utc))
    report = calibrator.calibrate(
        market_snapshots={"BTCUSDT": _snapshot("BTCUSDT", 200_000.0)},
        segments=segments,
        risk_report=_risk_report(220.0),
    )

    json_path = tmp_path / "calibration.json"
    csv_path = tmp_path / "calibration.csv"
    sig_path = tmp_path / "calibration.sig"

    payload = write_calibration_json(report, json_path)
    write_calibration_csv(report, csv_path)
    signature = write_calibration_signature(payload, sig_path, key=b"key", key_id="stage6", target=json_path.name)

    assert json_path.exists()
    assert csv_path.exists()
    assert sig_path.exists()
    assert signature["schema"] == "stage6.risk.stress_lab.calibration.signature"
    assert signature["target"] == json_path.name
    assert payload["schema"] == "stage6.risk.stress_lab.calibration"
    csv_lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
    assert csv_lines[0].startswith("segment,symbol_count")
    assert any("core" in line for line in csv_lines[1:])
