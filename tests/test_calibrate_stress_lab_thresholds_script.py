import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from bot_core.risk.simulation import ProfileSimulationResult, RiskSimulationReport, StressTestResult
from bot_core.security.signing import build_hmac_signature


def _market_payload() -> dict[str, object]:
    now = datetime(2024, 1, 1, tzinfo=timezone.utc).isoformat()
    snapshots = {
        "BTCUSDT": {
            "interval": "1h",
            "bar_count": 168,
            "price_change_pct": 2.0,
            "volatility_pct": 5.0,
            "max_drawdown_pct": 3.0,
            "average_volume": 1_000_000.0,
            "liquidity_usd": 450_000.0,
            "momentum_score": 1.5,
            "start": now,
            "end": now,
        },
        "UNIUSDT": {
            "interval": "1h",
            "bar_count": 168,
            "price_change_pct": 1.0,
            "volatility_pct": 8.0,
            "max_drawdown_pct": 6.0,
            "average_volume": 250_000.0,
            "liquidity_usd": 120_000.0,
            "momentum_score": -0.5,
            "start": now,
            "end": now,
        },
    }
    return {
        "generated_at": now,
        "environment": "paper",
        "governor": "core",
        "interval": "1h",
        "lookback_bars": 168,
        "symbols": list(snapshots.keys()),
        "snapshots": snapshots,
    }


def _risk_report() -> RiskSimulationReport:
    stress = StressTestResult(
        name="latency_spike",
        status="warning",
        metrics={"avg_order_latency_ms": 340.0, "max_order_latency_ms": 420.0},
    )
    profile = ProfileSimulationResult(
        profile="balanced",
        base_equity=100_000.0,
        final_equity=101_000.0,
        total_return_pct=0.01,
        max_drawdown_pct=0.04,
        worst_daily_loss_pct=0.02,
        realized_volatility=0.09,
        breaches=(),
        stress_tests=(stress,),
        sample_size=240,
    )
    return RiskSimulationReport(
        generated_at="2024-01-02T00:00:00Z",
        base_equity=100_000.0,
        profiles=(profile,),
        synthetic_data=False,
    )


def test_calibrate_stress_lab_cli(tmp_path: Path) -> None:
    market_path = tmp_path / "market_intel.json"
    market_path.write_text(json.dumps(_market_payload()), encoding="utf-8")

    risk_report_path = tmp_path / "risk_report.json"
    _risk_report().write_json(risk_report_path)

    segments_path = tmp_path / "segments.json"
    segments_path.write_text(
        json.dumps([
            {"name": "core", "symbols": ["BTCUSDT"]},
            {"name": "alts", "symbols": ["UNIUSDT"]},
        ]),
        encoding="utf-8",
    )

    key_path = tmp_path / "hmac.key"
    key_path.write_bytes(b"stage6-key")

    output_json = tmp_path / "calibration.json"
    output_csv = tmp_path / "calibration.csv"
    signature_path = tmp_path / "calibration.sig"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_stress_lab_thresholds.py",
            "--market-intel",
            str(market_path),
            "--risk-report",
            str(risk_report_path),
            "--segments",
            str(segments_path),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
            "--signing-key",
            str(key_path),
            "--signing-key-id",
            "stage6",
            "--signature-path",
            str(signature_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_json.exists()
    assert output_csv.exists()
    assert signature_path.exists()
    assert "kalibrację Stress Lab" in result.stdout

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "stage6.risk.stress_lab.calibration"
    segments = {entry["segment"]: entry for entry in payload["segments"]}
    assert segments["core"]["liquidity_warning_threshold_usd"] >= 120_000.0
    assert payload["latency_warning_threshold_ms"] >= 340.0
    assert payload["latency_critical_threshold_ms"] == pytest.approx(408.0)

    recorded_signature = json.loads(signature_path.read_text(encoding="utf-8"))
    assert recorded_signature["schema"] == "stage6.risk.stress_lab.calibration.signature"
    assert recorded_signature["target"] == output_json.name
    expected_signature = build_hmac_signature(payload, key=b"stage6-key", key_id="stage6")
    assert recorded_signature["signature"] == expected_signature


def test_calibrate_cli_with_volume_segments(tmp_path: Path) -> None:
    payload = _market_payload()
    payload["snapshots"]["SOLUSDT"] = {
        "interval": "1h",
        "bar_count": 168,
        "price_change_pct": 0.5,
        "volatility_pct": 6.0,
        "max_drawdown_pct": 4.0,
        "average_volume": 180_000.0,
        "liquidity_usd": 160_000.0,
        "momentum_score": 0.1,
    }
    payload["snapshots"]["DOGEUSDT"] = {
        "interval": "1h",
        "bar_count": 168,
        "price_change_pct": -0.1,
        "volatility_pct": 9.0,
        "max_drawdown_pct": 7.0,
        "average_volume": 90_000.0,
        "liquidity_usd": 70_000.0,
        "momentum_score": -0.2,
    }

    market_path = tmp_path / "market_volume.json"
    market_path.write_text(json.dumps(payload), encoding="utf-8")

    output_json = tmp_path / "auto_segments.json"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/calibrate_stress_lab_thresholds.py",
            "--market-intel",
            str(market_path),
            "--output-json",
            str(output_json),
            "--volume-buckets",
            "2",
            "--volume-min-symbols",
            "2",
            "--volume-name-prefix",
            "liq",
            "--volume-risk-budget-prefix",
            "bucket-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload_json = json.loads(output_json.read_text(encoding="utf-8"))
    assert {segment["segment"] for segment in payload_json["segments"]} == {"liq_1", "liq_2"}
    first_segment = next(seg for seg in payload_json["segments"] if seg["segment"] == "liq_1")
    second_segment = next(seg for seg in payload_json["segments"] if seg["segment"] == "liq_2")
    # segment 1 powinien zawierać aktywa o najwyższej płynności
    assert first_segment["symbol_count"] >= second_segment["symbol_count"]
    assert "Zapisano kalibrację Stress Lab" in result.stdout
