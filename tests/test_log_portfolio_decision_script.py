from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from bot_core.security.signing import build_hmac_signature


def _write_config(path: Path, log_path: Path) -> None:
    path.write_text(
        f"""
environments:
  paper:
    exchange: binance
    environment: paper
    keychain_key: paper-key
    data_cache_path: {path.parent}
    risk_profile: balanced
    alert_channels: []
risk_profiles:
  balanced:
    max_daily_loss_pct: 5.0
    max_position_pct: 10.0
    target_volatility: 15.0
    max_leverage: 2.0
    stop_loss_atr_multiple: 3.0
    max_open_positions: 5
    hard_drawdown_pct: 25.0
portfolio_governors:
  core:
    name: core
    portfolio_id: core
    assets:
      - symbol: BTC_USDT
        target_weight: 0.5
        min_weight: 0.1
        max_weight: 0.6
runtime:
  portfolio_decision_log:
    path: {log_path}
    signing_key_env: PORTFOLIO_STAGE6_KEY
    signing_key_id: stage6-config
""",
        encoding="utf-8",
    )


def test_log_portfolio_decision_script_generates_signed_entry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "audit" / "portfolio_decision.jsonl"
    config_path = tmp_path / "core.yaml"
    _write_config(config_path, log_path)

    allocations_path = tmp_path / "allocations.json"
    allocations_path.write_text(json.dumps({"BTC_USDT": 0.25}), encoding="utf-8")

    market_intel_path = tmp_path / "market.json"
    market_payload = {
        "snapshots": {
            "BTC_USDT": {
                "interval": "1h",
                "bar_count": 24,
                "start": "2024-01-01T00:00:00Z",
                "end": "2024-01-02T00:00:00Z",
                "price_change_pct": 5.0,
                "volatility_pct": 12.0,
                "max_drawdown_pct": 8.0,
                "average_volume": 10_000.0,
                "liquidity_usd": 500_000.0,
                "momentum_score": 1.5,
            }
        }
    }
    market_intel_path.write_text(json.dumps(market_payload), encoding="utf-8")

    slo_path = tmp_path / "slo.json"
    slo_payload = {
        "results": {
            "latency": {
                "indicator": "router_latency_ms",
                "target": 250.0,
                "comparison": "<=",
                "status": "ok",
                "severity": "info",
                "value": 180.0,
                "warning_threshold": 200.0,
                "sample_size": 1000,
            }
        }
    }
    slo_path.write_text(json.dumps(slo_payload), encoding="utf-8")

    stress_path = tmp_path / "stress.json"
    stress_payload = {
        "overrides": [
            {
                "severity": "critical",
                "reason": "latency_spike",
                "symbol": "BTC_USDT",
                "weight_multiplier": 0.4,
                "force_rebalance": True,
            }
        ]
    }
    stress_path.write_text(json.dumps(stress_payload), encoding="utf-8")

    key_value = "stage6-secret"
    monkeypatch.setenv("PORTFOLIO_STAGE6_KEY", key_value)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/log_portfolio_decision.py",
            "--config",
            str(config_path),
            "--environment",
            "paper",
            "--governor",
            "core",
            "--allocations",
            str(allocations_path),
            "--portfolio-value",
            "150000",
            "--market-intel",
            str(market_intel_path),
            "--slo-status",
            str(slo_path),
            "--stress-overrides",
            str(stress_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert log_path.exists()
    assert "z podpisem HMAC" in result.stdout

    lines = [line for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["portfolio_id"] == "core"
    assert entry["metadata"]["inputs"]["stress_overrides"] == str(stress_path)
    assert entry["metadata"]["stress_overrides"][0]["reason"] == "latency_spike"
    signature = entry.pop("signature")
    expected_signature = build_hmac_signature(entry, key=key_value.encode("utf-8"), key_id="stage6-config")
    assert signature == expected_signature
