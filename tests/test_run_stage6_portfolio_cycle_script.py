import json
from pathlib import Path

import pytest

from scripts.run_stage6_portfolio_cycle import run as run_cycle


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
risk_profiles:
  balanced:
    max_daily_loss_pct: 5.0
    max_position_pct: 10.0
    target_volatility: 15.0
    max_leverage: 2.0
    stop_loss_atr_multiple: 3.0
    max_open_positions: 5
    hard_drawdown_pct: 25.0
""",
        encoding="utf-8",
    )


def test_run_stage6_portfolio_cycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "audit" / "decision.jsonl"
    config_path = tmp_path / "core.yaml"
    _write_config(config_path, log_path)

    allocations_path = tmp_path / "allocations.json"
    allocations_path.write_text(json.dumps({"BTC_USDT": 0.2}), encoding="utf-8")

    market_path = tmp_path / "market.json"
    market_payload = {
        "generated_at": "2024-04-10T09:00:00Z",
        "snapshots": {
            "BTC_USDT": {
                "interval": "1h",
                "bar_count": 24,
                "price_change_pct": 5.0,
                "volatility_pct": 9.0,
                "max_drawdown_pct": 6.5,
                "average_volume": 20000,
                "liquidity_usd": 400000,
                "momentum_score": 1.4,
            }
        },
    }
    market_path.write_text(json.dumps(market_payload), encoding="utf-8")

    slo_path = tmp_path / "slo.json"
    slo_path.write_text(
        json.dumps(
            {
                "results": {
                    "latency": {
                        "indicator": "latency_ms",
                        "target": 250,
                        "status": "ok",
                        "severity": "info",
                        "value": 180,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    stress_path = tmp_path / "stress.json"
    stress_path.write_text(
        json.dumps(
            {
                "overrides": [
                    {
                        "severity": "critical",
                        "reason": "latency_spike",
                        "symbol": "BTC_USDT",
                        "force_rebalance": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary_path = tmp_path / "outputs" / "summary.json"
    signature_path = tmp_path / "outputs" / "summary.sig"
    csv_path = tmp_path / "outputs" / "summary.csv"

    monkeypatch.setenv("PORTFOLIO_STAGE6_KEY", "cycle-secret")

    exit_code = run_cycle(
        [
            "--config",
            str(config_path),
            "--environment",
            "paper",
            "--governor",
            "core",
            "--allocations",
            str(allocations_path),
            "--portfolio-value",
            "200000",
            "--market-intel",
            str(market_path),
            "--slo-report",
            str(slo_path),
            "--stress-report",
            str(stress_path),
            "--summary",
            str(summary_path),
            "--summary-csv",
            str(csv_path),
            "--summary-signature",
            str(signature_path),
            "--signing-key",
            "summary-secret",
            "--signing-key-id",
            "ops",
        ]
    )

    assert exit_code == 0
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["adjustment_count"] == 1
    assert signature_path.exists()
    signature = json.loads(signature_path.read_text(encoding="utf-8"))
    assert signature["key_id"] == "ops"
    assert csv_path.exists()
    assert log_path.exists()
