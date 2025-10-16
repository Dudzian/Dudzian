import json
from datetime import timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from bot_core.portfolio.io import (
    load_allocations_file,
    load_market_intel_report,
    parse_slo_status_payload,
    parse_stress_overrides_payload,
    resolve_decision_log_config,
)


def test_load_market_intel_report_parses_snapshots(tmp_path: Path) -> None:
    report = {
        "generated_at": "2024-04-01T10:30:00Z",
        "environment": "paper",
        "governor": "core",
        "interval": "1h",
        "snapshots": {
            "BTC_USDT": {
                "interval": "1h",
                "bar_count": 24,
                "price_change_pct": 3.5,
                "volatility_pct": 8.2,
                "max_drawdown_pct": 6.1,
                "average_volume": 15000,
                "liquidity_usd": 420000,
                "momentum_score": 1.2,
                "metadata": {"bars_used": 24},
            }
        },
    }
    path = tmp_path / "market.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    snapshots, metadata = load_market_intel_report(path)
    assert "BTC_USDT" in snapshots
    snapshot = snapshots["BTC_USDT"]
    assert snapshot.volatility_pct == pytest.approx(8.2)
    assert snapshot.average_volume == pytest.approx(15000)
    assert metadata["environment"] == "paper"
    assert metadata["governor"] == "core"
    assert metadata["generated_at"].tzinfo == timezone.utc


def test_load_allocations_file_validates_structure(tmp_path: Path) -> None:
    alloc_path = tmp_path / "alloc.yaml"
    alloc_path.write_text("BTC_USDT: 0.4\nETH_USDT: 0.6\n", encoding="utf-8")

    allocations = load_allocations_file(alloc_path)
    assert allocations == {"BTC_USDT": 0.4, "ETH_USDT": 0.6}

    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_allocations_file(invalid_path)


def test_parse_slo_status_payload(tmp_path: Path) -> None:
    payload = {
        "results": {
            "latency": {
                "indicator": "router_latency_ms",
                "target": 250,
                "comparison": "<=",
                "status": "ok",
                "severity": "info",
                "value": 180,
                "warning_threshold": 200,
                "sample_size": 1200,
            }
        }
    }
    statuses = parse_slo_status_payload(payload)
    assert "latency" in statuses
    status = statuses["latency"]
    assert status.target == pytest.approx(250.0)
    assert status.value == pytest.approx(180.0)
    assert status.status == "ok"
    assert status.severity == "info"


def test_parse_stress_overrides_payload_handles_tags() -> None:
    payload = {
        "overrides": [
            {
                "severity": "critical",
                "reason": "latency_spike",
                "symbol": "BTC_USDT",
                "weight_multiplier": 0.3,
                "tags": ["stress", 42],
            }
        ]
    }
    overrides = parse_stress_overrides_payload(payload)
    assert len(overrides) == 1
    override = overrides[0]
    assert override.reason == "latency_spike"
    assert override.weight_multiplier == pytest.approx(0.3)
    assert override.tags == ("stress", "42")


def test_resolve_decision_log_config_reads_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    log_path = tmp_path / "decision.jsonl"
    monkeypatch.setenv("PORTFOLIO_KEY", "secret")
    config = SimpleNamespace(
        portfolio_decision_log=SimpleNamespace(
            enabled=True,
            path=str(log_path),
            signing_key_env="PORTFOLIO_KEY",
            signing_key_id="stage6",
            max_entries=128,
            jsonl_fsync=True,
        )
    )

    path, kwargs = resolve_decision_log_config(config)
    assert path == log_path
    assert kwargs["signing_key"] == b"secret"
    assert kwargs["signing_key_id"] == "stage6"
    assert kwargs["max_entries"] == 128
    assert kwargs["jsonl_fsync"] is True


def test_resolve_decision_log_config_disabled() -> None:
    config = SimpleNamespace(portfolio_decision_log=SimpleNamespace(enabled=False))
    path, kwargs = resolve_decision_log_config(config)
    assert path is None
    assert kwargs == {}
