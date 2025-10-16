import json
from pathlib import Path

import pytest

from bot_core.portfolio import (
    PortfolioAssetConfig,
    PortfolioCycleConfig,
    PortfolioCycleInputs,
    PortfolioCycleOutputConfig,
    PortfolioDecisionLog,
    PortfolioGovernor,
    PortfolioGovernorConfig,
    PortfolioHypercareCycle,
)
from bot_core.security.signing import verify_hmac_signature


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _build_governor(tmp_path: Path) -> tuple[PortfolioGovernor, PortfolioDecisionLog]:
    config = PortfolioGovernorConfig(
        name="core",
        portfolio_id="core",
        assets=(
            PortfolioAssetConfig(
                symbol="BTC_USDT",
                target_weight=0.5,
                min_weight=0.1,
                max_weight=0.6,
                max_volatility_pct=10.0,
            ),
        ),
        min_rebalance_value=1000.0,
        min_rebalance_weight=0.01,
    )
    log_path = tmp_path / "decision.jsonl"
    decision_log = PortfolioDecisionLog(
        jsonl_path=log_path,
        signing_key=b"decision-secret",
        signing_key_id="stage6-log",
    )
    return PortfolioGovernor(config, decision_log=decision_log), decision_log


def test_portfolio_hypercare_cycle_generates_signed_summary(tmp_path: Path) -> None:
    governor, decision_log = _build_governor(tmp_path)

    allocations_path = tmp_path / "inputs" / "allocations.json"
    _write_json(allocations_path, {"BTC_USDT": 0.2})

    market_path = tmp_path / "inputs" / "market.json"
    _write_json(
        market_path,
        {
            "generated_at": "2024-04-05T12:00:00Z",
            "snapshots": {
                "BTC_USDT": {
                    "interval": "1h",
                    "bar_count": 48,
                    "price_change_pct": 4.2,
                    "volatility_pct": 8.0,
                    "max_drawdown_pct": 6.0,
                    "average_volume": 12000,
                    "liquidity_usd": 300000,
                    "momentum_score": 1.1,
                }
            },
        },
    )

    slo_path = tmp_path / "inputs" / "slo.json"
    _write_json(
        slo_path,
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
        },
    )

    stress_path = tmp_path / "inputs" / "stress.json"
    _write_json(
        stress_path,
        {
            "overrides": [
                {
                    "severity": "critical",
                    "reason": "latency_spike",
                    "symbol": "BTC_USDT",
                    "force_rebalance": True,
                }
            ]
        },
    )

    summary_path = tmp_path / "outputs" / "summary.json"
    signature_path = tmp_path / "outputs" / "summary.sig"
    csv_path = tmp_path / "outputs" / "summary.csv"

    cycle_config = PortfolioCycleConfig(
        inputs=PortfolioCycleInputs(
            allocations_path=allocations_path,
            market_intel_path=market_path,
            portfolio_value=150_000.0,
            slo_report_path=slo_path,
            stress_report_path=stress_path,
        ),
        output=PortfolioCycleOutputConfig(
            summary_path=summary_path,
            signature_path=signature_path,
            csv_path=csv_path,
            pretty_json=True,
        ),
        signing_key=b"summary-secret",
        signing_key_id="ops",
        metadata={"run": "test"},
        log_context={"trigger": "unit-test"},
    )

    cycle = PortfolioHypercareCycle(governor, cycle_config)
    result = cycle.run()

    assert result.summary_path == summary_path
    assert summary_path.exists()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["rebalance_required"] is True
    assert payload["adjustment_count"] == 1
    assert payload["metadata"]["run"] == "test"
    assert payload["slo_statuses"]["latency"]["status"] == "ok"
    assert payload["stress_overrides"][0]["reason"] == "latency_spike"

    assert signature_path.exists()
    signature = json.loads(signature_path.read_text(encoding="utf-8"))
    assert signature["key_id"] == "ops"
    assert verify_hmac_signature(payload, signature, key=b"summary-secret")

    assert csv_path.exists()
    csv_content = csv_path.read_text(encoding="utf-8")
    assert "BTC_USDT" in csv_content

    assert decision_log.path is not None
    decision_entries = [
        line
        for line in decision_log.path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert decision_entries, "Decision log powinien zawierać wpis"


def test_portfolio_hypercare_cycle_requires_market_intel(tmp_path: Path) -> None:
    governor, _ = _build_governor(tmp_path)

    allocations_path = tmp_path / "alloc.json"
    _write_json(allocations_path, {"BTC_USDT": 0.5})
    market_path = tmp_path / "market.json"
    _write_json(market_path, {"snapshots": {}})

    cycle_config = PortfolioCycleConfig(
        inputs=PortfolioCycleInputs(
            allocations_path=allocations_path,
            market_intel_path=market_path,
            portfolio_value=100_000.0,
        ),
        output=PortfolioCycleOutputConfig(summary_path=tmp_path / "summary.json"),
    )

    cycle = PortfolioHypercareCycle(governor, cycle_config)
    with pytest.raises(ValueError):
        cycle.run()
