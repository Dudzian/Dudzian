from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from bot_core.portfolio import (
    PortfolioAdjustment,
    PortfolioAdvisory,
    PortfolioDecision,
    PortfolioDecisionLog,
)
from bot_core.security.signing import build_hmac_signature


def _decision() -> PortfolioDecision:
    now = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    adjustment = PortfolioAdjustment(
        symbol="BTC_USDT",
        current_weight=0.2,
        proposed_weight=0.4,
        reason="dryf alokacji",
        severity="warning",
        metadata={"target_weight": 0.4},
    )
    advisory = PortfolioAdvisory(
        code="risk_budget.core",
        severity="warning",
        message="volatility 25% > limit 20%",
        symbols=("BTC_USDT",),
        metrics={"volatility_pct": 25.0},
    )
    return PortfolioDecision(
        timestamp=now,
        portfolio_id="core",
        portfolio_value=150_000.0,
        adjustments=(adjustment,),
        advisories=(advisory,),
        rebalance_required=True,
    )


def test_portfolio_decision_log_generates_signed_entry(tmp_path: Path) -> None:
    log_path = tmp_path / "portfolio.jsonl"
    key = b"K" * 64
    log = PortfolioDecisionLog(
        jsonl_path=log_path,
        signing_key=key,
        signing_key_id="stage6",
        max_entries=4,
    )

    decision = _decision()
    entry = log.record(decision, metadata={"environment": "paper", "tags": ["core"]})

    assert entry["metadata"]["tags"] == ["core"]
    assert entry["signature"]["key_id"] == "stage6"
    assert log_path.exists()

    stored = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(stored) == 1
    recorded = stored[0]
    assert recorded["portfolio_id"] == "core"
    signature = recorded.pop("signature")
    expected_signature = build_hmac_signature(recorded, key=key, key_id="stage6")
    assert signature == expected_signature


def test_portfolio_decision_log_tail_returns_recent_entries(tmp_path: Path) -> None:
    log = PortfolioDecisionLog(max_entries=2)
    decision = _decision()
    log.record(decision)
    later = decision.to_dict()
    later["timestamp"] = datetime(2024, 1, 1, 12, 5, 0, tzinfo=timezone.utc).isoformat()
    log.record(decision, metadata={"snapshot": later["timestamp"]})

    entries = log.tail(limit=2)
    assert len(entries) == 2
    assert entries[-1]["metadata"]["snapshot"] == later["timestamp"]
