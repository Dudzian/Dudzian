from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from bot_core.ai import AdaptiveStrategyLearner, ModelRepository
from bot_core.runtime.journal import (
    AdaptiveDecisionJournal,
    InMemoryTradingDecisionJournal,
    TradingDecisionEvent,
)


class _StubOrchestrator:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def record_strategy_performance(
        self,
        strategy: str,
        regime: str,
        *,
        hit_rate: float,
        pnl: float,
        sharpe: float,
        observations: int,
        timestamp: datetime | None,
    ) -> None:
        self.calls.append((strategy, regime))


def test_adaptive_journal_updates_learner(tmp_path) -> None:
    registry = tmp_path / "models"
    repository = ModelRepository(registry)
    orchestrator = _StubOrchestrator()
    learner = AdaptiveStrategyLearner(repository=repository, orchestrator=orchestrator)
    learner.register_strategies("trend", ["trend_following"])

    base = InMemoryTradingDecisionJournal()
    journal = AdaptiveDecisionJournal(journal=base, learner=learner, persist_interval=1)

    event = TradingDecisionEvent(
        event_type="ai_inference",
        timestamp=datetime.now(timezone.utc),
        environment="paper",
        portfolio="autotrader",
        risk_profile="trend",
        strategy="trend_following",
        metadata={
            "ai_inference": json.dumps(
                {
                    "success_probability": 0.8,
                    "expected_return_bps": 120.0,
                    "signal_after_adjustment": 0.9,
                }
            ),
            "activation": json.dumps({"regime": "trend"}),
        },
    )

    journal.record(event)
    learner.persist()

    snapshot = learner.snapshot()
    assert "trend" in snapshot
    policies = snapshot["trend"]["strategies"]
    assert policies, "expected adaptive learner to accumulate strategy stats"
    stats = next(iter(policies))
    assert stats["plays"] >= 1
    assert stats["total_reward"] > 0
    assert orchestrator.calls == [("trend_following", "trend")]
    artifact_path = registry / "adaptive_strategy_policy.json"
    assert artifact_path.exists()


def test_adaptive_learner_build_dynamic_preset_uses_metrics(tmp_path) -> None:
    registry = tmp_path / "models"
    repository = ModelRepository(registry)
    learner = AdaptiveStrategyLearner(repository=repository, orchestrator=None)
    learner.register_strategies(
        "trend",
        ["trend_following", "mean_reversion", "volatility_target"],
    )

    now = datetime.now(timezone.utc)
    learner.observe(
        regime="trend",
        strategy="trend_following",
        metrics={"hit_rate": 0.78, "pnl": 40.0, "sharpe": 1.5},
        timestamp=now,
    )
    learner.observe(
        regime="trend",
        strategy="mean_reversion",
        metrics={"hit_rate": 0.55, "pnl": 8.0, "sharpe": 0.4},
        timestamp=now,
    )
    learner.observe(
        regime="trend",
        strategy="volatility_target",
        metrics={"hit_rate": 0.62, "pnl": 18.0, "sharpe": 0.9},
        timestamp=now,
    )

    metrics = {"confidence": 0.85, "risk_score": 0.3, "trend_strength": 0.9}
    preset = learner.build_dynamic_preset("trend", metrics=metrics)

    assert preset is not None
    assert preset["regime"] == "trend"
    assert "generated_at" in preset
    strategies = {
        entry["name"]: entry["weight"]
        for entry in preset["strategies"]
    }
    assert "trend_following" in strategies
    assert pytest.approx(sum(strategies.values()), rel=1e-9) == 1.0
    assert strategies["trend_following"] > strategies["mean_reversion"]
    assert "metrics" in preset
    assert pytest.approx(preset["metrics"]["confidence"], rel=1e-6) == 0.85
    assert pytest.approx(preset["metadata"]["confidence"], rel=1e-6) == 0.85
