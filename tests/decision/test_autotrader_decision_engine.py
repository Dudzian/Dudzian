from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import types

import pandas as pd

import pytest

from bot_core.ai.pipeline import register_model_artifact, train_gradient_boosting_model
from bot_core.auto_trader.app import AutoTrader
from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds
from bot_core.decision import DecisionOrchestrator
from bot_core.ai.regime import MarketRegime, MarketRegimeAssessment


class _Emitter:
    def __init__(self) -> None:
        self.logs: list[tuple[str, dict[str, object]]] = []

    def log(self, message: str, level: str | None = None, **kwargs: object) -> None:  # pragma: no cover - minimal emitter
        self.logs.append((message, {"level": level, **kwargs}))


class _GUI:
    def __init__(self) -> None:
        self._demo = True

    def is_demo_mode_active(self) -> bool:
        return self._demo


class _RiskEngine:
    def snapshot_state(self, profile: str) -> dict[str, object]:
        return {
            "profile": profile,
            "start_of_day_equity": 100_000.0,
            "last_equity": 100_500.0,
            "peak_equity": 101_000.0,
            "daily_realized_pnl": 0.0,
            "positions": {},
        }


def _orchestrator_config() -> DecisionEngineConfig:
    thresholds = DecisionOrchestratorThresholds(
        max_cost_bps=25.0,
        min_net_edge_bps=0.3,
        max_daily_loss_pct=0.12,
        max_drawdown_pct=0.25,
        max_position_ratio=0.5,
        max_open_positions=5,
        max_latency_ms=250.0,
    )
    return DecisionEngineConfig(
        orchestrator=thresholds,
        min_probability=0.5,
        require_cost_data=False,
        penalty_cost_bps=0.0,
    )


def _bootstrap_context(orchestrator: DecisionOrchestrator) -> SimpleNamespace:
    return SimpleNamespace(
        decision_orchestrator=orchestrator,
        risk_engine=_RiskEngine(),
        decision_engine_config=_orchestrator_config(),
        risk_profile_name="paper",
    )


def _train_artifact(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "close": [-2.0, -1.5, -1.0, 1.0, 1.5, 2.0],
            "volume": [80.0, 90.0, 100.0, 100.0, 110.0, 120.0],
            "target": [-30.0, -25.0, -15.0, 15.0, 25.0, 35.0],
        }
    )
    return train_gradient_boosting_model(
        frame,
        ("close", "volume"),
        "target",
        output_dir=tmp_path,
        model_name="autotrader",
        metadata={"unit": "bps"},
    )


def test_decision_orchestrator_allows_positive_signal(tmp_path: Path) -> None:
    artifact = _train_artifact(tmp_path)
    orchestrator = DecisionOrchestrator(_orchestrator_config())
    register_model_artifact(
        orchestrator,
        artifact,
        name="autotrader",
        repository_root=tmp_path,
        set_default=True,
    )
    trader = AutoTrader(_Emitter(), _GUI(), lambda: "BTCUSDT", bootstrap_context=_bootstrap_context(orchestrator))
    data = pd.DataFrame(
        {
            "close": [1.0, 1.5, 2.0],
            "volume": [110.0, 130.0, 150.0],
        }
    )
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.8,
        risk_score=0.2,
        metrics={},
    )
    evaluation = trader._evaluate_decision_candidate(
        symbol="BTCUSDT",
        signal="buy",
        market_data=data,
        assessment=assessment,
        last_return=0.015,
    )
    assert evaluation is not None
    assert evaluation.accepted is True
    assert evaluation.thresholds_snapshot is not None
    assert trader._ai_feature_columns(data) == ["close", "volume"]
    decision_metadata = evaluation.candidate.metadata["decision_engine"]
    assert decision_metadata["feature_columns"] == ["close", "volume"]
    assert decision_metadata["feature_columns_source"] == "default"
    assert "configured_feature_columns" not in decision_metadata
    decision = trader._build_risk_decision(
        "BTCUSDT",
        "buy",
        assessment,
        effective_risk=0.25,
        summary=None,
        cooldown_active=False,
        cooldown_remaining=0.0,
        cooldown_reason=None,
        guardrail_reasons=[],
        guardrail_triggers=[],
        decision_engine=evaluation,
    )
    assert decision.should_trade is True
    assert decision.details["decision_engine"]["accepted"] is True
    assert decision.details["decision_engine"]["thresholds"]["min_probability"] == pytest.approx(0.5)


def test_auto_trader_attaches_ai_context_to_decision(tmp_path: Path) -> None:
    artifact = _train_artifact(tmp_path)
    orchestrator = DecisionOrchestrator(_orchestrator_config())
    register_model_artifact(
        orchestrator,
        artifact,
        name="autotrader",
        repository_root=tmp_path,
        set_default=True,
    )
    trader = AutoTrader(_Emitter(), _GUI(), lambda: "ETHUSDT", bootstrap_context=_bootstrap_context(orchestrator))
    data = pd.DataFrame(
        {
            "close": [0.9, 1.1, 1.3],
            "volume": [95.0, 120.0, 140.0],
        }
    )
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.75,
        risk_score=0.22,
        metrics={},
    )
    ai_context = {
        "prediction": 0.003,
        "prediction_bps": 30.0,
        "threshold_bps": 5.0,
        "direction": "buy",
        "probability": 0.7,
        "evaluated_at": "2024-01-01T00:00:00+00:00",
    }
    evaluation = trader._evaluate_decision_candidate(
        symbol="ETHUSDT",
        signal="buy",
        market_data=data,
        assessment=assessment,
        last_return=0.012,
        ai_context=ai_context,
    )
    assert evaluation is not None
    assert evaluation.thresholds_snapshot is not None
    ai_meta = evaluation.candidate.metadata["decision_engine"]["ai"]
    assert ai_meta["prediction_bps"] == pytest.approx(30.0)
    assert ai_meta["direction"] == "buy"
    assert trader._ai_feature_columns(data) == ["close", "volume"]
    decision_meta = evaluation.candidate.metadata["decision_engine"]
    assert decision_meta["feature_columns"] == ["close", "volume"]
    assert decision_meta["feature_columns_source"] == "default"
    assert "configured_feature_columns" not in decision_meta

    decision = trader._build_risk_decision(
        "ETHUSDT",
        "buy",
        assessment,
        effective_risk=0.28,
        summary=None,
        cooldown_active=False,
        cooldown_remaining=0.0,
        cooldown_reason=None,
        guardrail_reasons=[],
        guardrail_triggers=[],
        decision_engine=evaluation,
        ai_context=ai_context,
    )
    ai_details = decision.details["decision_engine"]["ai"]
    assert ai_details["prediction_bps"] == pytest.approx(30.0)
    assert ai_details["direction"] == "buy"
    decision_meta = decision.details["decision_engine"]
    assert decision_meta["feature_columns"] == ["close", "volume"]
    assert decision_meta["feature_columns_source"] == "default"
    assert "configured_feature_columns" not in decision_meta


def test_auto_trader_uses_signal_service_feature_columns(tmp_path: Path) -> None:
    artifact = _train_artifact(tmp_path)
    orchestrator = DecisionOrchestrator(_orchestrator_config())
    register_model_artifact(
        orchestrator,
        artifact,
        name="autotrader",
        repository_root=tmp_path,
        set_default=True,
    )
    signal_service = types.SimpleNamespace(
        ai_feature_columns=("volume", "ema", "volume", "close", "close")
    )
    trader = AutoTrader(
        _Emitter(),
        _GUI(),
        lambda: "SOLUSDT",
        bootstrap_context=_bootstrap_context(orchestrator),
        signal_service=signal_service,
    )
    data = pd.DataFrame({"close": [1.0, 1.1], "volume": [90.0, 95.0]})
    assert trader._ai_feature_columns(data) == ["volume", "close"]
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.65,
        risk_score=0.18,
        metrics={},
    )
    evaluation = trader._evaluate_decision_candidate(
        symbol="SOLUSDT",
        signal="buy",
        market_data=data,
        assessment=assessment,
        last_return=0.01,
    )
    assert evaluation is not None
    decision_meta = evaluation.candidate.metadata["decision_engine"]
    assert decision_meta["feature_columns"] == ["volume", "close"]
    assert decision_meta["feature_columns_source"] == "configured"
    assert decision_meta["configured_feature_columns"] == ["volume", "ema", "close"]
    assert len(decision_meta["configured_feature_columns"]) == len(
        set(decision_meta["configured_feature_columns"])
    )
    assert "ema" not in decision_meta["features"]
    assert set(decision_meta["features"]).issuperset(
        {"volume", "close", "assessment_confidence", "assessment_risk", "signal_direction"}
    )
    decision = trader._build_risk_decision(
        "SOLUSDT",
        "buy",
        assessment,
        effective_risk=0.18,
        summary=None,
        cooldown_active=False,
        cooldown_remaining=0.0,
        cooldown_reason=None,
        guardrail_reasons=[],
        guardrail_triggers=[],
        decision_engine=evaluation,
    )
    final_meta = decision.details["decision_engine"]
    assert final_meta["feature_columns"] == ["volume", "close"]
    assert final_meta["feature_columns_source"] == "configured"
    assert final_meta["configured_feature_columns"] == ["volume", "ema", "close"]


def test_auto_trader_falls_back_when_signal_service_columns_missing(tmp_path: Path) -> None:
    artifact = _train_artifact(tmp_path)
    orchestrator = DecisionOrchestrator(_orchestrator_config())
    register_model_artifact(
        orchestrator,
        artifact,
        name="autotrader",
        repository_root=tmp_path,
        set_default=True,
    )
    signal_service = types.SimpleNamespace(ai_feature_columns=("ema", "vwma", "ema", "vwma"))
    trader = AutoTrader(
        _Emitter(),
        _GUI(),
        lambda: "ADAUSDT",
        bootstrap_context=_bootstrap_context(orchestrator),
        signal_service=signal_service,
    )
    data = pd.DataFrame({"close": [1.0, 1.05], "volume": [100.0, 102.0]})
    assert trader._ai_feature_columns(data) == ["close", "volume"]
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.6,
        risk_score=0.15,
        metrics={},
    )
    evaluation = trader._evaluate_decision_candidate(
        symbol="ADAUSDT",
        signal="buy",
        market_data=data,
        assessment=assessment,
        last_return=0.008,
    )
    assert evaluation is not None
    decision_meta = evaluation.candidate.metadata["decision_engine"]
    assert decision_meta["feature_columns"] == ["close", "volume"]
    assert decision_meta["feature_columns_source"] == "fallback"
    assert decision_meta["configured_feature_columns"] == ["ema", "vwma"]
    assert set(decision_meta["features"]).issuperset(
        {"close", "volume", "assessment_confidence", "assessment_risk", "signal_direction"}
    )
    decision = trader._build_risk_decision(
        "ADAUSDT",
        "buy",
        assessment,
        effective_risk=0.22,
        summary=None,
        cooldown_active=False,
        cooldown_remaining=0.0,
        cooldown_reason=None,
        guardrail_reasons=[],
        guardrail_triggers=[],
        decision_engine=evaluation,
    )
    final_meta = decision.details["decision_engine"]
    assert final_meta["feature_columns"] == ["close", "volume"]
    assert final_meta["feature_columns_source"] == "fallback"
    assert final_meta["configured_feature_columns"] == ["ema", "vwma"]


def test_auto_trader_exposes_feature_columns_without_decision_evaluation(tmp_path: Path) -> None:
    artifact = _train_artifact(tmp_path)
    orchestrator = DecisionOrchestrator(_orchestrator_config())
    register_model_artifact(
        orchestrator,
        artifact,
        name="autotrader",
        repository_root=tmp_path,
        set_default=True,
    )
    trader = AutoTrader(
        _Emitter(),
        _GUI(),
        lambda: "BTCUSDT",
        bootstrap_context=_bootstrap_context(orchestrator),
    )
    data = pd.DataFrame({"close": [1.0, 1.1], "volume": [100.0, 105.0]})
    assert trader._ai_feature_columns(data) == ["close", "volume"]
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.7,
        risk_score=0.2,
        metrics={},
    )
    decision = trader._build_risk_decision(
        "BTCUSDT",
        "hold",
        assessment,
        effective_risk=0.2,
        summary=None,
        cooldown_active=False,
        cooldown_remaining=0.0,
        cooldown_reason=None,
        guardrail_reasons=[],
        guardrail_triggers=[],
        decision_engine=None,
        ai_context=None,
    )
    decision_meta = decision.details["decision_engine"]
    assert decision_meta["feature_columns"] == ["close", "volume"]
    assert decision_meta["feature_columns_source"] == "default"
    assert "configured_feature_columns" not in decision_meta


def test_decision_orchestrator_blocks_negative_signal(tmp_path: Path) -> None:
    artifact = _train_artifact(tmp_path)
    orchestrator = DecisionOrchestrator(_orchestrator_config())
    register_model_artifact(
        orchestrator,
        artifact,
        name="autotrader",
        repository_root=tmp_path,
        set_default=True,
    )
    trader = AutoTrader(_Emitter(), _GUI(), lambda: "BTCUSDT", bootstrap_context=_bootstrap_context(orchestrator))
    original_builder = trader._build_decision_candidate

    def _builder_with_cost(self, **kwargs: object):
        candidate = original_builder(**kwargs)
        if candidate is not None:
            candidate.cost_bps_override = 100.0
        return candidate

    trader._build_decision_candidate = types.MethodType(_builder_with_cost, trader)
    data = pd.DataFrame(
        {
            "close": [-2.0, -2.0, -2.0],
            "volume": [80.0, 80.0, 80.0],
        }
    )
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.7,
        risk_score=0.35,
        metrics={},
    )
    evaluation = trader._evaluate_decision_candidate(
        symbol="BTCUSDT",
        signal="sell",
        market_data=data,
        assessment=assessment,
        last_return=-0.04,
    )
    assert evaluation is not None
    assert evaluation.accepted is False
    final_signal = "sell" if evaluation.accepted else "hold"
    decision = trader._build_risk_decision(
        "BTCUSDT",
        final_signal,
        assessment,
        effective_risk=0.35,
        summary=None,
        cooldown_active=False,
        cooldown_remaining=0.0,
        cooldown_reason=None,
        guardrail_reasons=[],
        guardrail_triggers=[],
        decision_engine=evaluation,
    )
    assert decision.should_trade is False
    decision_info = decision.details["decision_engine"]
    assert decision_info["accepted"] is False
    assert "thresholds" in decision_info
    assert decision_info["thresholds"]["max_cost_bps"] == pytest.approx(25.0)
