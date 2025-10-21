from __future__ import annotations

import os

import pandas as pd
import pytest

os.environ.setdefault("BOT_CORE_MINIMAL_EXCHANGES", "1")
os.environ.setdefault("BOT_CORE_MINIMAL_CORE", "1")
os.environ.setdefault("BOT_CORE_MINIMAL_DECISION", "1")

from bot_core.ai.manager import AIManager
from bot_core.ai.pipeline import train_gradient_boosting_model
from bot_core.config.models import (
    DecisionEngineConfig,
    DecisionOrchestratorThresholds,
)
from bot_core.decision.orchestrator import DecisionOrchestrator
from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.profiles.manual import ManualProfile


def _decision_engine_config() -> DecisionEngineConfig:
    return DecisionEngineConfig(
        orchestrator=DecisionOrchestratorThresholds(
            max_cost_bps=25.0,
            min_net_edge_bps=0.0,
            max_daily_loss_pct=0.15,
            max_drawdown_pct=0.3,
            max_position_ratio=0.8,
            max_open_positions=12,
            max_latency_ms=500.0,
        ),
        profile_overrides={},
        stress_tests=None,
        min_probability=0.0,
        require_cost_data=False,
        penalty_cost_bps=0.0,
    )


def _risk_profile(name: str = "integration") -> ManualProfile:
    return ManualProfile(
        name=name,
        max_positions=5,
        max_leverage=5.0,
        drawdown_limit=0.4,
        daily_loss_limit=0.2,
        max_position_pct=0.5,
        target_volatility=0.15,
        stop_loss_atr_multiple=2.0,
    )


def _snapshot(equity: float = 100_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        balances={"USDT": equity},
        total_equity=equity,
        available_margin=equity,
        maintenance_margin=0.0,
    )


@pytest.mark.parametrize("notional", [1_000.0])
def test_signal_flow_to_order_request(tmp_path, notional: float) -> None:
    features = ["feature_a", "feature_b"]
    frame = pd.DataFrame(
        {
            "feature_a": [float(1 + idx) for idx in range(32)],
            "feature_b": [float(2 + idx * 0.5) for idx in range(32)],
        }
    )
    frame["target"] = frame["feature_a"] * 0.002 + frame["feature_b"] * 0.001

    artifact_dir = tmp_path / "artifacts"
    artifact_path = train_gradient_boosting_model(
        frame,
        features,
        "target",
        output_dir=artifact_dir,
        model_name="autotrader",
        metadata={"training_rows": len(frame)},
    )

    ai_manager = AIManager(model_dir=tmp_path / "cache")
    orchestrator = DecisionOrchestrator(_decision_engine_config())
    ai_manager.attach_decision_orchestrator(orchestrator, default_model="autotrader")
    ai_manager.load_decision_artifact("autotrader", artifact_path, set_default=True)
    ai_manager.require_real_models()

    latest_row = frame.iloc[-1][features]
    feature_payload = {name: float(latest_row[name]) for name in features}
    score = ai_manager.score_decision_features(feature_payload)
    assert isinstance(score.expected_return_bps, float)

    decision_payload = ai_manager.build_decision_engine_payload(
        strategy="integration-test",
        action="enter",
        risk_profile="integration",
        symbol="BTCUSDT",
        notional=notional,
        features=feature_payload,
    )

    decision_metadata = {
        "features": dict(feature_payload),
        "candidate": decision_payload["candidate"],
        "ai": decision_payload["ai"],
    }
    order_metadata = {"decision_engine": decision_metadata}

    price = 20_000.0
    quantity = notional / price
    order = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=quantity,
        order_type="market",
        price=price,
        atr=100.0,
        stop_price=price - 200.0,
        metadata=order_metadata,
    )

    engine = ThresholdRiskEngine()
    engine.attach_decision_orchestrator(orchestrator)
    engine.register_profile(_risk_profile())
    result = engine.apply_pre_trade_checks(
        order,
        account=_snapshot(),
        profile_name="integration",
    )

    assert result.allowed is True
    assert result.metadata is not None
    decision_details = result.metadata.get("decision_orchestrator")
    assert isinstance(decision_details, dict)
    assert decision_details.get("accepted") is True
    assert decision_details.get("model_expected_return_bps") is not None
    candidate_details = decision_details.get("candidate", {})
    assert candidate_details.get("strategy") == "integration-test"
