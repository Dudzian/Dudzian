"""End-to-end coverage for DecisionOrchestrator ↔ AutoTrader integration."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Mapping

import pandas as pd
import pytest

from bot_core.auto_trader import AutoTrader
from bot_core.ai.regime import MarketRegime, MarketRegimeAssessment
from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds
from bot_core.decision.orchestrator import DecisionOrchestrator
from bot_core.exchanges.base import OrderRequest
from bot_core.runtime.journal import InMemoryTradingDecisionJournal


class _Emitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, Mapping[str, Any]]] = []

    def emit(self, event: str, **payload: Any) -> None:
        self.events.append((event, payload))

    def log(self, *_args: Any, **_kwargs: Any) -> None:  # pragma: no cover - optional
        return None


class _Var:
    def __init__(self, value: str) -> None:
        self._value = value

    def get(self) -> str:
        return self._value


class _GUI:
    def __init__(self) -> None:
        self.timeframe_var = _Var("1h")
        self._demo = False
        self.ai_mgr: Any | None = None

    def is_demo_mode_active(self) -> bool:
        return self._demo


class _MarketDataProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def get_historical(self, symbol: str, timeframe: str, limit: int = 256) -> pd.DataFrame:
        del symbol, timeframe, limit
        return self._frame


class _AIManagerStub:
    ai_threshold_bps = 6.0
    is_degraded = False

    def assess_market_regime(self, symbol: str, market_data: pd.DataFrame) -> MarketRegimeAssessment:
        del symbol, market_data
        return MarketRegimeAssessment(
            regime=MarketRegime.TREND,
            confidence=0.85,
            risk_score=0.25,
            metrics={"atr": 1.2},
        )

    def get_regime_summary(self, symbol: str) -> None:
        del symbol
        return None

    def predict_series(self, symbol: str, df: pd.DataFrame) -> pd.Series:
        del symbol
        index = df.index[-1:]
        return pd.Series([0.03], index=index)

    def build_decision_engine_payload(
        self,
        *,
        strategy: str,
        action: str,
        risk_profile: str,
        symbol: str,
        notional: float,
        features: Mapping[str, float],
    ) -> Mapping[str, object]:
        return {
            "candidate": {
                "strategy": strategy,
                "action": action,
                "risk_profile": risk_profile,
                "symbol": symbol,
                "notional": notional,
                "features": dict(features),
            },
            "ai": {
                "direction": "buy",
                "prediction_bps": 22.0,
                "probability": 0.82,
                "threshold_bps": self.ai_threshold_bps,
            },
        }


class _RiskServiceStub:
    def __init__(self) -> None:
        self.decisions: list[Any] = []
        self.last_payload: Mapping[str, object] | None = None

    def evaluate_decision(self, decision: Any) -> Any:
        self.decisions.append(decision)
        payload = decision.details.get("decision_engine") if hasattr(decision, "details") else None
        if isinstance(payload, Mapping):
            self.last_payload = payload
        return SimpleNamespace(approved=True)

    def attach_decision_orchestrator(self, orchestrator: DecisionOrchestrator) -> None:
        self._orchestrator = orchestrator

    def snapshot_state(self, profile: str) -> Mapping[str, Any]:
        return {
            "profile": profile,
            "positions": {},
            "start_of_day_equity": 250_000.0,
            "last_equity": 250_000.0,
            "peak_equity": 250_000.0,
            "daily_realized_pnl": 0.0,
        }


class _ExecutionServiceStub:
    def __init__(self) -> None:
        self.requests: list[OrderRequest] = []

    def execute(self, request: OrderRequest, context: Mapping[str, Any]) -> None:
        del context
        self.requests.append(request)


def _decision_engine_config() -> DecisionEngineConfig:
    thresholds = DecisionOrchestratorThresholds(
        max_cost_bps=12.0,
        min_net_edge_bps=4.0,
        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.1,
        max_position_ratio=0.35,
        max_open_positions=5,
        max_latency_ms=300.0,
    )
    return DecisionEngineConfig(
        orchestrator=thresholds,
        profile_overrides={},
        stress_tests=None,
        min_probability=0.55,
        require_cost_data=False,
        penalty_cost_bps=0.0,
    )


@pytest.fixture()
def _market_frame() -> pd.DataFrame:
    index = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        periods=96,
        freq="15min",
    )
    close = pd.Series(range(len(index)), index=index, dtype=float) / 10.0 + 100.0
    frame = pd.DataFrame(
        {
            "timestamp": index,
            "open": close - 0.2,
            "high": close + 0.4,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0,
        }
    )
    frame.set_index("timestamp", inplace=True)
    return frame


def test_autotrader_applies_orchestrator_strategy_and_risk_limits(_market_frame: pd.DataFrame) -> None:
    emitter = _Emitter()
    gui = _GUI()
    ai_manager = _AIManagerStub()
    gui.ai_mgr = ai_manager

    risk_service = _RiskServiceStub()
    execution_service = _ExecutionServiceStub()
    journal = InMemoryTradingDecisionJournal()

    orchestrator = DecisionOrchestrator(_decision_engine_config())
    orchestrator.record_strategy_performance(
        "baseline_mean_reversion",
        MarketRegime.TREND,
        hit_rate=0.55,
        pnl=8.0,
        sharpe=0.4,
    )
    orchestrator.record_strategy_performance(
        "momentum_alpha",
        MarketRegime.TREND,
        hit_rate=0.82,
        pnl=14.0,
        sharpe=1.1,
    )

    bootstrap = SimpleNamespace(
        decision_orchestrator=orchestrator,
        risk_engine=risk_service,
        decision_engine_config=_decision_engine_config(),
        risk_profile_name="paper",
    )

    trader = AutoTrader(
        emitter,
        gui,
        lambda: "BTCUSDT",
        market_data_provider=_MarketDataProvider(_market_frame),
        risk_service=risk_service,
        execution_service=execution_service,
        bootstrap_context=bootstrap,
        decision_journal=journal,
    )
    trader._auto_trade_user_confirmed = True

    trader.run_cycle_once()

    assert trader.current_strategy == "momentum_alpha"
    assert risk_service.decisions, "risk service should evaluate at least one decision"
    assert execution_service.requests, "execution should be triggered when risk approves"

    payload = risk_service.last_payload
    assert payload is not None and payload.get("accepted") is True
    thresholds = payload.get("thresholds") if isinstance(payload, Mapping) else None
    assert thresholds is not None
    assert thresholds["max_cost_bps"] == pytest.approx(12.0)
    assert payload.get("net_edge_bps") is not None
    assert payload.get("net_edge_bps") >= 4.0
