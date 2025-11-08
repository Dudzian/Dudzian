from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from bot_core.ai.regime import MarketRegime, MarketRegimeAssessment
from bot_core.auto_trader import AutoTrader
from bot_core.auto_trader.audit import DecisionAuditLog
from bot_core.execution import ExecutionService
from bot_core.runtime.journal import InMemoryTradingDecisionJournal

from tests.e2e.fixtures import FakeExecutionService


class _Emitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, name: str, **payload: object) -> None:
        self.events.append((name, dict(payload)))


@dataclass(slots=True)
class _StaticAIManager:
    assessment: MarketRegimeAssessment
    prediction: float
    probability: float
    model_name: str = "static_model"

    def assess_market_regime(self, symbol: str, market_data: pd.DataFrame) -> MarketRegimeAssessment:
        return self.assessment

    def get_regime_summary(self, symbol: str) -> None:
        return None

    def predict_series(self, symbol: str, market_data: pd.DataFrame) -> pd.Series:
        return pd.Series([self.prediction] * len(market_data), index=market_data.index)

    def prediction_probability(self, symbol: str, market_data: pd.DataFrame) -> float:
        return self.probability

    def get_active_model(self, symbol: str) -> str:
        return self.model_name


def _market_data(rows: int = 120) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=rows, freq="h")
    prices = pd.Series(100.0 + pd.Series(range(rows)).rolling(5, min_periods=1).mean(), index=index)
    return pd.DataFrame({
        "open": prices,
        "high": prices + 2.0,
        "low": prices - 2.0,
        "close": prices + 1.0,
        "volume": 1000 + pd.Series(range(rows), index=index),
    })


def _build_trader(
    ai_manager: _StaticAIManager,
    *,
    execution_service: ExecutionService | None = None,
    environment: str = "paper",
    symbol: str = "BTCUSDT",
) -> tuple[AutoTrader, InMemoryTradingDecisionJournal, _Emitter, DecisionAuditLog]:
    emitter = _Emitter()
    journal = InMemoryTradingDecisionJournal()
    audit_log = DecisionAuditLog()
    gui = type(
        "GuiStub",
        (),
        {"ai_mgr": ai_manager, "portfolio_manager": None, "decision_journal": journal},
    )()
    trader = AutoTrader(
        emitter=emitter,
        gui=gui,
        symbol_getter=lambda: symbol,
        market_data_provider=lambda *_, **__: _market_data(),
        enable_auto_trade=True,
        trusted_auto_confirm=True,
        decision_audit_log=audit_log,
        decision_journal=journal,
        execution_service=execution_service,
    )
    trader.risk_service = None
    trader.core_risk_engine = None
    trader._environment_name = environment
    trader._execution_context = None
    return trader, journal, emitter, audit_log


def test_autotrader_paper_switches_to_growth_profile() -> None:
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.86,
        risk_score=0.32,
        metrics={"trend_strength": 0.8},
        symbol="BTCUSDT",
    )
    ai_manager = _StaticAIManager(assessment=assessment, prediction=0.015, probability=0.71)
    trader, journal, emitter, _ = _build_trader(ai_manager)

    trader._auto_trade_loop()

    assert trader._risk_profile_name == "aggressive"
    assert trader.current_strategy == "trend_following"
    assert trader._decision_cycle_metadata.get("decision_state") == "trade"
    assert journal.export(), "journal should capture decision events"
    assert any(event[0] == "auto_trader.decision_audit" for event in emitter.events)


def test_autotrader_live_enforces_conservative_profile_on_high_risk() -> None:
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.DAILY,
        confidence=0.74,
        risk_score=0.88,
        metrics={"volatility": 0.12},
        symbol="ETHUSDT",
    )
    ai_manager = _StaticAIManager(assessment=assessment, prediction=0.0, probability=0.4)
    trader, journal, emitter, _ = _build_trader(ai_manager)

    trader._auto_trade_loop()

    assert trader._risk_profile_name == "conservative"
    assert trader.current_strategy == "capital_preservation"
    assert trader._decision_cycle_metadata.get("decision_state") == "hold"
    decision_events = [event for event in journal.export() if event.get("event") == "decision_composed"]
    assert decision_events and decision_events[0].get("risk_profile") == "conservative"
    assert any(
        event[0] == "auto_trader.decision_audit" and event[1].get("stage") == "risk_profile_transition"
        for event in emitter.events
    )


def test_autotrader_paper_executes_order_and_records_audit() -> None:
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.91,
        risk_score=0.28,
        metrics={"trend_strength": 0.9},
        symbol="BTCUSDT",
    )
    ai_manager = _StaticAIManager(assessment=assessment, prediction=0.022, probability=0.83)
    service = FakeExecutionService()
    trader, _, _, audit_log = _build_trader(ai_manager, execution_service=service)

    trader._auto_trade_loop()

    assert service.executed, "usługa egzekucji powinna zostać wywołana"
    recorded = service.executed[0]
    assert recorded.request.symbol == "BTCUSDT"
    assert recorded.request.quantity > 0
    assert recorded.context.environment == "paper"
    assert recorded.request.metadata is not None
    assert recorded.request.metadata.get("mode") == trader._schedule_mode

    stages = audit_log.to_dicts(limit=10)
    assert any(entry["stage"] == "execution_submitted" for entry in stages)
    assert any(
        entry["stage"] == "execution_submitted"
        and entry["payload"].get("order", {}).get("symbol") == "BTCUSDT"
        for entry in stages
    )


def test_autotrader_live_execution_failure_records_audit() -> None:
    assessment = MarketRegimeAssessment(
        regime=MarketRegime.TREND,
        confidence=0.88,
        risk_score=0.35,
        metrics={"trend_strength": 0.85},
        symbol="ETHUSDT",
    )
    ai_manager = _StaticAIManager(assessment=assessment, prediction=0.018, probability=0.78)
    service = FakeExecutionService(should_fail=True, failure=RuntimeError("boom"))
    trader, _, _, audit_log = _build_trader(
        ai_manager,
        execution_service=service,
        environment="live",
        symbol="ETHUSDT",
    )

    trader._auto_trade_loop()

    assert service.executed, "nawet w przypadku błędu powinien wystąpić jeden attempt"
    recorded = service.executed[0]
    assert recorded.request.symbol == "ETHUSDT"
    assert recorded.context.environment == "live"

    stages = audit_log.to_dicts(limit=10)
    assert any(entry["stage"] == "execution_failed" for entry in stages)
    assert not any(
        entry["stage"] == "execution_submitted"
        and entry["payload"].get("order", {}).get("symbol") == "ETHUSDT"
        for entry in stages
    )

