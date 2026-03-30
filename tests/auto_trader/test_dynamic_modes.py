import types
from types import SimpleNamespace

import pytest

from bot_core.ai.regime import MarketRegime, MarketRegimeAssessment, RiskLevel
from bot_core.auto_trader import AutoTrader
from bot_core.auto_trader.risk_bridge import GuardrailTrigger
from bot_core.config.models import (
    AutoTraderModeProfileConfig,
    AutoTraderModeParameterRange,
    RuntimeAutoTraderSettings,
)
from bot_core.runtime.journal import InMemoryTradingDecisionJournal


class _Emitter:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, object]]] = []

    def emit(self, event: str, **payload: object) -> None:
        self.events.append((event, dict(payload)))

    def log(self, message: str, *_, **context: object) -> None:
        payload = dict(context)
        payload["message"] = message
        self.events.append(("log", payload))


class _GUI:
    def __init__(self) -> None:
        self.timeframe_var = SimpleNamespace(get=lambda: "1h")


def _build_settings() -> RuntimeAutoTraderSettings:
    scalping = AutoTraderModeProfileConfig(
        description="Scalping",
        default_strategy="intraday_breakout",
        allowed_strategies=("intraday_breakout", "trend_following"),
        preferred_regimes=("trend",),
        required_inputs=("market_data", "ai_context"),
        guardrail_tags=("drawdown",),
        base_weight=1.2,
        leverage=AutoTraderModeParameterRange(min=0.2, max=2.0, default=0.8),
        position_size=AutoTraderModeParameterRange(min=0.02, max=0.2, default=0.05),
        risk_floor=0.05,
        risk_ceiling=0.6,
    )
    hedge = AutoTraderModeProfileConfig(
        description="Hedge",
        default_strategy="capital_preservation",
        allowed_strategies=("capital_preservation", "mean_reversion"),
        preferred_regimes=("mean_reversion",),
        required_inputs=("market_data",),
        guardrail_tags=("latency",),
        base_weight=0.9,
        leverage=AutoTraderModeParameterRange(min=0.0, max=1.0, default=0.25),
        position_size=AutoTraderModeParameterRange(min=0.0, max=0.35, default=0.1),
        risk_floor=0.0,
        risk_ceiling=0.45,
    )
    return RuntimeAutoTraderSettings(
        enabled=True,
        default_mode="scalping",
        modes={"scalping": scalping, "hedge": hedge},
    )


@pytest.fixture(name="trader")
def trader_fixture() -> AutoTrader:
    emitter = _Emitter()
    gui = _GUI()
    settings = _build_settings()
    trader = AutoTrader(emitter, gui, lambda: "BTCUSDT", mode_settings=settings)
    trader._decision_journal = InMemoryTradingDecisionJournal()
    trader._decision_journal_context = {
        "environment": "paper",
        "portfolio": "default",
        "risk_profile": "balanced",
    }
    return trader


def _assessment(regime: MarketRegime, risk: float) -> MarketRegimeAssessment:
    return MarketRegimeAssessment(
        regime=regime,
        confidence=0.8,
        risk_score=risk,
        metrics={"volatility": 0.01},
        symbol="BTCUSDT",
    )


def test_dynamic_mode_prefers_scalping_profile(trader: AutoTrader) -> None:
    summary = SimpleNamespace(regime=MarketRegime.TREND, risk_level=RiskLevel.BALANCED)
    assessment = _assessment(MarketRegime.TREND, 0.35)
    trader._evaluate_dynamic_mode(
        symbol="BTCUSDT",
        assessment=assessment,
        summary=summary,
        workflow_summary=None,
        ai_context={"direction": "long"},
    )
    assert trader._active_decision_mode == "scalping"

    trader._log_decision_event("cycle_started", symbol="BTCUSDT", status="pending")
    events = list(trader._decision_journal.export())
    assert events, "journal should contain events"
    assert events[-1].get("decision_mode") == "scalping"


def test_guardrail_penalty_switches_to_hedge(trader: AutoTrader) -> None:
    trigger = GuardrailTrigger(
        name="drawdown",
        label="drawdown",
        comparator=">",
        threshold=0.05,
        unit="ratio",
        value=0.08,
    )
    trader._handle_guardrail_trigger("BTCUSDT", ["drawdown"], [trigger])

    summary = SimpleNamespace(regime=MarketRegime.MEAN_REVERSION, risk_level=RiskLevel.ELEVATED)
    assessment = _assessment(MarketRegime.MEAN_REVERSION, 0.5)
    trader._evaluate_dynamic_mode(
        symbol="BTCUSDT",
        assessment=assessment,
        summary=summary,
        workflow_summary=None,
        ai_context={"direction": "flat"},
    )
    assert trader._active_decision_mode == "hedge"
