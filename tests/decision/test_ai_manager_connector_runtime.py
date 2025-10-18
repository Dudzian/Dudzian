import asyncio
from datetime import datetime

import pandas as pd

from bot_core.config.models import DecisionEngineConfig, DecisionOrchestratorThresholds
from bot_core.decision.ai_connector import AIManagerDecisionConnector
from bot_core.decision.models import RiskSnapshot
from bot_core.decision.orchestrator import DecisionOrchestrator
from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.execution.base import ExecutionContext
from bot_core.execution.paper import MarketMetadata, PaperTradingExecutionService
from bot_core.risk.engine import ThresholdRiskEngine


class DummyRiskProfile:
    name = "balanced"

    def max_positions(self) -> int:
        return 5

    def max_leverage(self) -> float:
        return 5.0

    def drawdown_limit(self) -> float:
        return 0.2

    def daily_loss_limit(self) -> float:
        return 0.05

    def max_position_exposure(self) -> float:
        return 0.5

    def target_volatility(self) -> float:
        return 0.1

    def stop_loss_atr_multiple(self) -> float:
        return 1.5


class StubAIManager:
    def __init__(self, *, signal: float = 0.015) -> None:
        self.signal = float(signal)
        self.ai_threshold_bps = 5.0

    async def predict_series(
        self,
        symbol: str,
        df: pd.DataFrame,
        *,
        model_types=None,
        feature_cols=None,
    ) -> pd.Series:
        return pd.Series([self.signal], index=df.index[-1:])


def test_connector_creates_candidate_and_executes_order() -> None:
    connector = AIManagerDecisionConnector(
        ai_manager=StubAIManager(signal=0.02),
        strategy="ai_core",
        risk_profile="balanced",
        default_notional=1_000.0,
        action="enter",
        min_probability=0.0,
    )

    df = pd.DataFrame(
        {
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [10_000.0, 11_000.0],
        },
        index=[datetime(2024, 1, 1, 12, 0), datetime(2024, 1, 1, 12, 1)],
    )

    candidates = asyncio.run(connector.generate_candidates("BTCUSDT", df))
    assert candidates

    thresholds = DecisionOrchestratorThresholds(
        max_cost_bps=15.0,
        min_net_edge_bps=1.0,
        max_daily_loss_pct=0.05,
        max_drawdown_pct=0.2,
        max_position_ratio=0.5,
        max_open_positions=5,
        max_latency_ms=250.0,
        max_trade_notional=10_000.0,
    )
    orchestrator = DecisionOrchestrator(
        DecisionEngineConfig(
            orchestrator=thresholds,
            profile_overrides={},
            stress_tests=None,
            min_probability=0.3,
            require_cost_data=False,
            penalty_cost_bps=0.0,
        )
    )

    snapshot = RiskSnapshot(
        profile="balanced",
        start_of_day_equity=50_000.0,
        daily_realized_pnl=0.0,
        peak_equity=50_000.0,
        last_equity=50_000.0,
        gross_notional=0.0,
        active_positions=0,
        symbols=(),
    )
    evaluation = orchestrator.evaluate_candidate(candidates[0], snapshot)
    assert evaluation.accepted

    risk_engine = ThresholdRiskEngine()
    risk_engine.register_profile(DummyRiskProfile())

    stop_price = 100.0
    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=candidates[0].notional / 102.0,
        order_type="market",
        price=102.0,
        atr=1.0,
        stop_price=stop_price,
        metadata={
            "decision_candidate": candidates[0].to_mapping(),
            "source": "test",
        },
    )

    account = AccountSnapshot(
        balances={"USDT": 100_000.0},
        total_equity=100_000.0,
        available_margin=100_000.0,
        maintenance_margin=1_000.0,
    )
    risk_result = risk_engine.apply_pre_trade_checks(
        request,
        account=account,
        profile_name="balanced",
    )
    assert risk_result.allowed

    execution = PaperTradingExecutionService(
        markets={"BTCUSDT": MarketMetadata(base_asset="BTC", quote_asset="USDT")},
        initial_balances={"USDT": 100_000.0},
    )
    context = ExecutionContext(
        portfolio_id="test",
        risk_profile="balanced",
        environment="paper",
        metadata={"source": "test"},
    )
    result = execution.execute(request, context)
    assert result.status == "filled"
    assert result.filled_quantity > 0
