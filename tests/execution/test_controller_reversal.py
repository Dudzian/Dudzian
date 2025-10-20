from datetime import timedelta

from bot_core.exchanges.base import AccountSnapshot, OrderRequest, OrderResult
from bot_core.execution import ExecutionService
from bot_core.runtime import TradingController
from bot_core.runtime.journal import TradingDecisionEvent
from bot_core.strategies import StrategySignal


class DummyRouter:
    def dispatch(self, message) -> None:  # pragma: no cover - trivial stub
        return None

    def health_snapshot(self):  # pragma: no cover - trivial stub
        return {}


class DummyExecutionService(ExecutionService):
    def __init__(self) -> None:
        self.requests: list[OrderRequest] = []

    def execute(self, request: OrderRequest, context) -> OrderResult:  # type: ignore[override]
        self.requests.append(request)
        return OrderResult(
            order_id="close-1" if request.metadata.get("action") == "close" else "open-1",
            status="filled",
            filled_quantity=request.quantity,
            avg_price=request.price,
            raw_response={},
        )

    def cancel(self, order_id: str, context) -> None:  # type: ignore[override]
        return None

    def flush(self) -> None:
        return None


class DummyRiskEngine:
    def apply_pre_trade_checks(self, request: OrderRequest, *, account: AccountSnapshot, profile_name: str):  # type: ignore[override]
        return type("RiskResult", (), {"allowed": True, "reason": None, "adjustments": {}})()

    def should_liquidate(self, *, profile_name: str) -> bool:  # type: ignore[override]
        return False


class DummyJournal:
    def __init__(self) -> None:
        self.events: list[TradingDecisionEvent] = []

    def record(self, event: TradingDecisionEvent) -> None:
        self.events.append(event)


def _account_snapshot() -> AccountSnapshot:
    return AccountSnapshot(
        balances={"USDT": 10_000.0},
        total_equity=10_000.0,
        available_margin=9_000.0,
        maintenance_margin=500.0,
    )


def test_reversal_pipeline_executes_close_then_open():
    execution = DummyExecutionService()
    controller = TradingController(
        risk_engine=DummyRiskEngine(),
        execution_service=execution,
        alert_router=DummyRouter(),
        account_snapshot_provider=_account_snapshot,
        portfolio_id="paper-1",
        environment="paper",
        risk_profile="balanced",
        health_check_interval=timedelta(hours=1),
        decision_journal=DummyJournal(),
    )

    signal = StrategySignal(
        symbol="BTC/USDT",
        side="SELL",
        confidence=0.8,
        metadata={
            "quantity": "1.0",
            "price": "100",
            "order_type": "market",
            "current_position_qty": "0.4",
            "current_position_side": "LONG",
        },
    )

    controller.process_signals([signal])

    assert len(execution.requests) == 2
    close_request, open_request = execution.requests
    assert close_request.side == "SELL"
    assert close_request.metadata["action"] == "close"
    assert open_request.side == "SELL"
