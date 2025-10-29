from __future__ import annotations

from dataclasses import dataclass

from bot_core.execution.base import ExecutionContext
from bot_core.execution.bridge import ExchangeAdapterExecutionService, decision_to_order_request
from bot_core.execution.paper import MarketMetadata
from bot_core.exchanges.base import ExchangeAdapter, ExchangeCredentials, OrderRequest, OrderResult
from bot_core.runtime.journal import InMemoryTradingDecisionJournal, aggregate_decision_statistics


def _execution_context() -> ExecutionContext:
    return ExecutionContext(
        portfolio_id="paper-demo",
        risk_profile="default",
        environment="paper",
        metadata={"source": "test"},
    )


@dataclass
class _RecordedOrder:
    request: OrderRequest
    attempts: int


class _StubExchangeAdapter(ExchangeAdapter):
    def __init__(self) -> None:
        super().__init__(ExchangeCredentials(key_id="stub"))
        self.requests: list[_RecordedOrder] = []
        self.fail_once: bool = False

    def configure_network(self, *, ip_allowlist=None) -> None:  # pragma: no cover - not used
        return None

    def fetch_account_snapshot(self):  # pragma: no cover - not used
        raise NotImplementedError

    def fetch_symbols(self):  # pragma: no cover - not used
        return []

    def fetch_ohlcv(self, symbol, interval, start=None, end=None, limit=None):  # pragma: no cover - not used
        return []

    def place_order(self, request: OrderRequest) -> OrderResult:
        attempts = len(self.requests) + 1
        self.requests.append(_RecordedOrder(request=request, attempts=attempts))
        if self.fail_once:
            self.fail_once = False
            raise TimeoutError("temporary outage")
        price = request.price if request.price is not None else 100.0
        return OrderResult(
            order_id=f"order-{attempts}",
            status="filled",
            filled_quantity=request.quantity,
            avg_price=price,
            raw_response={"fee": 0.0, "fee_asset": "USDT"},
        )

    def cancel_order(self, order_id: str, *, symbol=None) -> None:  # pragma: no cover - not used
        return None

    def stream_public_data(self, *, channels):  # pragma: no cover - not used
        raise NotImplementedError

    def stream_private_data(self, *, channels):  # pragma: no cover - not used
        raise NotImplementedError


def test_execute_with_retry_and_journal_logging() -> None:
    adapter = _StubExchangeAdapter()
    journal = InMemoryTradingDecisionJournal()
    service = ExchangeAdapterExecutionService(
        adapter=lambda: adapter,
        journal=journal,
        max_attempts=2,
        backoff_base=0.0,
    )

    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=1.5,
        order_type="market",
        price=20500.0,
    )

    adapter.fail_once = True
    result = service.execute(request, _execution_context())

    assert result.order_id == "order-2"
    assert len(adapter.requests) == 2

    summary = aggregate_decision_statistics(journal)
    assert summary["total"] == 2  # submitted + filled
    assert summary["by_status"]["submitted"] == 1
    assert summary["by_status"]["filled"] == 1
    assert summary["by_symbol"]["BTCUSDT"] == 2


def test_decision_to_order_request_from_mapping() -> None:
    decision = {
        "candidate": {
            "symbol": "ETHUSDT",
            "action": "enter",
            "notional": 2000.0,
        },
        "metadata": {"strategy": "mean_reversion"},
    }
    market = MarketMetadata(base_asset="ETH", quote_asset="USDT", min_quantity=0.0001, min_notional=10.0)
    order = decision_to_order_request(decision, price=2000.0, market=market)

    assert order.symbol == "ETHUSDT"
    assert order.side == "buy"
    assert order.quantity == 1.0
    assert order.metadata is not None
    assert order.metadata["strategy"] == "mean_reversion"
