from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import pytest

from bot_core.execution.base import ExecutionContext
from bot_core.execution.live_router import LiveExecutionRouter, RouteDefinition
from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.exchanges.errors import ExchangeNetworkError
from bot_core.observability import MetricsRegistry


class DummyAdapter(ExchangeAdapter):
    """Minimalna implementacja adaptera giełdowego do testów routera live."""

    def __init__(self, name: str, responses: Sequence[OrderResult | Exception]) -> None:
        super().__init__(ExchangeCredentials(key_id=name, environment=Environment.LIVE))
        self.name = name
        self._responses: Iterator[OrderResult | Exception] = iter(responses)
        self.placed: list[OrderRequest] = []
        self.cancelled: list[str] = []

    # ------------------------------------------------------------------
    # ExchangeAdapter API (nieużywane metody implementujemy symbolicznie)
    # ------------------------------------------------------------------
    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # noqa: ARG002
        return None

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(balances={}, total_equity=0.0, available_margin=0.0, maintenance_margin=0.0)

    def fetch_symbols(self) -> Iterable[str]:
        return ()

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        start: int | None = None,
        end: int | None = None,
        limit: int | None = None,
    ) -> Sequence[Sequence[float]]:  # noqa: ARG002
        return ()

    def place_order(self, request: OrderRequest) -> OrderResult:
        self.placed.append(request)
        try:
            response = next(self._responses)
        except StopIteration as exc:  # pragma: no cover - defensywnie
            raise RuntimeError("Brak przygotowanej odpowiedzi w dummy adapterze") from exc
        if isinstance(response, Exception):
            raise response
        return response

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # noqa: ARG002
        self.cancelled.append(order_id)

    def stream_public_data(self, *, channels: Sequence[str]):  # noqa: ARG002
        raise NotImplementedError

    def stream_private_data(self, *, channels: Sequence[str]):  # noqa: ARG002
        raise NotImplementedError


@dataclass(slots=True)
class FakeClock:
    value: float = 1000.0

    def __call__(self) -> float:
        current = self.value
        self.value += 0.05
        return current


def build_request(symbol: str = "BTCUSDT") -> OrderRequest:
    return OrderRequest(symbol=symbol, side="buy", quantity=1.0, order_type="market")


def build_context(route: str | None = None) -> ExecutionContext:
    metadata: Mapping[str, str] = {"execution_route": route} if route else {}
    return ExecutionContext(
        portfolio_id="core",
        risk_profile="balanced",
        environment="live",
        metadata=metadata,
    )


def read_decision_entries(path: Path) -> list[Mapping[str, object]]:
    entries: list[Mapping[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            entries.append(json.loads(line))
    return entries


def test_live_router_executes_on_primary(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    decision_key = b"L" * 48
    clock = FakeClock()
    success = OrderResult(order_id="1", status="FILLED", filled_quantity=1.0, avg_price=100.0, raw_response={})
    adapters = {
        "primary": DummyAdapter("primary", [success]),
        "fallback": DummyAdapter("fallback", []),
    }
    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary", "fallback"))],
        default_route="default",
        decision_log_path=tmp_path / "live_decision.jsonl",
        decision_log_hmac_key=decision_key,
        decision_log_key_id="live-router",
        metrics=registry,
        time_source=clock,
    )

    result = router.execute(build_request(), build_context())

    assert result.order_id == "1"
    assert registry.get("live_orders_total").value(labels={"exchange": "primary", "route": "default"}) == 1.0
    entries = read_decision_entries(tmp_path / "live_decision.jsonl")
    assert len(entries) == 1
    assert entries[0]["payload"]["fallback_used"] is False
    assert "signature" in entries[0]


def test_live_router_uses_fallback(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    clock = FakeClock()
    adapters = {
        "primary": DummyAdapter("primary", [ExchangeNetworkError("fail", None)]),
        "secondary": DummyAdapter(
            "secondary",
            [OrderResult(order_id="2", status="FILLED", filled_quantity=1.0, avg_price=99.5, raw_response={})],
        ),
    }
    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="balanced", exchanges=("primary", "secondary"))],
        decision_log_path=tmp_path / "fallback.jsonl",
        decision_log_hmac_key=b"A" * 48,
        metrics=registry,
        time_source=clock,
    )

    result = router.execute(build_request(), build_context("balanced"))

    assert result.order_id == "2"
    assert registry.get("live_orders_fallback_total").value(labels={"route": "balanced"}) == 1.0
    entries = read_decision_entries(tmp_path / "fallback.jsonl")
    assert entries[0]["payload"]["fallback_used"] is True


def test_live_router_raises_when_all_fail(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    clock = FakeClock()
    adapters = {
        "primary": DummyAdapter("primary", [ExchangeNetworkError("fail", None)]),
        "secondary": DummyAdapter("secondary", [ExchangeNetworkError("fail", None)]),
    }
    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary", "secondary"))],
        decision_log_path=tmp_path / "failed.jsonl",
        decision_log_hmac_key=b"B" * 48,
        metrics=registry,
        time_source=clock,
    )

    with pytest.raises(RuntimeError):
        router.execute(build_request(), build_context())

    assert registry.get("live_orders_failed_total").value(labels={"route": "default"}) == 1.0
    entries = read_decision_entries(tmp_path / "failed.jsonl")
    assert entries[0]["payload"].get("error")


def test_cancel_uses_recorded_exchange(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    clock = FakeClock()
    adapters = {
        "primary": DummyAdapter(
            "primary",
            [OrderResult(order_id="abc", status="FILLED", filled_quantity=2.0, avg_price=50.0, raw_response={})],
        ),
        "fallback": DummyAdapter("fallback", []),
    }
    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary", "fallback"))],
        decision_log_path=tmp_path / "cancel.jsonl",
        decision_log_hmac_key=b"C" * 48,
        metrics=registry,
        time_source=clock,
    )

    router.execute(build_request(), build_context())
    router.cancel("abc", build_context())

    assert adapters["primary"].cancelled == ["abc"]

