from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from bot_core.execution.base import ExecutionContext
from bot_core.execution.live_router import LiveExecutionRouter, RouteDefinition
from bot_core.exchanges.base import Environment, OrderRequest, OrderResult
from bot_core.exchanges.errors import ExchangeNetworkError
from bot_core.observability import MetricsRegistry

from tests._exchange_adapter_helpers import StubExchangeAdapter

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
        "primary": StubExchangeAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[success],
        ),
        "fallback": StubExchangeAdapter.from_name("fallback", environment=Environment.LIVE),
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
        "primary": StubExchangeAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[ExchangeNetworkError("fail", None)],
        ),
        "secondary": StubExchangeAdapter.from_name(
            "secondary",
            environment=Environment.LIVE,
            responses=[
                OrderResult(order_id="2", status="FILLED", filled_quantity=1.0, avg_price=99.5, raw_response={})
            ],
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
        "primary": StubExchangeAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[ExchangeNetworkError("fail", None)],
        ),
        "secondary": StubExchangeAdapter.from_name(
            "secondary",
            environment=Environment.LIVE,
            responses=[ExchangeNetworkError("fail", None)],
        ),
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
        "primary": StubExchangeAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[
                OrderResult(order_id="abc", status="FILLED", filled_quantity=2.0, avg_price=50.0, raw_response={})
            ],
        ),
        "fallback": StubExchangeAdapter.from_name("fallback", environment=Environment.LIVE),
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

