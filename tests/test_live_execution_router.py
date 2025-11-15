from __future__ import annotations

import asyncio
import json
import contextlib
import threading
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pytest

from bot_core.execution.base import ExecutionContext
from bot_core.execution.live_router import (
    LiveExecutionRouter,
    QoSConfig,
    RouteDefinition,
    RouterRuntimeStats,
)
from bot_core.exchanges.base import Environment, OrderRequest, OrderResult
from bot_core.exchanges.errors import ExchangeAPIError, ExchangeNetworkError
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

    try:
        result = router.execute(build_request(), build_context())

        assert result.order_id == "1"
        assert (
            registry.get("live_orders_total").value(
                labels={"exchange": "primary", "route": "default"}
            )
            == 1.0
        )
        entries = read_decision_entries(tmp_path / "live_decision.jsonl")
        assert len(entries) == 1
        assert entries[0]["payload"]["fallback_used"] is False
        assert "signature" in entries[0]

        attempts_metric = registry.get("live_orders_attempts_total")
        labels_tuple = next(iter(attempts_metric._values.keys()))  # type: ignore[attr-defined]
        labels_mapping = dict(labels_tuple)
        assert labels_mapping["exchange"] == "primary"
        assert labels_mapping["queued"] in {"true", "false"}
        assert "queue_wait_seconds" in labels_mapping
        assert float(labels_mapping["queue_wait_seconds"]) >= 0.0
        assert attempts_metric.value(labels=labels_mapping) == 1.0
    finally:
        router.close()


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

    try:
        result = router.execute(build_request(), build_context("balanced"))

        assert result.order_id == "2"
        assert registry.get("live_orders_fallback_total").value(labels={"route": "balanced"}) == 1.0
        entries = read_decision_entries(tmp_path / "fallback.jsonl")
        assert entries[0]["payload"]["fallback_used"] is True
    finally:
        router.close()


def test_live_router_blocks_disallowed_fallback(tmp_path: Path) -> None:
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
        decision_log_path=tmp_path / "blocked.jsonl",
        decision_log_hmac_key=b"F" * 48,
        metrics=registry,
        time_source=clock,
    )

    ctx = build_context("balanced")
    ctx.metadata = {"execution_route": "balanced", "risk_allow_fallback_categories": "throttling"}

    with pytest.raises(ExchangeNetworkError):
        router.execute(build_request(), ctx)

    log_path = tmp_path / "blocked.jsonl"
    if log_path.exists():
        entries = read_decision_entries(log_path)
        assert entries[0]["payload"].get("fallback_used") is False


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

    try:
        with pytest.raises(RuntimeError):
            router.execute(build_request(), build_context())

        assert registry.get("live_orders_failed_total").value(labels={"route": "default"}) == 1.0
        entries = read_decision_entries(tmp_path / "failed.jsonl")
        assert entries[0]["payload"].get("error")
    finally:
        router.close()


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

    try:
        router.execute(build_request(), build_context())
        router.cancel("abc", build_context())

        assert adapters["primary"].cancelled == ["abc"]
    finally:
        router.close()


def test_execute_async_returns_result(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    clock = FakeClock()
    adapters = {
        "primary": StubExchangeAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[
                OrderResult(order_id="async-1", status="FILLED", filled_quantity=1.0, avg_price=100.5, raw_response={})
            ],
        )
    }
    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary",))],
        decision_log_path=tmp_path / "async.jsonl",
        decision_log_hmac_key=b"D" * 48,
        metrics=registry,
        time_source=clock,
    )

    async def _scenario() -> None:
        try:
            result = await router.execute_async(build_request(), build_context())

            assert result.order_id == "async-1"
            attempts_metric = registry.get("live_orders_attempts_total")
            labels_tuple = next(iter(attempts_metric._values.keys()))  # type: ignore[attr-defined]
            labels_mapping = dict(labels_tuple)
            assert labels_mapping["exchange"] == "primary"
        finally:
            router.close()

    asyncio.run(_scenario())


def test_execute_async_cancellation_propagates(tmp_path: Path) -> None:
    start_event = threading.Event()
    release_event = threading.Event()

    class BlockingAdapter(StubExchangeAdapter):
        def place_order(self, request: OrderRequest) -> OrderResult:  # type: ignore[override]
            start_event.set()
            release_event.wait()
            return super().place_order(request)

    adapters = {
        "primary": BlockingAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[
                OrderResult(order_id="async-2", status="FILLED", filled_quantity=1.0, avg_price=101.5, raw_response={})
            ],
        )
    }
    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary",))],
        decision_log_path=tmp_path / "cancel.jsonl",
        decision_log_hmac_key=b"E" * 48,
        metrics=MetricsRegistry(),
        time_source=FakeClock(),
    )

    async def _scenario() -> None:
        task = asyncio.create_task(router.execute_async(build_request(), build_context()))
        try:
            await asyncio.to_thread(start_event.wait)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        finally:
            release_event.set()
            await asyncio.sleep(0)
            router.close()

    asyncio.run(_scenario())


def test_execute_async_cancellation_before_dispatch_skips_order(tmp_path: Path) -> None:
    start_event = threading.Event()
    release_event = threading.Event()
    call_symbols: list[str] = []

    class BlockingAdapter(StubExchangeAdapter):
        def place_order(self, request: OrderRequest) -> OrderResult:  # type: ignore[override]
            call_symbols.append(request.symbol)
            if request.symbol == "BTCUSDT":
                start_event.set()
                release_event.wait()
            return super().place_order(request)

    registry = MetricsRegistry()
    adapters = {
        "primary": BlockingAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[
                OrderResult(order_id="async-blocked-1", status="FILLED", filled_quantity=1.0, avg_price=102.5, raw_response={}),
                OrderResult(order_id="async-blocked-2", status="FILLED", filled_quantity=1.0, avg_price=103.5, raw_response={}),
            ],
        )
    }
    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary",))],
        decision_log_path=tmp_path / "cancel-before.jsonl",
        decision_log_hmac_key=b"F" * 48,
        metrics=registry,
        time_source=FakeClock(),
        qos=QoSConfig(worker_concurrency=1, max_queue_size=4),
    )

    async def _scenario() -> None:
        first_task = asyncio.create_task(router.execute_async(build_request("BTCUSDT"), build_context()))
        second_task: asyncio.Task[OrderResult] | None = None
        try:
            await asyncio.to_thread(start_event.wait)
            second_task = asyncio.create_task(router.execute_async(build_request("ETHUSDT"), build_context()))
            await asyncio.sleep(0.05)
            second_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await second_task
            release_event.set()
            result = await first_task
            assert result.order_id == "async-blocked-1"
            await asyncio.sleep(0)
            assert call_symbols == ["BTCUSDT"]
            attempts_metric = registry.get("live_orders_attempts_total")
            assert len(getattr(attempts_metric, "_values", {})) == 1
        finally:
            release_event.set()
            if second_task is not None and not second_task.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await second_task
            router.close()

    asyncio.run(_scenario())


def test_queue_timeout_rejects_waiting_order(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    block_event = threading.Event()

    adapters = {
        "primary": StubExchangeAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[
                OrderResult(
                    order_id="timeout-ok-1",
                    status="FILLED",
                    filled_quantity=1.0,
                    avg_price=104.5,
                    raw_response={},
                ),
            ],
        )
    }

    original_place = adapters["primary"].place_order

    def blocking_place(self: StubExchangeAdapter, request: OrderRequest) -> OrderResult:
        block_event.wait()
        return original_place(request)

    adapters["primary"].place_order = types.MethodType(blocking_place, adapters["primary"])  # type: ignore[assignment]

    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary",))],
        decision_log_path=tmp_path / "queue_timeout.jsonl",
        decision_log_hmac_key=b"Q" * 48,
        metrics=registry,
        qos=QoSConfig(worker_concurrency=1, max_queue_size=8, max_queue_wait_seconds=0.05),
    )

    async def _scenario() -> None:
        first_task = asyncio.create_task(router.execute_async(build_request("BTCUSDT"), build_context()))
        await asyncio.sleep(0.01)
        second_task = asyncio.create_task(router.execute_async(build_request("ETHUSDT"), build_context()))
        try:
            await asyncio.sleep(0.08)
            block_event.set()
            result = await first_task
            assert result.order_id == "timeout-ok-1"
            with pytest.raises(TimeoutError):
                await second_task
        finally:
            block_event.set()
            if not first_task.done():
                await first_task
            if not second_task.done():
                with contextlib.suppress(asyncio.CancelledError, TimeoutError):
                    await second_task

    try:
        asyncio.run(_scenario())

        failure_metric = registry.get("live_orders_failed_total")
        assert failure_metric.value(labels={"route": "default"}) == 1.0

        attempts_metric = registry.get("live_orders_attempts_total")
        queue_attempt_labels: dict[str, str] | None = None
        for labels_tuple in getattr(attempts_metric, "_values", {}):
            mapping = dict(labels_tuple)
            if mapping.get("result") == "queue_timeout":
                queue_attempt_labels = mapping
                break
        assert queue_attempt_labels is not None
        assert queue_attempt_labels["queued"] == "true"
        assert float(queue_attempt_labels["queue_wait_seconds"]) >= 0.05

        entries = read_decision_entries(tmp_path / "queue_timeout.jsonl")
        timeout_entries = [
            entry for entry in entries if entry["payload"]["attempts"][0]["status"] == "queue_timeout"
        ]
        assert timeout_entries, "decision log should contain queue timeout entry"
    finally:
        block_event.set()
        router.close()


def test_queue_overflow_rejects_new_orders(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    start_event = threading.Event()
    release_event = threading.Event()

    class BlockingAdapter(StubExchangeAdapter):
        def place_order(self, request: OrderRequest) -> OrderResult:
            start_event.set()
            if not release_event.wait(timeout=1.0):
                raise RuntimeError("release event not set")
            return super().place_order(request)

    adapters = {
        "primary": BlockingAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[
                OrderResult(
                    order_id="overflow-1",
                    status="FILLED",
                    filled_quantity=1.0,
                    avg_price=99.5,
                    raw_response={},
                ),
                OrderResult(
                    order_id="overflow-2",
                    status="FILLED",
                    filled_quantity=1.0,
                    avg_price=100.5,
                    raw_response={},
                ),
            ],
        )
    }

    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary",))],
        decision_log_path=tmp_path / "queue_overflow.jsonl",
        decision_log_hmac_key=b"O" * 48,
        metrics=registry,
        qos=QoSConfig(worker_concurrency=1, max_queue_size=1),
    )

    async def _scenario() -> None:
        first = asyncio.create_task(router.execute_async(build_request("BTCUSDT"), build_context()))
        await asyncio.to_thread(start_event.wait)
        second = asyncio.create_task(router.execute_async(build_request("ETHUSDT"), build_context()))
        await asyncio.sleep(0.01)
        third = asyncio.create_task(router.execute_async(build_request("LTCUSDT"), build_context()))

        try:
            with pytest.raises(TimeoutError):
                await third
        finally:
            release_event.set()

        result_first = await first
        result_second = await second

        assert result_first.order_id == "overflow-1"
        assert result_second.order_id == "overflow-2"

    try:
        asyncio.run(_scenario())

        failure_metric = registry.get("live_orders_failed_total")
        assert failure_metric.value(labels={"route": "default"}) == 1.0

        attempts_metric = registry.get("live_orders_attempts_total")
        overflow_labels: dict[str, str] | None = None
        for labels_tuple in getattr(attempts_metric, "_values", {}):
            mapping = dict(labels_tuple)
            if mapping.get("result") == "queue_overflow":
                overflow_labels = mapping
                break
        assert overflow_labels is not None
        assert overflow_labels["queued"] == "false"

        entries = read_decision_entries(tmp_path / "queue_overflow.jsonl")
        overflow_entries = [
            entry
            for entry in entries
            if entry["payload"]["attempts"][0]["status"] == "queue_overflow"
        ]
        assert overflow_entries, "decision log should contain queue overflow entry"
    finally:
        release_event.set()
        router.close()


def test_cancel_async_invokes_adapter(tmp_path: Path) -> None:
    adapters = {
        "primary": StubExchangeAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[
                OrderResult(order_id="async-3", status="FILLED", filled_quantity=1.0, avg_price=102.5, raw_response={})
            ],
        )
    }
    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary",))],
        decision_log_path=tmp_path / "cancel_async.jsonl",
        decision_log_hmac_key=b"F" * 48,
        metrics=MetricsRegistry(),
        time_source=FakeClock(),
    )

    try:
        result = router.execute(build_request(), build_context())
        assert result.order_id == "async-3"

        asyncio.run(router.cancel_async(result.order_id, build_context()))

        primary = adapters["primary"]
        assert primary.cancelled == [result.order_id]
    finally:
        router.close()


def test_flush_async_waits_for_pending_orders(tmp_path: Path) -> None:
    start_event = threading.Event()
    release_event = threading.Event()

    class BlockingAdapter(StubExchangeAdapter):
        def place_order(self, request: OrderRequest) -> OrderResult:
            start_event.set()
            if not release_event.wait(timeout=1.0):
                raise RuntimeError("timeout during blocking adapter test")
            return super().place_order(request)

    adapters = {
        "primary": BlockingAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[
                OrderResult(
                    order_id="async-flush-1",
                    status="FILLED",
                    filled_quantity=1.0,
                    avg_price=101.0,
                    raw_response={},
                )
            ],
        )
    }

    registry = MetricsRegistry()
    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary",))],
        decision_log_path=tmp_path / "flush_async.jsonl",
        decision_log_hmac_key=b"H" * 48,
        metrics=registry,
        time_source=FakeClock(),
        qos=QoSConfig(worker_concurrency=1, max_queue_size=2),
    )

    async def _scenario() -> None:
        task = asyncio.create_task(router.execute_async(build_request(), build_context()))
        await asyncio.to_thread(start_event.wait)
        flush_task = asyncio.create_task(router.flush_async())
        await asyncio.sleep(0.05)
        assert not flush_task.done()
        release_event.set()
        await flush_task
        result = await task
        assert result.order_id == "async-flush-1"

    try:
        asyncio.run(_scenario())
    finally:
        release_event.set()
        router.close()


def test_close_async_resets_queue_depth(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    adapters = {
        "primary": StubExchangeAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[
                OrderResult(
                    order_id="async-close-1",
                    status="FILLED",
                    filled_quantity=1.0,
                    avg_price=99.5,
                    raw_response={},
                )
            ],
        )
    }

    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary",))],
        decision_log_path=tmp_path / "close_async.jsonl",
        decision_log_hmac_key=b"C" * 48,
        metrics=registry,
        time_source=FakeClock(),
        qos=QoSConfig(worker_concurrency=1, max_queue_size=2),
    )

    async def _scenario() -> None:
        await router.execute_async(build_request(), build_context())
        await router.close_async()

    asyncio.run(_scenario())

    gauge = registry.get("live_execution_queue_depth")
    assert getattr(gauge, "value")() == 0.0

    router.close()


def test_live_router_context_manager_closes(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    adapters = {
        "primary": StubExchangeAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[
                OrderResult(
                    order_id="ctx-1",
                    status="FILLED",
                    filled_quantity=1.0,
                    avg_price=101.0,
                    raw_response={},
                )
            ],
        )
    }

    with LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary",))],
        decision_log_path=tmp_path / "ctx.jsonl",
        decision_log_hmac_key=b"D" * 48,
        metrics=registry,
        time_source=FakeClock(),
    ) as router:
        result = router.execute(build_request(), build_context())
        assert result.order_id == "ctx-1"

    gauge = registry.get("live_execution_queue_depth")
    assert getattr(gauge, "value")() == 0.0


def test_live_router_async_context_manager(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    adapters = {
        "primary": StubExchangeAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[
                OrderResult(
                    order_id="actx-1",
                    status="FILLED",
                    filled_quantity=1.0,
                    avg_price=102.0,
                    raw_response={},
                )
            ],
        )
    }

    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary",))],
        decision_log_path=tmp_path / "actx.jsonl",
        decision_log_hmac_key=b"E" * 48,
        metrics=registry,
        time_source=FakeClock(),
    )

    async def _scenario() -> None:
        async with router as ctx:
            result = await ctx.execute_async(build_request(), build_context())
            assert result.order_id == "actx-1"

    asyncio.run(_scenario())

    gauge = registry.get("live_execution_queue_depth")
    assert getattr(gauge, "value")() == 0.0


def test_runtime_stats_exposes_queue_and_inflight(tmp_path: Path) -> None:
    start_event = threading.Event()
    release_event = threading.Event()

    class BlockingAdapter(StubExchangeAdapter):
        def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            super().__init__(*args, **kwargs)
            self._calls = 0

        def place_order(self, request: OrderRequest) -> OrderResult:  # type: ignore[override]
            self._calls += 1
            if self._calls == 1:
                start_event.set()
                release_event.wait()
            return super().place_order(request)

    registry = MetricsRegistry()
    adapters = {
        "primary": BlockingAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[
                OrderResult(
                    order_id="stats-1",
                    status="FILLED",
                    filled_quantity=1.0,
                    avg_price=105.0,
                    raw_response={},
                ),
                OrderResult(
                    order_id="stats-2",
                    status="FILLED",
                    filled_quantity=1.0,
                    avg_price=106.0,
                    raw_response={},
                ),
            ],
        )
    }

    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary",))],
        decision_log_path=tmp_path / "stats.jsonl",
        decision_log_hmac_key=b"S" * 48,
        metrics=registry,
        time_source=FakeClock(),
        qos=QoSConfig(worker_concurrency=1, max_queue_size=4),
    )

    async def _scenario() -> None:
        first_task = asyncio.create_task(router.execute_async(build_request("BTCUSDT"), build_context()))
        second_task: asyncio.Task[OrderResult] | None = None
        try:
            await asyncio.to_thread(start_event.wait)
            second_task = asyncio.create_task(router.execute_async(build_request("ETHUSDT"), build_context()))
            await asyncio.sleep(0)
            stats = await router.get_runtime_stats_async()
            assert isinstance(stats, RouterRuntimeStats)
            assert stats.queue_depth == 1
            assert stats.queue_limit == 4
            assert stats.per_exchange_limits["primary"] == 1
            assert stats.inflight_by_exchange == {"primary": 1}
            release_event.set()
            first_result = await first_task
            assert first_result.order_id == "stats-1"
            second_result = await second_task
            assert second_result.order_id == "stats-2"
            await asyncio.sleep(0)
            stats_after = router.get_runtime_stats()
            assert stats_after.queue_depth == 0
            assert stats_after.inflight_by_exchange == {}
        finally:
            release_event.set()
            if not first_task.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await first_task
            if second_task is not None and not second_task.done():
                with contextlib.suppress(asyncio.CancelledError):
                    await second_task

    asyncio.run(_scenario())
    router.close()


def test_runtime_stats_after_close(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    adapters = {
        "primary": StubExchangeAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[
                OrderResult(
                    order_id="closed-1",
                    status="FILLED",
                    filled_quantity=1.0,
                    avg_price=107.0,
                    raw_response={},
                )
            ],
        )
    }

    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary",))],
        decision_log_path=tmp_path / "stats_closed.jsonl",
        decision_log_hmac_key=b"T" * 48,
        metrics=registry,
        time_source=FakeClock(),
    )

    router.close()

    stats = router.get_runtime_stats()
    assert isinstance(stats, RouterRuntimeStats)
    assert stats.closed is True
    assert stats.queue_depth == 0
    assert stats.inflight_by_exchange == {}


def test_router_validates_adapter_result(tmp_path: Path) -> None:
    registry = MetricsRegistry()
    clock = FakeClock()
    bad_result = OrderResult(order_id="", status="rejected", filled_quantity=-1.0, avg_price=None, raw_response={})
    adapters = {
        "primary": StubExchangeAdapter.from_name(
            "primary",
            environment=Environment.LIVE,
            responses=[bad_result],
        ),
    }
    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary",))],
        decision_log_path=tmp_path / "invalid.jsonl",
        decision_log_hmac_key=b"G" * 48,
        metrics=registry,
        time_source=clock,
    )

    with pytest.raises(ExchangeAPIError):
        router.execute(build_request(), build_context())


def test_live_router_acknowledgements_success() -> None:
    success = OrderResult(order_id="ex-1", status="filled", filled_quantity=1.0, avg_price=100.0, raw_response={})
    adapters = {
        "primary": StubExchangeAdapter.from_name("primary", environment=Environment.LIVE, responses=[success])
    }
    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary",))],
        default_route="default",
        qos=QoSConfig(worker_concurrency=1, max_queue_size=4, ack_queue_size=8),
    )

    request = build_request()
    request.client_order_id = "cid-ack-success"
    context = build_context()

    result = router.execute(request, context)

    ack_submit = router.get_acknowledgement(timeout=1.0)
    ack_done = router.get_acknowledgement(timeout=1.0)

    assert ack_submit.status == "ack"
    assert ack_submit.ack_id == "cid-ack-success"
    assert ack_done.status == "done"
    assert ack_done.order_id == result.order_id
    router.close()


def test_live_router_acknowledgements_failure() -> None:
    failure = ExchangeAPIError(message="reject", status_code=400, payload=None)
    adapters = {
        "primary": StubExchangeAdapter.from_name("primary", environment=Environment.LIVE, responses=[failure])
    }
    router = LiveExecutionRouter(
        adapters=adapters,
        routes=[RouteDefinition(name="default", exchanges=("primary",))],
        default_route="default",
        qos=QoSConfig(worker_concurrency=1, max_queue_size=2, ack_queue_size=8),
    )

    request = build_request()
    request.client_order_id = "cid-ack-failure"
    context = build_context()

    with pytest.raises(ExchangeAPIError):
        router.execute(request, context)

    ack_submit = router.get_acknowledgement(timeout=1.0)
    ack_fail = router.get_acknowledgement(timeout=1.0)

    assert ack_submit.status == "ack"
    assert ack_submit.ack_id == "cid-ack-failure"
    assert ack_fail.status == "nak"
    assert ack_fail.details.get("kind") == "api"
    router.close()

