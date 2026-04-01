from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Sequence

from bot_core.exchanges.base import (
    AccountSnapshot,
    Environment,
    ExchangeAdapter,
    ExchangeCredentials,
    OrderRequest,
    OrderResult,
)
from bot_core.exchanges.errors import ExchangeNetworkError, ExchangeThrottlingError


class StubExchangeAdapter(ExchangeAdapter):
    """Wielokrotnego użytku adapter giełdowy wykorzystywany w testach."""

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        name: str | None = None,
        responses: Sequence[OrderResult | Exception] | None = None,
        reconciled_order: OrderResult | None = None,
        reconcile_error: Exception | None = None,
    ) -> None:
        super().__init__(credentials)
        self.name = name or credentials.key_id
        self._responses: Deque[OrderResult | Exception] = deque(responses or ())
        self.placed: list[OrderRequest] = []
        self.cancelled: list[str] = []
        self.ip_allowlist: Sequence[str] | None = None
        self.reconciled_order = reconciled_order
        self.reconcile_error = reconcile_error
        self.reconcile_calls: list[tuple[str, str | None]] = []

    @classmethod
    def from_name(
        cls,
        name: str,
        *,
        environment: Environment = Environment.PAPER,
        responses: Sequence[OrderResult | Exception] | None = None,
        reconciled_order: OrderResult | None = None,
        reconcile_error: Exception | None = None,
    ) -> "StubExchangeAdapter":
        credentials = ExchangeCredentials(key_id=name, environment=environment)
        return cls(
            credentials,
            name=name,
            responses=responses,
            reconciled_order=reconciled_order,
            reconcile_error=reconcile_error,
        )

    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # noqa: D401
        self.ip_allowlist = ip_allowlist

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={},
            total_equity=0.0,
            available_margin=0.0,
            maintenance_margin=0.0,
        )

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
        if not self._responses:
            return OrderResult(
                order_id="1",
                status="filled",
                filled_quantity=0.0,
                avg_price=None,
                raw_response={},
            )
        response = self._responses.popleft()
        if isinstance(response, Exception):
            raise response
        return response

    def fetch_order_by_client_id(
        self,
        client_order_id: str,
        *,
        symbol: str | None = None,
    ) -> OrderResult | None:
        self.reconcile_calls.append((client_order_id, symbol))
        if self.reconcile_error is not None:
            raise self.reconcile_error
        return self.reconciled_order

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # noqa: ARG002
        self.cancelled.append(order_id)

    def stream_public_data(self, *, channels: Sequence[str]):  # noqa: ARG002
        return self

    def stream_private_data(self, *, channels: Sequence[str]):  # noqa: ARG002
        return self


@dataclass(slots=True, frozen=True)
class ChaosEvent:
    event_type: str
    order_id: str
    status: str
    filled_quantity: float = 0.0
    avg_price: float | None = None
    source: str = "exchange"


@dataclass(slots=True)
class ChaosOrderStep:
    action: str
    order_id: str
    status: str = "accepted"
    filled_quantity: float = 0.0
    avg_price: float | None = None
    release_after_ticks: int = 0
    events: Sequence[ChaosEvent] = ()
    reason: str = "network failure"


@dataclass(slots=True)
class ChaosCancelStep:
    action: str = "ack"
    release_after_ticks: int = 0
    events: Sequence[ChaosEvent] = ()
    reason: str = "network failure"


@dataclass(slots=True)
class _DelayedReconcile:
    release_tick: int
    result: OrderResult


class ChaosExchangeAdapter(ExchangeAdapter):
    """Deterministyczny adapter chaosowy do scenariuszy execution-layer."""

    def __init__(
        self,
        credentials: ExchangeCredentials,
        *,
        name: str | None = None,
        place_steps: Sequence[ChaosOrderStep] = (),
        cancel_steps: Sequence[ChaosCancelStep] = (),
    ) -> None:
        super().__init__(credentials)
        self.name = name or credentials.key_id
        self._place_steps: Deque[ChaosOrderStep] = deque(place_steps)
        self._cancel_steps: Deque[ChaosCancelStep] = deque(cancel_steps)
        self._current_tick = 0
        self.placed: list[OrderRequest] = []
        self.cancelled: list[str] = []
        self.emitted_events: list[ChaosEvent] = []
        self._released_events: Deque[ChaosEvent] = deque()
        self._delayed_events: list[tuple[int, ChaosEvent]] = []
        self._reconcile_by_client_order_id: dict[str, OrderResult] = {}
        self._client_order_id_by_order_id: dict[str, str] = {}
        self._delayed_reconcile: list[tuple[str, _DelayedReconcile]] = []

    def configure_network(self, *, ip_allowlist: Sequence[str] | None = None) -> None:  # noqa: D401, ARG002
        return None

    def fetch_account_snapshot(self) -> AccountSnapshot:
        return AccountSnapshot(
            balances={},
            total_equity=0.0,
            available_margin=0.0,
            maintenance_margin=0.0,
        )

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
        step = self._place_steps.popleft() if self._place_steps else ChaosOrderStep("ack", "order-1")
        result = OrderResult(
            order_id=step.order_id,
            status=step.status,
            filled_quantity=step.filled_quantity,
            avg_price=step.avg_price,
            raw_response={"action": step.action},
        )
        self._register_effects(step.events, result, request.client_order_id, step.release_after_ticks)
        if step.action == "ack":
            return result
        if step.action == "timeout_unknown":
            raise ExchangeNetworkError(step.reason, None)
        if step.action == "rate_limit":
            raise ExchangeThrottlingError(message=step.reason, status_code=429)
        if step.action == "network_failure":
            raise ExchangeNetworkError(step.reason, None)
        raise RuntimeError(f"Unsupported chaos action: {step.action}")

    def fetch_order_by_client_id(
        self,
        client_order_id: str,
        *,
        symbol: str | None = None,
    ) -> OrderResult | None:
        del symbol
        self._release_due_items()
        return self._reconcile_by_client_order_id.get(client_order_id)

    def cancel_order(self, order_id: str, *, symbol: str | None = None) -> None:  # noqa: ARG002
        self.cancelled.append(order_id)
        step = self._cancel_steps.popleft() if self._cancel_steps else ChaosCancelStep()
        if step.events:
            synthetic = OrderResult(
                order_id=order_id,
                status="cancelled",
                filled_quantity=0.0,
                avg_price=None,
                raw_response={"action": step.action},
            )
            self._register_effects(step.events, synthetic, None, step.release_after_ticks)
        if step.action == "ack":
            return None
        if step.action == "rate_limit":
            raise ExchangeThrottlingError(message=step.reason, status_code=429)
        if step.action == "network_failure":
            raise ExchangeNetworkError(step.reason, None)
        raise RuntimeError(f"Unsupported cancel chaos action: {step.action}")

    def stream_public_data(self, *, channels: Sequence[str]):  # noqa: ARG002
        return self

    def stream_private_data(self, *, channels: Sequence[str]):  # noqa: ARG002
        return self

    def next_private_event(self) -> ChaosEvent | None:
        self._release_due_items()
        if not self._released_events:
            return None
        event = self._released_events.popleft()
        self.emitted_events.append(event)
        return event

    def advance(self, ticks: int = 1) -> None:
        if ticks < 0:
            raise ValueError("ticks must be >= 0")
        if ticks == 0:
            return
        self._current_tick += ticks
        self._release_due_items()

    def _register_effects(
        self,
        events: Sequence[ChaosEvent],
        result: OrderResult,
        client_order_id: str | None,
        release_after_ticks: int,
    ) -> None:
        if client_order_id is None:
            client_order_id = self._client_order_id_by_order_id.get(result.order_id)
        else:
            self._client_order_id_by_order_id[result.order_id] = client_order_id
        release_tick = self._current_tick + max(0, release_after_ticks)
        for event in events:
            client_for_event = self._client_order_id_by_order_id.get(event.order_id, client_order_id)
            if client_for_event is not None:
                event_result = OrderResult(
                    order_id=event.order_id,
                    status=event.status,
                    filled_quantity=event.filled_quantity,
                    avg_price=event.avg_price,
                    raw_response={"event_type": event.event_type, "source": event.source},
                )
                if release_tick <= self._current_tick:
                    self._reconcile_by_client_order_id[client_for_event] = event_result
                else:
                    self._delayed_reconcile.append(
                        (
                            client_for_event,
                            _DelayedReconcile(release_tick=release_tick, result=event_result),
                        )
                    )
            if release_tick <= self._current_tick:
                self._released_events.append(event)
            else:
                self._delayed_events.append((release_tick, event))
        if client_order_id is None:
            return
        if events:
            return
        if release_tick <= self._current_tick:
            self._reconcile_by_client_order_id[client_order_id] = result
        else:
            self._delayed_reconcile.append(
                (client_order_id, _DelayedReconcile(release_tick=release_tick, result=result))
            )

    def _release_due_items(self) -> None:
        ready_events = [item for item in self._delayed_events if item[0] <= self._current_tick]
        self._delayed_events = [item for item in self._delayed_events if item[0] > self._current_tick]
        for _, event in ready_events:
            self._released_events.append(event)

        ready_reconcile = [
            item for item in self._delayed_reconcile if item[1].release_tick <= self._current_tick
        ]
        self._delayed_reconcile = [
            item for item in self._delayed_reconcile if item[1].release_tick > self._current_tick
        ]
        for client_order_id, delayed in ready_reconcile:
            self._reconcile_by_client_order_id[client_order_id] = delayed.result
