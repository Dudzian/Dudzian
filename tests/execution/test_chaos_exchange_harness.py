from __future__ import annotations

import pytest

from bot_core.exchanges.base import Environment, ExchangeCredentials, OrderRequest
from bot_core.exchanges.errors import ExchangeNetworkError, ExchangeThrottlingError
from tests._exchange_adapter_helpers import (
    ChaosCancelStep,
    ChaosEvent,
    ChaosExchangeAdapter,
    ChaosOrderStep,
)


def _request(*, client_order_id: str = "cid-1") -> OrderRequest:
    return OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=1.0,
        order_type="market",
        client_order_id=client_order_id,
    )


def _build_adapter(
    *,
    place_steps: tuple[ChaosOrderStep, ...] = (),
    cancel_steps: tuple[ChaosCancelStep, ...] = (),
) -> ChaosExchangeAdapter:
    return ChaosExchangeAdapter(
        ExchangeCredentials(key_id="chaos", environment=Environment.TESTNET),
        place_steps=place_steps,
        cancel_steps=cancel_steps,
    )


def test_harness_timeout_without_ack_certainty_and_late_reconcile() -> None:
    adapter = _build_adapter(
        place_steps=(
            ChaosOrderStep(
                action="timeout_unknown",
                order_id="ord-1",
                status="filled",
                filled_quantity=1.0,
                avg_price=100.0,
                release_after_ticks=2,
            ),
        )
    )

    with pytest.raises(ExchangeNetworkError):
        adapter.place_order(_request())
    assert adapter.fetch_order_by_client_id("cid-1", symbol="BTCUSDT") is None

    adapter.advance()
    assert adapter.fetch_order_by_client_id("cid-1", symbol="BTCUSDT") is None
    adapter.advance()
    reconciled = adapter.fetch_order_by_client_id("cid-1", symbol="BTCUSDT")
    assert reconciled is not None
    assert reconciled.order_id == "ord-1"


def test_harness_delayed_ack_and_duplicate_callbacks() -> None:
    duplicate = ChaosEvent("fill", "ord-2", "filled", filled_quantity=1.0, avg_price=101.0)
    adapter = _build_adapter(
        place_steps=(
            ChaosOrderStep(
                action="ack",
                order_id="ord-2",
                status="accepted",
                release_after_ticks=1,
                events=(duplicate, duplicate),
            ),
        )
    )

    result = adapter.place_order(_request(client_order_id="cid-2"))
    assert result.status == "accepted"
    assert adapter.next_private_event() is None
    adapter.advance()
    first = adapter.next_private_event()
    second = adapter.next_private_event()
    assert first == second
    assert adapter.next_private_event() is None


def test_harness_partial_fill_then_cancel_ack() -> None:
    partial = ChaosEvent("fill", "ord-3", "partially_filled", filled_quantity=0.4, avg_price=102.0)
    cancelled = ChaosEvent("cancel", "ord-3", "cancelled", filled_quantity=0.4, avg_price=102.0)
    adapter = _build_adapter(
        place_steps=(ChaosOrderStep(action="ack", order_id="ord-3", status="accepted", events=(partial,)),),
        cancel_steps=(ChaosCancelStep(action="ack", events=(cancelled,)),),
    )

    result = adapter.place_order(_request(client_order_id="cid-3"))
    assert result.order_id == "ord-3"
    partial_event = adapter.next_private_event()
    assert partial_event is not None
    assert partial_event.status == "partially_filled"
    adapter.cancel_order("ord-3", symbol="BTCUSDT")
    cancel_event = adapter.next_private_event()
    assert cancel_event is not None
    assert cancel_event.status == "cancelled"
    assert cancel_event.filled_quantity == pytest.approx(0.4)
    reconciled = adapter.fetch_order_by_client_id("cid-3", symbol="BTCUSDT")
    assert reconciled is not None
    assert reconciled.status == "cancelled"
    assert reconciled.filled_quantity == pytest.approx(0.4)


def test_harness_out_of_order_events_do_not_rollback_reconcile_state() -> None:
    adapter = _build_adapter(
        place_steps=(
            ChaosOrderStep(
                action="ack",
                order_id="ord-4",
                events=(
                    ChaosEvent("filled", "ord-4", "filled", filled_quantity=1.0),
                    ChaosEvent("partial", "ord-4", "partially_filled", filled_quantity=0.5),
                ),
            ),
        )
    )

    adapter.place_order(_request(client_order_id="cid-4"))
    reconciled = adapter.fetch_order_by_client_id("cid-4", symbol="BTCUSDT")
    assert reconciled is not None
    assert reconciled.status == "filled"
    assert reconciled.filled_quantity == pytest.approx(1.0)

    first = adapter.next_private_event()
    second = adapter.next_private_event()
    assert first is not None and second is not None
    assert first.status == "filled"
    assert second.status == "partially_filled"
    assert adapter.fetch_order_by_client_id("cid-4", symbol="BTCUSDT") == reconciled
    assert adapter.cancelled == []
    assert adapter.placed[0].client_order_id == "cid-4"


def test_harness_rate_limit_and_temp_network_failure_paths() -> None:
    adapter = _build_adapter(
        place_steps=(
            ChaosOrderStep(action="rate_limit", order_id="ord-5", reason="retry later"),
            ChaosOrderStep(action="network_failure", order_id="ord-6", reason="temporary dns"),
        )
    )

    with pytest.raises(ExchangeThrottlingError):
        adapter.place_order(_request(client_order_id="cid-5"))
    with pytest.raises(ExchangeNetworkError):
        adapter.place_order(_request(client_order_id="cid-6"))


def test_harness_advance_zero_is_noop() -> None:
    adapter = _build_adapter(
        place_steps=(
            ChaosOrderStep(
                action="timeout_unknown",
                order_id="ord-7",
                status="filled",
                filled_quantity=1.0,
                release_after_ticks=1,
            ),
        )
    )

    with pytest.raises(ExchangeNetworkError):
        adapter.place_order(_request(client_order_id="cid-7"))
    adapter.advance(0)
    assert adapter.fetch_order_by_client_id("cid-7", symbol="BTCUSDT") is None
    adapter.advance(1)
    assert adapter.fetch_order_by_client_id("cid-7", symbol="BTCUSDT") is not None
