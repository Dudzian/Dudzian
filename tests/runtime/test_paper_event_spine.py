"""Unit tests for the deterministic local paper event spine."""

from __future__ import annotations

import pytest

from bot_core.runtime.paper_event_spine import (
    PaperEventSpine,
    PaperOrderEventType,
    PaperOrderStatus,
    PaperOrderTransitionError,
)
from bot_core.runtime.preview_modes import PreviewModeContractError, RuntimeCapability


def test_paper_spine_uses_preview_mode_contract() -> None:
    spine = PaperEventSpine(mode="paper")

    assert RuntimeCapability.PAPER_ORDER_SUBMIT in spine.policy.capabilities
    assert RuntimeCapability.PAPER_ORDER_LIFECYCLE in spine.policy.capabilities

    for blocked in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
    ):
        with pytest.raises(PreviewModeContractError):
            PaperEventSpine(capabilities=(RuntimeCapability.PAPER_ORDER_SUBMIT, blocked))

    with pytest.raises(PreviewModeContractError):
        PaperEventSpine(capabilities=(RuntimeCapability.PAPER_ORDER_SUBMIT, "unknown"))


def test_happy_path_sequences_and_ids_are_deterministic() -> None:
    spine = PaperEventSpine(created_at="2026-06-16T00:00:00Z")

    accepted = spine.submit_order(
        order_id="ord-1",
        client_order_id="client-1",
        symbol="BTC/USDT",
        side="buy",
        quantity=3.0,
        metadata={"strategy": "unit"},
    )
    partial = spine.partial_fill("ord-1", fill_quantity=1.0)
    filled = spine.fill_order("ord-1")

    assert [event.event_type for event in spine.all_events()] == [
        PaperOrderEventType.PAPER_ORDER_ACCEPTED,
        PaperOrderEventType.PAPER_ORDER_PARTIALLY_FILLED,
        PaperOrderEventType.PAPER_ORDER_FILLED,
    ]
    assert [event.sequence for event in spine.all_events()] == [1, 2, 3]
    assert [event.event_id for event in spine.all_events()] == [
        "paper-000001",
        "paper-000002",
        "paper-000003",
    ]
    assert accepted.remaining_quantity == 3.0
    assert partial.filled_quantity == 1.0
    assert filled.filled_quantity == 3.0
    assert filled.remaining_quantity == 0.0
    assert filled.created_at == "2026-06-16T00:00:00Z"
    assert spine.order_state("ord-1").status is PaperOrderStatus.FILLED


def test_reject_order_is_terminal_and_later_transitions_fail() -> None:
    spine = PaperEventSpine()
    spine.submit_order(order_id="ord-2", symbol="ETH/USDT", side="sell", quantity=2.0)

    rejected = spine.reject_order("ord-2", reason="risk_block")

    assert rejected.event_type is PaperOrderEventType.PAPER_ORDER_REJECTED
    assert rejected.status is PaperOrderStatus.REJECTED
    assert spine.order_state("ord-2").status is PaperOrderStatus.REJECTED
    with pytest.raises(PaperOrderTransitionError, match="terminal"):
        spine.fill_order("ord-2")
    with pytest.raises(PaperOrderTransitionError, match="terminal"):
        spine.cancel_order("ord-2")


def test_cancel_order_is_terminal_and_cancel_after_filled_fails() -> None:
    spine = PaperEventSpine()
    spine.submit_order(order_id="ord-3", symbol="SOL/USDT", side="buy", quantity=2.0)

    cancelled = spine.cancel_order("ord-3", reason="user_cancel")

    assert cancelled.event_type is PaperOrderEventType.PAPER_ORDER_CANCELLED
    assert cancelled.status is PaperOrderStatus.CANCELLED
    assert spine.order_state("ord-3").status is PaperOrderStatus.CANCELLED

    spine.submit_order(order_id="ord-4", symbol="SOL/USDT", side="buy", quantity=1.0)
    spine.fill_order("ord-4")
    with pytest.raises(PaperOrderTransitionError, match="terminal"):
        spine.cancel_order("ord-4")


@pytest.mark.parametrize("quantity", [0.0, -1.0])
def test_quantity_must_be_positive(quantity: float) -> None:
    spine = PaperEventSpine()

    with pytest.raises(PaperOrderTransitionError, match="quantity must be > 0"):
        spine.submit_order(order_id="ord-bad", symbol="BTC/USDT", side="buy", quantity=quantity)


def test_overfill_unknown_order_and_secret_metadata_are_blocked() -> None:
    spine = PaperEventSpine()
    spine.submit_order(order_id="ord-5", symbol="BTC/USDT", side="buy", quantity=1.0)

    with pytest.raises(PaperOrderTransitionError, match="exceeds remaining"):
        spine.fill_order("ord-5", fill_quantity=2.0)
    with pytest.raises(PaperOrderTransitionError, match="Unknown paper order"):
        spine.fill_order("missing", fill_quantity=1.0)
    with pytest.raises(PaperOrderTransitionError, match="Unknown paper order"):
        spine.cancel_order("missing")
    with pytest.raises(PaperOrderTransitionError, match="forbidden credential-like"):
        spine.submit_order(
            order_id="ord-secret",
            symbol="BTC/USDT",
            side="buy",
            quantity=1.0,
            metadata={"api_key": "not-allowed"},
        )


def test_partial_fill_must_leave_remaining_quantity() -> None:
    spine = PaperEventSpine()
    spine.submit_order(order_id="ord-6", symbol="BTC/USDT", side="buy", quantity=1.0)

    with pytest.raises(PaperOrderTransitionError, match="partial fill must leave"):
        spine.partial_fill("ord-6", fill_quantity=1.0)
