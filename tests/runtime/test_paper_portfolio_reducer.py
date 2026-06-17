"""Unit tests for the deterministic local paper portfolio reducer."""

from __future__ import annotations

import os
import socket
from dataclasses import replace

import pytest

from bot_core.runtime.paper_event_spine import PaperEventSpine, PaperOrderEventType
from bot_core.runtime.paper_portfolio_reducer import (
    PaperPortfolioReducer,
    PaperPortfolioReducerError,
)
from bot_core.runtime.preview_modes import PreviewModeContractError, RuntimeCapability


def _fill_event(*, order_id: str, side: str, quantity: float, symbol: str = "BTC/USDT"):
    spine = PaperEventSpine(created_at="2026-06-17T00:00:00Z")
    return _fill_event_on_spine(
        spine, order_id=order_id, side=side, quantity=quantity, symbol=symbol
    )


def _fill_event_on_spine(
    spine: PaperEventSpine, *, order_id: str, side: str, quantity: float, symbol: str = "BTC/USDT"
):
    spine.submit_order(order_id=order_id, symbol=symbol, side=side, quantity=quantity)
    return spine.fill_order(order_id)


def test_reducer_uses_preview_mode_contract() -> None:
    reducer = PaperPortfolioReducer(mode="paper")

    assert RuntimeCapability.PAPER_ORDER_LIFECYCLE in reducer.policy.capabilities

    for blocked in (
        RuntimeCapability.LIVE_ORDER_SUBMIT,
        RuntimeCapability.LIVE_ACCOUNT_BALANCE_FETCH,
        RuntimeCapability.LIVE_CREDENTIALS_READ,
    ):
        with pytest.raises(PreviewModeContractError):
            PaperPortfolioReducer(capabilities=(RuntimeCapability.PAPER_ORDER_LIFECYCLE, blocked))

    with pytest.raises(PreviewModeContractError):
        PaperPortfolioReducer(capabilities=(RuntimeCapability.PAPER_ORDER_SUBMIT,))


def test_buy_fill_creates_trade_and_position() -> None:
    fill = _fill_event(order_id="buy-1", side="buy", quantity=2.0)
    reducer = PaperPortfolioReducer()

    snapshot = reducer.apply_event(fill, fill_price=100.0)

    assert len(snapshot.trades) == 1
    assert snapshot.trades[0].quantity == 2.0
    assert snapshot.trades[0].price == 100.0
    assert snapshot.trades[0].realized_pnl == 0.0
    assert len(snapshot.positions) == 1
    assert snapshot.positions[0].quantity == 2.0
    assert snapshot.positions[0].avg_entry_price == 100.0
    assert snapshot.positions[0].realized_pnl == 0.0


def test_partial_fills_update_weighted_average() -> None:
    spine = PaperEventSpine()
    spine.submit_order(order_id="buy-2", symbol="BTC/USDT", side="buy", quantity=2.0)
    first = spine.partial_fill("buy-2", fill_quantity=1.0)
    second = spine.fill_order("buy-2")
    reducer = PaperPortfolioReducer()

    reducer.apply_event(first, fill_price=100.0)
    snapshot = reducer.apply_event(second, fill_price=120.0)

    assert snapshot.positions[0].quantity == 2.0
    assert snapshot.positions[0].avg_entry_price == pytest.approx(110.0)


def test_sell_reduces_position_and_realizes_pnl() -> None:
    spine = PaperEventSpine()
    reducer = PaperPortfolioReducer()
    reducer.apply_event(
        _fill_event_on_spine(spine, order_id="buy-3", side="buy", quantity=2.0),
        fill_price=100.0,
    )

    snapshot = reducer.apply_event(
        _fill_event_on_spine(spine, order_id="sell-3", side="sell", quantity=1.0),
        fill_price=130.0,
    )

    assert snapshot.positions[0].quantity == 1.0
    assert snapshot.positions[0].avg_entry_price == 100.0
    assert snapshot.positions[0].realized_pnl == pytest.approx(30.0)
    assert snapshot.trades[-1].realized_pnl == pytest.approx(30.0)


def test_sell_closing_full_position_resets_average_entry_price() -> None:
    spine = PaperEventSpine()
    reducer = PaperPortfolioReducer()
    reducer.apply_event(
        _fill_event_on_spine(spine, order_id="buy-4", side="buy", quantity=1.0),
        fill_price=100.0,
    )

    snapshot = reducer.apply_event(
        _fill_event_on_spine(spine, order_id="sell-4", side="sell", quantity=1.0),
        fill_price=90.0,
    )

    assert snapshot.positions[0].quantity == 0.0
    assert snapshot.positions[0].avg_entry_price == 0.0
    assert snapshot.positions[0].realized_pnl == pytest.approx(-10.0)


def test_accepted_rejected_cancelled_do_not_create_trades() -> None:
    spine = PaperEventSpine()
    accepted = spine.submit_order(order_id="noop-1", symbol="BTC/USDT", side="buy", quantity=1.0)
    cancelled = spine.cancel_order("noop-1")
    spine.submit_order(order_id="noop-2", symbol="ETH/USDT", side="buy", quantity=1.0)
    rejected = spine.reject_order("noop-2", reason="risk")
    reducer = PaperPortfolioReducer()

    for event in (accepted, rejected, cancelled):
        snapshot = reducer.apply_event(event)

    assert snapshot.trades == ()
    assert snapshot.positions == ()


@pytest.mark.parametrize("fill_price", [0.0, -1.0])
def test_fill_price_must_be_positive(fill_price: float) -> None:
    reducer = PaperPortfolioReducer()

    with pytest.raises(PaperPortfolioReducerError, match="fill_price must be > 0"):
        reducer.apply_event(
            _fill_event(order_id=f"bad-price-{fill_price}", side="buy", quantity=1.0),
            fill_price=fill_price,
        )


def test_invalid_fill_price_does_not_mark_event_applied_and_can_be_retried() -> None:
    event = _fill_event(order_id="retry-price", side="buy", quantity=1.0)
    reducer = PaperPortfolioReducer()

    with pytest.raises(PaperPortfolioReducerError, match="fill_price must be > 0"):
        reducer.apply_event(event, fill_price=0.0)

    failed_snapshot = reducer.snapshot()
    assert failed_snapshot.trades == ()
    assert failed_snapshot.positions == ()
    assert event.event_id not in failed_snapshot.applied_event_ids

    snapshot = reducer.apply_event(event, fill_price=100.0)

    assert snapshot.applied_event_ids == (event.event_id,)
    assert len(snapshot.trades) == 1
    assert snapshot.positions[0].quantity == 1.0


def test_oversell_does_not_mark_event_applied_or_create_trade() -> None:
    event = _fill_event(order_id="failed-oversell", side="sell", quantity=1.0)
    reducer = PaperPortfolioReducer()

    with pytest.raises(PaperPortfolioReducerError, match="sell quantity exceeds"):
        reducer.apply_event(event, fill_price=100.0)

    snapshot = reducer.snapshot()
    assert snapshot.trades == ()
    assert snapshot.positions == ()
    assert event.event_id not in snapshot.applied_event_ids


def test_unknown_side_does_not_mark_event_applied_or_create_trade() -> None:
    event = replace(_fill_event(order_id="failed-side", side="buy", quantity=1.0), side="hold")
    reducer = PaperPortfolioReducer()

    with pytest.raises(PaperPortfolioReducerError, match="Unsupported paper order side"):
        reducer.apply_event(event, fill_price=100.0)

    snapshot = reducer.snapshot()
    assert snapshot.trades == ()
    assert snapshot.positions == ()
    assert event.event_id not in snapshot.applied_event_ids


def test_price_can_come_from_event_metadata_when_explicit_price_missing() -> None:
    fill = _fill_event(order_id="metadata-price", side="buy", quantity=1.0)
    fill = replace(fill, metadata={"price": 101.0, "strategy": "unit", "api_key": "filtered"})

    snapshot = PaperPortfolioReducer().apply_event(fill)

    assert snapshot.trades[0].price == 101.0
    assert dict(snapshot.trades[0].metadata) == {"strategy": "unit"}


def test_sell_more_than_long_raises() -> None:
    with pytest.raises(PaperPortfolioReducerError, match="sell quantity exceeds"):
        PaperPortfolioReducer().apply_event(
            _fill_event(order_id="oversell", side="sell", quantity=1.0), fill_price=100.0
        )


def test_duplicate_event_id_raises() -> None:
    event = _fill_event(order_id="dup", side="buy", quantity=1.0)
    reducer = PaperPortfolioReducer()
    reducer.apply_event(event, fill_price=100.0)

    with pytest.raises(PaperPortfolioReducerError, match="already applied"):
        reducer.apply_event(event, fill_price=100.0)


def test_unknown_side_raises() -> None:
    event = replace(_fill_event(order_id="side", side="buy", quantity=1.0), side="hold")

    with pytest.raises(PaperPortfolioReducerError, match="Unsupported paper order side"):
        PaperPortfolioReducer().apply_event(event, fill_price=100.0)


def test_unknown_event_type_raises() -> None:
    event = replace(
        _fill_event(order_id="event-type", side="buy", quantity=1.0),
        event_type="not_a_paper_event",
    )

    with pytest.raises(PaperPortfolioReducerError, match="Unsupported paper order event type"):
        PaperPortfolioReducer().apply_event(event, fill_price=100.0)


def test_snapshot_determinism() -> None:
    spine = PaperEventSpine()
    z_fill = _fill_event_on_spine(
        spine, order_id="z-buy", symbol="Z/USDT", side="buy", quantity=1.0
    )
    a_fill = _fill_event_on_spine(
        spine, order_id="a-buy", symbol="A/USDT", side="buy", quantity=1.0
    )
    reducer = PaperPortfolioReducer()

    reducer.apply_event(z_fill, fill_price=10.0)
    reducer.apply_event(a_fill, fill_price=20.0)
    snapshot = reducer.snapshot()

    assert [trade.sequence for trade in snapshot.trades] == [2, 4]
    assert [trade.order_id for trade in snapshot.trades] == ["z-buy", "a-buy"]
    assert [position.symbol for position in snapshot.positions] == ["A/USDT", "Z/USDT"]
    assert snapshot.applied_event_ids == ("paper-000002", "paper-000004")


def test_reducer_has_no_network_env_file_or_runtime_side_effects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_socket(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network must not be used")

    monkeypatch.setattr(socket, "socket", fail_socket)
    monkeypatch.setattr(
        os,
        "getenv",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("env must not be read")),
    )
    reducer = PaperPortfolioReducer()

    snapshot = reducer.apply_event(
        _fill_event(order_id="safe", side="buy", quantity=1.0), fill_price=100.0
    )

    assert snapshot.positions[0].quantity == 1.0
