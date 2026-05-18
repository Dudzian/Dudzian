from __future__ import annotations

from copy import deepcopy
from datetime import UTC, datetime, timedelta

from bot_core.runtime.canary_gate import evaluate_runtime_canary_gate


_APPROVED_CANARY = {
    "report_only": False,
    "canary_status": "approved",
    "canary_profile_id": "canary-live-btc",
    "allowed_exchanges": ["binance"],
    "allowed_symbols": ["BTC/USDT"],
    "max_order_notional": 1_000.0,
    "max_position_notional": 5_000.0,
    "review_required_at": "2026-05-19T00:00:00Z",
    "expires_at": "2026-05-20T00:00:00Z",
}


def _evaluate(canary_contract, **overrides):
    params = {
        "canary_contract": canary_contract,
        "mode": "live_autonomous",
        "exchange": "binance",
        "symbol": "BTC/USDT",
        "order_notional": 500.0,
        "position_notional": 1_000.0,
        "is_reduce_risk": False,
        "now": datetime(2026, 5, 18, tzinfo=UTC),
    }
    params.update(overrides)
    return evaluate_runtime_canary_gate(**params)


def test_live_autonomous_new_exposure_blocks_missing_canary_contract() -> None:
    decision = _evaluate(None)

    assert decision.allowed is False
    assert decision.blocking_reason == "accepted_autonomous_canary_contract_missing"
    assert decision.canary_status is None
    assert decision.canary_profile_id is None
    assert decision.snapshot["present"] is False


def test_live_autonomous_new_exposure_blocks_report_only_before_runtime_approval() -> None:
    canary = {**_APPROVED_CANARY, "report_only": True}

    decision = _evaluate(canary)

    assert decision.allowed is False
    assert decision.blocking_reason == "accepted_autonomous_canary_contract_report_only"
    assert decision.canary_status == "approved"
    assert decision.snapshot["report_only"] is True


def test_live_autonomous_new_exposure_blocks_unapproved_canary_contract() -> None:
    canary = {**_APPROVED_CANARY, "canary_status": "pending_review"}

    decision = _evaluate(canary)

    assert decision.allowed is False
    assert decision.blocking_reason == "accepted_autonomous_canary_contract_not_approved"
    assert decision.canary_status == "pending_review"


def test_live_autonomous_new_exposure_allows_approved_canary_within_scope() -> None:
    decision = _evaluate(_APPROVED_CANARY)

    assert decision.allowed is True
    assert decision.blocking_reason is None
    assert decision.canary_status == "approved"
    assert decision.canary_profile_id == "canary-live-btc"
    assert decision.snapshot == {
        "present": True,
        "report_only": False,
        "canary_status": "approved",
        "canary_profile_id": "canary-live-btc",
        "allowed_exchanges": ["binance"],
        "allowed_symbols": ["BTC/USDT"],
        "max_order_notional": 1_000.0,
        "max_position_notional": 5_000.0,
        "review_required_at": "2026-05-19T00:00:00Z",
        "expires_at": "2026-05-20T00:00:00Z",
    }


def test_live_autonomous_new_exposure_blocks_exchange_symbol_and_notional_mismatches() -> None:
    assert (
        _evaluate(_APPROVED_CANARY, exchange="kraken").blocking_reason
        == "accepted_autonomous_canary_exchange_not_allowed"
    )
    assert (
        _evaluate(_APPROVED_CANARY, symbol="ETH/USDT").blocking_reason
        == "accepted_autonomous_canary_symbol_not_allowed"
    )
    assert (
        _evaluate(_APPROVED_CANARY, order_notional=1_000.01).blocking_reason
        == "accepted_autonomous_canary_order_notional_exceeded"
    )
    assert (
        _evaluate(_APPROVED_CANARY, position_notional=5_000.01).blocking_reason
        == "accepted_autonomous_canary_position_notional_exceeded"
    )


def test_live_autonomous_new_exposure_blocks_expired_or_review_past_due_canary() -> None:
    expired = {
        **_APPROVED_CANARY,
        "review_required_at": (
            datetime(2026, 5, 18, tzinfo=UTC) - timedelta(seconds=1)
        ).isoformat(),
    }

    decision = _evaluate(expired)

    assert decision.allowed is False
    assert decision.blocking_reason == "accepted_autonomous_canary_expired"


def test_non_live_and_reduce_risk_requests_do_not_block_on_canary_contract() -> None:
    assert _evaluate(None, mode="paper_autonomous").allowed is True

    missing_reduce = _evaluate(None, is_reduce_risk=True)
    assert missing_reduce.allowed is True
    assert missing_reduce.blocking_reason is None

    unapproved_reduce = _evaluate(
        {**_APPROVED_CANARY, "report_only": True, "canary_status": "not_configured"},
        is_reduce_risk=True,
    )
    assert unapproved_reduce.allowed is True
    assert unapproved_reduce.blocking_reason is None


def test_runtime_canary_gate_does_not_mutate_input_mapping() -> None:
    canary = deepcopy(_APPROVED_CANARY)
    before = deepcopy(canary)

    decision = _evaluate(canary)

    assert decision.allowed is True
    assert canary == before
