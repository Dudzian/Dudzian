"""Testy profili ryzyka z wykorzystaniem rzeczywistych wartości ATR."""
from __future__ import annotations

import statistics
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.profiles.aggressive import AggressiveProfile
from bot_core.risk.profiles.balanced import BalancedProfile
from bot_core.risk.profiles.conservative import ConservativeProfile


@pytest.fixture()
def btc_daily_atr_series() -> list[float]:
    """Wycinek 14-dniowego ATR BTC/USDT z kwietnia 2024 (dzienny interwał)."""

    return [
        727.61,
        715.42,
        708.33,
        699.12,
        684.55,
        672.48,
        665.91,
        659.77,
        648.35,
        640.28,
        633.14,
        629.77,
        624.83,
        618.44,
    ]


def _recommended_quantity(*, profile, atr: float, equity: float, price: float, risk_pct: float) -> float:
    stop_distance = atr * profile.stop_loss_atr_multiple()
    risk_amount = equity * risk_pct
    raw_quantity = max(risk_amount / stop_distance, 0.0)
    max_quantity = profile.max_position_exposure() * equity / price
    return min(raw_quantity, max_quantity)


def test_profiles_scale_position_sizes_with_risk(btc_daily_atr_series: list[float]) -> None:
    price = 27_250.0
    equity = 120_000.0
    atr = statistics.mean(btc_daily_atr_series)

    conservative = ConservativeProfile()
    balanced = BalancedProfile()
    aggressive = AggressiveProfile()

    quantities = [
        _recommended_quantity(profile=profile, atr=atr, equity=equity, price=price, risk_pct=0.01)
        for profile in (conservative, balanced, aggressive)
    ]

    assert quantities[0] < quantities[1] < quantities[2]

    for profile, qty in zip((conservative, balanced, aggressive), quantities, strict=True):
        max_notional = profile.max_position_exposure() * equity
        assert qty * price <= max_notional + 1e-6

    stop_distances = [atr * profile.stop_loss_atr_multiple() for profile in (conservative, balanced, aggressive)]
    assert stop_distances[0] < stop_distances[1] < stop_distances[2]


def test_risk_engine_accepts_atr_informed_order(btc_daily_atr_series: list[float]) -> None:
    equity = 150_000.0
    price = 28_400.0
    atr = statistics.fmean(btc_daily_atr_series)

    profile = BalancedProfile()
    engine = ThresholdRiskEngine()
    engine.register_profile(profile)

    account = AccountSnapshot(
        balances={"USDT": equity},
        total_equity=equity,
        available_margin=equity,
        maintenance_margin=0.0,
    )

    quantity = _recommended_quantity(profile=profile, atr=atr, equity=equity, price=price, risk_pct=0.012)
    order = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=quantity,
        order_type="market",
        price=price,
        stop_price=price - atr * profile.stop_loss_atr_multiple(),
        atr=atr,
    )

    check = engine.apply_pre_trade_checks(order, account=account, profile_name=profile.name)
    assert check.allowed, check.reason

    oversized_quantity = quantity * 2.5
    oversized_order = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=oversized_quantity,
        order_type="market",
        price=price,
        stop_price=price - atr * profile.stop_loss_atr_multiple(),
        atr=atr,
    )

    denial = engine.apply_pre_trade_checks(oversized_order, account=account, profile_name=profile.name)
    assert not denial.allowed
    assert "limit" in (denial.reason or "").lower()
