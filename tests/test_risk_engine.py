"""Testy jednostkowe dla silnika zarządzania ryzykiem."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.risk.engine import ThresholdRiskEngine
from bot_core.risk.profiles.manual import ManualProfile


@pytest.fixture()
def manual_profile() -> ManualProfile:
    return ManualProfile(
        name="test-profile",
        max_positions=5,
        max_leverage=3.0,
        drawdown_limit=0.2,
        daily_loss_limit=0.005,
        max_position_pct=0.5,
        target_volatility=0.1,
        stop_loss_atr_multiple=2.0,
    )


def _snapshot(equity: float) -> AccountSnapshot:
    return AccountSnapshot(
        balances={"USDT": equity},
        total_equity=equity,
        available_margin=equity,
        maintenance_margin=0.0,
    )


def _order(price: float) -> OrderRequest:
    return OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.01,
        order_type="limit",
        price=price,
    )


def test_daily_loss_limit_blocks_on_first_day(manual_profile: ManualProfile) -> None:
    clock_value = datetime(2024, 1, 1, 12, 0, 0)
    engine = ThresholdRiskEngine(clock=lambda: clock_value)
    engine.register_profile(manual_profile)

    first_snapshot = _snapshot(1_000.0)
    request = _order(30_000.0)

    first_result = engine.apply_pre_trade_checks(
        request,
        account=first_snapshot,
        profile_name=manual_profile.name,
    )
    assert first_result.allowed is True

    state = engine._states[manual_profile.name]
    assert state.start_of_day_equity == pytest.approx(1_000.0)
    assert state.peak_equity == pytest.approx(1_000.0)

    engine.on_fill(
        profile_name=manual_profile.name,
        symbol=request.symbol,
        side="buy",
        position_value=0.0,
        pnl=-10.0,
        timestamp=clock_value,
    )

    second_snapshot = _snapshot(990.0)
    second_result = engine.apply_pre_trade_checks(
        request,
        account=second_snapshot,
        profile_name=manual_profile.name,
    )

    assert second_result.allowed is False
    assert "Przekroczono dzienny limit straty." in (second_result.reason or "")

