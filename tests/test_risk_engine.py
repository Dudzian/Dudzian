"""Testy jednostkowe dla silnika zarządzania ryzykiem."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.risk.engine import InMemoryRiskRepository, ThresholdRiskEngine
from bot_core.risk.repository import FileRiskRepository
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


def test_margin_check_blocks_when_available_margin_is_too_low() -> None:
    low_leverage_profile = ManualProfile(
        name="low-margin",
        max_positions=5,
        max_leverage=1.0,
        drawdown_limit=0.5,
        daily_loss_limit=0.5,
        max_position_pct=1.0,
        target_volatility=0.2,
        stop_loss_atr_multiple=2.0,
    )

    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    engine.register_profile(low_leverage_profile)

    snapshot = AccountSnapshot(
        balances={"USDT": 1_000.0},
        total_equity=1_000.0,
        available_margin=500.0,
        maintenance_margin=150.0,
    )

    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.05,
        order_type="limit",
        price=20_000.0,
    )

    result = engine.apply_pre_trade_checks(
        request,
        account=snapshot,
        profile_name=low_leverage_profile.name,
    )

    assert result.allowed is False
    assert result.reason == "Niewystarczający wolny margines na otwarcie lub powiększenie pozycji."
    assert result.adjustments is not None
    max_quantity = result.adjustments.get("max_quantity") if result.adjustments else None
    assert max_quantity is not None
    # Dostępny margines po uwzględnieniu maintenance to 350 USDT.
    assert max_quantity == pytest.approx(350.0 / 20_000.0, rel=1e-6)


def test_margin_check_passes_when_leverage_covers_notional() -> None:
    leveraged_profile = ManualProfile(
        name="leveraged",
        max_positions=5,
        max_leverage=4.0,
        drawdown_limit=0.5,
        daily_loss_limit=0.5,
        max_position_pct=1.0,
        target_volatility=0.2,
        stop_loss_atr_multiple=2.0,
    )

    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    engine.register_profile(leveraged_profile)

    snapshot = AccountSnapshot(
        balances={"USDT": 1_000.0},
        total_equity=1_000.0,
        available_margin=500.0,
        maintenance_margin=150.0,
    )

    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.05,
        order_type="limit",
        price=20_000.0,
    )

    result = engine.apply_pre_trade_checks(
        request,
        account=snapshot,
        profile_name=leveraged_profile.name,
    )

    assert result.allowed is True


def test_on_fill_normalizes_position_side_and_allows_growth(manual_profile: ManualProfile) -> None:
    engine = ThresholdRiskEngine(clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    engine.register_profile(manual_profile)

    snapshot = _snapshot(10_000.0)
    request = OrderRequest(
        symbol="BTCUSDT",
        side="buy",
        quantity=0.05,
        order_type="limit",
        price=30_000.0,
    )

    first_check = engine.apply_pre_trade_checks(request, account=snapshot, profile_name=manual_profile.name)
    assert first_check.allowed is True

    engine.on_fill(
        profile_name=manual_profile.name,
        symbol=request.symbol,
        side="buy",
        position_value=request.quantity * request.price,
        pnl=0.0,
        timestamp=datetime(2024, 1, 1, 13, 0, 0),
    )

    state = engine._states[manual_profile.name]
    assert state.positions[request.symbol].side == "long"

    second_check = engine.apply_pre_trade_checks(
        request,
        account=snapshot,
        profile_name=manual_profile.name,
    )

    assert second_check.allowed is True


def test_register_profile_normalizes_persisted_state(manual_profile: ManualProfile) -> None:
    repository = InMemoryRiskRepository()
    repository.store(
        manual_profile.name,
        {
            "profile": manual_profile.name,
            "current_day": "2024-01-01",
            "start_of_day_equity": 1_000.0,
            "daily_realized_pnl": 0.0,
            "peak_equity": 1_000.0,
            "force_liquidation": False,
            "last_equity": 1_000.0,
            "positions": {
                "BTCUSDT": {"side": "buy", "notional": 500.0},
            },
        },
    )

    engine = ThresholdRiskEngine(repository=repository, clock=lambda: datetime(2024, 1, 1, 12, 0, 0))
    engine.register_profile(manual_profile)

    state = engine._states[manual_profile.name]
    assert state.positions["BTCUSDT"].side == "long"


def test_file_risk_repository_persists_state(tmp_path: Path, manual_profile: ManualProfile) -> None:
    repository = FileRiskRepository(tmp_path)

    clock = lambda: datetime(2024, 1, 1, 12, 0, 0)
    engine = ThresholdRiskEngine(repository=repository, clock=clock)
    engine.register_profile(manual_profile)

    snapshot = _snapshot(1_000.0)
    request = _order(20_000.0)

    result = engine.apply_pre_trade_checks(
        request,
        account=snapshot,
        profile_name=manual_profile.name,
    )
    assert result.allowed is True

    engine.on_fill(
        profile_name=manual_profile.name,
        symbol="BTCUSDT",
        side="buy",
        position_value=500.0,
        pnl=-15.0,
        timestamp=datetime(2024, 1, 1, 13, 0, 0),
    )

    # Now instantiate a new engine sharing the same repository – state should persist.
    new_engine = ThresholdRiskEngine(repository=repository, clock=lambda: datetime(2024, 1, 1, 14, 0, 0))
    new_engine.register_profile(manual_profile)

    state = new_engine._states[manual_profile.name]
    assert state.start_of_day_equity == pytest.approx(1_000.0)
    assert state.daily_realized_pnl == pytest.approx(-15.0)
    assert "BTCUSDT" in state.positions
    btc_position = state.positions["BTCUSDT"]
    assert btc_position.side == "long"
    assert btc_position.notional == pytest.approx(500.0)

