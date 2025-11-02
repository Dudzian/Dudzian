import pytest
from datetime import datetime, timezone
from types import SimpleNamespace

from bot_core.exchanges.base import AccountSnapshot, OrderRequest
from bot_core.risk.engine import InMemoryRiskRepository, ThresholdRiskEngine
from bot_core.risk.events import RiskAlertLog
from bot_core.risk.profiles import BalancedProfile


def _fixed_clock() -> datetime:
    return datetime(2024, 1, 3, tzinfo=timezone.utc)


def _build_engine():
    alert_log = RiskAlertLog(clock=_fixed_clock)
    engine = ThresholdRiskEngine(
        repository=InMemoryRiskRepository(),
        clock=_fixed_clock,
        decision_log=None,
        alert_log=alert_log,
    )
    profile = BalancedProfile()
    engine.register_profile(profile)
    return engine, alert_log, profile


def _account(equity: float = 100_000.0) -> AccountSnapshot:
    return AccountSnapshot(
        balances={},
        total_equity=equity,
        available_margin=equity,
        maintenance_margin=0.0,
    )


def _order(quantity: float, *, stop_price: float = 98.0) -> OrderRequest:
    return OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        quantity=quantity,
        order_type="limit",
        price=100.0,
        stop_price=stop_price,
        atr=0.6,
        metadata={"stop_price": stop_price, "atr": 0.6},
    )


def test_trade_risk_limit_blocks_excessive_order():
    engine, alert_log, profile = _build_engine()
    account = _account()
    order = OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        quantity=30.0,
        order_type="limit",
        price=100.0,
        stop_price=20.0,
        atr=10.0,
        metadata={"stop_price": 20.0, "atr": 10.0},
    )

    result = engine.apply_pre_trade_checks(order, account=account, profile_name=profile.name)

    assert not result.allowed
    assert "limit ryzyka na pojedynczą transakcję" in (result.reason or "")
    assert engine.recent_alerts() == ()  # brak alertów dla naruszenia twardego limitu


def test_trade_risk_limit_allows_incremental_orders_within_cap():
    engine, alert_log, profile = _build_engine()
    account = _account()
    order = OrderRequest(
        symbol="BTC/USDT",
        side="buy",
        quantity=40.0,
        order_type="limit",
        price=100.0,
        stop_price=80.0,
        atr=5.0,
        metadata={"stop_price": 80.0, "atr": 5.0},
    )

    first = engine.apply_pre_trade_checks(order, account=account, profile_name=profile.name)
    assert first.allowed

    engine.on_fill(
        profile_name=profile.name,
        symbol="BTC/USDT",
        side="buy",
        position_value=4_000.0,
        pnl=0.0,
        timestamp=_fixed_clock(),
    )

    second = engine.apply_pre_trade_checks(order, account=account, profile_name=profile.name)
    assert second.allowed


def test_instrument_exposure_alert_and_limit():
    engine, alert_log, profile = _build_engine()
    account = _account()

    warning_order = _order(260.0)
    warning_result = engine.apply_pre_trade_checks(
        warning_order, account=account, profile_name=profile.name
    )

    assert warning_result.allowed
    alerts = engine.recent_alerts(profile_name=profile.name)
    assert any(alert["limit"] == "instrument_exposure_alert" for alert in alerts)

    blocking_order = _order(400.0)
    blocked = engine.apply_pre_trade_checks(
        blocking_order, account=account, profile_name=profile.name
    )

    assert not blocked.allowed
    assert "limit ekspozycji na instrument" in (blocked.reason or "")


def test_kill_switch_blocks_non_reducing_orders():
    engine, alert_log, profile = _build_engine()
    account = _account()
    state = engine._states[profile.name]
    state.start_of_day_equity = account.total_equity
    state.start_of_week_equity = account.total_equity
    state.weekly_realized_pnl = -2_500.0
    state.daily_realized_pnl = -2_500.0
    state.force_liquidation = False

    order = _order(280.0)
    result = engine.apply_pre_trade_checks(order, account=account, profile_name=profile.name)

    assert not result.allowed
    assert "kill-switch" in (result.reason or "")
    alerts = engine.recent_alerts(profile_name=profile.name)
    assert any(alert["limit"] == "kill_switch" for alert in alerts)
    assert engine._states[profile.name].force_liquidation is True


def test_cost_ratio_limit_blocks_orders():
    engine, alert_log, profile = _build_engine()
    account = _account()
    state = engine._states[profile.name]
    state.start_of_day_equity = account.total_equity
    state.start_of_week_equity = account.total_equity
    state.weekly_realized_pnl = 0.0
    state.daily_realized_pnl = 0.0
    state.rolling_profit_30d = 4_000.0
    state.rolling_costs_30d = 1_200.0

    result = engine.apply_pre_trade_checks(_order(280.0), account=account, profile_name=profile.name)

    assert not result.allowed
    assert "Koszty transakcyjne" in (result.reason or "")
    alerts = engine.recent_alerts(profile_name=profile.name)
    assert any(alert["limit"] == "cost_to_profit_ratio" for alert in alerts)


def test_account_mark_updates_extended_metadata():
    engine, alert_log, profile = _build_engine()

    payload = {
        "risk_profile": profile.name,
        "snapshot": {
            "total_equity": 120_000.0,
            "realized_pnl": -1_250.0,
            "weekly_realized_pnl": -3_000.0,
            "rollingProfit30d": 9_500.0,
            "costsRolling30d": 1_900.0,
        },
        "mode": "live",
    }

    engine._handle_account_mark(SimpleNamespace(payload=payload))

    state = engine._states[profile.name]
    assert state.last_equity == 120_000.0
    assert state.daily_realized_pnl == -1_250.0
    assert state.weekly_realized_pnl == -3_000.0
    assert state.rolling_profit_30d == 9_500.0
    assert state.rolling_costs_30d == 1_900.0
