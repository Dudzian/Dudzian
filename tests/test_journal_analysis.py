from datetime import datetime, timedelta, timezone

import pytest

from bot_core.runtime.journal import (
    InMemoryTradingDecisionJournal,
    log_decision_event,
)
from bot_core.runtime.journal_analysis import analyse_decision_journal


def test_analyse_decision_journal_computes_basic_metrics() -> None:
    journal = InMemoryTradingDecisionJournal()
    base_time = datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc)

    # Zarejestruj dwa sygnały decyzji i trzy zlecenia z PnL.
    log_decision_event(
        journal,
        event="decision_composed",
        environment="paper",
        portfolio="demo",
        risk_profile="balanced",
        timestamp=base_time,
        status="trade",
        metadata={"approved": True},
    )
    log_decision_event(
        journal,
        event="decision_composed",
        environment="paper",
        portfolio="demo",
        risk_profile="balanced",
        timestamp=base_time + timedelta(minutes=1),
        status="hold",
        metadata={"approved": False},
    )

    pnl_values = [35.0, -20.0, 15.0]
    for index, pnl in enumerate(pnl_values, start=1):
        log_decision_event(
            journal,
            event="order_filled",
            environment="paper",
            portfolio="demo",
            risk_profile="balanced",
            timestamp=base_time + timedelta(minutes=2 + index),
            symbol="BTCUSDT",
            side="buy",
            quantity=0.1,
            price=20000.0,
            status="filled",
            metadata={"pnl": pnl},
        )

    analytics = analyse_decision_journal(journal, window=10)
    assert analytics.trade_count == 3
    assert analytics.wins == 2
    assert analytics.losses == 1
    assert pytest.approx(analytics.win_rate, rel=1e-3) == 2 / 3
    assert pytest.approx(analytics.signal_accuracy, rel=1e-3) == 0.8
    assert analytics.max_drawdown > 0
    assert analytics.max_drawdown_pct > 0


def test_analyse_decision_journal_requires_positive_window() -> None:
    journal = InMemoryTradingDecisionJournal()
    with pytest.raises(ValueError):
        analyse_decision_journal(journal, window=0)

