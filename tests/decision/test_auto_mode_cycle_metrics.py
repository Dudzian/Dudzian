from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bot_core.auto_trader.performance import build_cycle_equity_summary


@pytest.mark.parametrize("window_hours", [24.0, None])
def test_build_cycle_equity_summary(window_hours: float | None) -> None:
    history = [
        {
            "finished_at": "2024-01-01T00:00:00Z",
            "orders": [{"pnl": 100.0}],
        },
        {
            "finished_at": "2024-01-01T06:00:00Z",
            "orders": [{"pnl": -50.0}],
        },
        {
            "finished_at": "2024-01-02T01:00:00Z",
            "orders": [{"pnl": 25.0}],
        },
    ]

    now_reference = datetime(2024, 1, 2, 2, 0, tzinfo=timezone.utc)
    points, metrics, window_metrics = build_cycle_equity_summary(
        history,
        tz=timezone.utc,
        now=now_reference,
        base_equity=100_000.0,
        window_hours=window_hours,
    )

    assert len(points) == 3
    assert points[-1]["value"] == pytest.approx(100_075.0)

    assert metrics["cycle_count"] == 3
    assert metrics["net_return_pct"] == pytest.approx(0.00075, rel=1e-6)
    expected_drawdown = (100_100.0 - 100_050.0) / 100_100.0
    assert metrics["max_drawdown_pct"] == pytest.approx(expected_drawdown, rel=1e-6)
    assert "avg_return_pct" in metrics and "volatility_pct" in metrics
    assert metrics["window"]["start"].startswith("2024-01-01")

    if window_hours is None:
        assert window_metrics == {}
    else:
        assert window_metrics["cycle_count"] == 2
        expected_window_net = (100_075.0 - 100_100.0) / 100_100.0
        assert window_metrics["net_return_pct"] == pytest.approx(expected_window_net, rel=1e-6)
        assert window_metrics["max_drawdown_pct"] == pytest.approx(expected_drawdown, rel=1e-6)
        assert "avg_return_pct" in window_metrics and "volatility_pct" in window_metrics
        assert window_metrics["window"]["start"].startswith("2024-01-01T06")
