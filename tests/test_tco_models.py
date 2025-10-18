from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from bot_core.tco.models import (
    CostBreakdown,
    ProfileCostSummary,
    StrategyCostSummary,
    TCOReport,
    TradeCostEvent,
)


class TestTradeCostEvent:
    def test_from_mapping_parses_types(self) -> None:
        payload = {
            "timestamp": "2024-03-01T12:30:00Z",
            "strategy": "mean_reversion",
            "risk_profile": "balanced",
            "instrument": "BTC/USDT",
            "exchange": "binance",
            "side": "BUY",
            "quantity": "0.75",
            "price": "20500.5",
            "commission": 4,
            "slippage": "1.25",
            "funding": 0,
            "other": "0.15",
            "metadata": {"source": "fixture"},
        }

        event = TradeCostEvent.from_mapping(payload)

        assert event.timestamp == datetime(2024, 3, 1, 12, 30, tzinfo=timezone.utc)
        assert event.side == "buy"
        assert event.quantity == Decimal("0.75")
        assert event.price == Decimal("20500.5")
        assert event.total_cost == Decimal("5.40")
        assert event.notional == Decimal("15375.375")
        # ensure metadata is copied and further mutations do not affect the event
        payload["metadata"]["source"] = "mutated"
        assert event.metadata == {"source": "fixture"}

    def test_to_dict_quantizes_values(self) -> None:
        event = TradeCostEvent(
            timestamp=datetime(2024, 1, 1, 0, 0, tzinfo=timezone.utc),
            strategy="trend",
            risk_profile="aggressive",
            instrument="ETH/USDT",
            exchange="kraken",
            side="sell",
            quantity=Decimal("2.5"),
            price=Decimal("1800.1234567"),
            commission=Decimal("3.3333333"),
            slippage=Decimal("1.0000004"),
            funding=Decimal("0"),
            other=Decimal("0.5"),
        )

        payload = event.to_dict()

        assert payload["quantity"] == pytest.approx(2.5)
        assert payload["price"] == pytest.approx(1800.123457)
        assert payload["commission"] == pytest.approx(3.333333)
        assert payload["slippage"] == pytest.approx(1.000000)
        assert payload["total_cost"] == pytest.approx(4.833333)
        assert payload["notional"] == pytest.approx(4500.308642)


class TestCostSummaries:
    def test_cost_breakdown_and_profile_summary(self) -> None:
        breakdown = CostBreakdown.zero()
        event = TradeCostEvent(
            timestamp=datetime(2024, 1, 1, tzinfo=timezone.utc),
            strategy="trend",
            risk_profile="balanced",
            instrument="BTC/USDT",
            exchange="binance",
            side="buy",
            quantity=Decimal("1"),
            price=Decimal("20000"),
            commission=Decimal("6.5"),
            slippage=Decimal("1.5"),
            funding=Decimal("0.25"),
            other=Decimal("0.75"),
        )

        breakdown.add_event(event)

        assert breakdown.to_dict() == {
            "commission": pytest.approx(6.5),
            "slippage": pytest.approx(1.5),
            "funding": pytest.approx(0.25),
            "other": pytest.approx(0.75),
            "total": pytest.approx(9.0),
        }

        profile = ProfileCostSummary(
            profile="balanced",
            trade_count=1,
            notional=Decimal("20000"),
            breakdown=breakdown,
        )

        assert profile.cost_per_trade == Decimal("9")
        assert profile.cost_bps == Decimal("4.5")

        empty_profile = ProfileCostSummary(
            profile="aggressive",
            trade_count=0,
            notional=Decimal("0"),
            breakdown=CostBreakdown.zero(),
        )
        assert empty_profile.cost_per_trade == Decimal("0")
        assert empty_profile.cost_bps == Decimal("0")

    def test_strategy_summary_and_report_serialization(self) -> None:
        balanced_breakdown = CostBreakdown(
            commission=Decimal("3.0"),
            slippage=Decimal("1.0"),
            funding=Decimal("0.2"),
            other=Decimal("0.3"),
        )
        balanced_profile = ProfileCostSummary(
            profile="balanced",
            trade_count=2,
            notional=Decimal("15000"),
            breakdown=balanced_breakdown,
        )
        total_breakdown = CostBreakdown(
            commission=Decimal("4.5"),
            slippage=Decimal("1.2"),
            funding=Decimal("0.3"),
            other=Decimal("0.5"),
        )
        total_profile = ProfileCostSummary(
            profile="__total__",
            trade_count=2,
            notional=Decimal("15000"),
            breakdown=total_breakdown,
        )
        strategy_summary = StrategyCostSummary(
            strategy="mean_reversion",
            profiles={"balanced": balanced_profile},
            total=total_profile,
        )
        report = TCOReport(
            generated_at=datetime(2024, 1, 2, 15, 0, tzinfo=timezone.utc),
            metadata={"environment": "paper"},
            strategies={"mean_reversion": strategy_summary},
            total=total_profile,
            alerts=["cost_limit_exceeded"],
        )

        strategy_dict = strategy_summary.to_dict()
        assert strategy_dict["strategy"] == "mean_reversion"
        assert strategy_dict["profiles"]["balanced"]["trade_count"] == 2
        assert strategy_dict["total"]["cost_bps"] == pytest.approx(float(total_profile.cost_bps))

        report_dict = report.to_dict()
        assert report_dict["generated_at"] == "2024-01-02T15:00:00+00:00"
        assert report_dict["alerts"] == ["cost_limit_exceeded"]
        assert report_dict["total"]["cost_bps"] == pytest.approx(float(total_profile.cost_bps))
