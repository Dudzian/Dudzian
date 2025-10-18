from __future__ import annotations

from decimal import Decimal

import pytest

from bot_core.tco.costs import CommissionCost, FundingCost, SlippageCost
from bot_core.tco.services import (
    BaseCostReportingService,
    CostAggregationContext,
    CostReportExtension,
)


class _MetadataExtension(CostReportExtension):
    """Prosta wtyczka ustawiająca dodatkowe metadane."""

    def __init__(self, **metadata: str) -> None:
        self._metadata = metadata

    def apply(self, context: CostAggregationContext) -> None:
        context.metadata.update(self._metadata)


def _commission(
    amount: str,
    *,
    strategy: str,
    profile: str,
    scheduler: str | None = None,
) -> CommissionCost:
    return CommissionCost(
        amount=Decimal(amount),
        metadata={
            "strategy": strategy,
            "risk_profile": profile,
            "scheduler": scheduler,
        },
    )


def _slippage(
    amount: str,
    *,
    strategy: str,
    profile: str,
    scheduler: str | None = None,
) -> SlippageCost:
    return SlippageCost(
        amount=Decimal(amount),
        metadata={
            "strategy": strategy,
            "risk_profile": profile,
            "scheduler": scheduler,
        },
    )


def _funding(
    amount: str,
    *,
    strategy: str,
    profile: str,
    scheduler: str | None = None,
) -> FundingCost:
    return FundingCost(
        amount=Decimal(amount),
        metadata={
            "strategy": strategy,
            "profile": profile,
            "scheduler": scheduler,
        },
    )


def test_aggregate_costs_groups_by_strategy_and_profile() -> None:
    service = BaseCostReportingService(currency="USD")

    report = service.aggregate_costs(
        (
            _commission(
                "1.25",
                strategy="alpha",
                profile="balanced",
                scheduler="cron.daily",
            ),
            _slippage(
                "0.50",
                strategy="alpha",
                profile="balanced",
                scheduler="cron.daily",
            ),
            _funding(
                "0.10",
                strategy="beta",
                profile="aggressive",
                scheduler="cron.nightly",
            ),
        ),
        metadata={"source": "unit-test"},
    )

    assert report.currency == "USD"
    assert report.component_count == 3
    assert report.metadata["source"] == "unit-test"
    assert report.metadata["strategy_count"] == 2
    assert report.metadata["scheduler_count"] == 2

    alpha_view = report.strategies["alpha"]
    alpha_profile = alpha_view.profiles["balanced"]
    assert alpha_profile.amounts["commission"] == Decimal("1.25")
    assert alpha_profile.amounts["slippage"] == Decimal("0.50")
    assert alpha_profile.total == Decimal("1.75")
    assert alpha_view.total.amounts["commission"] == Decimal("1.25")
    assert alpha_view.total.total == Decimal("1.75")

    beta_view = report.strategies["beta"]
    beta_profile = beta_view.profiles["aggressive"]
    assert beta_profile.amounts["funding"] == Decimal("0.10")
    assert beta_profile.total == Decimal("0.10")

    totals = report.totals
    assert totals.amounts["commission"] == Decimal("1.25")
    assert totals.amounts["slippage"] == Decimal("0.50")
    assert totals.amounts["funding"] == Decimal("0.10")
    assert totals.total == Decimal("1.85")

    schedulers = report.schedulers
    assert set(schedulers) == {"cron.daily", "cron.nightly"}
    cron_daily = schedulers["cron.daily"]
    assert cron_daily.total.amounts["commission"] == Decimal("1.25")
    assert cron_daily.total.amounts["slippage"] == Decimal("0.50")
    assert cron_daily.total.total == Decimal("1.75")
    assert cron_daily.strategies["alpha"].total == Decimal("1.75")

    cron_nightly = schedulers["cron.nightly"]
    assert cron_nightly.total.amounts["funding"] == Decimal("0.10")
    assert cron_nightly.total.total == Decimal("0.10")
    assert cron_nightly.strategies["beta"].total == Decimal("0.10")


def test_aggregate_costs_applies_extensions() -> None:
    service = BaseCostReportingService()
    service.register_extension(_MetadataExtension(stage="stage5"))

    report = service.aggregate_costs(
        (
            _commission("0.3", strategy="gamma", profile="balanced"),
        ),
        metadata={"source": "unit-test"},
    )

    assert report.metadata["source"] == "unit-test"
    assert report.metadata["stage"] == "stage5"
    assert report.metadata["component_count"] == 1


def test_missing_scheduler_uses_default_bucket() -> None:
    service = BaseCostReportingService(currency="USD")

    report = service.aggregate_costs(
        (
            CommissionCost(
                amount=Decimal("1"),
                metadata={"strategy": "alpha", "risk_profile": "balanced"},
            ),
        ),
    )

    assert report.schedulers["default"].total.total == Decimal("1")
    assert report.metadata["scheduler_count"] == 1


def test_aggregate_costs_rejects_mixed_currencies() -> None:
    service = BaseCostReportingService(currency=None)

    components = (
        CommissionCost(amount=Decimal("1"), currency="USD"),
        SlippageCost(amount=Decimal("1"), currency="EUR"),
    )

    with pytest.raises(ValueError):
        service.aggregate_costs(components)
