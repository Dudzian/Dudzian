"""Silnik analizy kosztów transakcyjnych (TCO)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterable, Mapping, MutableMapping

from .models import (
    CostBreakdown,
    ProfileCostSummary,
    SchedulerCostSummary,
    StrategyCostSummary,
    TCOReport,
    TradeCostEvent,
)


@dataclass(slots=True)
class _Accumulator:
    trade_count: int = 0
    notional: Decimal = Decimal("0")
    breakdown: CostBreakdown = field(default_factory=CostBreakdown.zero)

    def add_event(self, event: TradeCostEvent) -> None:
        self.trade_count += 1
        self.notional += event.notional
        self.breakdown.add_event(event)

    def to_summary(self, *, profile: str) -> ProfileCostSummary:
        return ProfileCostSummary(
            profile=profile,
            trade_count=self.trade_count,
            notional=self.notional,
            breakdown=self.breakdown,
        )


@dataclass(slots=True)
class _SchedulerAggregation:
    scheduler: str
    strategies: MutableMapping[str, _Accumulator] = field(default_factory=dict)
    total: _Accumulator = field(default_factory=_Accumulator)

    def add_event(self, event: TradeCostEvent) -> None:
        strategy_bucket = self.strategies.setdefault(event.strategy, _Accumulator())
        strategy_bucket.add_event(event)
        self.total.add_event(event)


def _resolve_scheduler(metadata: Mapping[str, Any]) -> str:
    for key in ("scheduler", "scheduler_id", "schedule"):
        value = metadata.get(key)
        if value in (None, ""):
            continue
        text = str(value).strip()
        if text:
            return text
    return "default"


class TCOAnalyzer:
    """Agreguje koszty transakcyjne i buduje raport TCO."""

    def __init__(self, *, cost_limit_bps: Decimal | None = None) -> None:
        self._cost_limit_bps = cost_limit_bps

    def analyze(
        self,
        events: Iterable[TradeCostEvent],
        *,
        metadata: Mapping[str, object] | None = None,
    ) -> TCOReport:
        metadata = dict(metadata or {})
        by_strategy: MutableMapping[str, MutableMapping[str, _Accumulator]] = {}
        by_scheduler: MutableMapping[str, _SchedulerAggregation] = {}
        totals = _Accumulator()

        for event in events:
            strategy_bucket = by_strategy.setdefault(event.strategy, {})
            profile_bucket = strategy_bucket.setdefault(event.risk_profile, _Accumulator())
            profile_bucket.add_event(event)
            totals.add_event(event)
            scheduler_name = _resolve_scheduler(event.metadata)
            scheduler_bucket = by_scheduler.setdefault(
                scheduler_name, _SchedulerAggregation(scheduler=scheduler_name)
            )
            scheduler_bucket.add_event(event)

        strategy_summaries: MutableMapping[str, StrategyCostSummary] = {}
        scheduler_summaries: MutableMapping[str, SchedulerCostSummary] = {}
        alerts: list[str] = []

        for strategy, profiles in sorted(by_strategy.items()):
            strategy_total = _Accumulator()
            profile_summaries: MutableMapping[str, ProfileCostSummary] = {}
            for profile, accumulator in sorted(profiles.items()):
                profile_summary = accumulator.to_summary(profile=profile)
                profile_summaries[profile] = profile_summary
                strategy_total.trade_count += accumulator.trade_count
                strategy_total.notional += accumulator.notional
                strategy_total.breakdown.commission += accumulator.breakdown.commission
                strategy_total.breakdown.slippage += accumulator.breakdown.slippage
                strategy_total.breakdown.funding += accumulator.breakdown.funding
                strategy_total.breakdown.other += accumulator.breakdown.other
                self._check_threshold(strategy, profile_summary, alerts)
            strategy_summary = StrategyCostSummary(
                strategy=strategy,
                profiles=dict(profile_summaries),
                total=strategy_total.to_summary(profile="all"),
            )
            strategy_summaries[strategy] = strategy_summary
            self._check_threshold(strategy, strategy_summary.total, alerts)

        for scheduler_name, aggregation in sorted(by_scheduler.items()):
            strategy_breakdowns: MutableMapping[str, ProfileCostSummary] = {}
            for strategy, accumulator in sorted(aggregation.strategies.items()):
                strategy_breakdowns[strategy] = accumulator.to_summary(profile=strategy)
            scheduler_summaries[scheduler_name] = SchedulerCostSummary(
                scheduler=scheduler_name,
                strategies=dict(strategy_breakdowns),
                total=aggregation.total.to_summary(profile="all"),
            )

        if self._cost_limit_bps is not None:
            metadata.setdefault("cost_limit_bps", float(self._cost_limit_bps))
        metadata.setdefault("events_count", totals.trade_count)
        metadata.setdefault("strategy_count", len(strategy_summaries))
        metadata.setdefault("scheduler_count", len(scheduler_summaries))
        generated_at = datetime.now(tz=timezone.utc)
        report = TCOReport(
            generated_at=generated_at,
            metadata=metadata,
            strategies=dict(strategy_summaries),
            total=totals.to_summary(profile="all"),
            alerts=alerts,
            schedulers=dict(scheduler_summaries),
        )
        return report

    def _check_threshold(
        self,
        strategy: str,
        summary: ProfileCostSummary,
        alerts: list[str],
    ) -> None:
        if self._cost_limit_bps is None:
            return
        if summary.cost_bps <= self._cost_limit_bps:
            return
        alerts.append(
            (
                "Strategia {strategy} / profil {profile} przekracza limit kosztów {limit:.2f} bps: "
                "{actual:.2f} bps"
            ).format(
                strategy=strategy,
                profile=summary.profile,
                limit=float(self._cost_limit_bps),
                actual=float(summary.cost_bps),
            )
        )
