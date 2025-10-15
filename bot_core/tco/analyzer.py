"""Silnik analizy kosztów transakcyjnych (TCO)."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from typing import Iterable, Mapping, MutableMapping

from .models import CostBreakdown, ProfileCostSummary, StrategyCostSummary, TCOReport, TradeCostEvent


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
        totals = _Accumulator()

        for event in events:
            strategy_bucket = by_strategy.setdefault(event.strategy, {})
            profile_bucket = strategy_bucket.setdefault(event.risk_profile, _Accumulator())
            profile_bucket.add_event(event)
            totals.add_event(event)

        strategy_summaries: MutableMapping[str, StrategyCostSummary] = {}
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

        if self._cost_limit_bps is not None:
            metadata.setdefault("cost_limit_bps", float(self._cost_limit_bps))
        metadata.setdefault("events_count", totals.trade_count)
        metadata.setdefault("strategy_count", len(strategy_summaries))
        generated_at = datetime.now(tz=timezone.utc)
        report = TCOReport(
            generated_at=generated_at,
            metadata=metadata,
            strategies=dict(strategy_summaries),
            total=totals.to_summary(profile="all"),
            alerts=alerts,
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
