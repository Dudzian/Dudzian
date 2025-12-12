"""Obsługa kosztów/TCO na potrzeby Decision engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, MutableMapping

from bot_core.decision.models import DecisionCandidate
from bot_core.tco.models import ProfileCostSummary, StrategyCostSummary, TCOReport


@dataclass(slots=True)
class CostIndex:
    lookup: MutableMapping[tuple[str, str], float]
    default_cost: float | None


class DecisionCostResolver:
    """Centralny serwis zarządzający indeksami kosztów/TCO."""

    def __init__(self, *, penalty_cost_bps: float = 0.0) -> None:
        self._cost_index = CostIndex(lookup={}, default_cost=None)
        self._penalty_cost_bps = float(penalty_cost_bps)

    def update_costs_from_report(self, report: TCOReport | Mapping[str, object]) -> None:
        """Buduje indeks kosztów (bps) na podstawie raportu TCO."""

        index: MutableMapping[tuple[str, str], float] = {}
        default_cost: float | None = None

        if isinstance(report, TCOReport):
            strategies = report.strategies.values()
            default_cost = float(report.total.cost_bps)
            for summary in strategies:
                self._ingest_strategy_summary(summary, index)
        else:
            data = dict(report)
            strategies_data = data.get("strategies", {}) or {}
            for strategy_name, summary_raw in strategies_data.items():
                index[(str(strategy_name), "__total__")] = self._extract_cost_bps(
                    summary_raw.get("total")
                )
                profiles = summary_raw.get("profiles", {}) or {}
                for profile_name, profile_raw in profiles.items():
                    index[(str(strategy_name), str(profile_name))] = self._extract_cost_bps(
                        profile_raw
                    )
            total_raw = data.get("total")
            if total_raw is not None:
                default_cost = self._extract_cost_bps(total_raw)

        self._cost_index = CostIndex(lookup=index, default_cost=default_cost)

    def resolve_cost(self, candidate: DecisionCandidate) -> tuple[float | None, bool]:
        if candidate.cost_bps_override is not None:
            return candidate.cost_bps_override, False
        cost = self._cost_index.lookup.get((candidate.strategy, candidate.risk_profile))
        if cost is None:
            cost = self._cost_index.lookup.get((candidate.strategy, "__total__"))
        if cost is None:
            cost = self._cost_index.default_cost
        missing = cost is None
        if cost is None and self._penalty_cost_bps > 0:
            cost = self._penalty_cost_bps
        return cost, missing

    # --------------------------------------------------------------- helpery --
    def _ingest_strategy_summary(
        self,
        summary: StrategyCostSummary,
        index: MutableMapping[tuple[str, str], float],
    ) -> None:
        index[(summary.strategy, "__total__")] = float(summary.total.cost_bps)
        for profile_name, profile_summary in summary.profiles.items():
            index[(summary.strategy, profile_name)] = float(profile_summary.cost_bps)

    def _extract_cost_bps(self, payload: object) -> float:
        if payload is None:
            return 0.0
        if isinstance(payload, ProfileCostSummary):
            return float(payload.cost_bps)
        if isinstance(payload, Mapping):
            value = payload.get("cost_bps")
            if value is None:
                return 0.0
            return float(value)
        return float(payload)


__all__ = ["CostIndex", "DecisionCostResolver"]
