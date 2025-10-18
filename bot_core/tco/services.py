"""Bazowe usługi raportowania kosztów transakcyjnych."""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Iterable, Mapping, MutableMapping, Protocol

from .costs import BaseCostComponent

__all__ = [
    "AggregatedCostReport",
    "BaseCostReportingService",
    "CostAggregationContext",
    "CostComponentSummary",
    "CostReportExtension",
    "StrategyCostView",
]

_DECIMAL_ZERO = Decimal("0")


@dataclass(slots=True, frozen=True)
class CostComponentSummary:
    """Zestawienie wartości kosztów dla pojedynczego widoku."""

    amounts: Mapping[str, Decimal]

    def __post_init__(self) -> None:
        object.__setattr__(self, "amounts", MappingProxyType(dict(self.amounts)))

    @property
    def total(self) -> Decimal:
        return sum(self.amounts.values(), _DECIMAL_ZERO)

    @classmethod
    def from_amounts(cls, amounts: Mapping[str, Decimal]) -> "CostComponentSummary":
        ordered = dict(sorted(amounts.items()))
        return cls(amounts=ordered)

    def to_dict(self) -> dict[str, Any]:
        payload = {name: str(amount) for name, amount in self.amounts.items()}
        payload["total"] = str(self.total)
        return payload


@dataclass(slots=True)
class _CostBucket:
    amounts: MutableMapping[str, Decimal] = field(default_factory=dict)

    def add_component(self, component: BaseCostComponent) -> None:
        current = self.amounts.get(component.component_type, _DECIMAL_ZERO)
        self.amounts[component.component_type] = current + component.amount

    def merge(self, other: "_CostBucket") -> None:
        for component_type, value in other.amounts.items():
            current = self.amounts.get(component_type, _DECIMAL_ZERO)
            self.amounts[component_type] = current + value

    def to_summary(self) -> CostComponentSummary:
        return CostComponentSummary.from_amounts(self.amounts)


@dataclass(slots=True, frozen=True)
class StrategyCostView:
    """Agregacja kosztów dla pojedynczej strategii."""

    strategy: str
    profiles: Mapping[str, CostComponentSummary]
    total: CostComponentSummary

    def __post_init__(self) -> None:
        ordered_profiles = dict(sorted(self.profiles.items()))
        object.__setattr__(self, "profiles", MappingProxyType(ordered_profiles))

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy,
            "profiles": {name: summary.to_dict() for name, summary in self.profiles.items()},
            "total": self.total.to_dict(),
        }


@dataclass(slots=True, frozen=True)
class AggregatedCostReport:
    """Struktura zawierająca zsumowane informacje o kosztach."""

    currency: str
    totals: CostComponentSummary
    strategies: Mapping[str, StrategyCostView]
    metadata: Mapping[str, Any]
    component_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "strategies", MappingProxyType(dict(self.strategies)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "totals": self.totals.to_dict(),
            "strategies": {name: view.to_dict() for name, view in self.strategies.items()},
            "metadata": dict(self.metadata),
            "component_count": self.component_count,
        }


@dataclass(slots=True)
class _StrategyAggregation:
    strategy: str
    profiles: MutableMapping[str, _CostBucket] = field(default_factory=dict)
    total: _CostBucket = field(default_factory=_CostBucket)

    def add_component(self, profile: str, component: BaseCostComponent) -> None:
        profile_bucket = self.profiles.setdefault(profile, _CostBucket())
        profile_bucket.add_component(component)
        self.total.add_component(component)


@dataclass(slots=True)
class CostAggregationContext:
    """Stan agregacji używany podczas budowy raportu."""

    currency: str | None
    metadata: MutableMapping[str, Any]
    totals: _CostBucket = field(default_factory=_CostBucket)
    strategies: MutableMapping[str, _StrategyAggregation] = field(default_factory=dict)
    components: list[BaseCostComponent] = field(default_factory=list)

    def add_component(self, component: BaseCostComponent) -> None:
        if self.currency is None:
            self.currency = component.currency
        elif component.currency and component.currency != self.currency:
            raise ValueError(
                "Wszystkie komponenty muszą być w tej samej walucie; "
                f"oczekiwano {self.currency!r}, otrzymano {component.currency!r}"
            )
        self.components.append(component)
        self.totals.add_component(component)
        strategy = str(component.metadata.get("strategy", "global"))
        profile = str(
            component.metadata.get(
                "risk_profile",
                component.metadata.get("profile", "default"),
            )
        )
        strategy_bucket = self.strategies.setdefault(strategy, _StrategyAggregation(strategy=strategy))
        strategy_bucket.add_component(profile, component)

    @property
    def component_count(self) -> int:
        return len(self.components)

    def to_report(self) -> AggregatedCostReport:
        strategies: dict[str, StrategyCostView] = {}
        for name, aggregation in sorted(self.strategies.items()):
            profiles = {
                profile: bucket.to_summary()
                for profile, bucket in sorted(aggregation.profiles.items())
            }
            strategies[name] = StrategyCostView(
                strategy=name,
                profiles=profiles,
                total=aggregation.total.to_summary(),
            )
        metadata = dict(self.metadata)
        metadata.setdefault("strategy_count", len(strategies))
        metadata.setdefault("component_count", self.component_count)
        report_currency = self.currency or ""
        return AggregatedCostReport(
            currency=report_currency,
            totals=self.totals.to_summary(),
            strategies=strategies,
            metadata=metadata,
            component_count=self.component_count,
        )


class CostReportExtension(Protocol):
    """Interfejs rozszerzeń raportowania kosztów."""

    def apply(self, context: CostAggregationContext) -> None:
        """Modyfikuje kontekst agregacji przed finalizacją raportu."""


class BaseCostReportingService:
    """Podstawowa usługa agregująca komponenty kosztowe."""

    def __init__(self, *, currency: str | None = "USD") -> None:
        self._currency = currency
        self._extensions: list[CostReportExtension] = []

    def register_extension(self, extension: CostReportExtension) -> None:
        self._extensions.append(extension)

    def aggregate_costs(
        self,
        components: Iterable[BaseCostComponent],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> AggregatedCostReport:
        context = CostAggregationContext(currency=self._currency, metadata=dict(metadata or {}))
        for component in components:
            context.add_component(component)
        for extension in self._extensions:
            extension.apply(context)
        return context.to_report()
