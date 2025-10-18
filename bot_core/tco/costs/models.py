"""Dataclasses opisujące podstawowe komponenty kosztowe w raportach TCO."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, MutableMapping

__all__ = [
    "BaseCostComponent",
    "CommissionCost",
    "CostComponent",
    "CostComponentFactory",
    "FundingCost",
    "SlippageCost",
]


def _ensure_decimal(value: Decimal | int | float | str) -> Decimal:
    """Konwertuje obsługiwane typy na :class:`~decimal.Decimal`."""
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float)):
        return Decimal(str(value))
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return Decimal("0")
        return Decimal(cleaned)
    raise TypeError(f"Nieobsługiwany typ wartości kosztu: {type(value)!r}")


def _ensure_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})
    if isinstance(metadata, Mapping):
        return MappingProxyType(dict(metadata))
    return MappingProxyType(dict(metadata))


def _ensure_timestamp(value: datetime | str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return datetime.now(timezone.utc)
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        timestamp = datetime.fromisoformat(cleaned)
    else:  # pragma: no cover - defensywne
        raise TypeError(f"Nieobsługiwany format timestampu: {type(value)!r}")
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


@dataclass(slots=True, frozen=True)
class BaseCostComponent:
    """Bazowa reprezentacja komponentu kosztowego."""

    amount: Decimal
    currency: str = "USD"
    timestamp: datetime | str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    component_type: ClassVar[str] = "base"

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", _ensure_decimal(self.amount))
        object.__setattr__(self, "metadata", _ensure_metadata(self.metadata))
        object.__setattr__(self, "timestamp", _ensure_timestamp(self.timestamp))
        object.__setattr__(self, "currency", str(self.currency))

    def with_metadata(self, **extra: Any) -> "BaseCostComponent":
        metadata: MutableMapping[str, Any] = dict(self.metadata)
        metadata.update(extra)
        return type(self)(
            amount=self.amount,
            currency=self.currency,
            timestamp=self.timestamp,
            metadata=metadata,
        )

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "type": self.component_type,
            "amount": str(self.amount),
            "currency": self.currency,
            "timestamp": self.timestamp.isoformat(),
            "metadata": dict(self.metadata),
        }
        return payload


@dataclass(slots=True, frozen=True)
class CommissionCost(BaseCostComponent):
    """Komponent reprezentujący koszty prowizji."""

    component_type: ClassVar[str] = "commission"


@dataclass(slots=True, frozen=True)
class SlippageCost(BaseCostComponent):
    """Komponent reprezentujący koszty slippage."""

    component_type: ClassVar[str] = "slippage"


@dataclass(slots=True, frozen=True)
class FundingCost(BaseCostComponent):
    """Komponent reprezentujący koszty finansowania."""

    component_type: ClassVar[str] = "funding"


CostComponent = BaseCostComponent


class CostComponentFactory:
    """Prosta fabryka komponentów kosztowych."""

    _registry: dict[str, type[BaseCostComponent]] = {
        CommissionCost.component_type: CommissionCost,
        SlippageCost.component_type: SlippageCost,
        FundingCost.component_type: FundingCost,
    }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> BaseCostComponent:
        try:
            component_type = str(payload["type"])
        except KeyError as exc:  # pragma: no cover - defensywne
            raise KeyError("Brak klucza 'type' w danych komponentu kosztowego") from exc
        try:
            component_cls = cls._registry[component_type]
        except KeyError as exc:
            raise ValueError(f"Nieznany typ komponentu kosztowego: {component_type!r}") from exc
        data = dict(payload)
        amount = _ensure_decimal(data.get("amount", Decimal("0")))
        currency = data.get("currency", "USD")
        timestamp = data.get("timestamp")
        metadata = data.get("metadata")
        return component_cls(
            amount=amount,
            currency=str(currency),
            timestamp=timestamp,
            metadata=metadata,
        )

    @classmethod
    def register(
        cls,
        component_type: str,
        component_cls: type[BaseCostComponent],
    ) -> None:
        if not issubclass(component_cls, BaseCostComponent):  # pragma: no cover - defensywne
            raise TypeError("Zarejestrowana klasa musi dziedziczyć po BaseCostComponent")
        if component_type in cls._registry:
            raise ValueError(f"Typ komponentu {component_type!r} jest już zarejestrowany")
        updated = dict(cls._registry)
        updated[component_type] = component_cls
        cls._registry = updated
