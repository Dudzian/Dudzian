"""Analiza kosztów całkowitych utrzymania (TCO) dla hypercare Stage5."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, MutableMapping, Sequence

from bot_core.security.signing import build_hmac_signature


@dataclass(slots=True)
class TcoCostItem:
    """Pojedynczy składnik kosztowy w analizie TCO."""

    name: str
    category: str
    monthly_cost: float
    currency: str = "USD"
    notes: str | None = None

    @property
    def annual_cost(self) -> float:
        """Koszt roczny przy założeniu 12 równych miesięcy."""

        return self.monthly_cost * 12.0

    def to_payload(self) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "name": self.name,
            "category": self.category,
            "monthly_cost": self.monthly_cost,
            "annual_cost": self.annual_cost,
            "currency": self.currency,
        }
        if self.notes:
            payload["notes"] = self.notes
        return payload


@dataclass(slots=True)
class TcoCategoryBreakdown:
    """Zbiorcze koszty dla pojedynczej kategorii."""

    category: str
    monthly_total: float
    annual_total: float

    def to_payload(self) -> Mapping[str, object]:
        return {
            "category": self.category,
            "monthly_total": self.monthly_total,
            "annual_total": self.annual_total,
        }


@dataclass(slots=True)
class TcoUsageMetrics:
    """Parametry wykorzystania służące do przeliczeń jednostkowych."""

    monthly_trades: float | None = None
    monthly_volume: float | None = None

    def to_payload(self, *, monthly_total: float) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {}
        if self.monthly_trades is not None:
            payload["monthly_trades"] = self.monthly_trades
            if self.monthly_trades > 0:
                payload["cost_per_trade"] = monthly_total / self.monthly_trades
        if self.monthly_volume is not None:
            payload["monthly_volume"] = self.monthly_volume
            if self.monthly_volume > 0:
                payload["cost_per_volume_unit"] = monthly_total / self.monthly_volume
        return payload


@dataclass(slots=True)
class TcoSummary:
    """Podsumowanie analizy kosztów TCO."""

    currency: str
    monthly_total: float
    annual_total: float
    categories: Sequence[TcoCategoryBreakdown]
    items: Sequence[TcoCostItem]

    def to_payload(
        self,
        *,
        generated_at: datetime,
        usage: TcoUsageMetrics | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> Mapping[str, object]:
        payload: MutableMapping[str, object] = {
            "currency": self.currency,
            "generated_at": generated_at.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "monthly_total": self.monthly_total,
            "annual_total": self.annual_total,
            "categories": [category.to_payload() for category in self.categories],
            "items": [item.to_payload() for item in self.items],
        }
        if usage is not None:
            usage_payload = usage.to_payload(monthly_total=self.monthly_total)
            if usage_payload:
                payload["usage"] = usage_payload
        if metadata:
            payload.update(metadata)
        return payload


def load_cost_items(path: Path) -> Sequence[TcoCostItem]:
    """Ładuje listę pozycji kosztowych z pliku JSON."""

    contents = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(contents, Mapping):
        default_currency = str(contents.get("currency", "USD"))
        raw_items = contents.get("items", [])
    elif isinstance(contents, list):
        default_currency = "USD"
        raw_items = contents
    else:
        raise TypeError("Nieobsługiwany format pliku wejściowego TCO")

    items: list[TcoCostItem] = []
    for entry in raw_items:
        if not isinstance(entry, Mapping):
            raise TypeError("Element listy kosztów musi być słownikiem")
        try:
            name = str(entry["name"])
            category = str(entry["category"])
            monthly_cost = float(entry["monthly_cost"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Nieprawidłowy wpis kosztowy: {entry!r}") from exc
        currency = str(entry.get("currency", default_currency))
        notes = entry.get("notes")
        items.append(
            TcoCostItem(
                name=name,
                category=category,
                monthly_cost=monthly_cost,
                currency=currency,
                notes=str(notes) if notes is not None else None,
            )
        )
    return items


def aggregate_costs(items: Sequence[TcoCostItem]) -> TcoSummary:
    """Zlicza koszty miesięczne i roczne w rozbiciu na kategorie."""

    if not items:
        return TcoSummary(
            currency="USD",
            monthly_total=0.0,
            annual_total=0.0,
            categories=[],
            items=[],
        )

    currency = items[0].currency
    for item in items:
        if item.currency != currency:
            raise ValueError("Wszystkie pozycje kosztowe muszą mieć tę samą walutę")

    monthly_total = sum(item.monthly_cost for item in items)
    annual_total = monthly_total * 12.0

    category_totals: dict[str, float] = {}
    for item in items:
        category_totals.setdefault(item.category, 0.0)
        category_totals[item.category] += item.monthly_cost

    categories = [
        TcoCategoryBreakdown(
            category=category,
            monthly_total=value,
            annual_total=value * 12.0,
        )
        for category, value in sorted(category_totals.items())
    ]

    return TcoSummary(
        currency=currency,
        monthly_total=monthly_total,
        annual_total=annual_total,
        categories=categories,
        items=list(items),
    )


def write_summary_json(
    summary: TcoSummary,
    path: Path,
    *,
    generated_at: datetime,
    usage: TcoUsageMetrics | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Zapisuje podsumowanie do pliku JSON i zwraca payload."""

    payload = summary.to_payload(generated_at=generated_at, usage=usage, metadata=metadata)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_summary_csv(summary: TcoSummary, path: Path) -> None:
    """Zapisuje rozbicie pozycji kosztowych do pliku CSV."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["category", "item", "monthly_cost", "annual_cost", "currency", "notes"])
        for item in summary.items:
            writer.writerow(
                [
                    item.category,
                    item.name,
                    f"{item.monthly_cost:.2f}",
                    f"{item.annual_cost:.2f}",
                    item.currency,
                    item.notes or "",
                ]
            )


def write_summary_signature(
    payload: Mapping[str, object],
    path: Path,
    *,
    key: bytes,
    key_id: str,
) -> Mapping[str, object]:
    """Zapisuje podpis HMAC dla podsumowania TCO."""

    signature = build_hmac_signature(payload, key=key, key_id=key_id)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(signature, ensure_ascii=False, indent=2), encoding="utf-8")
    return signature


__all__ = [
    "TcoCostItem",
    "TcoCategoryBreakdown",
    "TcoUsageMetrics",
    "TcoSummary",
    "load_cost_items",
    "aggregate_costs",
    "write_summary_json",
    "write_summary_csv",
    "write_summary_signature",
]
