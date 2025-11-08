"""Dostawcy kursów walutowych wykorzystywani przy raportowaniu podatkowym."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping, Protocol


class FXRateProvider(Protocol):
    """Interfejs dostawcy kursów walutowych.

    Implementacje powinny zwracać współczynnik konwersji do waluty bazowej.
    """

    def get_rate(self, currency: str, timestamp: datetime) -> float:  # pragma: no cover - protokół
        """Zwraca kurs dla waluty ``currency`` obowiązujący w chwili ``timestamp``."""


@dataclass(slots=True)
class StaticFXRateProvider:
    """Dostawca kursów o stałej wartości, niezależnej od czasu."""

    rates: Mapping[str, float]
    base_currency: str | None = None

    def get_rate(self, currency: str, timestamp: datetime) -> float:
        """Zwraca stały kurs dla wskazanej waluty.

        Jeżeli waluta jest zgodna z walutą bazową lub nie została podana,
        zwracany jest kurs ``1.0``.
        """

        if not currency:
            return 1.0
        currency_upper = currency.strip().upper()
        if not currency_upper:
            return 1.0
        base_upper = (self.base_currency or "").strip().upper()
        if base_upper and currency_upper == base_upper:
            return 1.0
        if currency_upper in self.rates:
            return float(self.rates[currency_upper])
        raise KeyError(currency_upper)


__all__ = ["FXRateProvider", "StaticFXRateProvider"]
