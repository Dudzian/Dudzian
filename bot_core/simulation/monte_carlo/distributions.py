"""Narzędzia do pracy z historycznymi rozkładami cen."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class HistoricalPriceSeries:
    """Reprezentacja szeregu cenowego wykorzystywanego w symulacji.

    Attributes:
        prices: Serie cenowe uporządkowane rosnąco według czasu.
        returns: Logarytmiczne stopy zwrotu obliczone na podstawie ``prices``.
    """

    prices: pd.Series
    returns: pd.Series

    @classmethod
    def from_prices(cls, prices: pd.Series) -> "HistoricalPriceSeries":
        if not isinstance(prices, pd.Series):
            raise TypeError("prices musi być instancją pandas.Series")
        sorted_prices = prices.sort_index()
        returns = compute_log_returns(sorted_prices)
        return cls(prices=sorted_prices, returns=returns)


def load_price_series(path: Path | str, price_column: str = "close") -> pd.Series:
    """Ładuje serie cenową z pliku CSV.

    Args:
        path: Ścieżka do pliku CSV zawierającego dane OHLCV.
        price_column: Nazwa kolumny z ceną wykorzystywaną w symulacji.

    Returns:
        ``pd.Series`` z indeksami czasowymi i cenami.
    """

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Nie znaleziono pliku z danymi historycznymi: {csv_path}")

    frame = pd.read_csv(csv_path)
    if price_column not in frame.columns:
        raise ValueError(f"Brak kolumny '{price_column}' w pliku {csv_path}")

    if "timestamp" in frame.columns:
        index = pd.to_datetime(frame["timestamp"])
    elif "date" in frame.columns:
        index = pd.to_datetime(frame["date"])
    else:
        index = pd.RangeIndex(start=0, stop=len(frame))

    series = pd.Series(frame[price_column].astype(float).to_numpy(), index=index, name="price")
    return series.sort_index()


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Oblicza logarytmiczne stopy zwrotu z szeregu cenowego."""

    return np.log(prices / prices.shift(1)).dropna()


def annualize_volatility(returns: pd.Series, periods_per_year: int = 252) -> float:
    """Szacuje roczną zmienność na podstawie stóp zwrotu."""

    return float(returns.std(ddof=1) * np.sqrt(periods_per_year))


def resample_returns(returns: Iterable[float], size: int, random_state: Optional[np.random.Generator] = None) -> np.ndarray:
    """Losuje próbkę stóp zwrotu z możliwością powtarzania."""

    generator = random_state or np.random.default_rng()
    returns_array = np.asarray(list(returns), dtype=float)
    if returns_array.size == 0:
        raise ValueError("Brak danych do resamplingu")
    indices = generator.integers(0, returns_array.size, size=size)
    return returns_array[indices]
