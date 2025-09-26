"""Pobieranie i normalizacja danych rynkowych z cache'em."""
from __future__ import annotations

import asyncio
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


class MarketDataError(RuntimeError):
    """Sygnalizuje problem z pobraniem lub walidacją danych rynkowych."""


@dataclass(frozen=True)
class MarketDataRequest:
    """Opisuje zakres danych OHLCV do pobrania."""

    symbol: str
    timeframe: str
    limit: int = 500
    since: Optional[int] = None

    def cache_key(self) -> Tuple[str, str, int, Optional[int]]:
        return (
            (self.symbol or "").upper(),
            (self.timeframe or "").lower(),
            int(self.limit),
            int(self.since) if self.since is not None else None,
        )


@dataclass
class _CacheEntry:
    df: pd.DataFrame
    fetched_at: float
    coverage: Tuple[int, int]


def _timeframe_to_seconds(value: str) -> int:
    mapping = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    if not value:
        raise MarketDataError("Timeframe nie może być pusty")
    unit = value[-1].lower()
    try:
        factor = mapping[unit]
    except KeyError as exc:
        raise MarketDataError(f"Nieznana jednostka w timeframe: {value}") from exc
    try:
        amount = int(value[:-1])
    except ValueError as exc:
        raise MarketDataError(f"Nieprawidłowy format timeframe: {value}") from exc
    if amount <= 0:
        raise MarketDataError("Timeframe musi być dodatni")
    return amount * factor


class MarketDataProvider:
    """Ujednolicony provider danych OHLCV z cache'em i walidacją luk."""

    def __init__(
        self,
        exchange: Any,
        *,
        cache_ttl_s: float = 120.0,
        max_candles: int = 5000,
    ) -> None:
        self._exchange = exchange
        self._cache_ttl_s = max(1.0, float(cache_ttl_s))
        self._max_candles = int(max(1, max_candles))
        self._cache: Dict[Tuple[str, str, int, Optional[int]], _CacheEntry] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------
    def get_historical(self, request: MarketDataRequest) -> pd.DataFrame:
        """Zwraca znormalizowany DataFrame OHLCV; korzysta z cache."""

        key = request.cache_key()
        now = time.time()
        with self._lock:
            entry = self._cache.get(key)
            if entry and now - entry.fetched_at < self._cache_ttl_s:
                logger.debug("Zwracam dane z cache dla %s", key)
                return entry.df.copy(deep=True)

        raw = self._fetch_ohlcv(request)
        df = self._normalize(raw)
        if df.empty:
            raise MarketDataError("Brak danych OHLCV z giełdy")
        timeframe_s = _timeframe_to_seconds(request.timeframe)
        df = self._ensure_regular_index(df, timeframe_s)
        if len(df) > request.limit:
            df = df.iloc[-request.limit :]

        with self._lock:
            self._cache[key] = _CacheEntry(df=df.copy(deep=True), fetched_at=now, coverage=(int(df.index[0].timestamp()), int(df.index[-1].timestamp())))
        return df

    async def get_historical_async(self, request: MarketDataRequest) -> pd.DataFrame:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_historical, request)

    def get_latest_price(self, symbol: str, default: Optional[float] = None) -> Optional[float]:
        ticker_fn = getattr(self._exchange, "fetch_ticker", None)
        if callable(ticker_fn):
            try:
                ticker = ticker_fn(symbol) or {}
                for key in ("last", "close", "bid", "ask"):
                    val = ticker.get(key)
                    if val is not None:
                        return float(val)
            except Exception:
                logger.exception("fetch_ticker(%s) nie powiodło się", symbol)
        # fallback do ostatniej świecy z cache
        with self._lock:
            for (sym, _tf, *_), entry in self._cache.items():
                if sym == (symbol or "").upper() and not entry.df.empty:
                    try:
                        return float(entry.df["close"].iloc[-1])
                    except Exception:  # pragma: no cover - bezpieczeństwo
                        continue
        return default

    # ------------------------------------------------------------------
    def _fetch_ohlcv(self, request: MarketDataRequest) -> Iterable[Any]:
        fetch_fn = getattr(self._exchange, "fetch_ohlcv", None)
        if not callable(fetch_fn):
            raise MarketDataError("Obiekt giełdy nie udostępnia metody fetch_ohlcv")
        limit = min(request.limit, self._max_candles)
        try:
            raw = fetch_fn(
                request.symbol,
                timeframe=request.timeframe,
                limit=limit,
                since=request.since,
            )
            if asyncio.iscoroutine(raw):
                raw = asyncio.run(raw)
            return raw or []
        except Exception as exc:
            logger.exception("fetch_ohlcv dla %s nie powiodło się", request)
            raise MarketDataError(f"fetch_ohlcv nie powiodło się: {exc}") from exc

    @staticmethod
    def _normalize(raw: Iterable[Any]) -> pd.DataFrame:
        if isinstance(raw, pd.DataFrame):
            df = raw.copy()
        else:
            try:
                df = pd.DataFrame(list(raw))
            except Exception as exc:
                raise MarketDataError("Nie można zbudować DataFrame z danych OHLCV") from exc
        if df.empty:
            return df
        expected = ["timestamp", "open", "high", "low", "close", "volume"]
        if len(df.columns) >= 6:
            rename_map = {df.columns[i]: expected[i] for i in range(6)}
            df = df.rename(columns=rename_map)
        missing = [col for col in expected if col not in df.columns]
        if missing:
            raise MarketDataError(f"Brak kolumn w danych OHLCV: {missing}")
        df = df[expected[: len(df.columns)]]
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
        df = df.set_index("timestamp")
        return df

    @staticmethod
    def _ensure_regular_index(df: pd.DataFrame, timeframe_s: int) -> pd.DataFrame:
        if df.empty:
            return df
        freq = pd.Timedelta(seconds=timeframe_s)
        expected_index = pd.date_range(start=df.index[0], end=df.index[-1], freq=freq)
        if len(expected_index) == len(df) and df.index.equals(expected_index):
            return df
        # Wypełnij luki metodą forward-fill dla cen i zerami dla wolumenów
        reindexed = df.reindex(expected_index)
        price_cols = ["open", "high", "low", "close"]
        reindexed[price_cols] = reindexed[price_cols].ffill()
        reindexed["volume"] = reindexed["volume"].fillna(0.0)
        missing = reindexed[price_cols].isna().sum().sum()
        if missing:
            raise MarketDataError("Nie udało się uzupełnić luk w danych OHLCV")
        return reindexed

