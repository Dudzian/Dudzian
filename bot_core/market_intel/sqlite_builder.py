"""Budowanie bazy SQLite z metrykami Market Intelligence (Stage6)."""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from statistics import pstdev
from typing import Mapping, Protocol, Sequence

from bot_core.config.models import MarketIntelConfig, MarketIntelSqliteConfig
from bot_core.market_intel.models import MarketIntelBaseline

_LOGGER = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Dane wejściowe
# ---------------------------------------------------------------------------


@dataclass(slots=True, frozen=True)
class OrderBookLevel:
    """Pojedynczy poziom orderbooka."""

    price: float
    quantity: float


@dataclass(slots=True, frozen=True)
class OrderBookSnapshot:
    """Ograniczony orderbook wykorzystywany do agregacji Stage6."""

    bids: Sequence[OrderBookLevel]
    asks: Sequence[OrderBookLevel]


@dataclass(slots=True, frozen=True)
class FundingSnapshot:
    """Informacje o aktualnej stawce fundingowej w punktach bazowych."""

    rate_bps: float


@dataclass(slots=True, frozen=True)
class SentimentSnapshot:
    """Ocena sentymentu rynku w skali [-1.0, 1.0]."""

    score: float


@dataclass(slots=True, frozen=True)
class OHLCVBar:
    """Świeca OHLCV potrzebna do obliczenia zrealizowanej zmienności."""

    close: float


class MarketIntelDataProvider(Protocol):
    """Dostawca danych źródłowych do budowy bazowych metryk Stage6."""

    def fetch_order_book(self, symbol: str, *, depth: int) -> OrderBookSnapshot:
        ...

    def fetch_funding(self, symbol: str) -> FundingSnapshot:
        ...

    def fetch_sentiment(self, symbol: str) -> SentimentSnapshot:
        ...

    def fetch_ohlcv(self, symbol: str, *, bars: int) -> Sequence[OHLCVBar]:
        ...

    def resolve_weight(self, symbol: str) -> float | None:
        ...


# ---------------------------------------------------------------------------
#  Budowa metryk i zapis do SQLite
# ---------------------------------------------------------------------------


class MarketIntelSqliteBuilder:
    """Automatyzuje zasilenie bazy SQLite na potrzeby Stage6."""

    def __init__(
        self,
        config: MarketIntelConfig,
        *,
        provider: MarketIntelDataProvider,
        depth_levels: int = 50,
        volatility_lookback: int = 240,
        logger: logging.Logger | None = None,
    ) -> None:
        if not isinstance(config, MarketIntelConfig):
            raise TypeError("config musi być instancją MarketIntelConfig")
        if config.sqlite is None:
            raise ValueError("MarketIntelConfig wymaga sekcji sqlite do budowy bazy")
        if depth_levels <= 0:
            raise ValueError("depth_levels musi być dodatnie")
        if volatility_lookback < 2:
            raise ValueError("volatility_lookback musi być >= 2")

        self._config = config
        self._sqlite_cfg: MarketIntelSqliteConfig = config.sqlite
        self._provider = provider
        self._depth_levels = int(depth_levels)
        self._lookback = int(volatility_lookback)
        self._logger = logger or _LOGGER

    # ------------------------------------------------------------------ API --
    def collect(self, *, symbols: Sequence[str] | None = None) -> tuple[MarketIntelBaseline, ...]:
        """Buduje listę bazowych metryk dla wskazanych symboli."""

        targets = tuple(symbols or self._config.required_symbols)
        if not targets:
            raise ValueError("Brak symboli wymaganych do budowy Market Intelligence")

        baselines: list[MarketIntelBaseline] = []
        for symbol in targets:
            snapshot = self._provider.fetch_order_book(symbol, depth=self._depth_levels)
            funding = self._provider.fetch_funding(symbol)
            sentiment = self._provider.fetch_sentiment(symbol)
            candles = self._provider.fetch_ohlcv(symbol, bars=self._lookback)
            weight = self._resolve_weight(symbol)

            baseline = MarketIntelBaseline(
                symbol=symbol,
                mid_price=_mid_price(snapshot),
                avg_depth_usd=_average_depth(snapshot, self._depth_levels),
                avg_spread_bps=_spread_bps(snapshot),
                funding_rate_bps=float(funding.rate_bps),
                sentiment_score=float(sentiment.score),
                realized_volatility=_realized_volatility(candles),
                weight=weight,
            )
            baselines.append(baseline)

        return tuple(baselines)

    def write_database(self, baselines: Sequence[MarketIntelBaseline]) -> Path:
        """Zapisuje metryki do bazy SQLite zgodnie z konfiguracją."""

        db_path = Path(self._sqlite_cfg.path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        table = _validate_identifier(self._sqlite_cfg.table, "table")
        columns = _sql_columns(self._sqlite_cfg)

        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(_create_table_sql(table, columns))
            conn.execute(f"DELETE FROM {table}")
            insert_sql = _insert_sql(table, columns)
            payloads = [_row_payload(baseline, columns) for baseline in baselines]
            conn.executemany(insert_sql, payloads)
            conn.commit()

        self._logger.info("Zapisano %d rekordów Market Intel do %s", len(baselines), db_path)
        return db_path

    def validate_checksums(self, baselines: Sequence[MarketIntelBaseline]) -> None:
        """Porównuje sumy kontrolne rekordów z oczekiwanymi wartościami."""

        expected = {baseline.symbol: _checksum(baseline) for baseline in baselines}
        actual: dict[str, str] = {}
        for baseline in self.read_database():
            actual[baseline.symbol] = _checksum(baseline)

        if expected != actual:
            missing = sorted(set(expected) ^ set(actual))
            if missing:
                raise ValueError(
                    "Niepoprawne sumy kontrolne Market Intel (różne symbole): " + ", ".join(missing)
                )
            mismatched = [symbol for symbol, digest in expected.items() if actual.get(symbol) != digest]
            raise ValueError(
                "Niepoprawne sumy kontrolne Market Intel dla: " + ", ".join(sorted(mismatched))
            )

        self._logger.debug("Zweryfikowano sumy kontrolne Market Intel: %s", json.dumps(actual, indent=2))

    def read_database(self) -> tuple[MarketIntelBaseline, ...]:
        """Odczytuje rekordy z aktualnej bazy SQLite."""

        db_path = Path(self._sqlite_cfg.path)
        table = _validate_identifier(self._sqlite_cfg.table, "table")
        columns = _sql_columns(self._sqlite_cfg)
        if not db_path.exists():
            raise FileNotFoundError(f"Brak bazy Market Intelligence: {db_path}")

        query = _select_sql(table, columns)
        baselines: list[MarketIntelBaseline] = []
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute(query)
            for row in cursor.fetchall():
                payload = {name: row[idx] for idx, name in enumerate(columns.keys())}
                baselines.append(_baseline_from_row(payload, self._config))
        return tuple(baselines)

    # ------------------------------------------------------------- helpers --
    def _resolve_weight(self, symbol: str) -> float:
        weight = None
        if hasattr(self._provider, "resolve_weight"):
            try:
                weight = self._provider.resolve_weight(symbol)
            except Exception as exc:  # pragma: no cover - diagnostyka providerów
                self._logger.warning("resolve_weight(%s) zgłosiło wyjątek: %s", symbol, exc)
        if weight is None:
            return float(self._config.default_weight)
        return float(weight)


# ---------------------------------------------------------------------------
#  Funkcje pomocnicze – przetwarzanie danych
# ---------------------------------------------------------------------------


def _mid_price(order_book: OrderBookSnapshot) -> float:
    if not order_book.bids or not order_book.asks:
        raise ValueError("Orderbook wymaga przynajmniej jednego bid i ask do wyliczenia mid price")
    best_bid = float(order_book.bids[0].price)
    best_ask = float(order_book.asks[0].price)
    if best_bid <= 0 or best_ask <= 0:
        raise ValueError("Orderbook zawiera niepoprawne ceny (<= 0)")
    return (best_bid + best_ask) / 2.0


def _average_depth(order_book: OrderBookSnapshot, depth_levels: int) -> float:
    depth = max(1, int(depth_levels))
    levels = list(order_book.bids[:depth]) + list(order_book.asks[:depth])
    if not levels:
        raise ValueError("Orderbook nie zawiera poziomów do obliczenia głębokości")
    notionals = [float(level.price) * float(level.quantity) for level in levels]
    return sum(notionals) / len(notionals)


def _spread_bps(order_book: OrderBookSnapshot) -> float:
    mid = _mid_price(order_book)
    best_bid = float(order_book.bids[0].price)
    best_ask = float(order_book.asks[0].price)
    spread = best_ask - best_bid
    return (spread / mid) * 10_000.0 if mid else 0.0


def _realized_volatility(candles: Sequence[OHLCVBar]) -> float:
    closes = [float(bar.close) for bar in candles if float(bar.close) > 0]
    if len(closes) < 2:
        return 0.0
    returns = []
    for previous, current in zip(closes[:-1], closes[1:]):
        if previous <= 0:
            continue
        returns.append(current / previous - 1.0)
    if len(returns) < 2:
        return 0.0
    volatility = pstdev(returns)
    return float(volatility * len(returns) ** 0.5 * 100.0)


def _checksum(baseline: MarketIntelBaseline) -> str:
    payload = json.dumps(baseline.to_mapping(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
#  Funkcje pomocnicze – SQL
# ---------------------------------------------------------------------------


def _validate_identifier(value: str, context: str) -> str:
    text = (value or "").strip()
    if not text.isidentifier():
        raise ValueError(f"{context} musi być poprawnym identyfikatorem SQL")
    return text


def _sql_columns(config: MarketIntelSqliteConfig) -> Mapping[str, str]:
    columns: dict[str, str] = {}
    columns[config.symbol_column] = "TEXT PRIMARY KEY"
    columns[config.mid_price_column] = "REAL NOT NULL"
    columns[config.depth_column] = "REAL NOT NULL"
    columns[config.spread_column] = "REAL NOT NULL"
    columns[config.funding_column] = "REAL NOT NULL"
    columns[config.sentiment_column] = "REAL NOT NULL"
    columns[config.volatility_column] = "REAL NOT NULL"
    if config.weight_column not in (None, ""):
        columns[str(config.weight_column)] = "REAL NOT NULL"
    return { _validate_identifier(key, "column"): value for key, value in columns.items() }


def _create_table_sql(table: str, columns: Mapping[str, str]) -> str:
    cols = ", ".join(f"{name} {definition}" for name, definition in columns.items())
    return f"CREATE TABLE IF NOT EXISTS {table} ({cols})"


def _insert_sql(table: str, columns: Mapping[str, str]) -> str:
    names = list(columns.keys())
    placeholders = ", ".join(["?"] * len(names))
    cols = ", ".join(names)
    return f"INSERT OR REPLACE INTO {table} ({cols}) VALUES ({placeholders})"


def _select_sql(table: str, columns: Mapping[str, str]) -> str:
    return f"SELECT {', '.join(columns.keys())} FROM {table}"


def _row_payload(baseline: MarketIntelBaseline, columns: Mapping[str, str]) -> tuple[object, ...]:
    mapping = baseline.to_mapping()
    return tuple(mapping.get(column) for column in columns.keys())


def _baseline_from_row(row: Mapping[str, object], config: MarketIntelConfig) -> MarketIntelBaseline:
    sqlite_cfg = config.sqlite
    assert sqlite_cfg is not None
    symbol = str(row[sqlite_cfg.symbol_column])
    weight = row.get(str(sqlite_cfg.weight_column))
    if weight is None:
        weight = config.default_weight
    return MarketIntelBaseline(
        symbol=symbol,
        mid_price=float(row[sqlite_cfg.mid_price_column]),
        avg_depth_usd=float(row[sqlite_cfg.depth_column]),
        avg_spread_bps=float(row[sqlite_cfg.spread_column]),
        funding_rate_bps=float(row[sqlite_cfg.funding_column]),
        sentiment_score=float(row[sqlite_cfg.sentiment_column]),
        realized_volatility=float(row[sqlite_cfg.volatility_column]),
        weight=float(weight),
    )


__all__ = [
    "MarketIntelSqliteBuilder",
    "MarketIntelDataProvider",
    "OrderBookLevel",
    "OrderBookSnapshot",
    "FundingSnapshot",
    "SentimentSnapshot",
    "OHLCVBar",
]

