"""Agregator Market Intelligence (Stage6) – warstwa zgodności.

Obsługiwane dwa tryby:

1) Tryb „cache” (wstecznie kompatybilny):
   - Konstruktor: MarketIntelAggregator(storage: CacheStorage, *, price_column=..., volume_column=..., time_column=...)
   - API: build_snapshot(MarketIntelQuery) -> MarketIntelSnapshot, build_many(iterable[MarketIntelQuery]) -> dict

2) Tryb „sqlite” (nowy Stage6):
   - Konstruktor: MarketIntelAggregator(config: MarketIntelConfig)
   - API: build() -> tuple[MarketIntelBaseline, ...], write_outputs(...): list[Path]
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from math import sqrt
from statistics import mean, pstdev
from pathlib import Path
from typing import Iterable, Mapping, MutableMapping, Sequence, Any

from bot_core.data.base import CacheStorage
from bot_core.config.models import MarketIntelConfig, MarketIntelSqliteConfig

# ----------------------------- Wspólne DTO / dataclasses -----------------------------

@dataclass(slots=True)
class MarketIntelQuery:
    """Zapytanie o znormalizowane metryki rynkowe (tryb 'cache')."""
    symbol: str
    interval: str
    lookback_bars: int


@dataclass(slots=True)
class MarketIntelSnapshot:
    """Zestaw metryk rynkowych (tryb 'cache')."""
    symbol: str
    interval: str
    start: datetime | None
    end: datetime | None
    bar_count: int
    price_change_pct: float | None
    volatility_pct: float | None
    max_drawdown_pct: float | None
    average_volume: float | None
    liquidity_usd: float | None
    momentum_score: float | None
    metadata: Mapping[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "symbol": self.symbol,
            "interval": self.interval,
            "bar_count": self.bar_count,
            "start": self.start.isoformat() if self.start else None,
            "end": self.end.isoformat() if self.end else None,
            "price_change_pct": self.price_change_pct,
            "volatility_pct": self.volatility_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "average_volume": self.average_volume,
            "liquidity_usd": self.liquidity_usd,
            "momentum_score": self.momentum_score,
        }
        if self.metadata:
            payload["metadata"] = dict(self.metadata)
        return payload


@dataclass(slots=True, frozen=True)
class MarketIntelSourceInfo:
    """Opis źródła danych wykorzystanego do wygenerowania metryk (tryb 'sqlite')."""
    type: str
    path: str
    table: str

    def to_mapping(self) -> Mapping[str, str]:
        return {"type": self.type, "path": self.path, "table": self.table}


@dataclass(slots=True, frozen=True)
class MarketIntelBaseline:
    """Bazowe metryki rynku (tryb 'sqlite')."""
    symbol: str
    mid_price: float
    avg_depth_usd: float
    avg_spread_bps: float
    funding_rate_bps: float
    sentiment_score: float
    realized_volatility: float
    weight: float

    def to_mapping(self) -> Mapping[str, float | str]:
        return {
            "symbol": self.symbol,
            "mid_price": self.mid_price,
            "avg_depth_usd": self.avg_depth_usd,
            "avg_spread_bps": self.avg_spread_bps,
            "funding_rate_bps": self.funding_rate_bps,
            "sentiment_score": self.sentiment_score,
            "realized_volatility": self.realized_volatility,
            "weight": self.weight,
        }


# --------------------------------- Implementacja ---------------------------------

_ALLOWED_IDENTIFIER = re.compile(r"^[A-Za-z_][0-9A-Za-z_]*$")


class MarketIntelAggregator:
    """Jedna klasa wspierająca dwa tryby pracy (cache/sqlite) dla zgodności gałęzi."""

    # --- konstruktor & tryb ---
    def __init__(
        self,
        storage_or_config: CacheStorage | MarketIntelConfig,
        *,
        price_column: str = "close",
        volume_column: str = "volume",
        time_column: str = "open_time",
    ) -> None:
        if isinstance(storage_or_config, CacheStorage):
            # Tryb CACHE (stary)
            self._mode = "cache"
            self._storage: CacheStorage = storage_or_config
            self._price_column = price_column
            self._volume_column = volume_column
            self._time_column = time_column
            self._sqlite_config = None
            self._config = None
        elif isinstance(storage_or_config, MarketIntelConfig):
            # Tryb SQLITE (nowy)
            self._mode = "sqlite"
            self._config: MarketIntelConfig = storage_or_config
            if not self._config.enabled:
                raise ValueError("Sekcja market_intel w konfiguracji jest wyłączona")
            if self._config.sqlite is None:
                raise ValueError("market_intel wymaga zdefiniowanego źródła sqlite")
            self._sqlite_config: MarketIntelSqliteConfig = self._config.sqlite
            self._storage = None
            # kolumny „cache” nie są używane w tym trybie
            self._price_column = price_column
            self._volume_column = volume_column
            self._time_column = time_column
        else:
            raise TypeError(
                "MarketIntelAggregator wymaga CacheStorage (tryb 'cache') lub MarketIntelConfig (tryb 'sqlite')"
            )

    # --- API trybu CACHE (wsteczna kompatybilność) ---
    def build_many(self, queries: Iterable[MarketIntelQuery]) -> dict[str, MarketIntelSnapshot]:
        self._require_mode("cache")
        results: dict[str, MarketIntelSnapshot] = {}
        for query in queries:
            results[query.symbol] = self.build_snapshot(query)
        return results

    def build_snapshot(self, query: MarketIntelQuery) -> MarketIntelSnapshot:
        self._require_mode("cache")
        key = f"{query.symbol}::{query.interval}"
        payload: Mapping[str, Any] = self._storage.read(key)  # type: ignore[union-attr]
        columns = tuple(payload.get("columns", ()))
        rows: Sequence[Sequence[float]] = payload.get("rows", ())

        if not rows:
            raise ValueError(f"Brak danych OHLCV dla {key}")

        price_index = self._column_index(columns, self._price_column)
        volume_index = self._column_index(columns, self._volume_column)
        time_index = self._column_index(columns, self._time_column)

        selected_rows = self._select_tail(rows, query.lookback_bars)
        closes = [float(row[price_index]) for row in selected_rows]
        volumes = [float(row[volume_index]) for row in selected_rows]
        timestamps = [float(row[time_index]) for row in selected_rows]

        bar_count = len(selected_rows)
        start_dt = self._timestamp_to_dt(timestamps[0]) if timestamps else None
        end_dt = self._timestamp_to_dt(timestamps[-1]) if timestamps else None

        price_change_pct = self._price_change(closes)
        volatility_pct = self._volatility(closes)
        max_drawdown_pct = self._max_drawdown(closes)
        average_volume = mean(volumes) if volumes else None
        liquidity_usd = self._liquidity_score(closes, volumes)
        momentum_score = self._momentum(closes)

        metadata: MutableMapping[str, float] = {}
        if bar_count:
            metadata["bars_used"] = float(bar_count)
        if price_change_pct is not None:
            metadata["price_change_abs"] = price_change_pct / 100.0
        if volatility_pct is not None:
            metadata["volatility_abs"] = volatility_pct / 100.0

        return MarketIntelSnapshot(
            symbol=query.symbol,
            interval=query.interval,
            start=start_dt,
            end=end_dt,
            bar_count=bar_count,
            price_change_pct=price_change_pct,
            volatility_pct=volatility_pct,
            max_drawdown_pct=max_drawdown_pct,
            average_volume=average_volume,
            liquidity_usd=liquidity_usd,
            momentum_score=momentum_score,
            metadata=metadata,
        )

    # --- API trybu SQLITE (nowy Stage6) ---
    @staticmethod
    def _validate_identifier(value: str, *, context: str) -> str:
        text = str(value).strip()
        if not text or not _ALLOWED_IDENTIFIER.fullmatch(text):
            raise ValueError(f"{context} musi być poprawnym identyfikatorem SQL")
        return text

    def _read_rows(self) -> list[MarketIntelBaseline]:
        assert self._mode == "sqlite" and self._sqlite_config is not None
        sqlite_cfg = self._sqlite_config
        db_path = Path(sqlite_cfg.path)
        if not db_path.exists():
            raise FileNotFoundError(f"Brak pliku bazy Market Intelligence: {db_path}")

        table = self._validate_identifier(sqlite_cfg.table, context="market_intel.sqlite.table")
        columns = {
            "symbol": self._validate_identifier(sqlite_cfg.symbol_column, context="market_intel.sqlite.symbol_column"),
            "mid_price": self._validate_identifier(sqlite_cfg.mid_price_column, context="market_intel.sqlite.mid_price_column"),
            "avg_depth_usd": self._validate_identifier(sqlite_cfg.depth_column, context="market_intel.sqlite.depth_column"),
            "avg_spread_bps": self._validate_identifier(sqlite_cfg.spread_column, context="market_intel.sqlite.spread_column"),
            "funding_rate_bps": self._validate_identifier(sqlite_cfg.funding_column, context="market_intel.sqlite.funding_column"),
            "sentiment_score": self._validate_identifier(sqlite_cfg.sentiment_column, context="market_intel.sqlite.sentiment_column"),
            "realized_volatility": self._validate_identifier(sqlite_cfg.volatility_column, context="market_intel.sqlite.volatility_column"),
        }
        weight_column: str | None = None
        if sqlite_cfg.weight_column not in (None, "", False):
            weight_column = self._validate_identifier(
                str(sqlite_cfg.weight_column), context="market_intel.sqlite.weight_column"
            )

        query_columns = [columns[k] for k in (
            "symbol",
            "mid_price",
            "avg_depth_usd",
            "avg_spread_bps",
            "funding_rate_bps",
            "sentiment_score",
            "realized_volatility",
        )]
        if weight_column is not None:
            query_columns.append(weight_column)

        sql = "SELECT " + ", ".join(query_columns) + f" FROM {table}"
        rows: list[MarketIntelBaseline] = []
        with sqlite3.connect(str(db_path)) as conn:
            cursor = conn.execute(sql)
            for db_row in cursor.fetchall():
                symbol = str(db_row[0]).strip()
                if not symbol:
                    continue
                mid_price = float(db_row[1])
                depth = float(db_row[2])
                spread = float(db_row[3])
                funding = float(db_row[4])
                sentiment = float(db_row[5])
                volatility = float(db_row[6])
                weight = float(db_row[7]) if weight_column is not None else (self._config.default_weight)  # type: ignore[union-attr]
                rows.append(
                    MarketIntelBaseline(
                        symbol=symbol,
                        mid_price=mid_price,
                        avg_depth_usd=depth,
                        avg_spread_bps=spread,
                        funding_rate_bps=funding,
                        sentiment_score=sentiment,
                        realized_volatility=volatility,
                        weight=max(0.0, weight),
                    )
                )
        if not rows:
            raise ValueError("Źródło market_intel nie zwróciło żadnych rekordów")
        return rows

    def build(self) -> tuple[MarketIntelBaseline, ...]:
        """Zwraca tuple bazowych metryk (tryb 'sqlite')."""
        self._require_mode("sqlite")
        rows = self._read_rows()
        required = {s.upper() for s in (self._config.required_symbols or [])}  # type: ignore[union-attr]
        available = {row.symbol.upper() for row in rows}
        missing = sorted(required - available)
        if missing:
            raise ValueError(f"Brakuje wymaganych symboli market_intel: {', '.join(missing)}")
        return tuple(rows)

    def write_outputs(
        self,
        *,
        output_directory: Path | None = None,
        manifest_path: Path | None = None,
    ) -> list[Path]:
        """Zapisuje JSON-y per-symbol + manifest (tryb 'sqlite')."""
        self._require_mode("sqlite")
        baselines = self.build()
        timestamp = datetime.now(timezone.utc).isoformat()

        output_dir = Path(output_directory or self._config.output_directory)  # type: ignore[union-attr]
        output_dir.mkdir(parents=True, exist_ok=True)

        manifest_entries: list[Mapping[str, object]] = []
        written: list[Path] = []
        source = MarketIntelSourceInfo(
            type="sqlite",
            path=str(Path(self._sqlite_config.path).resolve()),
            table=self._sqlite_config.table,
        )

        for baseline in baselines:
            payload = {
                "generated_at": timestamp,
                "baseline": dict(baseline.to_mapping()),
                "source": source.to_mapping(),
            }
            file_name = f"{baseline.symbol.lower()}.json"
            target = output_dir / file_name
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            manifest_entries.append(
                {"symbol": baseline.symbol, "path": str(target), "weight": baseline.weight}
            )
            written.append(target)

        if manifest_path is not None:
            resolved_manifest = Path(manifest_path)
        elif self._config.manifest_path is not None:  # type: ignore[union-attr]
            resolved_manifest = Path(self._config.manifest_path)  # type: ignore[union-attr]
        else:
            resolved_manifest = output_dir / "manifest.json"
        resolved_manifest.parent.mkdir(parents=True, exist_ok=True)
        manifest_payload = {
            "generated_at": timestamp,
            "source": source.to_mapping(),
            "count": len(manifest_entries),
            "entries": manifest_entries,
        }
        resolved_manifest.write_text(
            json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        written.append(resolved_manifest)
        return written

    # --- helpers (cache) ---
    @staticmethod
    def _select_tail(rows: Sequence[Sequence[float]], lookback_bars: int) -> Sequence[Sequence[float]]:
        if lookback_bars <= 0 or lookback_bars >= len(rows):
            return rows
        return rows[-lookback_bars:]

    @staticmethod
    def _column_index(columns: Sequence[str], name: str) -> int:
        try:
            return columns.index(name)
        except ValueError as exc:  # pragma: no cover
            raise ValueError(f"Brak kolumny '{name}' w danych OHLCV") from exc

    @staticmethod
    def _timestamp_to_dt(timestamp_ms: float) -> datetime:
        return datetime.fromtimestamp(float(timestamp_ms) / 1000.0, tz=timezone.utc)

    @staticmethod
    def _price_change(closes: Sequence[float]) -> float | None:
        if len(closes) < 2:
            return None
        start, end = closes[0], closes[-1]
        if start == 0:
            return None
        return (end / start - 1.0) * 100.0

    @staticmethod
    def _volatility(closes: Sequence[float]) -> float | None:
        if len(closes) < 3:
            return None
        returns: list[float] = []
        for i in range(1, len(closes)):
            prev, cur = closes[i - 1], closes[i]
            if prev <= 0:
                continue
            returns.append(cur / prev - 1.0)
        if len(returns) < 2:
            return None
        vol = pstdev(returns)
        scaled = vol * sqrt(len(returns))
        return scaled * 100.0

    @staticmethod
    def _max_drawdown(closes: Sequence[float]) -> float | None:
        if len(closes) < 2:
            return None
        peak = closes[0]
        max_dd = 0.0
        for price in closes:
            if price > peak:
                peak = price
                continue
            if peak <= 0:
                continue
            dd = (price - peak) / peak
            if dd < max_dd:
                max_dd = dd
        return abs(max_dd) * 100.0 if max_dd < 0 else 0.0

    @staticmethod
    def _liquidity_score(closes: Sequence[float], volumes: Sequence[float]) -> float | None:
        if not closes or not volumes or len(closes) != len(volumes):
            return None
        notional = [p * v for p, v in zip(closes, volumes)]
        return mean(notional) if notional else None

    @staticmethod
    def _momentum(closes: Sequence[float]) -> float | None:
        if len(closes) < 2:
            return None
        mid = len(closes) // 2
        first, second = closes[:mid], closes[mid:]
        if not first or not second:
            return None
        a, b = mean(first), mean(second)
        if a == 0:
            return None
        return (b / a - 1.0) * 100.0

    # --- guard ---
    def _require_mode(self, expected: str) -> None:
        if getattr(self, "_mode", None) != expected:
            raise RuntimeError(f"Ta metoda jest dostępna tylko w trybie '{expected}' (aktualnie: '{self._mode}')")


__all__ = [
    "MarketIntelAggregator",
    "MarketIntelQuery",
    "MarketIntelSnapshot",
    "MarketIntelBaseline",
    "MarketIntelSourceInfo",
]
