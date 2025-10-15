"""Budowanie metryk Market Intelligence na potrzeby Stress Lab Stage6."""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

from bot_core.config.models import MarketIntelConfig, MarketIntelSqliteConfig

_ALLOWED_IDENTIFIER = re.compile(r"^[A-Za-z_][0-9A-Za-z_]*$")


@dataclass(slots=True, frozen=True)
class MarketIntelSourceInfo:
    """Opis źródła danych wykorzystanego do wygenerowania metryk."""

    type: str
    path: str
    table: str

    def to_mapping(self) -> Mapping[str, str]:
        return {
            "type": self.type,
            "path": self.path,
            "table": self.table,
        }


@dataclass(slots=True, frozen=True)
class MarketIntelBaseline:
    """Zestaw bazowych metryk rynku."""

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


class MarketIntelAggregator:
    """Buduje metryki Market Intelligence na podstawie konfiguracji Stage6."""

    def __init__(self, config: MarketIntelConfig):
        if not config.enabled:
            raise ValueError("Sekcja market_intel w konfiguracji jest wyłączona")
        if config.sqlite is None:
            raise ValueError("market_intel wymaga zdefiniowanego źródła sqlite")
        self._config = config
        self._sqlite_config = config.sqlite

    @staticmethod
    def _validate_identifier(value: str, *, context: str) -> str:
        text = str(value).strip()
        if not text or not _ALLOWED_IDENTIFIER.fullmatch(text):
            raise ValueError(f"{context} musi być poprawnym identyfikatorem SQL")
        return text

    def _read_rows(self) -> list[MarketIntelBaseline]:
        sqlite_cfg = self._sqlite_config
        assert sqlite_cfg is not None  # dla mypy
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

        query_columns = [columns[field] for field in (
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
                weight = float(db_row[7]) if weight_column is not None else self._config.default_weight
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
        rows = self._read_rows()
        required = set(symbol.upper() for symbol in self._config.required_symbols)
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
        baselines = self.build()
        timestamp = datetime.now(timezone.utc).isoformat()

        output_dir = Path(output_directory or self._config.output_directory)
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
                {
                    "symbol": baseline.symbol,
                    "path": str(target),
                    "weight": baseline.weight,
                }
            )
            written.append(target)

        resolved_manifest: Path
        if manifest_path is not None:
            resolved_manifest = Path(manifest_path)
        elif self._config.manifest_path is not None:
            resolved_manifest = Path(self._config.manifest_path)
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


__all__ = ["MarketIntelAggregator", "MarketIntelBaseline", "MarketIntelSourceInfo"]
