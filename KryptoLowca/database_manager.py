# -*- coding: utf-8 -*-
"""Warstwa zgodności dla menedżera bazy danych.

Nowa implementacja znajduje się w ``bot_core.database.manager``. Ten moduł
zapewnia przyjazne aliasy (``DBOptions``) oraz ujednolicone wyjątki, tak aby
starsze testy oraz skrypty mogły działać bez zmian.
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from bot_core.database import manager as _core
from bot_core.database.manager import DatabaseManager as _CoreDatabaseManager

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint, select
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Mapped, mapped_column

__all__ = [
    "DatabaseManager",
    "DBOptions",
    "DatabaseConnectionError",
    "MigrationError",
]


class DatabaseConnectionError(RuntimeError):
    """Błąd inicjalizacji lub połączenia z bazą danych."""


class MigrationError(RuntimeError):
    """Błąd podczas uruchamiania migracji schematu."""


@dataclass(slots=True)
class DBOptions:
    """Opcje tworzenia ``DatabaseManager`` w starszym API."""

    db_url: str = "sqlite+aiosqlite:///trading.db"
    timeout_s: float = 30.0
    echo: bool = False


class _UserConfigModel(_core.Base):
    """Tabela przechowująca presety użytkowników dla legacy API."""

    __tablename__ = "user_configs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    payload: Mapped[str] = mapped_column(Text, nullable=False)
    updated_at: Mapped[dt.datetime] = mapped_column(
        DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow, nullable=False
    )

    __table_args__ = (UniqueConstraint("user_id", "name", name="uq_user_configs_user_name"),)


class DatabaseManager(_CoreDatabaseManager):
    """Rozszerzenie nowego menedżera o pomocnicze metody fabryczne i legacy API."""

    @classmethod
    async def create(cls, options: Optional[DBOptions] = None) -> "DatabaseManager":
        opts = options or DBOptions()
        manager = cls(db_url=opts.db_url)
        try:
            await manager.init_db(create=True)
        except Exception as exc:  # pragma: no cover - propagacja do testów
            raise DatabaseConnectionError(str(exc)) from exc
        return manager

    async def run_migrations(self) -> None:
        try:
            await self.init_db(create=True)
        except Exception as exc:  # pragma: no cover - propagacja do testów
            raise MigrationError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Legacy API - konfiguracje użytkowników
    # ------------------------------------------------------------------
    async def save_user_config(self, user_id: int, name: str, config: Dict[str, Any]) -> None:
        if user_id <= 0:
            raise ValueError("user_id must be positive")
        preset = (name or "").strip()
        if not preset:
            raise ValueError("Preset name cannot be empty")
        if not isinstance(config, dict):
            raise ValueError("config must be a dict")

        payload = json.dumps(config, ensure_ascii=False, sort_keys=True)
        async with self.transaction() as session:
            stmt = select(_UserConfigModel).where(
                _UserConfigModel.user_id == user_id, _UserConfigModel.name == preset
            )
            existing = (await session.execute(stmt)).scalar_one_or_none()
            if existing:
                existing.payload = payload
            else:
                session.add(
                    _UserConfigModel(user_id=user_id, name=preset, payload=payload)
                )

    async def load_user_config(self, user_id: int, name: str) -> Dict[str, Any]:
        preset = (name or "").strip()
        if user_id <= 0 or not preset:
            raise ValueError("user_id and preset name are required")

        async with self.session() as session:
            stmt = select(_UserConfigModel).where(
                _UserConfigModel.user_id == user_id, _UserConfigModel.name == preset
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
        if row is None:
            raise KeyError(f"Preset '{preset}' for user {user_id} not found")
        return json.loads(row.payload)

    async def delete_user_config(self, user_id: int, name: str) -> None:
        preset = (name or "").strip()
        if user_id <= 0 or not preset:
            raise ValueError("user_id and preset name are required")
        async with self.transaction() as session:
            stmt = select(_UserConfigModel).where(
                _UserConfigModel.user_id == user_id, _UserConfigModel.name == preset
            )
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row:
                await session.delete(row)

    # ------------------------------------------------------------------
    # Legacy API - logi
    # ------------------------------------------------------------------
    async def get_logs(
        self,
        user_id: Optional[int] = None,
        level: Optional[str] = None,
        category: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        stmt = select(_core.LogEntry).order_by(_core.LogEntry.ts.desc()).limit(limit)
        if user_id is not None:
            stmt = stmt.where(_core.LogEntry.user_id == user_id)
        if level:
            stmt = stmt.where(_core.LogEntry.level == level.upper())
        if category:
            stmt = stmt.where(_core.LogEntry.category == category)

        async with self.session() as session:
            rows = (await session.execute(stmt)).scalars().all()

        result: List[Dict[str, Any]] = []
        for row in rows:
            payload = self._decode_extra(row.extra)
            result.append(
                {
                    "id": row.id,
                    "timestamp": row.ts.isoformat() if row.ts else None,
                    "level": row.level,
                    "message": row.message,
                    "category": row.category,
                    "context": payload.get("context") or payload,
                    "user_id": row.user_id,
                }
            )
        return result

    async def export_logs(self, rows: Iterable[Dict[str, Any]], fmt: str = "csv") -> str:
        entries = list(rows)
        if fmt.lower() == "json":
            return json.dumps(entries, ensure_ascii=False, indent=2)

        if fmt.lower() != "csv":
            raise ValueError("Supported formats: csv, json")

        import io
        import csv

        if not entries:
            headers = ["id", "timestamp", "level", "message", "context", "category", "user_id"]
        else:
            headers = list(entries[0].keys())
            if "message" in headers and "context" in headers:
                headers.remove("context")
                headers.insert(headers.index("message") + 1, "context")

        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=headers)
        writer.writeheader()
        for row in entries:
            data = dict(row)
            if isinstance(data.get("context"), (dict, list)):
                data["context"] = json.dumps(data["context"], ensure_ascii=False)
            writer.writerow(data)
        return buffer.getvalue()

    # ------------------------------------------------------------------
    # Legacy API - handel / statystyki
    # ------------------------------------------------------------------
    async def insert_trade(self, user_id: int, trade: Dict[str, Any]) -> int:
        if user_id <= 0:
            raise ValueError("user_id must be positive")
        if not isinstance(trade, dict):
            raise ValueError("trade must be a dict")

        normalised = self._normalise_trade_payload(user_id, trade)
        trade_id = await self.record_trade(normalised["record"])

        trade_ts = normalised.get("timestamp")
        if trade_ts is not None:
            async with self.transaction() as session:
                db_row = await session.get(_core.Trade, trade_id)
                if db_row is not None:
                    db_row.ts = trade_ts
        return trade_id

    async def batch_insert_trades(self, user_id: int, trades: Iterable[Dict[str, Any]]) -> None:
        for payload in trades:
            await self.insert_trade(user_id, payload)

    async def upsert_position(
        self,
        user_id: int,
        symbol: str,
        quantity: float,
        avg_entry: float,
        *,
        mode: str = "paper",
    ) -> int:
        if user_id <= 0:
            raise ValueError("user_id must be positive")
        symbol_norm = (symbol or "").upper()
        if not symbol_norm:
            raise ValueError("symbol is required")
        qty = float(quantity)
        if qty < 0:
            raise ValueError("quantity must be non-negative in legacy adapter")
        price = float(avg_entry)
        side = "FLAT"
        if qty > 0:
            side = "LONG"

        payload = {
            "symbol": symbol_norm,
            "side": side,
            "quantity": qty,
            "avg_price": price,
            "unrealized_pnl": 0.0,
            "mode": mode,
        }
        return await super().upsert_position(payload)

    async def feed_reporting_trades(
        self, user_id: int, symbol: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        if user_id <= 0:
            raise ValueError("user_id must be positive")
        rows = await self.fetch_trades(limit=10_000)
        feed: List[Dict[str, Any]] = []
        for row in rows:
            extra = self._decode_extra(row.get("extra"))
            if extra.get("user_id") != user_id:
                continue
            if symbol and row.get("symbol") != symbol:
                continue
            feed.append(
                {
                    "timestamp": row.get("ts"),
                    "symbol": row.get("symbol"),
                    "side": row.get("side"),
                    "qty": row.get("quantity"),
                    "price": row.get("price"),
                    "pnl": row.get("pnl"),
                    "commission": extra.get("commission", 0.0),
                    "slippage": extra.get("slippage", 0.0),
                }
            )
        return feed

    async def get_pnl_by_symbol(
        self,
        user_id: int,
        symbol: Optional[str] = None,
        since_ts: Optional[dt.datetime] = None,
        until_ts: Optional[dt.datetime] = None,
        group_by: str = "symbol",
    ) -> Dict[str, Any]:
        rows = await self.fetch_trades(limit=10_000)
        by_symbol: Dict[str, float] = {}
        by_day: Dict[str, Dict[str, float]] = {}

        for row in rows:
            extra = self._decode_extra(row.get("extra"))
            if extra.get("user_id") != user_id:
                continue
            sym = str(row.get("symbol") or "").upper()
            if symbol and sym != symbol:
                continue
            ts_raw = row.get("ts")
            ts = self._parse_timestamp(ts_raw)
            if since_ts and ts and ts < since_ts:
                continue
            if until_ts and ts and ts > until_ts:
                continue
            pnl = float(row.get("pnl") or 0.0)
            by_symbol[sym] = by_symbol.get(sym, 0.0) + pnl
            if ts is not None:
                day_key = ts.date().isoformat()
                bucket = by_day.setdefault(day_key, {})
                bucket[sym] = bucket.get(sym, 0.0) + pnl

        if group_by.lower() == "day":
            return by_day
        return by_symbol

    # ------------------------------------------------------------------
    # Legacy API - backup
    # ------------------------------------------------------------------
    async def backup_database(self, destination: Path | str) -> Path:
        sqlite_path = self._sqlite_path()
        if sqlite_path is None:
            raise RuntimeError("Backup is only supported for SQLite databases in legacy API")

        dest_path = Path(destination)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, sqlite_path, dest_path)
        return dest_path

    async def restore_database(self, source: Path | str) -> None:
        sqlite_path = self._sqlite_path()
        if sqlite_path is None:
            raise RuntimeError("Restore is only supported for SQLite databases in legacy API")
        src_path = Path(source)
        if not src_path.exists():
            raise FileNotFoundError(str(source))
        sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(shutil.copy2, src_path, sqlite_path)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _sqlite_path(self) -> Optional[Path]:
        url = make_url(self.db_url)
        if not url.database:
            return None
        if not url.get_backend_name().startswith("sqlite"):
            return None
        return Path(url.database).expanduser().resolve()

    @staticmethod
    def _decode_extra(extra: Any) -> Dict[str, Any]:
        if isinstance(extra, dict):
            return extra
        if isinstance(extra, str) and extra:
            try:
                return json.loads(extra)
            except json.JSONDecodeError:
                return {"raw": extra}
        return {}

    @staticmethod
    def _parse_timestamp(value: Any) -> Optional[dt.datetime]:
        if value is None:
            return None
        if isinstance(value, dt.datetime):
            return value
        if isinstance(value, str) and value:
            try:
                parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
                if parsed.tzinfo is not None:
                    parsed = parsed.astimezone(dt.timezone.utc).replace(tzinfo=None)
                return parsed
            except ValueError:
                return None
        return None

    def _normalise_trade_payload(self, user_id: int, trade: Dict[str, Any]) -> Dict[str, Any]:
        symbol = str(trade.get("symbol") or trade.get("pair") or "").upper()
        if not symbol:
            raise ValueError("Trade payload requires 'symbol'")
        side = str(trade.get("side") or trade.get("direction") or "").upper()
        if side not in {"BUY", "SELL"}:
            raise ValueError("Trade side must be BUY or SELL")

        qty = trade.get("qty", trade.get("quantity"))
        if qty is None:
            raise ValueError("Trade payload requires qty/quantity")
        quantity = float(qty)
        if quantity <= 0:
            raise ValueError("Trade quantity must be positive")

        entry_price = trade.get("entry")
        exit_price = trade.get("exit") or trade.get("price")
        price_source = exit_price if exit_price is not None else entry_price
        if price_source is None:
            raise ValueError("Trade payload requires price/entry/exit")
        price = float(price_source)

        mode = str(trade.get("mode") or "paper").lower()
        pnl = float(trade.get("pnl", 0.0))
        fee = float(trade.get("fee", trade.get("commission", 0.0)) or 0.0)

        timestamp = self._parse_timestamp(trade.get("ts"))

        extra = {
            "user_id": user_id,
            "entry": entry_price,
            "exit": exit_price,
            "commission": fee,
            "slippage": float(trade.get("slippage", 0.0) or 0.0),
            "raw": trade,
        }

        record = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "fee": fee,
            "pnl": pnl,
            "mode": mode,
            "extra": extra,
        }

        return {"record": record, "timestamp": timestamp, "extra": extra}
