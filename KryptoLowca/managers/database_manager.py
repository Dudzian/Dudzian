# managers/database_manager.py
# -*- coding: utf-8 -*-
"""
Production-grade DatabaseManager dla bota tradingowego.

✅ Funkcje kluczowe:
- Async SQLAlchemy (SQLite + aiosqlite; łatwa zmiana na Postgres/MySQL).
- Tabele: orders, trades, positions, equity_curve, logs, schema_version.
- Transakcje (context manager), idempotencja po client_order_id, indeksy.
- Sync wrapper (łatwe użycie z GUI/innymi miejscami).
- Eksport CSV/JSON, proste backupy, walidacja Pydantic.
- Gotowe pod Paper Trading (te same tabele i metody co w trybie live).

Użycie (async):
    db = DatabaseManager("sqlite+aiosqlite:///trading.db")
    await db.init_db()
    await db.record_order(...)
    await db.record_trade(...)
    await db.upsert_position(...)

Użycie (sync):
    db = DatabaseManager("sqlite+aiosqlite:///trading.db")
    db.sync.init_db()
    db.sync.record_order(...)
    ...

Autor: Krok 1 – konsolidacja DB pod dalszy rozwój (paper trading, backtest, raporty).
"""
from __future__ import annotations

import asyncio
import contextlib
import csv
import datetime as dt
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Union

from pydantic import BaseModel, ValidationError, field_validator

from sqlalchemy import (
    Integer,
    String,
    Float,
    DateTime,
    Text,
    Index,
    UniqueConstraint,
    select,
    func,
    text,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


# --- Logowanie strukturalne ---
logger = logging.getLogger("database_manager")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        fmt='[%(asctime)s] %(levelname)s database_manager %(message)s'
    ))
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

# --- SQLAlchemy Base ---
class Base(DeclarativeBase):
    pass


# --- Modele tabel ---
class SchemaVersion(Base):
    __tablename__ = "schema_version"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    applied_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, nullable=False)


class EngineUser(Base):
    __tablename__ = "engine_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(254), unique=True, nullable=False, index=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, nullable=False)


class Order(Base):
    __tablename__ = "orders"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    symbol: Mapped[str] = mapped_column(String(50), index=True)
    side: Mapped[str] = mapped_column(String(4))  # BUY/SELL
    type: Mapped[str] = mapped_column(String(10))  # MARKET/LIMIT/STOP/STOP_LIMIT
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # dla LIMIT/STOP
    status: Mapped[str] = mapped_column(String(20), default="NEW", index=True)  # NEW/FILLED/PARTIALLY_FILLED/CANCELED
    exchange_order_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    client_order_id: Mapped[Optional[str]] = mapped_column(String(128), nullable=True, index=True)
    mode: Mapped[str] = mapped_column(String(10), default="live")  # live/paper
    extra: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON

    __table_args__ = (
        UniqueConstraint("client_order_id", name="uq_orders_client_order_id"),
        Index("ix_orders_symbol_ts", "symbol", "ts"),
    )


class Trade(Base):
    __tablename__ = "trades"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    symbol: Mapped[str] = mapped_column(String(50), index=True)
    side: Mapped[str] = mapped_column(String(4))  # BUY/SELL
    quantity: Mapped[float] = mapped_column(Float)
    price: Mapped[float] = mapped_column(Float)
    fee: Mapped[float] = mapped_column(Float, default=0.0)
    order_id: Mapped[Optional[int]] = mapped_column(Integer, index=True)  # FK logiczna (bez constraintu dla prostoty)
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)  # Realized PnL (jeśli dotyczy)
    mode: Mapped[str] = mapped_column(String(10), default="live")
    extra: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_trades_symbol_ts", "symbol", "ts"),
    )


class Position(Base):
    __tablename__ = "positions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)
    symbol: Mapped[str] = mapped_column(String(50), index=True, unique=True)
    side: Mapped[str] = mapped_column(String(5))  # LONG/SHORT/FLAT
    quantity: Mapped[float] = mapped_column(Float)
    avg_price: Mapped[float] = mapped_column(Float)
    unrealized_pnl: Mapped[float] = mapped_column(Float, default=0.0)
    mode: Mapped[str] = mapped_column(String(10), default="live")

    __table_args__ = (
        Index("ix_positions_symbol_mode", "symbol", "mode"),
    )


class EquityCurve(Base):
    __tablename__ = "equity_curve"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    equity: Mapped[float] = mapped_column(Float)    # wartość portfela
    balance: Mapped[float] = mapped_column(Float)   # wolne środki
    pnl: Mapped[float] = mapped_column(Float)       # dzienny/okresowy PnL
    mode: Mapped[str] = mapped_column(String(10), default="live")


class LogEntry(Base):
    __tablename__ = "logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    level: Mapped[str] = mapped_column(String(10), index=True)  # INFO/WARN/ERROR
    source: Mapped[str] = mapped_column(String(50), index=True) # trading_engine/strategy/exchange/...
    message: Mapped[str] = mapped_column(Text)
    extra: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    category: Mapped[Optional[str]] = mapped_column(String(50), nullable=True, index=True)

    __table_args__ = (
        Index("ix_logs_source_ts", "source", "ts"),
        Index("ix_logs_level_ts", "level", "ts"),
        Index("ix_logs_user_ts", "user_id", "ts"),
    )


class PerformanceMetric(Base):
    __tablename__ = "performance_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    metric: Mapped[str] = mapped_column(String(64), index=True)
    value: Mapped[float] = mapped_column(Float)
    window: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    symbol: Mapped[Optional[str]] = mapped_column(String(50), index=True, nullable=True)
    mode: Mapped[str] = mapped_column(String(10), default="live", index=True)
    extra: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_performance_metric_metric_ts", "metric", "ts"),
        Index("ix_performance_metric_symbol_ts", "symbol", "ts"),
    )


class RiskLimitSnapshot(Base):
    __tablename__ = "risk_limits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    symbol: Mapped[str] = mapped_column(String(50), index=True)
    max_fraction: Mapped[float] = mapped_column(Float)
    recommended_size: Mapped[float] = mapped_column(Float)
    mode: Mapped[str] = mapped_column(String(10), default="live", index=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_risk_limits_symbol_ts", "symbol", "ts"),
        Index("ix_risk_limits_mode_ts", "mode", "ts"),
    )


class ApiRateLimitSnapshot(Base):
    __tablename__ = "api_rate_limits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    bucket_name: Mapped[str] = mapped_column(String(64), index=True)
    window_seconds: Mapped[float] = mapped_column(Float)
    capacity: Mapped[int] = mapped_column(Integer)
    count: Mapped[int] = mapped_column(Integer)
    usage: Mapped[float] = mapped_column(Float)
    max_usage: Mapped[float] = mapped_column(Float)
    reset_in_seconds: Mapped[float] = mapped_column(Float)
    mode: Mapped[str] = mapped_column(String(10), default="live", index=True)
    endpoint: Mapped[Optional[str]] = mapped_column(String(120), nullable=True, index=True)
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    __table_args__ = (
        Index("ix_api_rate_limits_bucket_ts", "bucket_name", "ts"),
        Index("ix_api_rate_limits_endpoint_ts", "endpoint", "ts"),
    )


class SecurityAuditLog(Base):
    __tablename__ = "security_audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    action: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), index=True)
    detail: Mapped[str] = mapped_column(Text)
    actor: Mapped[Optional[str]] = mapped_column(String(120), nullable=True, index=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True, index=True)
    metadata_json: Mapped[Optional[str]] = mapped_column("metadata", Text, nullable=True)

    __table_args__ = (
        Index("ix_security_audit_action_ts", "action", "ts"),
        Index("ix_security_audit_user_ts", "user_id", "ts"),
    )


# --- Pydantic modele wejściowe (walidacja) ---
class OrderIn(BaseModel):
    symbol: str
    side: str  # BUY/SELL
    type: str  # MARKET/LIMIT/STOP/STOP_LIMIT
    quantity: float
    price: float | None = None
    status: str = "NEW"
    exchange_order_id: str | None = None
    client_order_id: str | None = None
    mode: str = "live"
    extra: dict | None = None

    @field_validator("side")
    @classmethod
    def _side(cls, v: str) -> str:
        v = v.upper()
        if v not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        return v

    @field_validator("type")
    @classmethod
    def _type(cls, v: str) -> str:
        if v.upper() not in {"MARKET", "LIMIT", "STOP", "STOP_LIMIT"}:
            raise ValueError("type must be MARKET/LIMIT/STOP/STOP_LIMIT")
        return v.upper()

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        v = v.lower()
        if v not in {"live", "paper"}:
            raise ValueError("mode must be 'live' or 'paper'")
        return v


class TradeIn(BaseModel):
    symbol: str
    side: str  # BUY/SELL
    quantity: float
    price: float
    fee: float = 0.0
    order_id: int | None = None
    pnl: float | None = None
    mode: str = "live"
    extra: dict | None = None

    @field_validator("side")
    @classmethod
    def _side(cls, v: str) -> str:
        v = v.upper()
        if v not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        return v

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        v = v.lower()
        if v not in {"live", "paper"}:
            raise ValueError("mode must be 'live' or 'paper'")
        return v


class PositionIn(BaseModel):
    symbol: str
    side: str  # LONG/SHORT/FLAT
    quantity: float
    avg_price: float
    unrealized_pnl: float = 0.0
    mode: str = "live"

    @field_validator("side")
    @classmethod
    def _side(cls, v: str) -> str:
        v = v.upper()
        if v not in {"LONG", "SHORT", "FLAT"}:
            raise ValueError("side must be LONG/SHORT/FLAT")
        return v

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        v = v.lower()
        if v not in {"live", "paper"}:
            raise ValueError("mode must be 'live' or 'paper'")
        return v


class EquityIn(BaseModel):
    equity: float
    balance: float
    pnl: float
    mode: str = "live"

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        v = v.lower()
        if v not in {"live", "paper"}:
            raise ValueError("mode must be 'live' or 'paper'")
        return v


class PerformanceMetricIn(BaseModel):
    metric: str
    value: float
    window: int | None = None
    symbol: str | None = None
    mode: str = "live"
    context: Dict[str, Any] | None = None

    @field_validator("metric")
    @classmethod
    def _metric(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("metric is required")
        return v

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        v = v.lower()
        if v not in {"live", "paper"}:
            raise ValueError("mode must be 'live' or 'paper'")
        return v


class RiskLimitIn(BaseModel):
    symbol: str
    max_fraction: float
    recommended_size: float
    mode: str = "live"
    details: Dict[str, Any] | None = None

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, v: str) -> str:
        v = (v or "").strip().upper()
        if not v:
            raise ValueError("symbol is required")
        return v

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        v = v.lower()
        if v not in {"live", "paper"}:
            raise ValueError("mode must be 'live' or 'paper'")
        return v


class RateLimitSnapshotIn(BaseModel):
    bucket_name: str
    window_seconds: float
    capacity: int
    count: int
    usage: float
    max_usage: float
    reset_in_seconds: float
    mode: str = "live"
    endpoint: str | None = None
    context: Dict[str, Any] | None = None

    @field_validator("bucket_name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("bucket_name is required")
        return v

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        v = v.lower()
        if v not in {"live", "paper"}:
            raise ValueError("mode must be 'live' or 'paper'")
        return v


class SecurityAuditEventIn(BaseModel):
    action: str
    status: str
    detail: str
    actor: str | None = None
    user_id: int | None = None
    metadata: Dict[str, Any] | None = None

    @field_validator("action")
    @classmethod
    def _action(cls, v: str) -> str:
        v = (v or "").strip()
        if not v:
            raise ValueError("action is required")
        return v

    @field_validator("status")
    @classmethod
    def _status(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in {"ok", "error", "denied"}:
            raise ValueError("status must be ok/error/denied")
        return v


# --- Database Manager ---
@dataclass
class _EngineState:
    engine: Optional[AsyncEngine] = None
    session_factory: Optional[sessionmaker] = None


class DatabaseManager:
    """
    Główna klasa obsługująca bazę danych (async) + sync wrapper.
    """

    def __init__(self, db_url: str = "sqlite+aiosqlite:///trading.db") -> None:
        self.db_url = db_url
        self._state = _EngineState()
        self._lock = asyncio.Lock()

    # ---------- Inicjalizacja ----------
    async def init_db(self, *, create: bool = True) -> None:
        """
        Tworzy silnik, sesję i (opcjonalnie) schemat bazy, jeśli nie istnieje.
        DDL (create_all) wykonujemy na połączeniu engine, NIE na sesji.
        """
        async with self._lock:
            if self._state.engine is None:
                self._state.engine = create_async_engine(self.db_url, future=True)
                self._state.session_factory = sessionmaker(
                    bind=self._state.engine,
                    expire_on_commit=False,
                    class_=AsyncSession,
                    autoflush=False,
                    autocommit=False,
                )

        if create:
            # 1) DDL na połączeniu
            async with self._state.engine.begin() as conn:  # type: ignore[union-attr]
                await conn.run_sync(Base.metadata.create_all)

        await self._apply_migrations()

        if create:
            logger.info("Database initialized (url=%s)", self.db_url)

    async def _apply_migrations(self) -> None:
        if self._state.session_factory is None:
            return

        async with self.session() as session:
            existing = (
                await session.execute(select(SchemaVersion.version))
            ).scalars().all()
            applied = set(int(v) for v in existing)
            current = max(applied) if applied else 0
            target = int(CURRENT_SCHEMA_VERSION)
            updated = False

            for version in range(current + 1, target + 1):
                migration = MIGRATIONS.get(version)
                logger.info("Applying database migration -> version %s", version)
                if migration is not None:
                    await migration(session, self)
                session.add(SchemaVersion(version=version))
                await session.flush()
                updated = True

            if updated:
                await session.commit()

    async def get_schema_version(self) -> int:
        if self._state.session_factory is None:
            return 0

        async with self.session() as session:
            result = await session.execute(select(func.max(SchemaVersion.version)))
            version = result.scalar()
            return int(version or 0)

    @contextlib.asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        if self._state.session_factory is None:
            raise RuntimeError("Call init_db() first.")
        session: AsyncSession = self._state.session_factory()
        try:
            yield session
        finally:
            await session.close()

    @contextlib.asynccontextmanager
    async def transaction(self) -> AsyncIterator[AsyncSession]:
        """
        Context manager transakcyjny:
            async with db.transaction() as s:
                ... operacje ...
        """
        async with self.session() as s:
            try:
                await s.begin()
                yield s
                await s.commit()
            except Exception:
                await s.rollback()
                logger.exception("Transaction rollback due to exception")
                raise

    # ---------- USERS / LOGGING BRIDGE ----------
    async def ensure_user(self, email: str) -> int:
        email_norm = (email or "").strip().lower()
        if not email_norm:
            raise ValueError("email is required")

        async with self.transaction() as s:
            q = await s.execute(select(EngineUser).where(EngineUser.email == email_norm))
            existing = q.scalar_one_or_none()
            if existing:
                return existing.id

            rec = EngineUser(email=email_norm)
            s.add(rec)
            await s.flush()
            return rec.id

    async def log(
        self,
        user_id: Optional[int],
        level: str,
        message: str,
        *,
        category: str = "general",
        context: Optional[Dict[str, Any]] = None,
    ) -> int:
        if not level:
            raise ValueError("level is required")
        if not message:
            raise ValueError("message is required")

        payload = {
            "user_id": user_id,
            "category": category,
            "context": context or {},
        }
        async with self.transaction() as s:
            rec = LogEntry(
                user_id=user_id,
                level=level.upper(),
                source=category,
                category=category,
                message=message,
                extra=json.dumps(payload),
            )
            s.add(rec)
            await s.flush()
            return rec.id

    async def get_positions(self, user_id: Optional[int], *, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        """Compatibility helper used by TradingEngine/GUI."""
        rows = await self.get_open_positions(mode=mode)
        out: List[Dict[str, Any]] = []
        for row in rows:
            out.append({
                "symbol": row.get("symbol"),
                "qty": row.get("quantity"),
                "avg_entry": row.get("avg_price"),
                "side": row.get("side"),
                "mode": row.get("mode"),
            })
        return out

    # ---------- OPERACJE: Orders ----------
    async def record_order(self, order: Union[OrderIn, Dict[str, Any]]) -> int:
        """
        Zapisuje zlecenie (idempotencja po client_order_id).
        Zwraca ID rekordu.
        """
        try:
            o = order if isinstance(order, OrderIn) else OrderIn(**order)
        except ValidationError as e:
            logger.error("Order validation error: %s", e)
            raise

        async with self.transaction() as s:
            if o.client_order_id:
                # idempotencja
                q = await s.execute(select(Order).where(Order.client_order_id == o.client_order_id))
                existing = q.scalar_one_or_none()
                if existing:
                    # ewentualna aktualizacja statusu/price itp.
                    existing.status = o.status
                    if o.price is not None:
                        existing.price = o.price
                    if o.exchange_order_id:
                        existing.exchange_order_id = o.exchange_order_id
                    if o.extra is not None:
                        existing.extra = json.dumps(o.extra)
                    await s.flush()
                    return existing.id

            rec = Order(
                symbol=o.symbol,
                side=o.side,
                type=o.type,
                quantity=o.quantity,
                price=o.price,
                status=o.status,
                exchange_order_id=o.exchange_order_id,
                client_order_id=o.client_order_id,
                mode=o.mode,
                extra=json.dumps(o.extra) if o.extra is not None else None,
            )
            s.add(rec)
            await s.flush()
            return rec.id

    async def update_order_status(
        self,
        *,
        order_id: Optional[int] = None,
        client_order_id: Optional[str] = None,
        status: str,
        price: Optional[float] = None,
        exchange_order_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not order_id and not client_order_id:
            raise ValueError("Provide order_id or client_order_id.")

        async with self.transaction() as s:
            stmt = select(Order)
            if order_id:
                stmt = stmt.where(Order.id == order_id)
            else:
                stmt = stmt.where(Order.client_order_id == client_order_id)
            q = await s.execute(stmt)
            rec = q.scalar_one_or_none()
            if not rec:
                raise ValueError("Order not found.")

            rec.status = status
            if price is not None:
                rec.price = price
            if exchange_order_id:
                rec.exchange_order_id = exchange_order_id
            if extra is not None:
                rec.extra = json.dumps(extra)
            await s.flush()

    # ---------- OPERACJE: Trades ----------
    async def record_trade(self, trade: Union[TradeIn, Dict[str, Any]]) -> int:
        """
        Zapisuje trade. Zwraca ID.
        """
        try:
            t = trade if isinstance(trade, TradeIn) else TradeIn(**trade)
        except ValidationError as e:
            logger.error("Trade validation error: %s", e)
            raise

        async with self.transaction() as s:
            rec = Trade(
                symbol=t.symbol,
                side=t.side,
                quantity=t.quantity,
                price=t.price,
                fee=t.fee,
                order_id=t.order_id,
                pnl=t.pnl,
                mode=t.mode,
                extra=json.dumps(t.extra) if t.extra is not None else None,
            )
            s.add(rec)
            await s.flush()
            return rec.id

    async def fetch_trades(
        self,
        *,
        symbol: Optional[str] = None,
        mode: Optional[str] = None,
        limit: int = 1000,
        since: Optional[dt.datetime] = None,
    ) -> List[Dict[str, Any]]:
        async with self.session() as s:
            stmt = select(Trade).order_by(Trade.ts.desc()).limit(limit)
            if symbol:
                stmt = stmt.where(Trade.symbol == symbol)
            if mode:
                stmt = stmt.where(Trade.mode == mode)
            if since:
                stmt = stmt.where(Trade.ts >= since)
            rows = (await s.execute(stmt)).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    # ---------- OPERACJE: Positions ----------
    async def upsert_position(self, pos: Union[PositionIn, Dict[str, Any]]) -> int:
        """
        Wstawia/aktualizuje po symbolu (unique), zwraca id.
        """
        try:
            p = pos if isinstance(pos, PositionIn) else PositionIn(**pos)
        except ValidationError as e:
            logger.error("Position validation error: %s", e)
            raise

        async with self.transaction() as s:
            q = await s.execute(select(Position).where(Position.symbol == p.symbol))
            existing = q.scalar_one_or_none()
            if existing:
                existing.side = p.side
                existing.quantity = p.quantity
                existing.avg_price = p.avg_price
                existing.unrealized_pnl = p.unrealized_pnl
                existing.mode = p.mode
                await s.flush()
                return existing.id

            rec = Position(
                symbol=p.symbol,
                side=p.side,
                quantity=p.quantity,
                avg_price=p.avg_price,
                unrealized_pnl=p.unrealized_pnl,
                mode=p.mode,
            )
            s.add(rec)
            await s.flush()
            return rec.id

    async def close_position(self, symbol: str) -> None:
        async with self.transaction() as s:
            q = await s.execute(select(Position).where(Position.symbol == symbol))
            existing = q.scalar_one_or_none()
            if existing:
                await s.delete(existing)

    async def get_open_positions(self, *, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        async with self.session() as s:
            stmt = select(Position)
            if mode:
                stmt = stmt.where(Position.mode == mode)
            rows = (await s.execute(stmt)).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    # ---------- OPERACJE: Equity ----------
    async def log_equity(self, equity: Union[EquityIn, Dict[str, Any]]) -> int:
        try:
            e = equity if isinstance(equity, EquityIn) else EquityIn(**equity)
        except ValidationError as ex:
            logger.error("Equity validation error: %s", ex)
            raise

        async with self.transaction() as s:
            rec = EquityCurve(
                equity=e.equity,
                balance=e.balance,
                pnl=e.pnl,
                mode=e.mode,
            )
            s.add(rec)
            await s.flush()
            return rec.id

    async def fetch_equity_curve(self, *, limit: int = 1000, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        async with self.session() as s:
            stmt = select(EquityCurve).order_by(EquityCurve.ts.asc()).limit(limit)
            if mode:
                stmt = stmt.where(EquityCurve.mode == mode)
            rows = (await s.execute(stmt)).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    # ---------- OPERACJE: Performance metrics ----------
    async def log_performance_metric(
        self, metric: Union[PerformanceMetricIn, Dict[str, Any]]
    ) -> int:
        try:
            payload = metric if isinstance(metric, PerformanceMetricIn) else PerformanceMetricIn(**metric)
        except ValidationError as exc:
            logger.error("Performance metric validation error: %s", exc)
            raise

        async with self.transaction() as session:
            rec = PerformanceMetric(
                metric=payload.metric,
                value=float(payload.value),
                window=payload.window,
                symbol=payload.symbol,
                mode=payload.mode,
                extra=json.dumps(payload.context) if payload.context is not None else None,
            )
            session.add(rec)
            await session.flush()
            return rec.id

    async def fetch_performance_metrics(
        self,
        *,
        metric: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        async with self.session() as session:
            stmt = select(PerformanceMetric).order_by(PerformanceMetric.ts.desc()).limit(limit)
            if metric:
                stmt = stmt.where(PerformanceMetric.metric == metric)
            if symbol:
                stmt = stmt.where(PerformanceMetric.symbol == symbol)
            rows = (await session.execute(stmt)).scalars().all()
            return [self._row_to_dict(row) for row in rows]

    # ---------- OPERACJE: Risk limits ----------
    async def log_risk_limit(
        self, snapshot: Union[RiskLimitIn, Dict[str, Any]]
    ) -> int:
        try:
            payload = snapshot if isinstance(snapshot, RiskLimitIn) else RiskLimitIn(**snapshot)
        except ValidationError as exc:
            logger.error("Risk limit validation error: %s", exc)
            raise

        async with self.transaction() as session:
            rec = RiskLimitSnapshot(
                symbol=payload.symbol,
                max_fraction=float(payload.max_fraction),
                recommended_size=float(payload.recommended_size),
                mode=payload.mode,
                details=json.dumps(payload.details) if payload.details is not None else None,
            )
            session.add(rec)
            await session.flush()
            return rec.id

    async def fetch_risk_limits(
        self,
        *,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        async with self.session() as session:
            stmt = select(RiskLimitSnapshot).order_by(RiskLimitSnapshot.ts.desc()).limit(limit)
            if symbol:
                stmt = stmt.where(RiskLimitSnapshot.symbol == symbol)
            rows = (await session.execute(stmt)).scalars().all()
            return [self._row_to_dict(row) for row in rows]

    # ---------- OPERACJE: API rate limits ----------
    async def log_rate_limit_snapshot(
        self,
        snapshot: Union[RateLimitSnapshotIn, Dict[str, Any]],
    ) -> int:
        try:
            payload = (
                snapshot
                if isinstance(snapshot, RateLimitSnapshotIn)
                else RateLimitSnapshotIn(**snapshot)
            )
        except ValidationError as exc:
            logger.error("Rate limit snapshot validation error: %s", exc)
            raise

        async with self.transaction() as session:
            rec = ApiRateLimitSnapshot(
                bucket_name=payload.bucket_name,
                window_seconds=float(payload.window_seconds),
                capacity=int(payload.capacity),
                count=int(payload.count),
                usage=float(payload.usage),
                max_usage=float(payload.max_usage),
                reset_in_seconds=float(payload.reset_in_seconds),
                mode=payload.mode,
                endpoint=payload.endpoint,
                context=json.dumps(payload.context) if payload.context is not None else None,
            )
            session.add(rec)
            await session.flush()
            return rec.id

    async def fetch_rate_limit_snapshots(
        self,
        *,
        bucket: Optional[str] = None,
        endpoint: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        async with self.session() as session:
            stmt = select(ApiRateLimitSnapshot).order_by(ApiRateLimitSnapshot.ts.desc()).limit(limit)
            if bucket:
                stmt = stmt.where(ApiRateLimitSnapshot.bucket_name == bucket)
            if endpoint:
                stmt = stmt.where(ApiRateLimitSnapshot.endpoint == endpoint)
            rows = (await session.execute(stmt)).scalars().all()
            return [self._row_to_dict(row) for row in rows]

    # ---------- OPERACJE: Security audit ----------
    async def log_security_audit(
        self,
        event: Union[SecurityAuditEventIn, Dict[str, Any]],
    ) -> int:
        try:
            payload = (
                event if isinstance(event, SecurityAuditEventIn) else SecurityAuditEventIn(**event)
            )
        except ValidationError as exc:
            logger.error("Security audit validation error: %s", exc)
            raise

        async with self.transaction() as session:
            rec = SecurityAuditLog(
                action=payload.action,
                status=payload.status,
                detail=payload.detail,
                actor=payload.actor,
                user_id=payload.user_id,
                metadata_json=json.dumps(payload.metadata) if payload.metadata is not None else None,
            )
            session.add(rec)
            await session.flush()
            return rec.id

    async def fetch_security_audit(
        self,
        *,
        action: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        async with self.session() as session:
            stmt = select(SecurityAuditLog).order_by(SecurityAuditLog.ts.desc()).limit(limit)
            if action:
                stmt = stmt.where(SecurityAuditLog.action == action)
            if status:
                stmt = stmt.where(SecurityAuditLog.status == status)
            rows = (await session.execute(stmt)).scalars().all()
            return [self._row_to_dict(row) for row in rows]

    # ---------- OPERACJE: Logi ----------
    async def add_log(self, *, level: str, source: str, message: str, extra: Optional[Dict[str, Any]] = None) -> int:
        payload = extra or {}
        async with self.transaction() as s:
            rec = LogEntry(
                level=level.upper(),
                source=source,
                category=payload.get("category", source),
                user_id=payload.get("user_id"),
                message=message,
                extra=json.dumps(payload) if payload else None,
            )
            s.add(rec)
            await s.flush()
            return rec.id

    async def fetch_logs(
        self, *, level: Optional[str] = None, source: Optional[str] = None, limit: int = 1000
    ) -> List[Dict[str, Any]]:
        async with self.session() as s:
            stmt = select(LogEntry).order_by(LogEntry.ts.desc()).limit(limit)
            if level:
                stmt = stmt.where(LogEntry.level == level.upper())
            if source:
                stmt = stmt.where(LogEntry.source == source)
            rows = (await s.execute(stmt)).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    # ---------- Eksport / Backup ----------
    async def export_trades_csv(self, *, path: Union[str, Path]) -> Path:
        rows = await self.fetch_trades(limit=10_000)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            if rows:
                headers = list(rows[0].keys())
            else:
                headers = ["id", "ts", "symbol", "side", "quantity", "price", "fee", "order_id", "pnl", "mode", "extra"]
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        return p

    async def export_table_json(self, *, table: str, path: Union[str, Path]) -> Path:
        mapper = {
            "orders": Order,
            "trades": Trade,
            "positions": Position,
            "equity_curve": EquityCurve,
            "logs": LogEntry,
        }
        model = mapper.get(table)
        if model is None:
            raise ValueError("Unknown table name")

        async with self.session() as s:
            rows = (await s.execute(select(model))).scalars().all()
            data = [self._row_to_dict(r) for r in rows]

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return p

    # ---------- Narzędzia ----------
    @staticmethod
    def _row_to_dict(obj: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for c in obj.__table__.columns:  # type: ignore[attr-defined]
            attr_name = c.key
            col_name = c.name
            val = getattr(obj, attr_name)
            if isinstance(val, dt.datetime):
                out[attr_name] = val.isoformat()
                continue

            if isinstance(val, str) and val:
                # dekodujemy popularne pola JSON
                json_fields = {
                    "extra",
                    "details",
                    "context",
                    "metadata_json",
                }
                target_name = attr_name
                if attr_name.endswith("_json"):
                    target_name = attr_name[:-5]
                    json_fields.add(attr_name)
                if attr_name in json_fields or target_name in {"extra", "details", "context", "metadata"}:
                    try:
                        out[target_name] = json.loads(val)
                        continue
                    except json.JSONDecodeError:
                        logger.warning("Nie udało się zdekodować JSON z kolumny %s", col_name)
            out[attr_name] = val
        return out

    # ---------- Sync wrapper ----------
    class _SyncWrapper:
        def __init__(self, outer: "DatabaseManager") -> None:
            self._outer = outer

        def _run(self, coro):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                # W razie wywołania z kontekstu async — użytkownik powinien użyć metod async.
                raise RuntimeError("Use async methods inside running event loop.")
            return asyncio.run(coro)

        def init_db(self, *, create: bool = True) -> None:
            return self._run(self._outer.init_db(create=create))

        def ensure_user(self, email: str) -> int:
            return self._run(self._outer.ensure_user(email))

        def log(
            self,
            user_id: Optional[int],
            level: str,
            message: str,
            *,
            category: str = "general",
            context: Optional[Dict[str, Any]] = None,
        ) -> int:
            return self._run(self._outer.log(user_id, level, message, category=category, context=context))

        def record_order(self, order: Union[OrderIn, Dict[str, Any]]) -> int:
            return self._run(self._outer.record_order(order))

        def update_order_status(
            self,
            *,
            order_id: Optional[int] = None,
            client_order_id: Optional[str] = None,
            status: str,
            price: Optional[float] = None,
            exchange_order_id: Optional[str] = None,
            extra: Optional[Dict[str, Any]] = None,
        ) -> None:
            return self._run(self._outer.update_order_status(
                order_id=order_id,
                client_order_id=client_order_id,
                status=status,
                price=price,
                exchange_order_id=exchange_order_id,
                extra=extra,
            ))

        def record_trade(self, trade: Union[TradeIn, Dict[str, Any]]) -> int:
            return self._run(self._outer.record_trade(trade))

        def fetch_trades(
            self,
            *,
            symbol: Optional[str] = None,
            mode: Optional[str] = None,
            limit: int = 1000,
            since: Optional[dt.datetime] = None,
        ) -> List[Dict[str, Any]]:
            return self._run(self._outer.fetch_trades(symbol=symbol, mode=mode, limit=limit, since=since))

        def upsert_position(self, pos: Union[PositionIn, Dict[str, Any]]) -> int:
            return self._run(self._outer.upsert_position(pos))

        def close_position(self, symbol: str) -> None:
            return self._run(self._outer.close_position(symbol))

        def get_open_positions(self, *, mode: Optional[str] = None) -> List[Dict[str, Any]]:
            return self._run(self._outer.get_open_positions(mode=mode))

        def get_positions(self, user_id: Optional[int], *, mode: Optional[str] = None) -> List[Dict[str, Any]]:
            return self._run(self._outer.get_positions(user_id, mode=mode))

        def log_equity(self, equity: Union[EquityIn, Dict[str, Any]]) -> int:
            return self._run(self._outer.log_equity(equity))

        def fetch_equity_curve(self, *, limit: int = 1000, mode: Optional[str] = None) -> List[Dict[str, Any]]:
            return self._run(self._outer.fetch_equity_curve(limit=limit, mode=mode))

        def log_performance_metric(self, metric: Union[PerformanceMetricIn, Dict[str, Any]]) -> int:
            return self._run(self._outer.log_performance_metric(metric))

        def fetch_performance_metrics(
            self,
            *,
            metric: Optional[str] = None,
            symbol: Optional[str] = None,
            limit: int = 100,
        ) -> List[Dict[str, Any]]:
            return self._run(
                self._outer.fetch_performance_metrics(metric=metric, symbol=symbol, limit=limit)
            )

        def log_risk_limit(self, snapshot: Union[RiskLimitIn, Dict[str, Any]]) -> int:
            return self._run(self._outer.log_risk_limit(snapshot))

        def fetch_risk_limits(
            self,
            *,
            symbol: Optional[str] = None,
            limit: int = 100,
        ) -> List[Dict[str, Any]]:
            return self._run(self._outer.fetch_risk_limits(symbol=symbol, limit=limit))

        def log_rate_limit_snapshot(
            self, snapshot: Union[RateLimitSnapshotIn, Dict[str, Any]]
        ) -> int:
            return self._run(self._outer.log_rate_limit_snapshot(snapshot))

        def fetch_rate_limit_snapshots(
            self,
            *,
            bucket: Optional[str] = None,
            endpoint: Optional[str] = None,
            limit: int = 100,
        ) -> List[Dict[str, Any]]:
            return self._run(
                self._outer.fetch_rate_limit_snapshots(
                    bucket=bucket, endpoint=endpoint, limit=limit
                )
            )

        def log_security_audit(
            self, event: Union[SecurityAuditEventIn, Dict[str, Any]]
        ) -> int:
            return self._run(self._outer.log_security_audit(event))

        def fetch_security_audit(
            self,
            *,
            action: Optional[str] = None,
            status: Optional[str] = None,
            limit: int = 100,
        ) -> List[Dict[str, Any]]:
            return self._run(
                self._outer.fetch_security_audit(
                    action=action, status=status, limit=limit
                )
            )

        def get_schema_version(self) -> int:
            return self._run(self._outer.get_schema_version())

        def add_log(self, *, level: str, source: str, message: str, extra: Optional[Dict[str, Any]] = None) -> int:
            return self._run(self._outer.add_log(level=level, source=source, message=message, extra=extra))

        def export_trades_csv(self, *, path: Union[str, Path]) -> Path:
            return self._run(self._outer.export_trades_csv(path=path))

        def export_table_json(self, *, table: str, path: Union[str, Path]) -> Path:
            return self._run(self._outer.export_table_json(table=table, path=path))

    @property
    def sync(self) -> "DatabaseManager._SyncWrapper":
        return DatabaseManager._SyncWrapper(self)


async def _migration_initial(session: AsyncSession, manager: "DatabaseManager") -> None:
    """Początkowa migracja – struktury tworzone przez ``create_all``."""
    # Brak dodatkowych działań – pozostawiamy jako znacznik wersji 1.
    return None


async def _migration_performance_tables(session: AsyncSession, manager: "DatabaseManager") -> None:
    """Migracja dodająca tabele metryk i limitów ryzyka."""
    # Struktury tabel dodawane są przez ``Base.metadata.create_all`` w ``init_db``.
    # Migracja pozostawiona dla kompatybilności – w razie potrzeby można dodać transformacje danych.
    return None


async def _migration_api_rate_limits(session: AsyncSession, manager: "DatabaseManager") -> None:
    """Migracja przygotowująca widoki łączące metryki i limity ryzyka."""
    await session.execute(text("DROP VIEW IF EXISTS vw_strategy_health"))
    await session.execute(
        text(
            """
            CREATE VIEW IF NOT EXISTS vw_strategy_health AS
            WITH latest_metrics AS (
                SELECT
                    pm.symbol,
                    pm.metric,
                    pm.value,
                    pm.ts,
                    pm.mode,
                    ROW_NUMBER() OVER (
                        PARTITION BY pm.symbol, pm.metric, pm.mode
                        ORDER BY pm.ts DESC
                    ) AS rn
                FROM performance_metrics pm
            ),
            latest_limits AS (
                SELECT
                    rl.symbol,
                    rl.max_fraction,
                    rl.recommended_size,
                    rl.details,
                    rl.ts,
                    rl.mode,
                    ROW_NUMBER() OVER (
                        PARTITION BY rl.symbol, rl.mode
                        ORDER BY rl.ts DESC
                    ) AS rn
                FROM risk_limits rl
            )
            SELECT
                lm.symbol,
                lm.mode,
                MAX(CASE WHEN lm.metric = 'auto_trader_expectancy' THEN lm.value END) AS expectancy,
                MAX(CASE WHEN lm.metric = 'auto_trader_profit_factor' THEN lm.value END) AS profit_factor,
                MAX(CASE WHEN lm.metric = 'auto_trader_win_rate' THEN lm.value END) AS win_rate,
                rl.max_fraction,
                rl.recommended_size,
                rl.details,
                lm.ts AS metric_ts,
                rl.ts AS risk_ts
            FROM latest_metrics lm
            LEFT JOIN latest_limits rl
                ON rl.symbol = lm.symbol
                AND rl.mode = lm.mode
                AND rl.rn = 1
            WHERE lm.rn = 1
            GROUP BY
                lm.symbol,
                lm.mode,
                rl.max_fraction,
                rl.recommended_size,
                rl.details,
                lm.ts,
                rl.ts;
            """
        )
    )


MIGRATIONS: Dict[int, Callable[[AsyncSession, "DatabaseManager"], Any]] = {
    1: _migration_initial,
    2: _migration_performance_tables,
    3: _migration_api_rate_limits,
}

CURRENT_SCHEMA_VERSION = max(MIGRATIONS)
