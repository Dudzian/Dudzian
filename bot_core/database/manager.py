# -*- coding: utf-8 -*-
"""Natywny menedżer bazy danych dla ``bot_core``.

Moduł zawiera kompletną implementację asynchronicznego ``DatabaseManagera``,
wykorzystywanego przez pipeline tradingowy i backtestowy. Zachowuje ten sam
interfejs co dotychczasowa wersja z monolitu, zapewniając transakcje,
eksporty i logowanie metryk/zdarzeń.
"""
from __future__ import annotations

import asyncio
import atexit
import contextlib
import concurrent.futures
import csv
import datetime as dt
import functools
import json
import logging
import os
import sys
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Coroutine, Dict, List, Optional, TypeVar, Union

from pydantic import BaseModel, ValidationError, field_validator

from sqlalchemy import (
    Boolean,
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
from sqlalchemy.pool import NullPool


# --- Logowanie strukturalne ---
logger = logging.getLogger("bot_core.database.manager")
logger.propagate = True

_active_lock = threading.Lock()
_active_instances: "weakref.WeakSet[DatabaseManager]" = weakref.WeakSet()
_background_lock = threading.Lock()
_background_loop: asyncio.AbstractEventLoop | None = None
_background_thread: threading.Thread | None = None
_background_ready = threading.Event()
_background_disabled_logged = False


def _register_instance(instance: "DatabaseManager") -> None:
    with _active_lock:
        _active_instances.add(instance)


def _discard_instance(instance: "DatabaseManager") -> None:
    with _active_lock:
        _active_instances.discard(instance)


def _is_test_mode() -> bool:
    return (
        bool(os.getenv("PYTEST_CURRENT_TEST"))
        or bool(os.getenv("PYTEST_ADDOPTS"))
        or bool(os.getenv("CI"))
        or "pytest" in sys.modules
    )


def _env_true(var_name: str) -> bool:
    value = os.getenv(var_name, "").strip().lower()
    return value in {"1", "true", "yes"}


def _disable_background_loop() -> bool:
    return _env_true("BOT_CORE_DISABLE_DB_BACKGROUND_LOOP") or (
        sys.platform == "win32" and _is_test_mode()
    )


def _default_timeout(base: float, *, test: float, windows: float) -> float:
    if sys.platform == "win32":
        return windows
    if _is_test_mode():
        return test
    return base


def _background_shutdown_timeout() -> float:
    return _default_timeout(5.0, test=12.0, windows=15.0)


def _ensure_background_loop() -> asyncio.AbstractEventLoop | None:
    global _background_loop, _background_thread
    global _background_disabled_logged
    if _disable_background_loop():
        if sys.platform == "win32" and _is_test_mode() and not _background_disabled_logged:
            _background_disabled_logged = True
            logger.debug(
                "db_manager/background_loop_disabled: using caller loop (platform=%s pid=%s)",
                sys.platform,
                os.getpid(),
            )
        return None
    with _background_lock:
        if _background_loop:
            try:
                if _background_loop.is_running() and not _background_loop.is_closed():
                    return _background_loop
            except Exception:
                pass
        if _background_thread and _background_thread.is_alive():
            ready = _background_ready
        else:
            _background_ready.clear()
            ready = _background_ready

        def _runner() -> None:
            global _background_loop
            if sys.platform == "win32":
                try:
                    loop = asyncio.SelectorEventLoop()
                except Exception:
                    loop = asyncio.new_event_loop()
            else:
                loop = asyncio.new_event_loop()
            try:
                loop.set_name("DatabaseManagerBackgroundLoop")
            except Exception:
                pass
            asyncio.set_event_loop(loop)
            logger.debug(
                "DatabaseManager background loop started (loop_type=%s loop_module=%s loop_repr=%s thread=%s platform=%s)",
                _loop_type(loop),
                _loop_module(loop),
                _loop_repr(loop),
                threading.current_thread().name,
                sys.platform,
            )
            with _background_lock:
                _background_loop = loop
                ready.set()
            try:
                loop.run_forever()
            finally:
                try:
                    try:
                        closed = loop.is_closed()
                    except Exception:
                        closed = False
                    if not closed:
                        try:
                            pending = asyncio.all_tasks(loop)
                        except Exception as exc:  # pragma: no cover - defensywne
                            logger.debug(
                                "db_manager/background_loop_shutdown_error: pending task snapshot failed: %s",
                                exc,
                            )
                            pending = []
                        for task in pending:
                            try:
                                task.cancel()
                            except Exception:
                                continue
                        if pending:
                            try:
                                loop.run_until_complete(
                                    asyncio.gather(*pending, return_exceptions=True)
                                )
                            except Exception as exc:  # pragma: no cover - defensywne
                                logger.debug(
                                    "db_manager/background_loop_shutdown_error: gather failed: %s",
                                    exc,
                                )
                        try:
                            loop.run_until_complete(loop.shutdown_asyncgens())
                        except Exception as exc:  # pragma: no cover - defensywne
                            logger.debug(
                                "db_manager/background_loop_shutdown_error: shutdown_asyncgens failed: %s",
                                exc,
                            )
                        shutdown_default_executor = getattr(loop, "shutdown_default_executor", None)
                        if shutdown_default_executor is not None:
                            try:
                                loop.run_until_complete(shutdown_default_executor())
                            except Exception as exc:  # pragma: no cover - defensywne
                                logger.debug(
                                    "db_manager/background_loop_shutdown_error: shutdown_default_executor failed: %s",
                                    exc,
                                )
                except Exception as exc:  # pragma: no cover - defensywne
                    logger.debug("db_manager/background_loop_shutdown_error: %s", exc)
                try:
                    loop.close()
                except Exception as exc:  # pragma: no cover - defensywne
                    logger.debug(
                        "db_manager/background_loop_shutdown_error: close failed: %s",
                        exc,
                    )

        if _background_thread and _background_thread.is_alive():
            pass
        else:
            thread = threading.Thread(
                target=_runner,
                name="DatabaseManagerBackgroundLoopThread",
                daemon=sys.platform == "win32",
            )
            _background_thread = thread
            thread.start()
    if not ready.wait(timeout=1.0):
        logger.error("DatabaseManager background loop failed to start.")
        return None
    with _background_lock:
        return _background_loop


def _background_diagnostics() -> dict[str, Any]:
    active_instances = _snapshot_active_instances()
    suspicious_threads = [
        thread.name
        for thread in threading.enumerate()
        if thread.is_alive()
        and ("aiosqlite" in thread.name.lower() or "anyio" in thread.name.lower())
    ]
    return {
        "active_instances": len(active_instances),
        "suspicious_threads": suspicious_threads,
    }


def _aiosqlite_thread_snapshot(*, limit: int = 20) -> dict[str, Any]:
    threads = [
        thread
        for thread in threading.enumerate()
        if thread.is_alive() and "aiosqlite" in thread.name.lower()
    ]
    details = [
        f"{thread.name} (ident={thread.ident}, daemon={thread.daemon})"
        for thread in threads[:limit]
    ]
    if len(threads) > limit:
        details.append(f"... +{len(threads) - limit} more")
    return {
        "count": len(threads),
        "details": details,
    }


def _log_aiosqlite_threads(phase: str) -> None:
    snapshot = _aiosqlite_thread_snapshot()
    logger.debug(
        "DatabaseManager aiosqlite threads %s: count=%s details=%s",
        phase,
        snapshot["count"],
        snapshot["details"] or "<none>",
    )


def wait_for_aiosqlite_threads(timeout: float = 5.0, poll_interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        threads = [
            thread
            for thread in threading.enumerate()
            if thread.is_alive() and "aiosqlite" in thread.name.lower()
        ]
        if not threads:
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            details = "; ".join(
                f"{thread.name} (ident={thread.ident}, daemon={thread.daemon})"
                for thread in threads
            )
            if not details:
                details = "<unknown>"
            suspicious = [
                thread
                for thread in threading.enumerate()
                if thread.is_alive()
                and ("aiosqlite" in thread.name.lower() or "anyio" in thread.name.lower())
            ]
            suspicious_details = "; ".join(
                f"{thread.name} (ident={thread.ident}, daemon={thread.daemon})"
                for thread in suspicious
            )
            if suspicious and not suspicious_details:
                suspicious_details = "<unknown>"
            suspicious_suffix = f"; suspicious={suspicious_details}" if suspicious_details else ""
            logger.debug(
                "db_manager/aiosqlite_join_timeout: %s%s",
                details,
                suspicious_suffix,
            )
            return False
        join_timeout = min(poll_interval, max(0.0, remaining))
        for thread in threads:
            try:
                thread.join(timeout=join_timeout)
            except Exception:
                continue
        time.sleep(poll_interval)


def _format_loop_tasks(tasks: List[asyncio.Task[Any]]) -> str:
    details: List[str] = []
    for task in tasks:
        try:
            name = task.get_name()
        except Exception:  # pragma: no cover - defensywne
            name = None
        try:
            done = task.done()
        except Exception:  # pragma: no cover - defensywne
            done = "<unknown>"
        details.append(f"{task!r} name={name} done={done}")
    return "; ".join(details)


def _format_task_snapshot(tasks: List[asyncio.Task[Any]], *, limit: int = 20) -> str:
    if not tasks:
        return "<none>"
    snapshot = tasks[:limit]
    details = _format_loop_tasks(snapshot)
    if len(tasks) > limit:
        details = f"{details}; ... +{len(tasks) - limit} more"
    return details


def _coro_qualname(coro: Coroutine[Any, Any, Any]) -> str | None:
    try:
        name = getattr(coro, "__qualname__", None) or getattr(coro, "__name__", None)
    except Exception:  # pragma: no cover - defensywne
        name = None
    if name:
        return name
    try:
        return getattr(getattr(coro, "cr_code", None), "co_qualname", None)
    except Exception:  # pragma: no cover - defensywne
        return None


def shutdown_background_loop(timeout: float | None = None) -> None:
    global _background_loop, _background_thread, _background_ready
    if _disable_background_loop():
        with _background_lock:
            _background_loop = None
            _background_thread = None
            _background_ready.clear()
        return
    if timeout is None:
        timeout = _background_shutdown_timeout()
    with _background_lock:
        loop = _background_loop
        thread = _background_thread
    _log_aiosqlite_threads("before background loop shutdown")
    closed = False
    if loop:
        try:
            closed = loop.is_closed()
        except Exception:
            closed = False
        try:
            running = loop.is_running()
        except Exception:
            running = False
        logger.debug(
            "DatabaseManager background loop shutdown start (loop_type=%s loop_module=%s running=%s closed=%s thread=%s)",
            _loop_type(loop),
            _loop_module(loop),
            running,
            closed,
            thread.name if thread else None,
        )
    if loop and not closed:
        try:
            if loop.is_running():
                try:
                    all_tasks = list(asyncio.all_tasks(loop))
                except Exception as exc:  # pragma: no cover - defensywne
                    logger.debug("DatabaseManager background loop task snapshot failed: %s", exc)
                else:
                    if all_tasks:
                        logger.debug(
                            "DatabaseManager background loop tasks before stop: %s",
                            _format_loop_tasks(all_tasks),
                        )
                try:
                    pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
                except Exception as exc:  # pragma: no cover - defensywne
                    logger.debug("DatabaseManager background loop pending tasks check failed: %s", exc)
                else:
                    if pending:
                        logger.debug(
                            "DatabaseManager background loop pending tasks before stop: count=%s",
                            len(pending),
                        )
                        logger.debug(
                            "DatabaseManager background loop pending task snapshot: %s",
                            _format_task_snapshot(pending),
                        )
                loop.call_soon_threadsafe(loop.stop)
        except Exception as exc:  # pragma: no cover - defensywne
            logger.debug("DatabaseManager background loop stop failed: %s", exc)
    joined = True
    if thread:
        thread.join(timeout=timeout)
        joined = not thread.is_alive()
        if not joined:
            diagnostics = _background_diagnostics()
            logger.debug(
                "db_manager/background_loop_join_timeout: suspicious_threads=%s",
                diagnostics["suspicious_threads"],
            )
    diagnostics = _background_diagnostics()
    _log_aiosqlite_threads("after background loop shutdown")
    if not joined:
        logger.error(
            "DatabaseManager background loop thread still alive after shutdown: active_instances=%s suspicious_threads=%s",
            diagnostics["active_instances"],
            diagnostics["suspicious_threads"],
        )
        if _is_test_mode():
            thread_details = [
                f"{thr.name} (ident={thr.ident}, daemon={thr.daemon})"
                for thr in threading.enumerate()
                if thr.is_alive()
            ]
            logger.error(
                "DatabaseManager background loop shutdown thread snapshot: %s",
                thread_details or "<none>",
            )
            if loop and not closed:
                try:
                    loop_tasks = list(asyncio.all_tasks(loop))
                except Exception as exc:  # pragma: no cover - defensywne
                    logger.debug("DatabaseManager background loop task snapshot failed: %s", exc)
                else:
                    if loop_tasks:
                        logger.error(
                            "DatabaseManager background loop tasks during failed shutdown: %s",
                            _format_task_snapshot(loop_tasks),
                        )
            wait_for_aiosqlite_threads(timeout=max(timeout, 1.0), poll_interval=0.05)
            raise RuntimeError(
                "DatabaseManager background loop thread failed to shut down in test/CI mode."
            )
        if diagnostics["active_instances"] == 0:
            raise RuntimeError("DatabaseManager background loop thread failed to shut down.")
        logger.error(
            "DatabaseManager background loop shutdown incomplete; deferring to best-effort cleanup."
        )
        logger.debug(
            "DatabaseManager skipping wait_for_aiosqlite_threads (joined=False, active_instances=%s) "
            "during best-effort shutdown.",
            diagnostics["active_instances"],
        )
        return
    wait_for_aiosqlite_threads(timeout=max(timeout, 1.0), poll_interval=0.05)
    if joined:
        with _background_lock:
            _background_loop = None
            _background_thread = None
            _background_ready.clear()
    logger.debug(
        "DatabaseManager background loop shutdown: joined=%s active_instances=%s suspicious_threads=%s",
        joined,
        diagnostics["active_instances"],
        diagnostics["suspicious_threads"],
    )


def _loop_name(loop: asyncio.AbstractEventLoop | None) -> str | None:
    if loop is None:
        return None
    try:
        return loop.get_name()
    except Exception:
        return None


def _loop_type(loop: asyncio.AbstractEventLoop | None) -> str | None:
    if loop is None:
        return None
    return type(loop).__name__


def _loop_module(loop: asyncio.AbstractEventLoop | None) -> str | None:
    if loop is None:
        return None
    try:
        return type(loop).__module__
    except Exception:
        return None


def _loop_repr(loop: asyncio.AbstractEventLoop | None, *, limit: int = 300) -> str | None:
    if loop is None:
        return None
    try:
        value = repr(loop)
    except Exception:
        return None
    if len(value) > limit:
        truncated = value[: max(0, limit - 3)]
        return f"{truncated}..."
    return value


def _log_close_timeout(instance: "DatabaseManager", loop: asyncio.AbstractEventLoop | None) -> None:
    logger.debug(
        "DatabaseManager close timeout (instance=%s thread=%s loop=%s running=%s)",
        id(instance),
        threading.current_thread().name,
        _loop_name(loop) or loop,
        loop.is_running() if loop else False,
    )


def _log_background_timeout(instance: "DatabaseManager", loop: asyncio.AbstractEventLoop | None) -> None:
    logger.debug(
        "DatabaseManager background close timeout (instance=%s thread=%s loop=%s running=%s closed=%s)",
        id(instance),
        threading.current_thread().name,
        _loop_name(loop) or loop,
        loop.is_running() if loop else False,
        loop.is_closed() if loop else False,
    )


def _log_close_request_inflight(instance: "DatabaseManager", where: str) -> None:
    # Pure diagnostics: informs that close is still running somewhere; we do NOT cancel.
    logger.debug(
        "DatabaseManager close still in-flight (instance=%s where=%s thread=%s)",
        id(instance),
        where,
        threading.current_thread().name,
    )


def _schedule_close(instance: "DatabaseManager", *, blocking: bool, timeout: float) -> None:
    instance_loop = instance._loop
    if instance_loop is not None:
        try:
            loop_running = instance_loop.is_running()
        except Exception:
            loop_running = False
        try:
            loop_closed = instance_loop.is_closed()
        except Exception:
            loop_closed = True
        if loop_running and not loop_closed:
            if blocking:
                try:
                    future = asyncio.run_coroutine_threadsafe(instance.close(), instance_loop)
                except Exception as exc:  # pragma: no cover - defensywne
                    logger.debug(
                        "close_all_active: failed to schedule close on instance loop: %s",
                        exc,
                    )
                else:
                    try:
                        future.result(timeout=timeout)
                        return
                    except concurrent.futures.TimeoutError:
                        _log_close_timeout(instance, instance_loop)
                        _log_close_request_inflight(instance, "instance_loop")
                        # Deterministycznie NIE anulujemy coroutine.
                        # Jeśli blocking=True, timeout jest twardym błędem (teardown ma być przewidywalny).
                        raise RuntimeError("DatabaseManager close timed out on instance loop.") from None
                    except Exception as exc:  # pragma: no cover - defensywne
                        logger.debug("close_all_active: close failed: %s", exc)
                        return
            else:
                try:
                    def _schedule_on_loop() -> None:
                        task = asyncio.create_task(instance.close())

                        def _log_task_exception(done_task: asyncio.Task[None]) -> None:
                            try:
                                done_task.result()
                            except Exception as exc:  # pragma: no cover - defensywne
                                logger.debug("close_all_active: close failed: %s", exc)

                        task.add_done_callback(_log_task_exception)

                    instance_loop.call_soon_threadsafe(_schedule_on_loop)
                    return
                except Exception as exc:  # pragma: no cover - defensywne
                    logger.debug(
                        "close_all_active: failed to schedule close on instance loop: %s",
                        exc,
                    )
    if _disable_background_loop():
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        if running_loop and running_loop.is_running():
            try:
                running_loop.create_task(instance.close())
            except Exception as exc:  # pragma: no cover - defensywne
                logger.debug(
                    "close_all_active: failed to schedule close on caller loop: %s",
                    exc,
                )
            return
        if blocking:
            async def _runner() -> None:
                loop = asyncio.get_running_loop()
                task = asyncio.create_task(instance.close())
                deadline = time.monotonic() + timeout
                while not task.done():
                    if time.monotonic() >= deadline:
                        _log_close_timeout(instance, loop)
                        _log_close_request_inflight(instance, "caller_loop")
                        raise RuntimeError("DatabaseManager close timed out on caller loop.") from None
                    await asyncio.sleep(0.05)
                task.result()

            asyncio.run(_runner())
            return

        def _run_close_in_thread() -> None:
            async def _runner() -> None:
                await instance.close()

            try:
                asyncio.run(_runner())
            except Exception:  # pragma: no cover - defensywne
                logger.debug(
                    "db_manager/close_thread_failed: close failed in background thread. (instance=%s)",
                    id(instance),
                    exc_info=True,
                )

        thread = threading.Thread(
            target=_run_close_in_thread,
            name=f"DatabaseManagerCloseThread-{id(instance)}",
            daemon=True,
        )
        thread.start()
        return
    background_loop = _ensure_background_loop()
    if not background_loop:
        logger.error("close_all_active: background loop unavailable for instance=%s", id(instance))
        if blocking:
            raise RuntimeError("DatabaseManager background loop unavailable for blocking close.")
        return
    try:
        future = asyncio.run_coroutine_threadsafe(instance.close(), background_loop)
    except Exception as exc:  # pragma: no cover - defensywne
        logger.debug(
            "close_all_active: failed to schedule close on background loop: %s",
            exc,
        )
        if blocking:
            raise RuntimeError("DatabaseManager background loop scheduling failed.") from exc
        return
    if not blocking:
        return
    try:
        future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        _log_background_timeout(instance, background_loop)
        _log_close_request_inflight(instance, "background_loop")
        # Deterministycznie NIE anulujemy coroutine.
        # W trybie blocking timeout ma failować test/teardown zamiast zostawiać UB.
        raise RuntimeError("DatabaseManager close timed out on background loop.") from None
    except Exception as exc:  # pragma: no cover - defensywne
        logger.debug("close_all_active: close failed: %s", exc)


def _snapshot_active_instances() -> List["DatabaseManager"]:
    with _active_lock:
        return list(_active_instances)

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
    order_id: Mapped[Optional[int]] = mapped_column(Integer, index=True)  # FK logiczna
    pnl: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
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
    level: Mapped[str] = mapped_column(String(10), index=True)   # INFO/WARN/ERROR
    source: Mapped[str] = mapped_column(String(50), index=True)  # trading_engine/strategy/exchange/...
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


class RiskAuditLog(Base):
    __tablename__ = "risk_audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ts: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow, index=True)
    symbol: Mapped[str] = mapped_column(String(50), index=True)
    side: Mapped[Optional[str]] = mapped_column(String(5), nullable=True, index=True)  # BUY/SELL
    state: Mapped[str] = mapped_column(String(16), index=True)
    reason: Mapped[Optional[str]] = mapped_column(String(80), nullable=True, index=True)
    fraction: Mapped[float] = mapped_column(Float)
    price: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    mode: Mapped[str] = mapped_column(String(10), default="live", index=True)
    schema_version: Mapped[int] = mapped_column(Integer, default=1)
    limit_events: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    stop_loss_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    take_profit_pct: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    should_trade: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    __table_args__ = (
        Index("ix_risk_audit_symbol_ts", "symbol", "ts"),
        Index("ix_risk_audit_state_ts", "state", "ts"),
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


class RiskAuditIn(BaseModel):
    symbol: str
    state: str
    fraction: float
    # opcjonalne pola
    side: str | None = None              # BUY/SELL
    reason: str | None = None
    price: float | None = None
    mode: str = "live"                   # akceptujemy też 'demo' w walidatorze
    schema_version: int = 1
    limit_events: List[str] | None = None
    details: Dict[str, Any] | None = None
    stop_loss_pct: float | None = None
    take_profit_pct: float | None = None
    should_trade: bool | None = None
    ts: float | None = None              # epoch seconds

    @field_validator("symbol")
    @classmethod
    def _symbol(cls, v: str) -> str:
        v = (v or "").strip().upper()
        if not v:
            raise ValueError("symbol is required")
        return v

    @field_validator("side")
    @classmethod
    def _side(cls, v: str | None) -> str | None:
        if v is None:
            return v
        vv = v.strip().upper()
        if vv not in {"BUY", "SELL"}:
            raise ValueError("side must be BUY or SELL")
        return vv

    @field_validator("state")
    @classmethod
    def _state(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if not v:
            raise ValueError("state is required")
        return v

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        v = (v or "").strip().lower()
        if v not in {"live", "paper", "demo"}:
            raise ValueError("mode must be live/demo/paper")
        return v

    @field_validator("schema_version")
    @classmethod
    def _schema_version(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("schema_version must be positive")
        return int(v)


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


def _allow_when_closing(
    method: Callable[..., Coroutine[Any, Any, Any]]
) -> Callable[..., Coroutine[Any, Any, Any]]:
    setattr(method, "_allow_when_closing", True)
    return method


class DatabaseManager:
    """
    Główna klasa obsługująca bazę danych (async) + sync wrapper.
    """

    _ASYNC_DISPATCH_TIMEOUT_BASE = 5.0
    _DispatchT = TypeVar("_DispatchT")
    _TrackT = TypeVar("_TrackT")
    _CLOSE_INFLIGHT_TIMEOUT_BASE = 5.0
    _CLOSE_INFLIGHT_GRACE_PERIOD = 0.5
    _CLOSE_INFLIGHT_POLL_INTERVAL = 0.05
    _FINALIZE_CLOSE_DEADLINE = 30.0

    def __init__(self, db_url: str = "sqlite+aiosqlite:///trading.db") -> None:
        self.db_url = db_url
        self._state = _EngineState()
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._inflight_tasks: set[asyncio.Task[Any]] = set()
        self._inflight_lock = threading.Lock()
        self._closing = False
        self._finalize_close_task: asyncio.Task[None] | None = None
        self._closing_started_at: float | None = None
        _register_instance(self)

    def __del__(self) -> None:  # pragma: no cover - sprzątanie defensywne
        _discard_instance(self)

    # ---------- Inicjalizacja ----------
    @staticmethod
    def _dispatch_to_db_loop(
        method: Callable[..., Coroutine[Any, Any, _DispatchT]]
    ) -> Callable[..., Coroutine[Any, Any, _DispatchT]]:
        @functools.wraps(method)
        async def _wrapper(
            self: "DatabaseManager", *args: Any, **kwargs: Any
        ) -> _DispatchT:
            allow_when_closing = bool(getattr(method, "_allow_when_closing", False))
            return await self._dispatch(
                method(self, *args, **kwargs),
                allow_when_closing=allow_when_closing,
            )

        return _wrapper

    def _ensure_db_loop(self) -> asyncio.AbstractEventLoop:
        if _disable_background_loop():
            try:
                running = asyncio.get_running_loop()
            except RuntimeError as exc:
                raise RuntimeError(
                    "DatabaseManager background loop disabled; no running event loop."
                ) from exc
            if self._loop is None:
                self._loop = running
                return running
            if self._loop is running:
                return running
            old_running = False
            old_closed = True
            try:
                old_running = self._loop.is_running()
            except Exception:
                old_running = False
            try:
                old_closed = self._loop.is_closed()
            except Exception:
                old_closed = True
            logger.debug(
                "DatabaseManager loop reassigned to caller loop (instance=%s old=%s new=%s running=%s closed=%s)",
                id(self),
                _loop_name(self._loop) or self._loop,
                _loop_name(running) or running,
                old_running,
                old_closed,
            )
            self._loop = running
            return running
        background_loop = _ensure_background_loop()
        if not background_loop:
            raise RuntimeError("DatabaseManager background loop unavailable.")
        if self._loop is None:
            self._loop = background_loop
            return background_loop
        if self._loop is background_loop:
            return background_loop
        old_running = False
        old_closed = True
        try:
            old_running = self._loop.is_running()
        except Exception:
            old_running = False
        try:
            old_closed = self._loop.is_closed()
        except Exception:
            old_closed = True
        logger.debug(
            "DatabaseManager loop reassigned to background loop (instance=%s old=%s new=%s running=%s closed=%s)",
            id(self),
            _loop_name(self._loop) or self._loop,
            _loop_name(background_loop) or background_loop,
            old_running,
            old_closed,
        )
        self._loop = background_loop
        return background_loop

    async def _track(self, coro: Coroutine[Any, Any, _TrackT]) -> _TrackT:
        task = asyncio.current_task()
        if task is not None:
            with self._inflight_lock:
                self._inflight_tasks.add(task)
        try:
            return await coro
        finally:
            if task is not None:
                with self._inflight_lock:
                    self._inflight_tasks.discard(task)

    def _snapshot_inflight_tasks(self) -> List[asyncio.Task[Any]]:
        with self._inflight_lock:
            return list(self._inflight_tasks)

    async def _drain_db_loop(self, *, ticks: int = 3) -> None:
        for _ in range(max(1, ticks)):
            await asyncio.sleep(0)

    async def _dispatch(
        self,
        coro: Coroutine[Any, Any, _DispatchT],
        *,
        timeout: float | None = None,
        allow_when_closing: bool = False,
    ) -> _DispatchT:
        if self._closing and not allow_when_closing:
            raise RuntimeError("DatabaseManager is closing; refusing new work.")
        db_loop = self._ensure_db_loop()
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is db_loop:
            return await self._track(coro)
        if timeout is None:
            timeout = _default_timeout(self._ASYNC_DISPATCH_TIMEOUT_BASE, test=30.0, windows=45.0)
        try:
            future = asyncio.run_coroutine_threadsafe(self._track(coro), db_loop)
        except Exception as exc:  # pragma: no cover - defensywne
            raise RuntimeError("DatabaseManager failed to schedule async call on background loop.") from exc
        instance_id = id(self)

        def _consume_future_result(done_future: concurrent.futures.Future[Any]) -> None:
            try:
                done_future.result()
            except Exception as exc:  # pragma: no cover - defensywne
                logger.debug(
                    "DatabaseManager async call failed (late) instance=%s: %s",
                    instance_id,
                    exc,
                    exc_info=True,
                )

        future.add_done_callback(_consume_future_result)
        wrapped = asyncio.wrap_future(future)
        done, _pending = await asyncio.wait({wrapped}, timeout=timeout)
        if wrapped in done:
            return wrapped.result()
        inflight_snapshot = self._snapshot_inflight_tasks()
        logger.debug(
            "DatabaseManager async call timed out; leaving in-flight (instance=%s thread=%s loop=%s coro=%s inflight_count=%s inflight_tasks=%s)",
            id(self),
            threading.current_thread().name,
            _loop_name(db_loop) or db_loop,
            _coro_qualname(coro) or type(coro).__name__,
            len(inflight_snapshot),
            _format_task_snapshot(inflight_snapshot),
        )
        raise RuntimeError("DatabaseManager async call timed out on background loop.") from None

    @_dispatch_to_db_loop
    async def init_db(self, *, create: bool = True) -> None:
        """
        Tworzy silnik, sesję i (opcjonalnie) schemat bazy, jeśli nie istnieje.
        DDL (create_all) wykonujemy na połączeniu engine, NIE na sesji.
        """
        self._ensure_db_loop()
        async with self._lock:
            if self._state.engine is None:
                engine_kwargs: dict[str, Any] = {"future": True}
                if self.db_url.startswith("sqlite+aiosqlite://") and (
                    sys.platform == "win32" or _is_test_mode()
                ):
                    engine_kwargs["poolclass"] = NullPool
                    logger.debug(
                        "DatabaseManager using NullPool for sqlite+aiosqlite (url=%s platform=%s test_mode=%s)",
                        self.db_url,
                        sys.platform,
                        _is_test_mode(),
                    )
                self._state.engine = create_async_engine(self.db_url, **engine_kwargs)
                self._state.session_factory = sessionmaker(
                    bind=self._state.engine,
                    expire_on_commit=False,
                    class_=AsyncSession,
                    autoflush=False,
                    autocommit=False,
                )

        if create:
            async with self._state.engine.begin() as conn:  # type: ignore[union-attr]
                await conn.run_sync(Base.metadata.create_all)

        await self._apply_migrations()

        if create:
            logger.info("Database initialized (url=%s)", self.db_url)

    @_dispatch_to_db_loop
    @_allow_when_closing
    async def close(self) -> None:
        self._closing = True
        if self._closing_started_at is None:
            self._closing_started_at = time.monotonic()
        engine = self._state.engine
        if engine is None:
            self._loop = None
            _discard_instance(self)
            return
        current_task = asyncio.current_task()
        close_timeout = _default_timeout(self._CLOSE_INFLIGHT_TIMEOUT_BASE, test=20.0, windows=45.0)
        deadline = time.monotonic() + close_timeout

        def _pending_inflight() -> List[asyncio.Task[Any]]:
            if current_task is None:
                return self._snapshot_inflight_tasks()
            with self._inflight_lock:
                return [task for task in self._inflight_tasks if task is not current_task]

        async def _wait_for_inflight(*, until: float) -> bool:
            while True:
                pending = _pending_inflight()
                if not pending:
                    return True
                if time.monotonic() >= until:
                    return False
                await asyncio.sleep(self._CLOSE_INFLIGHT_POLL_INTERVAL)

        inflight_cleared = await _wait_for_inflight(until=deadline)
        if not inflight_cleared:
            pending = _pending_inflight()
            logger.error(
                "DatabaseManager close timed out waiting for in-flight tasks (instance=%s tasks=%s)",
                id(self),
                _format_loop_tasks(pending),
            )
            grace_deadline = time.monotonic() + self._CLOSE_INFLIGHT_GRACE_PERIOD
            inflight_cleared = await _wait_for_inflight(until=grace_deadline)
            if not inflight_cleared:
                pending = _pending_inflight()
                logger.error(
                    "DatabaseManager close entering deferred finalize due to in-flight tasks (instance=%s loop=%s tasks=%s)",
                    id(self),
                    _loop_name(self._loop) or self._loop,
                    _format_loop_tasks(pending),
                )
                finalize_task = self._finalize_close_task
                if finalize_task is None or finalize_task.done():
                    finalize_task = asyncio.create_task(self._finalize_close(engine))
                    try:
                        finalize_task.set_name(f"DatabaseManagerFinalizeClose-{id(self)}")
                    except Exception:  # pragma: no cover - defensywne
                        pass
                    self._finalize_close_task = finalize_task
                return

        self._state.engine = None
        self._state.session_factory = None
        try:
            try:
                await engine.dispose()
            except Exception as exc:  # pragma: no cover - defensywne
                logger.debug(
                    "Error while disposing database engine (instance=%s): %s",
                    id(self),
                    exc,
                )
            try:
                await self._drain_db_loop()
            except Exception as exc:  # pragma: no cover - defensywne
                logger.debug(
                    "Error while draining database loop (instance=%s): %s",
                    id(self),
                    exc,
                )
        finally:
            self._loop = None
            _discard_instance(self)

    @_dispatch_to_db_loop
    @_allow_when_closing
    async def aclose(self) -> None:
        await self.close()

    async def _finalize_close(self, engine: AsyncEngine) -> None:
        current_task = asyncio.current_task()
        deadline = time.monotonic() + self._FINALIZE_CLOSE_DEADLINE
        timed_out = False
        try:
            while True:
                pending = [
                    task for task in self._snapshot_inflight_tasks()
                    if current_task is None or task is not current_task
                ]
                if not pending:
                    break
                if time.monotonic() >= deadline and not timed_out:
                    timed_out = True
                    logger.error(
                        "DatabaseManager finalize close deadline exceeded; pending tasks remain (instance=%s tasks=%s)",
                        id(self),
                        _format_loop_tasks(pending),
                    )
                await asyncio.sleep(self._CLOSE_INFLIGHT_POLL_INTERVAL)
            self._state.engine = None
            self._state.session_factory = None
            try:
                await engine.dispose()
                logger.debug(
                    "DatabaseManager deferred close disposed engine (instance=%s)",
                    id(self),
                )
            except Exception as exc:  # pragma: no cover - defensywne
                logger.debug(
                    "Error while disposing database engine (deferred, instance=%s): %s",
                    id(self),
                    exc,
                )
            try:
                await self._drain_db_loop()
            except Exception as exc:  # pragma: no cover - defensywne
                logger.debug(
                    "Error while draining database loop (deferred, instance=%s): %s",
                    id(self),
                    exc,
                )
        finally:
            self._loop = None
            _discard_instance(self)
            if self._finalize_close_task is current_task:
                self._finalize_close_task = None
            logger.debug(
                "DatabaseManager deferred close finalized (instance=%s)",
                id(self),
            )

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

    @_dispatch_to_db_loop
    async def get_schema_version(self) -> int:
        if self._state.session_factory is None:
            return 0

        async with self.session() as session:
            result = await session.execute(select(func.max(SchemaVersion.version)))
            version = result.scalar()
            return int(version or 0)

    @contextlib.asynccontextmanager
    async def session(self, *, allow_when_closing: bool = False) -> AsyncIterator[AsyncSession]:
        if self._closing and not allow_when_closing:
            raise RuntimeError("DatabaseManager is closing; refusing new work.")
        db_loop = self._ensure_db_loop()
        running = asyncio.get_running_loop()
        if running is not db_loop:
            raise RuntimeError("DatabaseManager session must be used on the background loop.")
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
    @_dispatch_to_db_loop
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

    @_dispatch_to_db_loop
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

    @_dispatch_to_db_loop
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
    @_dispatch_to_db_loop
    async def record_order(self, order: Union[OrderIn, Dict[str, Any]]) -> int:
        """Zapisuje zlecenie (idempotencja po client_order_id). Zwraca ID rekordu."""
        try:
            o = order if isinstance(order, OrderIn) else OrderIn(**order)
        except ValidationError as e:
            logger.error("Order validation error: %s", e)
            raise

        async with self.transaction() as s:
            if o.client_order_id:
                q = await s.execute(select(Order).where(Order.client_order_id == o.client_order_id))
                existing = q.scalar_one_or_none()
                if existing:
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

    @_dispatch_to_db_loop
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
    @_dispatch_to_db_loop
    async def record_trade(self, trade: Union[TradeIn, Dict[str, Any]]) -> int:
        """Zapisuje trade. Zwraca ID."""
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

    @_dispatch_to_db_loop
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
    @_dispatch_to_db_loop
    async def upsert_position(self, pos: Union[PositionIn, Dict[str, Any]]) -> int:
        """Wstawia/aktualizuje po symbolu (unique), zwraca id."""
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

    @_dispatch_to_db_loop
    async def close_position(self, symbol: str) -> None:
        async with self.transaction() as s:
            q = await s.execute(select(Position).where(Position.symbol == symbol))
            existing = q.scalar_one_or_none()
            if existing:
                await s.delete(existing)

    @_dispatch_to_db_loop
    async def get_open_positions(self, *, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        async with self.session() as s:
            stmt = select(Position)
            if mode:
                stmt = stmt.where(Position.mode == mode)
            rows = (await s.execute(stmt)).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    # ---------- OPERACJE: Equity ----------
    @_dispatch_to_db_loop
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

    @_dispatch_to_db_loop
    async def fetch_equity_curve(self, *, limit: int = 1000, mode: Optional[str] = None) -> List[Dict[str, Any]]:
        async with self.session() as s:
            stmt = select(EquityCurve).order_by(EquityCurve.ts.asc()).limit(limit)
            if mode:
                stmt = stmt.where(EquityCurve.mode == mode)
            rows = (await s.execute(stmt)).scalars().all()
            return [self._row_to_dict(r) for r in rows]

    # ---------- OPERACJE: Performance metrics ----------
    @_dispatch_to_db_loop
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

    @_dispatch_to_db_loop
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
    @_dispatch_to_db_loop
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

    @_dispatch_to_db_loop
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

    # ---------- OPERACJE: Risk audit logs ----------
    @_dispatch_to_db_loop
    async def log_risk_audit(
        self,
        event: Union[RiskAuditIn, Dict[str, Any]],
    ) -> int:
        try:
            payload = event if isinstance(event, RiskAuditIn) else RiskAuditIn(**event)
        except ValidationError as exc:
            logger.error("Risk audit validation error: %s", exc)
            raise

        ts_value = (
            dt.datetime.utcfromtimestamp(float(payload.ts))
            if payload.ts is not None
            else dt.datetime.utcnow()
        )

        async with self.transaction() as session:
            rec = RiskAuditLog(
                ts=ts_value,
                symbol=payload.symbol,
                side=payload.side,
                state=payload.state,
                reason=payload.reason,
                fraction=float(payload.fraction),
                price=float(payload.price) if payload.price is not None else None,
                mode=payload.mode,
                schema_version=int(payload.schema_version),
                limit_events=json.dumps(payload.limit_events) if payload.limit_events is not None else None,
                details=json.dumps(payload.details) if payload.details is not None else None,
                stop_loss_pct=float(payload.stop_loss_pct) if payload.stop_loss_pct is not None else None,
                take_profit_pct=float(payload.take_profit_pct) if payload.take_profit_pct is not None else None,
                should_trade=payload.should_trade,
            )
            session.add(rec)
            await session.flush()
            return rec.id

    @_dispatch_to_db_loop
    async def fetch_risk_audits(
        self,
        *,
        symbol: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        async with self.session() as session:
            stmt = select(RiskAuditLog).order_by(RiskAuditLog.ts.desc()).limit(limit)
            if symbol:
                stmt = stmt.where(RiskAuditLog.symbol == symbol.upper())
            rows = (await session.execute(stmt)).scalars().all()

        result: List[Dict[str, Any]] = []
        for row in rows:
            details = json.loads(row.details) if row.details else None
            limit_events = json.loads(row.limit_events) if row.limit_events else None
            result.append(
                {
                    "id": row.id,
                    "timestamp": row.ts.isoformat() if row.ts else None,
                    "symbol": row.symbol,
                    "side": row.side,
                    "state": row.state,
                    "reason": row.reason,
                    "fraction": row.fraction,
                    "price": row.price,
                    "mode": row.mode,
                    "schema_version": row.schema_version,
                    "limit_events": limit_events,
                    "details": details,
                    "stop_loss_pct": row.stop_loss_pct,
                    "take_profit_pct": row.take_profit_pct,
                    "should_trade": row.should_trade,
                }
            )
        return result

    # ---------- OPERACJE: API rate limits ----------
    @_dispatch_to_db_loop
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

    @_dispatch_to_db_loop
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
    @_dispatch_to_db_loop
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

    @_dispatch_to_db_loop
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
    @_dispatch_to_db_loop
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

    @_dispatch_to_db_loop
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
    @_dispatch_to_db_loop
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

    @_dispatch_to_db_loop
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
                    "limit_events",
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
        _DEFAULT_TIMEOUT_BASE = 5.0

        def __init__(self, outer: "DatabaseManager") -> None:
            self._outer = outer

        def _run(self, coro, *, timeout: float | None = None):
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None
            if loop and loop.is_running():
                raise RuntimeError("Use async methods inside running event loop.")
            if self._outer._closing:
                raise RuntimeError("DatabaseManager is closing; refusing new work.")

            if timeout is None:
                timeout = _default_timeout(self._DEFAULT_TIMEOUT_BASE, test=30.0, windows=45.0)

            if _disable_background_loop():
                async def _runner():
                    self._outer._loop = asyncio.get_running_loop()
                    try:
                        return await self._outer._track(coro)
                    finally:
                        self._outer._loop = None

                return asyncio.run(_runner())

            background_loop = self._outer._ensure_db_loop()
            try:
                running = background_loop.is_running()
                closed = background_loop.is_closed()
                if closed or not running:
                    logger.debug(
                        "DatabaseManager background loop state for sync call (loop=%s running=%s closed=%s)",
                        _loop_name(background_loop) or background_loop,
                        running,
                        closed,
                    )
            except Exception:  # pragma: no cover - defensywne
                pass
            try:
                future = asyncio.run_coroutine_threadsafe(
                    self._outer._track(coro),
                    background_loop,
                )
            except Exception as exc:  # pragma: no cover - defensywne
                raise RuntimeError("DatabaseManager failed to schedule sync call on background loop.") from exc

            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                inflight_snapshot = self._outer._snapshot_inflight_tasks()
                logger.debug(
                    "DatabaseManager sync call timed out (instance=%s thread=%s loop=%s coro=%s inflight_count=%s inflight_tasks=%s)",
                    id(self._outer),
                    threading.current_thread().name,
                    _loop_name(background_loop) or background_loop,
                    _coro_qualname(coro) or type(coro).__name__,
                    len(inflight_snapshot),
                    _format_task_snapshot(inflight_snapshot),
                )
                # Deterministycznie NIE anulujemy coroutine.
                raise RuntimeError("DatabaseManager sync call timed out.") from None

        def init_db(self, *, create: bool = True) -> None:
            return self._run(self._outer.init_db(create=create))

        def close(self) -> None:
            return self._run(self._outer.close())

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

        def log_risk_audit(self, event: Union[RiskAuditIn, Dict[str, Any]]) -> int:
            return self._run(self._outer.log_risk_audit(event))

        def fetch_risk_audits(
            self,
            *,
            symbol: Optional[str] = None,
            limit: int = 100,
        ) -> List[Dict[str, Any]]:
            return self._run(self._outer.fetch_risk_audits(symbol=symbol, limit=limit))

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

    @classmethod
    def active_instances(cls) -> List["DatabaseManager"]:
        return _snapshot_active_instances()

    @classmethod
    def close_all_active(cls, *, blocking: bool = False, timeout: float = 1.5) -> None:
        # Runtime: fire-and-forget. Tests/teardown: blocking=True for deterministic shutdown.
        _log_aiosqlite_threads("before close_all_active")
        if blocking and _disable_background_loop():
            try:
                running_loop = asyncio.get_running_loop()
            except RuntimeError:
                running_loop = None
            if running_loop and running_loop.is_running():
                raise RuntimeError(
                    "close_all_active called from running event loop; use close_all_active_async."
                )
            asyncio.run(cls.close_all_active_async(timeout=timeout))
            wait_for_aiosqlite_threads(timeout=max(timeout, 1.0), poll_interval=0.05)
            _log_aiosqlite_threads("after close_all_active")
            return
        instances = _snapshot_active_instances()
        for instance in instances:
            _schedule_close(instance, blocking=blocking, timeout=timeout)
        if blocking:
            if _is_test_mode():
                close_timeout = _default_timeout(
                    cls._CLOSE_INFLIGHT_TIMEOUT_BASE, test=20.0, windows=45.0
                )
                deadline = time.monotonic() + max(timeout, close_timeout, 5.0) + 1.0
            else:
                deadline = time.monotonic() + max(timeout, 1.0) + 1.0
            while True:
                pending_finalize = []
                for instance in _snapshot_active_instances():
                    task = instance._finalize_close_task
                    if task is not None and not task.done():
                        pending_finalize.append(task)
                if not pending_finalize:
                    break
                if time.monotonic() >= deadline:
                    pending_details = _format_loop_tasks(pending_finalize)
                    logger.error(
                        "DatabaseManager finalize close tasks still running after timeout: %s",
                        pending_details,
                    )
                    if _is_test_mode():
                        raise RuntimeError(
                            "DatabaseManager finalize close tasks still running after timeout: "
                            + pending_details
                        )
                    break
                time.sleep(0.05)
            wait_for_aiosqlite_threads(timeout=max(timeout, 1.0), poll_interval=0.05)
            _log_aiosqlite_threads("after close_all_active")

    @classmethod
    def shutdown_background_loop(cls, *, timeout: float | None = None) -> None:
        shutdown_background_loop(timeout=timeout)

    @classmethod
    def wait_for_aiosqlite_threads(
        cls, *, timeout: float = 5.0, poll_interval: float = 0.05
    ) -> bool:
        return wait_for_aiosqlite_threads(timeout=timeout, poll_interval=poll_interval)

    @classmethod
    def windows_test_cleanup(cls, *, timeout: float = 2.0) -> None:
        if sys.platform != "win32":
            return
        cls.close_all_active(blocking=True, timeout=timeout)
        cls.wait_for_aiosqlite_threads(timeout=timeout, poll_interval=0.05)
        cls.shutdown_background_loop(timeout=timeout)

    @classmethod
    async def close_all_active_async(cls, *, timeout: float = 1.5) -> None:
        # UWAGA: asyncio.wait_for() anuluje coroutine na timeout -> zakazane w tym projekcie.
        instances = _snapshot_active_instances()
        for instance in instances:
            task = asyncio.create_task(instance.close())
            done, pending = await asyncio.wait({task}, timeout=timeout)
            if task in done:
                try:
                    task.result()
                except Exception as exc:  # pragma: no cover - defensywne
                    logger.debug("close_all_active_async: close failed: %s", exc)
            else:
                # Bez cancel: zostawiamy task w toku, ale logujemy twardo.
                logger.debug(
                    "close_all_active_async: close timed out (instance=%s); leaving in-flight",
                    id(instance),
                )
                instance_id = id(instance)

                def _consume_task_result(
                    done_task: asyncio.Task[None],
                    *,
                    _instance_id: int = instance_id,
                ) -> None:
                    try:
                        done_task.result()
                    except Exception as exc:  # pragma: no cover - defensywne
                        logger.debug(
                            "close_all_active_async: close failed (late) instance=%s: %s",
                            _instance_id,
                            exc,
                        )

                task.add_done_callback(_consume_task_result)


async def _migration_initial(session: AsyncSession, manager: "DatabaseManager") -> None:
    """Początkowa migracja – struktury tworzone przez ``create_all``."""
    return None


async def _migration_performance_tables(session: AsyncSession, manager: "DatabaseManager") -> None:
    """Migracja dodająca tabele metryk i limitów ryzyka."""
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


async def _migration_risk_audit_logs(session: AsyncSession, manager: "DatabaseManager") -> None:
    """Migracja dodająca tabelę logów audytu ryzyka."""
    return None


MIGRATIONS: Dict[int, Callable[[AsyncSession, "DatabaseManager"], Any]] = {
    1: _migration_initial,
    2: _migration_performance_tables,
    3: _migration_api_rate_limits,
    4: _migration_risk_audit_logs,
}

CURRENT_SCHEMA_VERSION = max(MIGRATIONS)


def _atexit_cleanup() -> None:
    try:
        DatabaseManager.close_all_active(blocking=True, timeout=5.0)
    except Exception as exc:  # pragma: no cover - defensywne
        try:
            logger.debug("DatabaseManager atexit close_all_active failed: %s", exc)
        except Exception:
            pass
    try:
        DatabaseManager.shutdown_background_loop()
    except Exception as exc:  # pragma: no cover - defensywne
        try:
            logger.debug("DatabaseManager atexit shutdown background loop failed: %s", exc)
        except Exception:
            pass
    try:
        wait_for_aiosqlite_threads(timeout=5.0, poll_interval=0.05)
    except Exception as exc:  # pragma: no cover - defensywne
        try:
            logger.debug("DatabaseManager atexit wait_for_aiosqlite_threads failed: %s", exc)
        except Exception:
            pass


atexit.register(_atexit_cleanup)
