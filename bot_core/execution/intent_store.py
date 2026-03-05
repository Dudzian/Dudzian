"""Trwały magazyn intencji/bindingów zleceń dla LiveExecutionRouter (SQLite)."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any


class IntentStore:
    """Lekki persistence layer dla order bindings i (opcjonalnie) intentów."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self._db_path), timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _init_schema(self) -> None:
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS order_bindings (
                        order_id TEXT PRIMARY KEY,
                        exchange TEXT NOT NULL,
                        symbol TEXT,
                        client_order_id TEXT,
                        created_at REAL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS order_intents (
                        intent_id TEXT PRIMARY KEY,
                        client_order_id TEXT,
                        symbol TEXT,
                        created_at REAL,
                        state TEXT,
                        last_error TEXT
                    )
                    """
                )

    def save_order_binding(
        self,
        *,
        order_id: str,
        exchange: str,
        symbol: str | None = None,
        client_order_id: str | None = None,
        created_at: float | None = None,
    ) -> None:
        created = float(created_at if created_at is not None else time.time())
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO order_bindings(order_id, exchange, symbol, client_order_id, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (order_id, exchange, symbol, client_order_id, created),
                )

    def get_order_binding(self, order_id: str) -> dict[str, Any] | None:
        with self._lock:
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT order_id, exchange, symbol, client_order_id, created_at
                    FROM order_bindings
                    WHERE order_id = ?
                    """,
                    (order_id,),
                ).fetchone()
        if row is None:
            return None
        return {
            "order_id": row[0],
            "exchange": row[1],
            "symbol": row[2],
            "client_order_id": row[3],
            "created_at": row[4],
        }


__all__ = ["IntentStore"]
