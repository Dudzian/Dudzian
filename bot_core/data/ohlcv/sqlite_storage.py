"""Implementacja magazynu OHLCV opartego o SQLite."""
from __future__ import annotations

import sqlite3
from collections.abc import MutableMapping
from pathlib import Path
from typing import Mapping, Sequence

from bot_core.data.base import CacheStorage

_COLUMNS = ("open_time", "open", "high", "low", "close", "volume")


class _SQLiteMetadata(MutableMapping[str, str]):
    """Lekki adapter słownika mapujący na tabelę metadata."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._connection = connection

    def __getitem__(self, key: str) -> str:
        cursor = self._connection.execute("SELECT value FROM metadata WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row is None:
            raise KeyError(key)
        return str(row[0])

    def __setitem__(self, key: str, value: str) -> None:
        with self._connection:
            self._connection.execute(
                "INSERT INTO metadata(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def __delitem__(self, key: str) -> None:
        with self._connection:
            affected = self._connection.execute("DELETE FROM metadata WHERE key = ?", (key,)).rowcount
        if affected == 0:
            raise KeyError(key)

    def __iter__(self):
        cursor = self._connection.execute("SELECT key FROM metadata")
        return (row[0] for row in cursor.fetchall())

    def __len__(self) -> int:
        cursor = self._connection.execute("SELECT COUNT(1) FROM metadata")
        value = cursor.fetchone()
        return int(value[0] if value else 0)


class SQLiteCacheStorage(CacheStorage):
    """Przechowuje dane OHLCV lub pełni rolę manifestu metadanych."""

    def __init__(self, database_path: str | Path, *, store_rows: bool = True) -> None:
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, detect_types=sqlite3.PARSE_DECLTYPES)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._store_rows = store_rows
        self._initialize()

    def _initialize(self) -> None:
        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ohlcv (
                    symbol TEXT NOT NULL,
                    interval TEXT NOT NULL,
                    open_time INTEGER NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    PRIMARY KEY(symbol, interval, open_time)
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )

    def read(self, key: str) -> Mapping[str, Sequence[Sequence[float]]]:
        if not self._store_rows:
            raise KeyError(key)
        symbol, interval = key.split("::", maxsplit=1)
        cursor = self._connection.execute(
            """
            SELECT open_time, open, high, low, close, volume
            FROM ohlcv
            WHERE symbol = ? AND interval = ?
            ORDER BY open_time
            """,
            (symbol, interval),
        )
        rows = [[float(col) for col in row] for row in cursor.fetchall()]
        return {"columns": _COLUMNS, "rows": rows}

    def write(self, key: str, payload: Mapping[str, Sequence[Sequence[float]]]) -> None:
        symbol, interval = key.split("::", maxsplit=1)
        rows = [tuple(row) for row in payload.get("rows", []) if row]
        if not rows:
            return
        # Aktualizujemy metadane manifestu niezależnie od tego, czy przechowujemy świeczki.
        max_timestamp = max(float(row[0]) for row in rows)
        metadata = self.metadata()
        metadata[f"last_timestamp::{symbol}::{interval}"] = str(int(max_timestamp))
        metadata[f"row_count::{symbol}::{interval}"] = str(len(rows))
        if not self._store_rows:
            return
        with self._connection:
            self._connection.executemany(
                """
                INSERT INTO ohlcv(symbol, interval, open_time, open, high, low, close, volume)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, interval, open_time) DO UPDATE SET
                    open = excluded.open,
                    high = excluded.high,
                    low = excluded.low,
                    close = excluded.close,
                    volume = excluded.volume
                """,
                [
                    (
                        symbol,
                        interval,
                        float(row[0]),
                        float(row[1]),
                        float(row[2]),
                        float(row[3]),
                        float(row[4]),
                        float(row[5]),
                    )
                    for row in rows
                ],
            )

    def metadata(self) -> MutableMapping[str, str]:
        return _SQLiteMetadata(self._connection)

    def latest_timestamp(self, key: str) -> float | None:
        symbol, interval = key.split("::", maxsplit=1)
        if not self._store_rows:
            metadata = self.metadata()
            value = metadata.get(f"last_timestamp::{symbol}::{interval}")
            if value is None:
                return None
            try:
                return float(value)
            except (TypeError, ValueError):  # pragma: no cover - niepoprawny wpis metadanych
                return None
        cursor = self._connection.execute(
            """
            SELECT MAX(open_time) FROM ohlcv
            WHERE symbol = ? AND interval = ?
            """,
            (symbol, interval),
        )
        row = cursor.fetchone()
        if row is None or row[0] is None:
            return None
        return float(row[0])


__all__ = ["SQLiteCacheStorage"]
