# -*- coding: utf-8 -*-
"""Warstwa kompatybilności z historycznym API ``database_manager``.

Nowa implementacja znajduje się w ``managers.database_manager``. Ten plik
zapewnia jedynie przyjazne aliasy (``DBOptions``) oraz ujednolicone wyjątki,
tak aby starsze testy oraz skrypty mogły działać bez zmian.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from managers.database_manager import DatabaseManager as _CoreDatabaseManager

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


class DatabaseManager(_CoreDatabaseManager):
    """Rozszerzenie nowego menedżera o pomocnicze metody fabryczne."""

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
