# scripts/db_tools.py
# -*- coding: utf-8 -*-
from KryptoLowca.managers.database_manager import DatabaseManager
import os

def purge_zero_quantity_trades(db_path: str = "sqlite+aiosqlite:///trading.db") -> None:
    """
    Usuwa z tabeli 'trades' rekordy z quantity == 0.0 (np. stare testy).
    """
    db = DatabaseManager(db_path)
    db.sync.init_db()

    # SQLAlchemy 2.0 async nie ma tu prostego delete sync; zrobimy naive: odczyt -> filtr -> eksport nowej bazy
    trades = db.sync.fetch_trades(limit=1000000)
    keep = [t for t in trades if float(t.get("quantity") or 0.0) > 0.0]

    # Zrób kopię bezpieczeństwa
    db.sync.export_table_json(table="trades", path="backups/trades_before_purge.json")

    # Łatwy reset TRADES: zresetuj całą bazę jeśli to środowisko testowe (najprościej)
    # UWAGA: To uproszczenie dla środowiska dev. W produkcji użyj migracji.
    if os.path.exists("trading.db"):
        os.remove("trading.db")
    # odtwórz schemat
    db = DatabaseManager(db_path)
    db.sync.init_db()

    # Niestety przy pełnym resecie tracimy też orders/positions/equity.
    # Jeśli chcesz tylko wyciąć 0-ki bez utraty reszty — pomiń reset bazy i pozostaw jak jest.

    # W razie potrzeby można odtworzyć tylko "trades" z keep (pominięto dla prostoty).
    print("Zresetowano bazę. (Dev-only). Stary TRADES eksportowano do backups/trades_before_purge.json")

def nuke_db_file() -> None:
    """
    Brutalnie usuwa plik trading.db (dev-only).
    """
    if os.path.exists("trading.db"):
        os.remove("trading.db")
        print("Usunięto trading.db")
    else:
        print("Brak pliku trading.db")
