# quick_live_orders_test.py
# -*- coding: utf-8 -*-

from dataclasses import asdict, is_dataclass
from pathlib import Path
import sys


def _ensure_repo_root() -> None:
    current_dir = Path(__file__).resolve().parent
    for candidate in (current_dir, *current_dir.parents):
        package_init = candidate / "KryptoLowca" / "__init__.py"
        if package_init.exists():
            candidate_str = str(candidate)
            if candidate_str not in sys.path:
                sys.path.insert(0, candidate_str)
            break


if __package__ in (None, ""):
    _ensure_repo_root()


import ccxt
from KryptoLowca.database_manager import DatabaseManager
from KryptoLowca.exchange_adapter import ExchangeAdapter
from KryptoLowca.managers.database_manager import DatabaseManager
from KryptoLowca.managers.exchange_manager import ExchangeManager


SYMBOL = "BTC/USDT"
DB_URL = "sqlite+aiosqlite:///trading.db"

# >>> WPROWADZONE KLUCZE TESTNET (na Twoją prośbę, zapisane w kodzie) <<<
API_KEY = "AmmYwBf8i1blZ8shREGu6rf56VdOL73lIitsnQ3EYdNOWYKs0hSgD3WagJDsU0U1"
API_SECRET = "Ek0aMUBws5lLw2uJn6E42U9V8p8UdpONQkhBpbYX8QhGUjaeU08pf5G1w4gmnGFD"


def _pretty(obj):
    return asdict(obj) if is_dataclass(obj) else obj


def _build_manager() -> ExchangeManager:
    manager = ExchangeManager(exchange_id="binance", db_url=DB_URL)
    manager.set_credentials(API_KEY, API_SECRET)
    manager.set_mode(spot=True, testnet=True)
    manager.load_markets()
    return manager


def main() -> None:
    db = DatabaseManager(DB_URL)
    db.sync.init_db()

    manager = _build_manager()

    ticker = manager.fetch_ticker(SYMBOL) or {}
    last_price = ticker.get("last") or ticker.get("close")
    print(f"Ostatnia cena {SYMBOL}: {last_price}")

    qty = 0.0002

    try:
        print("Wysyłam MARKET BUY na testnet:", qty, "BTC")
        order = manager.create_order(
            SYMBOL,
            side="BUY",
            type="MARKET",
            quantity=qty,
            client_order_id="tn-mkt-buy-1",
        )
        print("Złożono zlecenie:", _pretty(order))
    except ccxt.AuthenticationError as exc:
        print("\n[AUTH ERROR] Binance zwrócił błąd autoryzacji.")
        print("  → Najczęściej to złe/nie-spotowe klucze albo literówka w kluczu.")
        print("  → Upewnij się, że to **Binance Spot Testnet** (https://testnet.binance.vision), nie Futures.")
        print("  → Sprawdź, czy klucz nie ma dodatkowego znaku/spacji i czy jest aktywny.")
        print("Szczegóły:", str(exc))
        return
    except ccxt.BaseError as exc:
        print("\n[CCXT ERROR] API zwróciło błąd:", str(exc))
        return

    print("\nOPEN ORDERS (live/testnet):")
    for order in manager.fetch_open_orders(SYMBOL):
        print(_pretty(order))

    print("\nPOZYCJE (zarejestrowane w DB):")
    for position in manager.fetch_positions(SYMBOL):
        print(_pretty(position))

    # Eksport
    db.sync.export_trades_csv(path="backups/trades_live_testnet.csv")
    db.sync.export_table_json(table="positions", path="backups/positions_live_testnet.json")
    print("\nEksport: backups/trades_live_testnet.csv, backups/positions_live_testnet.json")


if __name__ == "__main__":
    main()
