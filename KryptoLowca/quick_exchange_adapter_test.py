# quick_exchange_adapter_test.py
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


from KryptoLowca.database_manager import DatabaseManager
from KryptoLowca.exchange_adapter import ExchangeAdapter


SYMBOL = "BTC/USDT"
DB_URL = "sqlite+aiosqlite:///trading.db"


def _pretty(obj):
    return asdict(obj) if is_dataclass(obj) else obj


def _build_paper_manager() -> ExchangeManager:
    manager = ExchangeManager(exchange_id="binance", db_url=DB_URL)
    manager.set_mode(paper=True)
    manager.set_paper_balance(10_000.0)
    manager.set_paper_fee_rate(0.001)
    manager.load_markets()
    return manager


def main() -> None:
    manager = _build_paper_manager()

    prices = [50_000, 50_800, 51_300, 51_900, 52_100, 51_400]
    manager.process_paper_tick(SYMBOL, prices[0])

    manager.create_order(
        SYMBOL,
        side="BUY",
        type="MARKET",
        quantity=0.02,
        client_order_id="qa-mkt-1",
    )

    manager.create_order(
        SYMBOL,
        side="SELL",
        type="LIMIT",
        quantity=0.02,
        price=52_000,
        client_order_id="qa-lim-1",
    )

    for price in prices[1:]:
        manager.process_paper_tick(SYMBOL, price)

    print("\nOPEN ORDERS:")
    for order in manager.fetch_open_orders(SYMBOL):
        print(_pretty(order))

    print("\nPOZYCJE:")
    for position in manager.fetch_positions(SYMBOL):
        print(_pretty(position))

    # Eksport na koniec – korzystamy z tej samej bazy co ExchangeManager
    db = DatabaseManager(DB_URL)
    db.sync.export_trades_csv(path="backups/trades_adapter.csv")
    db.sync.export_table_json(table="positions", path="backups/positions_adapter.json")
    print("\nEksport: backups/trades_adapter.csv, backups/positions_adapter.json")


if __name__ == "__main__":
    main()
