# quick_exchange_adapter_test.py
# -*- coding: utf-8 -*-

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

def main():
    db = DatabaseManager("sqlite+aiosqlite:///trading.db")
    db.sync.init_db()

    ex = ExchangeAdapter(
        db,
        mode="paper",                 # tu w przyszłości podmienisz na 'live'
        symbol="BTC/USDT",
        starting_balance=10_000.0,
        fee_rate=0.001,
        slippage_bps=5,
        max_partial_fill_qty=0.01,
    )

    prices = [50_000, 50_800, 51_300, 51_900, 52_100, 51_400]
    ex.process_tick(prices[0])

    # Market buy 0.02
    ex.create_order(side="BUY", type="MARKET", quantity=0.02, client_order_id="qa-mkt-1")

    # Limit sell 0.02 @ 52_000
    ex.create_order(side="SELL", type="LIMIT", quantity=0.02, price=52_000, client_order_id="qa-lim-1")

    for p in prices[1:]:
        ex.process_tick(p)

    print("\nOPEN ORDERS:")
    for o in ex.get_open_orders():
        print(o)

    print("\nPOSITION:")
    print(ex.get_position())

    # Eksport na koniec
    db.sync.export_trades_csv(path="backups/trades_adapter.csv")
    db.sync.export_table_json(table="positions", path="backups/positions_adapter.json")
    print("\nEksport: backups/trades_adapter.csv, backups/positions_adapter.json")

if __name__ == "__main__":
    main()
