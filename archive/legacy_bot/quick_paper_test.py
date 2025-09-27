# quick_paper_test.py
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


from KryptoLowca.managers.database_manager import DatabaseManager
from KryptoLowca.managers.paper_exchange import PaperExchange
import datetime as dt

def main():
    # 1) DB init
    db = DatabaseManager("sqlite+aiosqlite:///trading.db")
    db.sync.init_db()

    # 2) PaperExchange ze startowym kapitałem 10k USDT
    px = PaperExchange(
        db,
        symbol="BTC/USDT",
        starting_balance=10_000.0,
        fee_rate=0.001,        # 0.1%
        slippage_bps=5,        # 0.05%
        max_partial_fill_qty=0.01,  # częściowe fill'e do 0.01 BTC na tick
    )

    # 3) Strumień cen (symulacja)
    prices = [50_000, 49_900, 50_100, 51_000, 51_500, 51_900, 52_100, 51_200, 50_800]

    # 4) Pierwszy tick – ustalamy cenę rynkową
    px.process_tick(prices[0])

    # 5) MARKET BUY 0.02 BTC
    px.create_order(side="BUY", type="MARKET", quantity=0.02, client_order_id="mkt-buy-1")

    # 6) LIMIT SELL 0.02 na 52_000
    px.create_order(side="SELL", type="LIMIT", quantity=0.02, price=52_000, client_order_id="lim-sell-1")

    # 7) STOP BUY (np. momentum) 0.01 na 51_800 -> po triggerze staje się MARKET
    px.create_order(side="BUY", type="STOP", quantity=0.01, stop_price=51_800, client_order_id="stop-buy-1")

    # 8) Przetapiaj ceny
    for p in prices[1:]:
        px.process_tick(p)

    # 9) Podsumowanie
    print("\n--- OPEN ORDERS ---")
    for o in px.get_open_orders():
        print(o)

    print("\n--- POSITION ---")
    print(px.get_position())

    print("\n--- TRADES SAMPLE ---")
    # pokaż ostatnie 5 trade’ów z DB:
    trades = db.sync.fetch_trades(limit=5)
    for t in trades:
        print(t)

    print("\nEksport do backups/:")
    db.sync.export_trades_csv(path="backups/trades_paper.csv")
    db.sync.export_table_json(table="positions", path="backups/positions_paper.json")
    print("OK -> backups/trades_paper.csv, backups/positions_paper.json")

if __name__ == "__main__":
    main()
