from pathlib import Path
import sys
import datetime as dt

from bot_core.exchanges.core import Mode, OrderSide, OrderType
from KryptoLowca.managers.exchange_manager import ExchangeManager


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
from KryptoLowca.paper_exchange import PaperExchange
import datetime as dt
def main() -> None:
    # 1) ExchangeManager w trybie papierowym
    manager = ExchangeManager()
    manager.set_mode(paper=True)
    manager.set_paper_balance(10_000.0, asset="USDT")
    manager.set_paper_fee_rate(0.001)
    manager.load_markets()

    # 2) Strumień cen (symulacja)
    prices = [50_000, 49_900, 50_100, 51_000, 51_500, 51_900, 52_100, 51_200, 50_800]

    # 3) Pierwszy tick – ustalamy cenę rynkową
    now = dt.datetime.utcnow()
    manager.process_paper_tick("BTC/USDT", prices[0], timestamp=now)

    # 4) MARKET BUY 0.02 BTC
    manager.create_order(
        "BTC/USDT",
        OrderSide.BUY.value,
        OrderType.MARKET.value,
        0.02,
        client_order_id="mkt-buy-1",
    )

    # 5) LIMIT SELL 0.02 na 52_000
    manager.create_order(
        "BTC/USDT",
        OrderSide.SELL.value,
        OrderType.LIMIT.value,
        0.02,
        price=52_000.0,
        client_order_id="lim-sell-1",
    )

    # 6) Przetapiaj ceny
    for idx, price in enumerate(prices[1:], start=1):
        manager.process_paper_tick(
            "BTC/USDT",
            price,
            timestamp=now + dt.timedelta(seconds=idx),
        )

    # 7) Podsumowanie
    print("\n--- OPEN ORDERS ---")
    for order in manager.fetch_open_orders("BTC/USDT"):
        print(order.model_dump())

    print("\n--- POSITION ---")
    for pos in manager.fetch_positions("BTC/USDT"):
        print(pos.model_dump())

    print("\n--- TRADES SAMPLE ---")
    db = manager._ensure_db()
    trades = db.sync.fetch_trades(limit=5, mode=Mode.PAPER.value)
    for trade in trades:
        print(trade)

    print("\nEksport do backups/:")
    db.sync.export_trades_csv(path="backups/trades_paper.csv")
    db.sync.export_table_json(table="positions", path="backups/positions_paper.json")
    print("OK -> backups/trades_paper.csv, backups/positions_paper.json")


if __name__ == "__main__":
    main()
