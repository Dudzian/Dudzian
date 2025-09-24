# quick_live_orders_test.py
# -*- coding: utf-8 -*-

import ccxt
from KryptoLowca.managers.database_manager import DatabaseManager
from KryptoLowca.managers.exchange_adapter import ExchangeAdapter

# >>> WPROWADZONE KLUCZE TESTNET (na Twoją prośbę, zapisane w kodzie) <<<
API_KEY = "AmmYwBf8i1blZ8shREGu6rf56VdOL73lIitsnQ3EYdNOWYKs0hSgD3WagJDsU0U1"
API_SECRET = "Ek0aMUBws5lLw2uJn6E42U9V8p8UdpONQkhBpbYX8QhGUjaeU08pf5G1w4gmnGFD"

def main():
    db = DatabaseManager("sqlite+aiosqlite:///trading.db")
    db.sync.init_db()

    ex = ExchangeAdapter(
        db,
        mode="live",
        symbol="BTC/USDT",
        exchange_id="binance",
        apiKey=API_KEY,
        secret=API_SECRET,
        enable_rate_limit=True,
        testnet=True,   # KLUCZOWE
    )

    # pobierz cenę (wewnętrznie z giełdy)
    ex.process_tick(price=None)

    # mały sanity-check na minNotional: jeśli ~10 USDT, to 0.0002 BTC zwykle wystarcza przy ~50k
    qty = 0.0002

    try:
        print("Wysyłam MARKET BUY na testnet:", qty, "BTC")
        oid = ex.create_order(side="BUY", type="MARKET", quantity=qty, client_order_id="tn-mkt-buy-1")
        print("Order zapisany w DB z id:", oid)
    except ccxt.AuthenticationError as e:
        print("\n[AUTH ERROR] Binance zwrócił błąd autoryzacji.")
        print("  → Najczęściej to złe/nie-spotowe klucze albo literówka w kluczu.")
        print("  → Upewnij się, że to **Binance Spot Testnet** (https://testnet.binance.vision), nie Futures.")
        print("  → Sprawdź, czy klucz nie ma dodatkowego znaku/spacji i czy jest aktywny.")
        print("Szczegóły:", str(e))
        return
    except ccxt.BaseError as e:
        print("\n[CCXT ERROR] API zwróciło błąd:", str(e))
        return

    print("\nOPEN ORDERS (live/testnet):")
    for o in ex.get_open_orders():
        print(o)

    print("\nPOSITION (z naszej DB):")
    print(ex.get_position())

    # Eksport
    db.sync.export_trades_csv(path="backups/trades_live_testnet.csv")
    db.sync.export_table_json(table="positions", path="backups/positions_live_testnet.json")
    print("\nEksport: backups/trades_live_testnet.csv, backups/positions_live_testnet.json")

if __name__ == "__main__":
    main()
