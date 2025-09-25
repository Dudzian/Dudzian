# quick_live_readonly_test.py
# -*- coding: utf-8 -*-

from KryptoLowca.managers.database_manager import DatabaseManager
from KryptoLowca.managers.exchange_adapter import ExchangeAdapter

def main():
    db = DatabaseManager("sqlite+aiosqlite:///trading.db")
    db.sync.init_db()

    # Tryb live bez kluczy – tylko odczyt ceny
    ex = ExchangeAdapter(
        db,
        mode="live",
        symbol="BTC/USDT",
        exchange_id="binance",  # możesz podmienić na 'kraken', 'coinbase', itd.
    )

    # pobierz aktualną cenę i zapisz do adaptera (process_tick sam pobierze z giełdy)
    ex.process_tick(price=None)

    # nic nie składamy – brak kluczy
    print("LIVE (readonly) OK: pobrano cenę i zaktualizowano stan (jeśli dostępne).")

if __name__ == "__main__":
    main()
