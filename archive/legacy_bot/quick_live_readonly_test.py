# quick_live_readonly_test.py
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
