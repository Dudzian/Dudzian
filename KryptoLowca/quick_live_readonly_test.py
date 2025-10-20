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


from KryptoLowca.managers.exchange_manager import ExchangeManager


def main() -> None:
    manager = ExchangeManager(exchange_id="binance")
    manager.set_mode(spot=True)
    manager.load_markets()

    ticker = manager.fetch_ticker("BTC/USDT") or {}
    last_price = ticker.get("last") or ticker.get("close") or ticker.get("bid")
    print("LIVE (readonly) OK – ostatnia cena BTC/USDT:", last_price)


if __name__ == "__main__":
    main()
