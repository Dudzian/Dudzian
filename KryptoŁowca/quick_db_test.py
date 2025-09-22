# quick_db_test.py
# -*- coding: utf-8 -*-

from managers.database_manager import DatabaseManager

def main():
    # 1) Łączymy się z lokalną bazą SQLite (plik trading.db utworzy się sam)
    db = DatabaseManager("sqlite+aiosqlite:///trading.db")
    db.sync.init_db()  # tworzy tabele, jeśli ich nie ma

    # 2) Dodajemy zlecenie (order) - tryb "paper" (papierowy/symulowany)
    oid = db.sync.record_order({
        "symbol": "BTC/USDT",
        "side": "BUY",
        "type": "MARKET",
        "quantity": 0.01,
        "mode": "paper",
        "client_order_id": "test-123"   # unikalny identyfikator dla idempotencji
    })

    # 3) Aktualizujemy status zlecenia na "FILLED" (wypełnione)
    db.sync.update_order_status(order_id=oid, status="FILLED")

    # 4) Rejestrujemy trade (czyli faktyczne wykonanie/kontrakt)
    tid = db.sync.record_trade({
        "symbol": "BTC/USDT",
        "side": "BUY",
        "quantity": 0.01,
        "price": 50000.0,
        "fee": 0.5,
        "order_id": oid,   # powiązanie z powyższym zleceniem
        "mode": "paper"
    })

    # 5) Uaktualniamy/wstawiamy pozycję (Position) — mamy LONG 0.01 BTC po 50k
    db.sync.upsert_position({
        "symbol": "BTC/USDT",
        "side": "LONG",
        "quantity": 0.01,
        "avg_price": 50000.0,
        "unrealized_pnl": 0.0,
        "mode": "paper"
    })

    # 6) Podgląd danych
    trades = db.sync.fetch_trades(mode="paper")[:1]
    positions = db.sync.get_open_positions(mode="paper")

    print("Trades (pierwszy rekord):", trades)
    print("Open positions:", positions)

    # 7) (Opcjonalnie) Backup/eksport danych do plików
    # Tworzy folder "backups" i zapisuje CSV + JSON
    db.sync.export_trades_csv(path="backups/trades_backup.csv")
    db.sync.export_table_json(table="positions", path="backups/positions.json")
    print("Zrobiono eksport: backups/trades_backup.csv oraz backups/positions.json")

if __name__ == "__main__":
    main()
