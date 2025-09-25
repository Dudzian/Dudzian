# run_trading_gui_paper.py
# -*- coding: utf-8 -*-
"""
Launcher PAPER do istniejącego trading_gui.py:
- Nie wymaga żadnych kluczy API.
- Dodaje okno "Quick Paper Trade" do ręcznego wysyłania MARKET BUY/SELL w trybie paper.
- Zapisuje do bazy: orders, trades, positions (korzysta z managers.database_manager.DatabaseManager).
- Pokazuje ostatnie pozycje i 10 ostatnich transakcji z trybu paper.

NIC nie modyfikuje trading_gui.py – importujemy i „doklejamy” funkcjonalność.
"""

from __future__ import annotations

import math
import traceback
from typing import Optional, List, Dict, Any

import tkinter as tk
from tkinter import ttk, messagebox

import KryptoLowca.trading_gui  # Twój oryginalny plik GUI (NIE MODYFIKUJEMY GO)

# DB manager (używamy tego samego co w testach earlier)
from KryptoLowca.managers.database_manager import DatabaseManager


# ----------------- POMOCNICZE -----------------

def _fmt_float(x: float, max_dec: int = 8) -> float:
    """Prosta normalizacja floatów do czytelnego zapisu."""
    s = f"{float(x):.{max_dec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return float(s) if s else 0.0


def _get_last_price(app: trading_gui.TradingGUI, symbol: str) -> Optional[float]:
    """
    Pobiera cenę ostatnią z publicznego tickera przez ExchangeManager (bez kluczy).
    Jeśli się nie uda, zwraca None.
    """
    try:
        app._ensure_exchange()  # utwórz/lub użyj publicznego klienta z GUI
        ex = getattr(app, "ex_mgr", None)
        if not ex:
            return None
        t = ex.fetch_ticker(symbol)
        if t:
            for k in ("last", "close", "bid", "ask"):
                v = t.get(k)
                if v is not None:
                    return float(v)
    except Exception:
        pass
    return None


# ----------------- PATCH – zostawiamy paper po staremu -----------------

# Oryginalna metoda GUI (paper – nie ruszamy jej, tylko zachowujemy dla ręcznych BUY/SELL w GUI)
_ORIG_EXEC = trading_gui.TradingGUI._bridge_execute_trade

def _patched_bridge_execute_trade(self, symbol: str, side: str, mkt_price: float):
    """
    Pozostawiamy PAPER tak jak było: wywołujemy oryginalną metodę GUI.
    (Ten launcher NIE wysyła nic do CCXT; całość jest paper.)
    """
    try:
        return _ORIG_EXEC(self, symbol, side, mkt_price)
    except Exception as e:
        tb = traceback.format_exc()
        self._log(f"[Paper] Wyjątek w _bridge_execute_trade: {e}\n{tb}", "ERROR")
        messagebox.showerror("Paper", f"Nieoczekiwany błąd: {e}")

trading_gui.TradingGUI._bridge_execute_trade = _patched_bridge_execute_trade


# ----------------- OKNO QUICK PAPER TRADE -----------------

class QuickPaperTrade(tk.Toplevel):
    """
    Ręczne MARKET BUY/SELL w trybie paper:
    - liczy ilość z kwoty USDT / bieżącej ceny (z publicznego tickera),
    - zapisuje zlecenie, trade i aktualizuje pozycję w DB,
    - pokazuje pozycje (paper) i 10 ostatnich transakcji (paper).
    """
    def __init__(self, app: trading_gui.TradingGUI):
        super().__init__(app.root)
        self.title("Quick Paper Trade")
        self.app = app

        # DB setup
        self.db = DatabaseManager("sqlite+aiosqlite:///trading.db")
        self.db.sync.init_db()

        self._build_ui()

        # podnieś okno na start
        self.after(100, self.lift)
        self.after(150, lambda: self.attributes("-topmost", True))
        self.after(500, lambda: self.attributes("-topmost", False))

    def _build_ui(self):
        frm_top = ttk.Frame(self)
        frm_top.pack(fill="x", padx=8, pady=6)

        ttk.Label(frm_top, text="Symbol:").grid(row=0, column=0, sticky="w")
        default_symbol = getattr(self.app, "symbol_var", None)
        default_symbol = (default_symbol.get() if default_symbol else "BTC/USDT")
        self.symbol_var = tk.StringVar(value=default_symbol)
        ttk.Entry(frm_top, textvariable=self.symbol_var, width=18).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(frm_top, text="Kwota (USDT):").grid(row=0, column=2, sticky="w", padx=(12,0))
        self.notional_var = tk.StringVar(value="12")
        ttk.Entry(frm_top, textvariable=self.notional_var, width=10).grid(row=0, column=3, sticky="w", padx=4)

        ttk.Button(frm_top, text="Market BUY", command=self._on_buy).grid(row=0, column=4, padx=(12,4))
        ttk.Button(frm_top, text="Market SELL", command=self._on_sell).grid(row=0, column=5, padx=4)

        # Listy: Pozycje + Ostatnie transakcje
        sep = ttk.Separator(self)
        sep.pack(fill="x", padx=8, pady=6)

        lbl_pos = ttk.Label(self, text="Pozycje (paper)")
        lbl_pos.pack(anchor="w", padx=8)
        self.tree_pos = ttk.Treeview(self, columns=("symbol","side","qty","avg_price","unrl_pnl"), show="headings", height=6)
        for c, w in (("symbol",120), ("side",80), ("qty",120), ("avg_price",120), ("unrl_pnl",120)):
            self.tree_pos.heading(c, text=c.upper()); self.tree_pos.column(c, width=w, anchor="center")
        self.tree_pos.pack(fill="x", padx=8, pady=(2,6))

        lbl_tr = ttk.Label(self, text="Trades (ostatnie 10, paper)")
        lbl_tr.pack(anchor="w", padx=8)
        self.tree_tr = ttk.Treeview(self, columns=("id","time","symbol","side","qty","price","fee"), show="headings", height=8)
        for c, w in (("id",60), ("time",160), ("symbol",120), ("side",80), ("qty",100), ("price",110), ("fee",100)):
            self.tree_tr.heading(c, text=c.upper()); self.tree_tr.column(c, width=w, anchor="center")
        self.tree_tr.pack(fill="both", expand=True, padx=8, pady=(2,8))

        frm_bot = ttk.Frame(self)
        frm_bot.pack(fill="x", padx=8, pady=(0,8))
        ttk.Button(frm_bot, text="Odśwież", command=self._refresh_views).pack(side="right")

        self._refresh_views()

    # --- Akcje BUY/SELL (paper) ---

    def _on_buy(self):
        self._market(side="BUY")

    def _on_sell(self):
        self._market(side="SELL")

    def _market(self, side: str):
        symbol = (self.symbol_var.get() or "BTC/USDT").strip().upper()
        try:
            notional = float(self.notional_var.get())
        except Exception:
            messagebox.showerror("Input", "Niepoprawna kwota USDT.")
            return
        if notional <= 0:
            messagebox.showerror("Input", "Kwota musi być > 0.")
            return

        price = _get_last_price(self.app, symbol)
        if not price:
            messagebox.showerror("Price", f"Brak ceny dla {symbol}. Najpierw użyj 'Load Markets', wybierz symbol.")
            return

        qty = _fmt_float(notional / price, 8)
        if qty <= 0:
            messagebox.showerror("Calc", "Wyliczona ilość wyszła 0. Zwiększ kwotę lub wybierz tańszy symbol.")
            return

        fee_rate = 0.001  # 0.1% – typowy maker/taker w testach paperowych
        fee = _fmt_float(qty * price * fee_rate, 8)

        try:
            # 1) Order → FILLED
            oid = self.db.sync.record_order({
                "symbol": symbol,
                "side": side,
                "type": "MARKET",
                "quantity": qty,
                "mode": "paper",
                "client_order_id": None
            })
            self.db.sync.update_order_status(order_id=oid, status="FILLED")

            # 2) Trade
            self.db.sync.record_trade({
                "symbol": symbol,
                "side": side,
                "quantity": qty,
                "price": price,
                "fee": fee,
                "order_id": oid,
                "mode": "paper"
            })

            # 3) Pozycja (prosty model long-only):
            #    - BUY zwiększa ilość i aktualizuje średnią
            #    - SELL zmniejsza ilość (nie otwieramy shortów w tym prostym panelu)
            positions = self.db.sync.get_open_positions(mode="paper")
            pos = next((p for p in positions if p.get("symbol")==symbol and p.get("side")=="LONG"), None)

            if side == "BUY":
                if pos:
                    old_qty = float(pos["quantity"])
                    old_avg = float(pos["avg_price"])
                    new_qty = _fmt_float(old_qty + qty, 8)
                    new_avg = _fmt_float((old_qty*old_avg + qty*price) / new_qty, 8) if new_qty > 0 else 0.0
                else:
                    new_qty = qty
                    new_avg = price
                self.db.sync.upsert_position({
                    "symbol": symbol,
                    "side": "LONG",
                    "quantity": new_qty,
                    "avg_price": new_avg,
                    "unrealized_pnl": 0.0,
                    "mode": "paper"
                })
            else:  # SELL
                if pos:
                    old_qty = float(pos["quantity"])
                    new_qty = _fmt_float(old_qty - qty, 8)
                    if new_qty < 0:
                        new_qty = 0.0  # nie wchodzimy w shorty w tym prostym panelu
                    new_avg = float(pos["avg_price"]) if new_qty > 0 else 0.0
                    self.db.sync.upsert_position({
                        "symbol": symbol,
                        "side": "LONG",
                        "quantity": new_qty,
                        "avg_price": new_avg,
                        "unrealized_pnl": 0.0,
                        "mode": "paper"
                    })
                # jeśli nie było pozycji – traktujemy SELL jako „nic do zamknięcia”: tylko trade + order zapisane

            self.app._log(f"[Paper] MARKET {side} {symbol} qty={qty} price={price} (fee≈{fee})", "INFO")
            messagebox.showinfo("Paper", f"Zaksięgowano PAPER {side} {symbol}\nIlość: {qty}\nCena: {price}")
            self._refresh_views()

        except Exception as e:
            tb = traceback.format_exc()
            self.app._log(f"[Paper] Błąd paper trade: {e}\n{tb}", "ERROR")
            messagebox.showerror("Paper", f"Błąd zapisu transakcji: {e}")

    # --- Widoki (pozycje + trades) ---

    def _refresh_views(self):
        # Pozycje (paper)
        try:
            pos = self.db.sync.get_open_positions(mode="paper")
        except Exception as e:
            pos = []
        for i in self.tree_pos.get_children():
            self.tree_pos.delete(i)
        for p in pos:
            self.tree_pos.insert("", "end", values=(
                p.get("symbol"), p.get("side"),
                _fmt_float(p.get("quantity", 0.0)), _fmt_float(p.get("avg_price", 0.0)),
                _fmt_float(p.get("unrealized_pnl", 0.0))
            ))

        # Trades (ostatnie 10 – paper)
        try:
            tr = self.db.sync.fetch_trades(mode="paper")
        except Exception as e:
            tr = []
        tr = sorted(tr, key=lambda x: x.get("id", 0), reverse=True)[:10]
        for i in self.tree_tr.get_children():
            self.tree_tr.delete(i)
        for t in tr:
            self.tree_tr.insert("", "end", values=(
                t.get("id"), t.get("ts"),
                t.get("symbol"), t.get("side"),
                _fmt_float(t.get("quantity", 0.0)), _fmt_float(t.get("price", 0.0)),
                _fmt_float(t.get("fee", 0.0))
            ))


# --- Start GUI + auto-okno ----------------------------------------------------

def _open_paper_panel_on_start(app: trading_gui.TradingGUI):
    app.root.after(800, lambda: QuickPaperTrade(app))

if __name__ == "__main__":
    root = tk.Tk()
    app = trading_gui.TradingGUI(root)
    _open_paper_panel_on_start(app)
    root.mainloop()
