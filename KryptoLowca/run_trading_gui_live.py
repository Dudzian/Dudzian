# run_trading_gui_paper.py
# -*- coding: utf-8 -*-
"""
Launcher PAPER dla modułowego Trading GUI (pakiet ``KryptoLowca.ui.trading``):
- Nie wymaga żadnych kluczy API.
- Dodaje okno "Quick Paper Trade" do ręcznego wysyłania MARKET BUY/SELL w trybie paper.
- Zapisuje do bazy: orders, trades, positions (korzysta z managers.database_manager.DatabaseManager).
- Pokazuje ostatnie pozycje i 10 ostatnich transakcji z trybu paper.

NIC nie modyfikuje modułowego Trading GUI – importujemy i „doklejamy” funkcjonalność.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys
import math
import traceback
from typing import Optional, List, Dict, Any, Tuple

import tkinter as tk
from tkinter import ttk, messagebox


from bot_core.runtime.paths import DesktopAppPaths, build_desktop_app_paths
from KryptoLowca.logging_utils import DEFAULT_LOG_FILE, LOGS_DIR
from KryptoLowca.runtime.bootstrap import FrontendBootstrap, bootstrap_frontend_services


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


from KryptoLowca.ui.trading import (
    TradingGUI,
    build_risk_profile_hint as _build_hint,
    compute_default_notional as _compute_notional,
    format_notional as _fmt_notional,
    snapshot_from_app,
)


DEFAULT_NOTIONAL_USDT = 12.0


_FRONTEND_PATHS: DesktopAppPaths | None = None


def _build_frontend_bootstrap(
    *,
    core_config_path: str | Path | None = None,
    core_environment: str | None = None,
) -> Tuple[DesktopAppPaths | None, FrontendBootstrap | None]:
    """Przygotowuje wspólne usługi frontowe dla launchera GUI."""

    global _FRONTEND_PATHS

    if _FRONTEND_PATHS is None:
        module_file: Path | None = None
        try:
            trading_app = import_module("KryptoLowca.ui.trading.app")
            candidate = getattr(trading_app, "__file__", None)
            if candidate is not None:
                module_file = Path(candidate)
        except Exception:
            module_file = None
        if module_file is not None:
            try:
                _FRONTEND_PATHS = build_desktop_app_paths(
                    module_file,
                    logs_dir=LOGS_DIR,
                    text_log_file=DEFAULT_LOG_FILE,
                )
            except Exception:
                _FRONTEND_PATHS = None

    services = bootstrap_frontend_services(
        paths=_FRONTEND_PATHS,
        config_path=core_config_path,
        environment=core_environment,
    )
    return _FRONTEND_PATHS, services


def _compute_default_notional(app: TradingGUI) -> float:
    snapshot = snapshot_from_app(app)
    value = _compute_notional(snapshot, default_notional=DEFAULT_NOTIONAL_USDT)
    settings = getattr(snapshot, "settings", None)
    balance = getattr(snapshot, "paper_balance", 0.0)
    if not settings or balance <= 0:
        return value
    try:
        risk_per_trade = float(settings.max_risk_per_trade)
    except Exception:
        risk_per_trade = 0.0
    if risk_per_trade <= 0:
        return value
    risk_limit = float(balance) * risk_per_trade
    return max(value, risk_limit)


def _format_notional(value: float) -> str:
    return _fmt_notional(value)


def _build_risk_profile_hint(app: TradingGUI) -> Optional[str]:
    snapshot = snapshot_from_app(app)
    return _build_hint(snapshot)

# DB manager (używamy tego samego co w testach earlier)
from KryptoLowca.managers.database_manager import DatabaseManager


# ----------------- POMOCNICZE -----------------

def _fmt_float(x: float, max_dec: int = 8) -> float:
    """Prosta normalizacja floatów do czytelnego zapisu."""
    s = f"{float(x):.{max_dec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return float(s) if s else 0.0


def _get_last_price(app: TradingGUI, symbol: str) -> Optional[float]:
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


# ----------------- EXECUTOR PAPER – delikatne rozszerzenie -----------------

def _paper_trade_executor(gui: TradingGUI, symbol: str, side: str, mkt_price: float) -> None:
    """Wywołuje domyślną symulację i przechwytuje błędy dla launchera paper."""

    try:
        gui._bridge_execute_trade(symbol, side, mkt_price)
    except Exception as exc:
        tb = traceback.format_exc()
        gui._log(f"[Paper] Wyjątek podczas symulacji transakcji: {exc}\n{tb}", "ERROR")
        messagebox.showerror("Paper", f"Nieoczekiwany błąd: {exc}")


# ----------------- OKNO QUICK PAPER TRADE -----------------

class QuickPaperTrade(tk.Toplevel):
    """
    Ręczne MARKET BUY/SELL w trybie paper:
    - liczy ilość z kwoty USDT / bieżącej ceny (z publicznego tickera),
    - zapisuje zlecenie, trade i aktualizuje pozycję w DB,
    - pokazuje pozycje (paper) i 10 ostatnich transakcji (paper).
    """
    def __init__(self, app: TradingGUI):
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
        notional_value = _format_notional(_compute_default_notional(self.app))
        self.notional_var = tk.StringVar(value=notional_value)
        ttk.Entry(frm_top, textvariable=self.notional_var, width=10).grid(row=0, column=3, sticky="w", padx=4)

        ttk.Button(frm_top, text="Market BUY", command=self._on_buy).grid(row=0, column=4, padx=(12,4))
        ttk.Button(frm_top, text="Market SELL", command=self._on_sell).grid(row=0, column=5, padx=4)

        risk_hint = _build_risk_profile_hint(self.app)
        if risk_hint:
            ttk.Label(frm_top, text=risk_hint, foreground="gray25").grid(
                row=1, column=0, columnspan=6, sticky="w", pady=(6, 0)
            )

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

def _open_paper_panel_on_start(app: TradingGUI):
    app.root.after(800, lambda: QuickPaperTrade(app))

if __name__ == "__main__":
    root = tk.Tk()
    paths, services = _build_frontend_bootstrap()
    gui_kwargs: Dict[str, Any] = {}
    if paths is not None:
        gui_kwargs["paths"] = paths
    if services is not None:
        gui_kwargs["frontend_services"] = services
    try:
        app = TradingGUI(root, trade_executor=_paper_trade_executor, **gui_kwargs)
    except TypeError:
        app = TradingGUI(root, trade_executor=_paper_trade_executor)
    _open_paper_panel_on_start(app)
    root.mainloop()
