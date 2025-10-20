# run_trading_gui_paper.py
# -*- coding: utf-8 -*-
"""
Launcher PAPER dla modułowego Trading GUI (pakiet ``KryptoLowca.ui.trading``).
Funkcje:
- Kierunek: LONG / SHORT (futures paper, bez nettingu – dwie niezależne pozycje na symbol).
- Market/Limit: otwieranie i zamykanie dla obu kierunków.
- Risk sizing: Spot i Futures (cap po marginie) + tryb ATR-aware (jak w Cryptohopperze).
- SL/TP (twarde ceny), Partial TP (TP1/2/TP3) z PRESETAMI udziałów, Trailing – dla LONG i SHORT.
- ATR-aware: SL/TP/Trailing/Sizing z krotności ATR na wybranym interwale.
- Zapis do DB: orders/trades/positions (side=LONG/SHORT), unrealized_pnl tutaj nie liczymy (0).

Uwaga:
- Short tylko w trybie Futures (w Spot – wyłączamy SHORT).
- Zamykanie częściowe przez TP1/TP2/TP3 działa proporcjonalnie.
- Trailing: LONG używa 'peak', SHORT używa 'trough'.
- Nowość: przełącznik „Zawsze na wierzchu” i komunikaty wiązane z tym oknem (nie chowają się).
"""

from __future__ import annotations

from pathlib import Path
import sys
import traceback
import logging
import math
from typing import Optional, List, Dict, Any, Set, Tuple

import tkinter as tk
from tkinter import ttk, messagebox


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
    RiskSnapshot,
    TradingGUI,
    build_risk_profile_hint as _build_hint,
    compute_default_notional as _compute_notional,
    format_decimal as _format_decimal,
    format_notional as _format_notional,
    snapshot_from_app,
)
from KryptoLowca.ui.trading.risk_helpers import apply_runtime_risk_context
from KryptoLowca.managers.database_manager import DatabaseManager


# ----------------- KONFIG / POMOCNICZE -----------------

DEFAULT_NOTIONAL_USDT = 12.0
DEFAULT_CAPITAL_USDT = 10_000.0
DEFAULT_RISK_PCT = 1.0
DEFAULT_PORTFOLIO_PCT = 20.0

FEE_RATE = 0.001       # 0.1% koszt (paper)
ENGINE_TICK_MS = 1500  # ms: cykl silnika symulacji
SAFETY_DELTA = 0.003   # 0.3% korekta, gdy SL/TP bez sensu
MIN_SL_PCT = 0.0005    # min. 0.05% dla sizingu – anty-kosmiczne pozycje

DEFAULT_ATR_LEN = 14
DEFAULT_ATR_TF = "5m"
TIMEFRAMES = ["1m","3m","5m","15m","30m","1h","2h","4h","6h","12h","1d"]

# alias typu
LevelsKey = Tuple[str, str]  # (symbol, side) side in {"LONG","SHORT"}

# Presety udziałów TP (wartości w ułamkach 0..1)
TP_PORTION_PRESETS: Dict[str, Tuple[float,float,float]] = {
    "Zbalansowany (33/33/34)": (0.33, 0.33, 0.34),
    "Konserwatywny (25/35/40)": (0.25, 0.35, 0.40),
    "Agresywny (50/30/20)": (0.50, 0.30, 0.20),
    "Jedno-TP (100/0/0)": (1.00, 0.00, 0.00),
    "Dwa-TP (50/50/0)": (0.50, 0.50, 0.00),
}

logger = logging.getLogger(__name__)


def _derive_risk_defaults(snapshot: RiskSnapshot) -> tuple[float, float, float, float]:
    """Zwraca (kapitał, ryzyko %, ekspozycja %, notional) zgodne z profilem runtime."""

    capital = snapshot.paper_balance if snapshot.paper_balance > 0 else DEFAULT_CAPITAL_USDT

    if snapshot.settings is not None:
        try:
            risk_pct = max(float(snapshot.settings.max_risk_per_trade) * 100.0, 0.0)
        except Exception:
            risk_pct = DEFAULT_RISK_PCT
        try:
            portfolio_pct = max(float(snapshot.settings.max_portfolio_risk) * 100.0, 0.0)
        except Exception:
            portfolio_pct = DEFAULT_PORTFOLIO_PCT
    else:
        risk_pct = DEFAULT_RISK_PCT
        portfolio_pct = DEFAULT_PORTFOLIO_PCT

    if risk_pct <= 0:
        risk_pct = DEFAULT_RISK_PCT
    if portfolio_pct <= 0:
        portfolio_pct = DEFAULT_PORTFOLIO_PCT

    notional = _compute_notional(snapshot, default_notional=DEFAULT_NOTIONAL_USDT)
    limit_notional = _compute_notional(snapshot, default_notional=float("inf"))
    if math.isfinite(limit_notional) and limit_notional > notional:
        notional = limit_notional

    return capital, risk_pct, portfolio_pct, notional


def _fmt_float(x: float, max_dec: int = 8) -> float:
    s = f"{float(x):.{max_dec}f}"
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    return float(s) if s else 0.0

def _get_last_price(app: TradingGUI, symbol: str) -> Optional[float]:
    try:
        app._ensure_exchange()
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

def _fetch_ohlcv(app: TradingGUI, symbol: str, timeframe: str, limit: int) -> Optional[List[List[float]]]:
    """Pobiera świece przez exchange manager GUI (ccxt). Format: [ts, o, h, l, c, v] rosnąco."""
    try:
        app._ensure_exchange()
        ex = getattr(app, "ex_mgr", None)
        if not ex:
            return None
        return ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    except Exception:
        return None

def _compute_atr(ohlcv: List[List[float]], length: int) -> Optional[float]:
    """Wilder ATR: initial SMA(TR[0:N]), potem wygładzanie."""
    if not ohlcv or len(ohlcv) < length + 1:
        return None
    trs: List[float] = []
    prev_close = ohlcv[0][4]
    for i in range(1, len(ohlcv)):
        h = float(ohlcv[i][2]); l = float(ohlcv[i][3]); c_prev = float(prev_close)
        tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
        trs.append(tr)
        prev_close = float(ohlcv[i][4])
    if len(trs) < length:
        return None
    atr = sum(trs[:length]) / float(length)
    for tr in trs[length:]:
        atr = (atr * (length - 1) + tr) / float(length)
    return float(atr)


# ----------------- OKNO QUICK PAPER TRADE -----------------

class QuickPaperTrade(tk.Toplevel):
    _last_instance: "QuickPaperTrade" = None  # referencja do ostatniego okna (dla podbijania komunikatów)

    def __init__(self, app: TradingGUI):
        super().__init__(app.root)
        self.title("Quick Paper Trade")
        self.app = app

        # DB
        self.db = DatabaseManager("sqlite+aiosqlite:///trading.db")
        self.db.sync.init_db()

        # Stan symulatora
        self.open_limit_orders: List[Dict[str, Any]] = []
        self.levels: Dict[LevelsKey, Dict[str, Any]] = {}
        self.watch_symbols: Set[str] = set()

        # „Zawsze na wierzchu”
        self.always_on_top_var = tk.IntVar(value=1)

        self._build_ui()

        # Zawsze-na-wierzchu: ustaw po starcie i podbij okno
        self.after(100, self.lift)
        self.after(150, self._apply_topmost_from_var)

        self._engine_running = True
        self._engine_tick()

        QuickPaperTrade._last_instance = self

    # ---------- Pomoc: topmost i messagebox ----------

    def _apply_topmost_from_var(self):
        try:
            on = bool(self.always_on_top_var.get())
            self.attributes("-topmost", on)
            if on:
                self.lift()
        except Exception:
            pass

    def _with_topmost_msg(self, kind: str, title: str, text: str):
        """Pokazuje messagebox modalnie względem TEGO okna i pilnuje topmost."""
        try:
            prev = bool(self.attributes("-topmost"))
        except Exception:
            prev = False
        try:
            self.attributes("-topmost", True)
            self.lift()
            if kind == "error":
                messagebox.showerror(title, text, parent=self)
            elif kind == "warning":
                messagebox.showwarning(title, text, parent=self)
            else:
                messagebox.showinfo(title, text, parent=self)
        finally:
            # przywróć do stanu wg przełącznika lub poprzedniego
            target = bool(self.always_on_top_var.get()) or prev
            try:
                self.attributes("-topmost", target)
            except Exception:
                pass

    def _error(self, title: str, text: str):   self._with_topmost_msg("error", title, text)
    def _warning(self, title: str, text: str): self._with_topmost_msg("warning", title, text)
    def _info(self, title: str, text: str):    self._with_topmost_msg("info", title, text)

    # ---------- UI ----------

    def _build_ui(self):
        main = ttk.Notebook(self)
        main.pack(fill="both", expand=True)

        # --- Zakładka: Trade (Market/LIMIT + Risk + Futures + ATR + Kierunek) ---
        tab_trade = ttk.Frame(main); main.add(tab_trade, text="Trade")

        # Górny pasek: symbol + kwota + market
        snapshot = snapshot_from_app(self.app)
        capital_default, risk_pct_default, portfolio_pct_default, notional_default = _derive_risk_defaults(snapshot)

        frm_top = ttk.Frame(tab_trade); frm_top.pack(fill="x", padx=8, pady=6)
        ttk.Label(frm_top, text="Symbol:").grid(row=0, column=0, sticky="w")
        default_symbol = getattr(self.app, "symbol_var", None)
        default_symbol = (default_symbol.get() if default_symbol else "BTC/USDT")
        self.symbol_var = tk.StringVar(value=default_symbol)
        ttk.Entry(frm_top, textvariable=self.symbol_var, width=18).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(frm_top, text="Kwota (USDT):").grid(row=0, column=2, sticky="w", padx=(12,0))
        self.notional_var = tk.StringVar(value=_format_notional(notional_default))
        ttk.Entry(frm_top, textvariable=self.notional_var, width=10).grid(row=0, column=3, sticky="w", padx=4)

        # Przełącznik „Zawsze na wierzchu”
        ttk.Checkbutton(frm_top, text="Zawsze na wierzchu", variable=self.always_on_top_var,
                        command=self._apply_topmost_from_var).grid(row=0, column=9, sticky="e", padx=(12,0))

        # Kierunek (tylko Futures)
        ttk.Label(frm_top, text="Kierunek:").grid(row=0, column=4, sticky="w", padx=(12,0))
        self.side_mode_var = tk.StringVar(value="LONG")
        self.rb_long = ttk.Radiobutton(frm_top, text="LONG", value="LONG", variable=self.side_mode_var)
        self.rb_short = ttk.Radiobutton(frm_top, text="SHORT", value="SHORT", variable=self.side_mode_var)
        self.rb_long.grid(row=0, column=5, sticky="w")
        self.rb_short.grid(row=0, column=6, sticky="w")

        ttk.Button(frm_top, text="Market BUY",  command=self._on_mkt_buy).grid(row=0, column=7, padx=(12,4))
        ttk.Button(frm_top, text="Market SELL", command=self._on_mkt_sell).grid(row=0, column=8, padx=4)

        risk_hint = _build_hint(snapshot)
        if risk_hint:
            ttk.Label(frm_top, text=risk_hint, foreground="gray25").grid(
                row=1, column=0, columnspan=9, sticky="w", pady=(6, 0)
            )

        # Risk sizing
        frm_risk = ttk.LabelFrame(tab_trade, text="Risk sizing (paper)")
        frm_risk.pack(fill="x", padx=8, pady=(0,6))

        ttk.Label(frm_risk, text="Kapitał (USDT):").grid(row=0, column=0, sticky="w")
        self.capital_var = tk.StringVar(
            value=_format_decimal(capital_default, decimals=2, fallback=str(int(DEFAULT_CAPITAL_USDT)))
        )
        ttk.Entry(frm_risk, textvariable=self.capital_var, width=10).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(frm_risk, text="Ryzyko % na trade:").grid(row=0, column=2, sticky="w", padx=(12,0))
        self.risk_pct_var = tk.StringVar(
            value=_format_decimal(risk_pct_default, decimals=2, fallback=str(DEFAULT_RISK_PCT))
        )
        ttk.Entry(frm_risk, textvariable=self.risk_pct_var, width=8).grid(row=0, column=3, sticky="w", padx=4)

        # FUTURES / LEVERAGE
        self.futures_var = tk.IntVar(value=1)
        chk = ttk.Checkbutton(frm_risk, text="Futures / Leverage", variable=self.futures_var, command=self._toggle_futures_fields)
        chk.grid(row=0, column=4, sticky="w", padx=(12,0))

        ttk.Label(frm_risk, text="Dźwignia (x):").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.leverage_var = tk.StringVar(value="10")
        self.ent_leverage = ttk.Entry(frm_risk, textvariable=self.leverage_var, width=8)
        self.ent_leverage.grid(row=1, column=1, sticky="w", padx=4, pady=(6,0))

        ttk.Label(frm_risk, text="Max margin % kapitału:").grid(row=1, column=2, sticky="w", padx=(12,0), pady=(6,0))
        self.max_margin_pct_var = tk.StringVar(
            value=_format_decimal(portfolio_pct_default, decimals=2, fallback=str(DEFAULT_PORTFOLIO_PCT))
        )
        self.ent_max_margin = ttk.Entry(frm_risk, textvariable=self.max_margin_pct_var, width=8)
        self.ent_max_margin.grid(row=1, column=3, sticky="w", padx=4, pady=(6,0))

        ttk.Label(frm_risk, text="Max wielkość pozycji % kapitału:").grid(row=1, column=4, sticky="w", padx=(12,0), pady=(6,0))
        self.max_notional_pct_var = tk.StringVar(
            value=_format_decimal(portfolio_pct_default, decimals=2, fallback=str(DEFAULT_PORTFOLIO_PCT))
        )
        self.ent_max_notional = ttk.Entry(frm_risk, textvariable=self.max_notional_pct_var, width=8)
        self.ent_max_notional.grid(row=1, column=5, sticky="w", padx=4, pady=(6,0))

        ttk.Button(frm_risk, text="Wylicz kwotę z ryzyka", command=self._on_calc_risk_notional).grid(row=1, column=6, padx=(12,4), pady=(6,0))
        ttk.Button(frm_risk, text="Market (z ryzyka)", command=self._on_market_by_risk).grid(row=1, column=7, padx=4, pady=(6,0))

        # domyślnie futures ON → SHORT dostępny; przy SPOT – SHORT blokujemy
        self._toggle_futures_fields()

        # ATR-aware panel
        frm_atr = ttk.LabelFrame(tab_trade, text="ATR-aware (jak w Cryptohopperze)")
        frm_atr.pack(fill="x", padx=8, pady=(0,6))

        self.use_atr_var = tk.IntVar(value=1)
        ttk.Checkbutton(frm_atr, text="Użyj ATR do SL/TP/Trailing i sizingu", variable=self.use_atr_var, command=self._toggle_atr_fields).grid(row=0, column=0, sticky="w")

        ttk.Label(frm_atr, text="ATR length:").grid(row=0, column=1, sticky="w", padx=(12,0))
        self.atr_len_var = tk.StringVar(value=str(DEFAULT_ATR_LEN))
        self.ent_atr_len = ttk.Entry(frm_atr, textvariable=self.atr_len_var, width=8)
        self.ent_atr_len.grid(row=0, column=2, sticky="w", padx=4)

        ttk.Label(frm_atr, text="Interwał ATR:").grid(row=0, column=3, sticky="w", padx=(12,0))
        self.atr_tf_var = tk.StringVar(value=DEFAULT_ATR_TF)
        self.cmb_atr_tf = ttk.Combobox(frm_atr, textvariable=self.atr_tf_var, values=TIMEFRAMES, state="readonly", width=6)
        self.cmb_atr_tf.grid(row=0, column=4, sticky="w", padx=4)

        ttk.Label(frm_atr, text="SL k×ATR:").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.k_sl_var = tk.StringVar(value="1.5")
        self.ent_k_sl = ttk.Entry(frm_atr, textvariable=self.k_sl_var, width=8)
        self.ent_k_sl.grid(row=1, column=1, sticky="w", padx=4, pady=(6,0))

        ttk.Label(frm_atr, text="TP1/TP2/TP3 k×ATR:").grid(row=1, column=2, sticky="w", padx=(12,0), pady=(6,0))
        self.k_tp1_var = tk.StringVar(value="1.0")
        self.k_tp2_var = tk.StringVar(value="2.0")
        self.k_tp3_var = tk.StringVar(value="3.0")
        self.ent_k_tp1 = ttk.Entry(frm_atr, textvariable=self.k_tp1_var, width=6); self.ent_k_tp1.grid(row=1, column=3, sticky="w", padx=2, pady=(6,0))
        self.ent_k_tp2 = ttk.Entry(frm_atr, textvariable=self.k_tp2_var, width=6); self.ent_k_tp2.grid(row=1, column=4, sticky="w", padx=2, pady=(6,0))
        self.ent_k_tp3 = ttk.Entry(frm_atr, textvariable=self.k_tp3_var, width=6); self.ent_k_tp3.grid(row=1, column=5, sticky="w", padx=2, pady=(6,0))

        ttk.Label(frm_atr, text="Trailing k×ATR:").grid(row=1, column=6, sticky="w", padx=(12,0), pady=(6,0))
        self.k_trail_var = tk.StringVar(value="1.0")
        self.ent_k_trail = ttk.Entry(frm_atr, textvariable=self.k_trail_var, width=8)
        self.ent_k_trail.grid(row=1, column=7, sticky="w", padx=4, pady=(6,0))

        ttk.Button(frm_atr, text="Wypełnij z ATR (jak CH)", command=self._apply_atr_levels).grid(row=0, column=5, padx=(12,4))
        ttk.Button(frm_atr, text="Policz ryzyko (ATR)", command=self._on_calc_risk_notional).grid(row=0, column=6, padx=4)

        # LIMIT sekcja
        frm_lim = ttk.Frame(tab_trade); frm_lim.pack(fill="x", padx=8, pady=(6,6))
        ttk.Label(frm_lim, text="LIMIT cena:").grid(row=0, column=0, sticky="w")
        self.limit_price_var = tk.StringVar(value="")
        ttk.Entry(frm_lim, textvariable=self.limit_price_var, width=12).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(frm_lim, text="Ilość (opcjonalnie, puste = z Kwoty):").grid(row=0, column=2, sticky="w", padx=(12,0))
        self.limit_qty_var = tk.StringVar(value="")
        ttk.Entry(frm_lim, textvariable=self.limit_qty_var, width=14).grid(row=0, column=3, sticky="w", padx=4)

        ttk.Button(frm_lim, text="Place LIMIT BUY",  command=self._on_limit_buy).grid(row=0, column=4, padx=(12,4))
        ttk.Button(frm_lim, text="Place LIMIT SELL", command=self._on_limit_sell).grid(row=0, column=5, padx=4)

        ttk.Label(tab_trade, text="Otwarte zlecenia LIMIT (paper)").pack(anchor="w", padx=8)
        cols = ("oid","symbol","side","intent","qty","limit_price","status")
        self.tree_lims = ttk.Treeview(tab_trade, columns=cols, show="headings", height=7)
        for c,w in (("oid",70),("symbol",110),("side",70),("intent",130),("qty",110),("limit_price",110),("status",90)):
            self.tree_lims.heading(c, text=c.upper()); self.tree_lims.column(c, width=w, anchor="center")
        self.tree_lims.pack(fill="x", padx=8, pady=(2,6))

        frm_lbot = ttk.Frame(tab_trade); frm_lbot.pack(fill="x", padx=8, pady=(0,8))
        ttk.Button(frm_lbot, text="Cancel selected LIMIT", command=self._cancel_selected_limit).pack(side="right")
        ttk.Button(frm_lbot, text="Odśwież", command=self._refresh_views).pack(side="right", padx=(0,8))

        # --- Zakładka: Stops / Targets ---
        tab_sl = ttk.Frame(main); main.add(tab_sl, text="Stops / Targets")

        # SL/TP twarde + SL % pomocniczo
        frm_sl = ttk.LabelFrame(tab_sl, text="Twardy SL / TP (ceny) + SL% dla sizingu")
        frm_sl.pack(fill="x", padx=8, pady=6)

        ttk.Label(frm_sl, text="Symbol:").grid(row=0, column=0, sticky="w")
        self.sl_symbol_var = tk.StringVar(value=self.symbol_var.get())
        ttk.Entry(frm_sl, textvariable=self.sl_symbol_var, width=18).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(frm_sl, text="Stop Loss (cena):").grid(row=0, column=2, sticky="w", padx=(12,0))
        self.sl_price_var = tk.StringVar(value="")
        self.ent_sl_price = ttk.Entry(frm_sl, textvariable=self.sl_price_var, width=12)
        self.ent_sl_price.grid(row=0, column=3, sticky="w", padx=4)

        ttk.Label(frm_sl, text="SL % (jeśli wolisz procent):").grid(row=0, column=4, sticky="w", padx=(12,0))
        self.sl_pct_var = tk.StringVar(value="")
        self.ent_sl_pct = ttk.Entry(frm_sl, textvariable=self.sl_pct_var, width=8)
        self.ent_sl_pct.grid(row=0, column=5, sticky="w", padx=4)

        ttk.Label(frm_sl, text="Take Profit (cena):").grid(row=0, column=6, sticky="w", padx=(12,0))
        self.tp_price_var = tk.StringVar(value="")
        self.ent_tp_price = ttk.Entry(frm_sl, textvariable=self.tp_price_var, width=12)
        self.ent_tp_price.grid(row=0, column=7, sticky="w", padx=4)

        ttk.Button(frm_sl, text="Ustaw SL/TP",   command=self._set_sl_tp).grid(row=0, column=8, padx=(12,4))
        ttk.Button(frm_sl, text="Wyczyść SL/TP", command=self._clear_sl_tp).grid(row=0, column=9, padx=4)

        # Partial TP (TP1/TP2/TP3) – w % od bazy
        frm_ptp = ttk.LabelFrame(tab_sl, text="Partial TP (od ceny bazowej – % dodatnie)")
        frm_ptp.pack(fill="x", padx=8, pady=6)

        # PRESET udziałów TP
        ttk.Label(frm_ptp, text="Preset udziałów:").grid(row=0, column=0, sticky="w")
        self.tp_portion_preset_var = tk.StringVar(value="Zbalansowany (33/33/34)")
        self.cmb_tp_preset = ttk.Combobox(frm_ptp, textvariable=self.tp_portion_preset_var,
                                          values=list(TP_PORTION_PRESETS.keys()), state="readonly", width=24)
        self.cmb_tp_preset.grid(row=0, column=1, sticky="w", padx=4)
        ttk.Button(frm_ptp, text="Zastosuj preset udziałów", command=self._apply_tp_portion_preset).grid(row=0, column=2, sticky="w", padx=(8,4))

        ttk.Label(frm_ptp, text="TP1 %:").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.tp1_pct_var = tk.StringVar(value="")
        ttk.Entry(frm_ptp, textvariable=self.tp1_pct_var, width=8).grid(row=1, column=1, sticky="w", padx=4, pady=(6,0))
        ttk.Label(frm_ptp, text="Udział TP1 (% pozycji):").grid(row=1, column=2, sticky="w", padx=(12,0), pady=(6,0))
        self.tp1_portion_pct_var = tk.StringVar(value="33")
        ttk.Entry(frm_ptp, textvariable=self.tp1_portion_pct_var, width=8).grid(row=1, column=3, sticky="w", padx=4, pady=(6,0))

        ttk.Label(frm_ptp, text="TP2 %:").grid(row=1, column=4, sticky="w", padx=(12,0), pady=(6,0))
        self.tp2_pct_var = tk.StringVar(value="")
        ttk.Entry(frm_ptp, textvariable=self.tp2_pct_var, width=8).grid(row=1, column=5, sticky="w", padx=4, pady=(6,0))
        ttk.Label(frm_ptp, text="Udział TP2 (% pozycji):").grid(row=1, column=6, sticky="w", padx=(12,0), pady=(6,0))
        self.tp2_portion_pct_var = tk.StringVar(value="33")
        ttk.Entry(frm_ptp, textvariable=self.tp2_portion_pct_var, width=8).grid(row=1, column=7, sticky="w", padx=4, pady=(6,0))

        ttk.Label(frm_ptp, text="TP3 %:").grid(row=1, column=8, sticky="w", padx=(12,0), pady=(6,0))
        self.tp3_pct_var = tk.StringVar(value="")
        ttk.Entry(frm_ptp, textvariable=self.tp3_pct_var, width=8).grid(row=1, column=9, sticky="w", padx=4, pady=(6,0))
        ttk.Label(frm_ptp, text="Udział TP3 (% pozycji):").grid(row=1, column=10, sticky="w", padx=(12,0), pady=(6,0))
        self.tp3_portion_pct_var = tk.StringVar(value="34")
        ttk.Entry(frm_ptp, textvariable=self.tp3_portion_pct_var, width=8).grid(row=1, column=11, sticky="w", padx=4, pady=(6,0))

        ttk.Button(frm_ptp, text="Ustaw Partial TP", command=self._set_partial_tp).grid(row=1, column=12, padx=(12,4), pady=(6,0))
        ttk.Button(frm_ptp, text="Wyczyść Partial",  command=self._clear_partial_tp).grid(row=1, column=13, padx=4, pady=(6,0))

        # Trailing Stop
        frm_tr = ttk.LabelFrame(tab_sl, text="Trailing Stop (od ceny bazowej – % dodatnie)")
        frm_tr.pack(fill="x", padx=8, pady=6)
        ttk.Label(frm_tr, text="Aktywacja po zysku %:").grid(row=0, column=0, sticky="w")
        self.tr_activate_pct_var = tk.StringVar(value="")
        ttk.Entry(frm_tr, textvariable=self.tr_activate_pct_var, width=8).grid(row=0, column=1, sticky="w", padx=4)
        ttk.Label(frm_tr, text="Szerokość traila %:").grid(row=0, column=2, sticky="w", padx=(12,0))
        self.tr_trail_pct_var = tk.StringVar(value="")
        ttk.Entry(frm_tr, textvariable=self.tr_trail_pct_var, width=8).grid(row=0, column=3, sticky="w", padx=4)
        ttk.Button(frm_tr, text="Ustaw Trailing",   command=self._set_trailing).grid(row=0, column=4, padx=(12,4))
        ttk.Button(frm_tr, text="Wyczyść Trailing", command=self._clear_trailing).grid(row=0, column=5, padx=4)

        self.lbl_sl_info = ttk.Label(tab_sl, text="Brak aktywnych poziomów.")
        self.lbl_sl_info.pack(anchor="w", padx=8, pady=(4,8))

        # --- Pozycje/Transakcje ---
        tab_pos = ttk.Frame(main); main.add(tab_pos, text="Pozycje/Transakcje")
        ttk.Label(tab_pos, text="Pozycje (paper)").pack(anchor="w", padx=8, pady=(8,0))
        self.tree_pos = ttk.Treeview(tab_pos, columns=("symbol","side","qty","avg_price","unrl_pnl"), show="headings", height=7)
        for c,w in (("symbol",120),("side",80),("qty",120),("avg_price",120),("unrl_pnl",120)):
            self.tree_pos.heading(c, text=c.upper()); self.tree_pos.column(c, width=w, anchor="center")
        self.tree_pos.pack(fill="x", padx=8, pady=(2,8))

        ttk.Label(tab_pos, text="Trades (ostatnie 10, paper)").pack(anchor="w", padx=8)
        self.tree_tr = ttk.Treeview(tab_pos, columns=("id","time","symbol","side","qty","price","fee"), show="headings", height=10)
        for c,w in (("id",60),("time",160),("symbol",120),("side",80),("qty",100),("price",110),("fee",100)):
            self.tree_tr.heading(c, text=c.upper()); self.tree_tr.column(c, width=w, anchor="center")
        self.tree_tr.pack(fill="both", expand=True, padx=8, pady=(2,8))

        # Synchronizacja symbolu
        def _sync_symbols(*_):
            new_sym = (self.symbol_var.get() or "").strip().upper()
            if new_sym:
                self.sl_symbol_var.set(new_sym)
        self.symbol_var.trace_add("write", _sync_symbols)

        self._toggle_atr_fields()
        self._refresh_views()
        self._refresh_sl_label()

    # ---------- MARKET ----------

    def _on_mkt_buy(self):  self._market_click("BUY")
    def _on_mkt_sell(self): self._market_click("SELL")

    def _market_click(self, click_side: str):
        """
        Mapowanie przycisku na zamiar (intent) w zależności od kierunku i futures/spot:
        - LONG: BUY = OPEN_LONG, SELL = CLOSE_LONG
        - SHORT (tylko futures): SELL = OPEN_SHORT, BUY = CLOSE_SHORT
        """
        symbol = (self.symbol_var.get() or "BTC/USDT").strip().upper()
        mode_side = self.side_mode_var.get() or "LONG"
        if self.futures_var.get() != 1 and mode_side == "SHORT":
            self._error("Futures", "SHORT dostępny tylko w trybie Futures.")
            return

        if click_side == "BUY":
            intent = "OPEN_LONG" if mode_side == "LONG" else "CLOSE_SHORT"
        else:  # SELL
            intent = "CLOSE_LONG" if mode_side == "LONG" else "OPEN_SHORT"

        try:
            notional = float(self.notional_var.get())
        except Exception:
            self._error("Input", "Niepoprawna kwota USDT.")
            return
        if notional <= 0:
            self._error("Input", "Kwota musi być > 0.")
            return

        price = _get_last_price(self.app, symbol)
        if not price:
            self._error("Price", f"Brak ceny dla {symbol}. Najpierw 'Load Markets'.")
            return

        qty = _fmt_float(notional / price, 8)
        if qty <= 0:
            self._error("Calc", "Wyliczona ilość wyszła 0. Zwiększ kwotę / wybierz tańszy symbol.")
            return

        if intent == "CLOSE_LONG":
            pos = self._get_position(symbol, "LONG")
            if not pos or float(pos.get("quantity", 0.0)) <= 0:
                self._error("Paper", "Brak pozycji LONG do zamknięcia.")
                return
            qty = min(qty, float(pos["quantity"]))
        elif intent == "CLOSE_SHORT":
            pos = self._get_position(symbol, "SHORT")
            if not pos or float(pos.get("quantity", 0.0)) <= 0:
                self._error("Paper", "Brak pozycji SHORT do zamknięcia.")
                return
            qty = min(qty, float(pos["quantity"]))

        side_on_trade = "BUY" if click_side == "BUY" else "SELL"
        self._fill_now(symbol, side_on_trade, qty, price, intent=intent)
        self._refresh_views()

    # ---------- FUTURES UI przełączanie ----------

    def _toggle_futures_fields(self):
        is_fut = self.futures_var.get() == 1
        self.ent_leverage.configure(state=("!disabled" if is_fut else "disabled"))
        self.ent_max_margin.configure(state=("!disabled" if is_fut else "disabled"))
        self.ent_max_notional.configure(state=("disabled" if is_fut else "!disabled"))
        if is_fut:
            self.rb_short.configure(state="!disabled")
        else:
            self.side_mode_var.set("LONG")
            self.rb_short.configure(state="disabled")

    # ---------- ATR UI przełączanie ----------

    def _toggle_atr_fields(self):
        use = self.use_atr_var.get() == 1
        state = ("!disabled" if use else "disabled")
        for w in (self.ent_atr_len, self.cmb_atr_tf, self.ent_k_sl, self.ent_k_tp1, self.ent_k_tp2, self.ent_k_tp3, self.ent_k_trail):
            w.configure(state=state)
        self.ent_sl_pct.configure(state=("disabled" if use else "!disabled"))
        self.ent_sl_price.configure(state=("disabled" if use else "!disabled"))
        self.ent_tp_price.configure(state=("disabled" if use else "!disabled"))

    # ---------- ATR compute & apply ----------

    def _apply_atr_levels(self):
        symbol = (self.symbol_var.get() or "BTC/USDT").strip().upper()
        side_mode = self.side_mode_var.get() or "LONG"
        base = self._get_base_price(symbol, side_mode)
        if base is None:
            self._error("ATR", "Brak ceny bazowej (pozycja/cena). Najpierw 'Load Markets'.")
            return

        try:
            use = self.use_atr_var.get() == 1
            n = max(int(self.atr_len_var.get()), 2)
            tf = (self.atr_tf_var.get() or DEFAULT_ATR_TF)
            k_sl = max(float(self.k_sl_var.get()), 0.01)
            k_tp1 = max(float(self.k_tp1_var.get()), 0.0)
            k_tp2 = max(float(self.k_tp2_var.get()), 0.0)
            k_tp3 = max(float(self.k_tp3_var.get()), 0.0)
            k_trail = max(float(self.k_trail_var.get()), 0.0)
        except Exception:
            self._error("ATR", "Upewnij się, że ATR length/krotności są poprawne.")
            return

        if not use:
            self._warning("ATR", "Zaznacz 'Użyj ATR...', aby zastosować poziomy z ATR.")
            return

        ohlcv = _fetch_ohlcv(self.app, symbol, tf, limit=n+100)
        if not ohlcv or len(ohlcv) < n+1:
            self._error("ATR", f"Za mało świec ({len(ohlcv) if ohlcv else 0}). Zwiększ limit/zmień interwał.")
            return
        atr = _compute_atr(ohlcv, n)
        if not atr:
            self._error("ATR", "Nie udało się policzyć ATR.")
            return

        key: LevelsKey = (symbol, side_mode)
        d = self.levels.get(key, {})

        if side_mode == "LONG":
            sl_abs = _fmt_float(base - k_sl * atr, 8)
            if sl_abs >= base:
                sl_abs = _fmt_float(base * (1.0 - SAFETY_DELTA), 8)
            tp1 = _fmt_float(base + k_tp1 * atr, 8) if k_tp1 > 0 else None
            tp2 = _fmt_float(base + k_tp2 * atr, 8) if k_tp2 > 0 else None
            tp3 = _fmt_float(base + k_tp3 * atr, 8) if k_tp3 > 0 else None
        else:
            sl_abs = _fmt_float(base + k_sl * atr, 8)
            if sl_abs <= base:
                sl_abs = _fmt_float(base * (1.0 + SAFETY_DELTA), 8)
            tp1 = _fmt_float(base - k_tp1 * atr, 8) if k_tp1 > 0 else None
            tp2 = _fmt_float(base - k_tp2 * atr, 8) if k_tp2 > 0 else None
            tp3 = _fmt_float(base - k_tp3 * atr, 8) if k_tp3 > 0 else None

        trail_pct = (k_trail * atr / base) if k_trail > 0 else 0.0
        trailing = {"activate_pct": trail_pct, "trail_pct": trail_pct, "active": False,
                    "peak": None, "trough": None, "dir": side_mode} if trail_pct > 0 else None

        self.sl_symbol_var.set(symbol)
        self.sl_price_var.set(f"{sl_abs}")
        self.sl_pct_var.set("")
        self.tp_price_var.set("")

        def _tp(price): return {"price": _fmt_float(price, 8), "portion": 0.0, "done": False}
        d["sl"] = sl_abs
        d["tp"] = None
        d["tp1"] = _tp(tp1) if tp1 else None
        d["tp2"] = _tp(tp2) if tp2 else None
        d["tp3"] = _tp(tp3) if tp3 else None
        d["trailing"] = trailing
        d["atr_snapshot"] = {"len": n, "tf": tf, "atr": float(atr)}
        self.levels[key] = d
        self.watch_symbols.add(symbol)

        self._apply_tp_portion_preset_to_levels(symbol, side_mode, log=False)

        self._log(f"[Paper] [ATR][{side_mode}] len={n}, tf={tf}, ATR≈{atr:.8f}; SL={sl_abs}; "
                  f"TPs={d['tp1']},{d['tp2']},{d['tp3']}; trailing≈{trail_pct*100:.3f}% ({side_mode}). "
                  f"Preset udziałów='{self.tp_portion_preset_var.get()}'.", "INFO")
        self._refresh_sl_label()

    # ---------- RISK SIZING ----------

    def _on_calc_risk_notional(self):
        symbol = (self.sl_symbol_var.get() or self.symbol_var.get() or "BTC/USDT").strip().upper()
        side_mode = self.side_mode_var.get() or "LONG"
        base = self._get_base_price(symbol, side_mode)
        if base is None:
            self._error("Risk sizing", "Brak ceny bazowej (pozycja/cena). Najpierw 'Load Markets'.")
            return

        try:
            capital = float(self.capital_var.get())
            risk_pct = float(self.risk_pct_var.get()) / 100.0
        except Exception:
            self._error("Risk sizing", "Podaj poprawny Kapitał i Ryzyko %.")
            return
        if capital <= 0 or risk_pct <= 0:
            self._error("Risk sizing", "Kapitał i Ryzyko muszą być > 0%.")
            return

        use_atr = self.use_atr_var.get() == 1
        sl_pct: Optional[float] = None

        if use_atr:
            try:
                n = max(int(self.atr_len_var.get()), 2)
                tf = (self.atr_tf_var.get() or DEFAULT_ATR_TF)
                k_sl = max(float(self.k_sl_var.get()), 0.01)
            except Exception:
                self._error("ATR", "Upewnij się, że ATR length i k_sl są poprawne.")
                return
            ohlcv = _fetch_ohlcv(self.app, symbol, tf, limit=n+100)
            if not ohlcv or len(ohlcv) < n+1:
                self._error("ATR", f"Za mało świec ({len(ohlcv) if ohlcv else 0}). Zmień interwał.")
                return
            atr = _compute_atr(ohlcv, n)
            if not atr:
                self._error("ATR", "Nie udało się policzyć ATR.")
                return
            sl_pct = max((k_sl * atr) / base, MIN_SL_PCT)
            self._log(f"[Paper] [ATR][{side_mode}] base={base}, ATR≈{atr:.8f}, k_sl={k_sl} → SL%≈{sl_pct*100:.3f}%.", "INFO")
        else:
            sl_pct_text = (self.sl_pct_var.get() or "").strip()
            sl_price_text = (self.sl_price_var.get() or "").strip()
            if sl_pct_text:
                try:
                    sl_pct = float(sl_pct_text) / 100.0
                except Exception:
                    self._error("Risk sizing", "SL % niepoprawne.")
                    return
            elif sl_price_text:
                try:
                    sl_price = float(sl_price_text)
                    if side_mode == "LONG":
                        sl_pct = max((base - sl_price) / base, 0.0)
                    else:
                        sl_pct = max((sl_price - base) / base, 0.0)
                except Exception:
                    self._error("Risk sizing", "SL (cena) niepoprawna.")
                    return
            else:
                self._error("Risk sizing", "Podaj SL (jako % lub jako cenę), albo włącz ATR.")
                return

        if sl_pct < MIN_SL_PCT:
            self._log(f"[Paper] SL% zbyt mały ({sl_pct*100:.4f}%) – podbity do {MIN_SL_PCT*100:.2f}%.", "ERROR")
            sl_pct = MIN_SL_PCT

        risk_usdt = capital * risk_pct
        req_notional = risk_usdt / sl_pct

        if self.futures_var.get() == 1:
            try:
                lev = max(float(self.leverage_var.get()), 1.0)
                max_margin_pct = max(float(self.max_margin_pct_var.get()) / 100.0, 0.0)
            except Exception:
                self._error("Futures", "Podaj poprawną dźwignię i Max margin %.") 
                return
            max_margin_pct = min(max_margin_pct, 1.0)
            req_margin = req_notional / lev
            margin_cap = capital * max_margin_pct
            notional_cap = margin_cap * lev
            if req_margin > margin_cap:
                self._log(f"[Paper] Kwota ograniczona marginem: wymagany margin~{req_margin:.2f} USDT > limit {margin_cap:.2f} USDT "
                          f"(max {max_margin_pct*100:.0f}% kapitału, lev x{lev}).", "ERROR")
            notional = min(req_notional, notional_cap)
        else:
            try:
                max_notional_pct = max(float(self.max_notional_pct_var.get()) / 100.0, 0.0)
            except Exception:
                self._error("Spot cap", "Podaj poprawny Max wielkość pozycji %.") 
                return
            max_notional_pct = min(max_notional_pct, 1.0)
            notional_cap = capital * max_notional_pct
            if req_notional > notional_cap:
                self._log(f"[Paper] Kwota ograniczona do limitu pozycji {notional_cap:.2f} USDT (max {max_notional_pct*100:.0f}% kapitału). "
                          f"Wymagane~{req_notional:.2f} USDT.", "ERROR")
            notional = min(req_notional, notional_cap)

        self.notional_var.set(f"{_fmt_float(notional, 2)}")
        mode = "FUTURES" if self.futures_var.get() == 1 else "SPOT"
        self._log(f"[Paper] [{mode}][{side_mode}] Risk sizing: base={base}, SL%={sl_pct*100:.3f}%, Risk={risk_usdt:.2f} USDT → Kwota={_fmt_float(notional,2)} USDT", "INFO")

    def _on_market_by_risk(self):
        self._on_calc_risk_notional()
        side_mode = self.side_mode_var.get() or "LONG"
        if side_mode == "LONG":
            self._market_click("BUY")
        else:
            self._market_click("SELL")

    # ---------- LIMIT ----------

    def _on_limit_buy(self):  self._place_limit(click_side="BUY")
    def _on_limit_sell(self): self._place_limit(click_side="SELL")

    def _place_limit(self, click_side: str):
        symbol = (self.symbol_var.get() or "BTC/USDT").strip().upper()
        side_mode = self.side_mode_var.get() or "LONG"
        if self.futures_var.get() != 1 and side_mode == "SHORT":
            self._error("Futures", "SHORT dostępny tylko w trybie Futures.")
            return

        try:
            limit_price = float(self.limit_price_var.get())
        except Exception:
            self._error("LIMIT", "Podaj poprawną cenę LIMIT.")
            return
        if limit_price <= 0:
            self._error("LIMIT", "Cena LIMIT musi być > 0.")
            return

        qty: Optional[float] = None
        if (self.limit_qty_var.get() or "").strip():
            try: qty = float(self.limit_qty_var.get())
            except Exception:
                self._error("LIMIT", "Niepoprawna ilość.")
                return
        else:
            try:
                notional = float(self.notional_var.get())
            except Exception:
                self._error("LIMIT", "Niepoprawna kwota USDT.")
                return
            if notional <= 0:
                self._error("LIMIT", "Kwota musi być > 0.")
                return
            qty = notional / limit_price

        qty = _fmt_float(qty, 8)
        if qty <= 0:
            self._error("LIMIT", "Ilość wyszła 0. Zwiększ kwotę/ilość.")
            return

        if side_mode == "LONG":
            intent = "OPEN_LONG" if click_side == "BUY" else "CLOSE_LONG"
        else:
            intent = "OPEN_SHORT" if click_side == "SELL" else "CLOSE_SHORT"

        if intent == "CLOSE_LONG":
            pos = self._get_position(symbol, "LONG")
            if not pos or float(pos.get("quantity", 0.0)) <= 0:
                self._error("LIMIT", "Brak pozycji LONG do sprzedaży.")
                return
            qty = min(qty, float(pos["quantity"]))
        elif intent == "CLOSE_SHORT":
            pos = self._get_position(symbol, "SHORT")
            if not pos or float(pos.get("quantity", 0.0)) <= 0:
                self._error("LIMIT", "Brak pozycji SHORT do odkupienia.")
                return
            qty = min(qty, float(pos["quantity"]))

        last = _get_last_price(self.app, symbol)
        if last:
            dev = abs(limit_price - last) / last
            if dev > 0.5:
                self._log(f"[Paper] Ostrzeżenie: LIMIT {click_side} {symbol} @ {limit_price} daleko od rynku (last~{last}).", "ERROR")

        oid = self.db.sync.record_order({
            "symbol": symbol, "side": click_side, "type": "LIMIT",
            "quantity": qty, "price": limit_price, "mode": "paper",
            "client_order_id": None
        })
        try: self.db.sync.update_order_status(order_id=oid, status="OPEN")
        except Exception: pass

        self.open_limit_orders.append({
            "oid": oid, "symbol": symbol, "side": click_side,
            "intent": intent, "qty": qty, "limit_price": limit_price, "status": "OPEN"
        })
        self.watch_symbols.add(symbol)
        self._log(f"[Paper] LIMIT {click_side} {symbol} qty={qty} @ {limit_price} ({intent}, OPEN)", "INFO")
        self._refresh_views()

    # ---------- SL/TP – ręczne ----------

    def _set_sl_tp(self):
        symbol = (self.sl_symbol_var.get() or "").strip().upper()
        side_mode = self.side_mode_var.get() or "LONG"
        if not symbol:
            self._error("SL/TP", "Podaj symbol.")
            return

        base = self._get_base_price(symbol, side_mode)

        sl_price = (self.sl_price_var.get() or "").strip()
        sl_pct_text = (self.sl_pct_var.get() or "").strip()
        tp_text = (self.tp_price_var.get() or "").strip()

        sl_p: Optional[float] = None
        tp_p: Optional[float] = None

        if sl_pct_text and self.use_atr_var.get() == 0:
            if base is None:
                self._error("SL/TP", "Brak ceny bazowej do przeliczenia SL%. Najpierw 'Load Markets'.")
                return
            try:
                pct = max(float(sl_pct_text) / 100.0, MIN_SL_PCT)
            except Exception:
                self._error("SL/TP", "SL% niepoprawne.")
                return
            if side_mode == "LONG":
                sl_p = _fmt_float(base * (1.0 - pct), 8)
            else:
                sl_p = _fmt_float(base * (1.0 + pct), 8)
            self.sl_price_var.set(f"{sl_p}")

        if sl_price:
            try: sl_p2 = float(sl_price)
            except Exception:
                self._error("SL/TP", "SL (cena) niepoprawna.")
                return
            if sl_p is None:
                sl_p = sl_p2

        if tp_text and self.use_atr_var.get() == 0:
            try:
                tp_p2 = float(tp_text)
            except Exception:
                self._error("SL/TP", "TP (cena) niepoprawna.")
                return
            tp_p = tp_p2

        if base:
            if side_mode == "LONG":
                if sl_p is not None and sl_p >= base:
                    corr = _fmt_float(base * (1.0 - SAFETY_DELTA), 8)
                    self._log(f"[Paper] Korekta SL: {sl_p} ≥ baza({base}) → {corr}", "ERROR")
                    sl_p = corr
                if tp_p is not None and tp_p <= base:
                    corr = _fmt_float(base * (1.0 + SAFETY_DELTA), 8)
                    self._log(f"[Paper] Korekta TP: {tp_p} ≤ baza({base}) → {corr}", "ERROR")
                    tp_p = corr
            else:  # SHORT
                if sl_p is not None and sl_p <= base:
                    corr = _fmt_float(base * (1.0 + SAFETY_DELTA), 8)
                    self._log(f"[Paper] Korekta SL(SHORT): {sl_p} ≤ baza({base}) → {corr}", "ERROR")
                    sl_p = corr
                if tp_p is not None and tp_p >= base:
                    corr = _fmt_float(base * (1.0 - SAFETY_DELTA), 8)
                    self._log(f"[Paper] Korekta TP(SHORT): {tp_p} ≥ baza({base}) → {corr}", "ERROR")
                    tp_p = corr

        key: LevelsKey = (symbol, side_mode)
        d = self.levels.get(key, {})
        d["sl"] = sl_p
        d["tp"] = tp_p if self.use_atr_var.get() == 0 else None  # przy ATR TP klasyczny pomijamy
        self.levels[key] = d
        self.watch_symbols.add(symbol)

        self._log(f"[Paper] Ustawiono SL/TP dla {symbol} [{side_mode}]: SL={sl_p}, TP={d['tp']}", "INFO")
        self._refresh_sl_label()

    def _clear_sl_tp(self):
        symbol = (self.sl_symbol_var.get() or "").strip().upper()
        side_mode = self.side_mode_var.get() or "LONG"
        key: LevelsKey = (symbol, side_mode)
        d = self.levels.get(key)
        if d:
            d["sl"] = None; d["tp"] = None
            self._log(f"[Paper] Wyczyszczono SL/TP dla {symbol} [{side_mode}]", "INFO")
        self._refresh_sl_label()

    # ---------- PRESET udziałów TP ----------

    def _apply_tp_portion_preset(self):
        preset_name = self.tp_portion_preset_var.get()
        portions = TP_PORTION_PRESETS.get(preset_name)
        if not portions:
            self._error("Preset", "Nie znaleziono presetu.")
            return
        p1, p2, p3 = portions
        self.tp1_portion_pct_var.set(f"{int(round(p1*100))}")
        self.tp2_portion_pct_var.set(f"{int(round(p2*100))}")
        self.tp3_portion_pct_var.set(f"{int(round(p3*100))}")

        symbol = (self.sl_symbol_var.get() or self.symbol_var.get() or "BTC/USDT").strip().upper()
        side_mode = self.side_mode_var.get() or "LONG"
        self._apply_tp_portion_preset_to_levels(symbol, side_mode, log=True)

    def _apply_tp_portion_preset_to_levels(self, symbol: str, side_mode: str, *, log: bool):
        preset_name = self.tp_portion_preset_var.get()
        portions = TP_PORTION_PRESETS.get(preset_name)
        if not portions:
            return
        key: LevelsKey = (symbol, side_mode)
        d = self.levels.get(key)
        if not d:
            return
        p1, p2, p3 = portions
        if d.get("tp1"): d["tp1"]["portion"] = float(p1)
        if d.get("tp2"): d["tp2"]["portion"] = float(p2)
        if d.get("tp3"): d["tp3"]["portion"] = float(p3)
        if log:
            self._log(f"[Paper] Zastosowano preset udziałów '{preset_name}' dla {symbol} [{side_mode}].", "INFO")
        self._refresh_sl_label()

    # ---------- Partial TP (TP1/TP2/TP3) ----------

    def _set_partial_tp(self):
        symbol = (self.sl_symbol_var.get() or "").strip().upper()
        side_mode = self.side_mode_var.get() or "LONG"
        if not symbol:
            self._error("Partial TP", "Podaj symbol.")
            return

        base = self._get_base_price(symbol, side_mode)
        if base is None:
            self._error("Partial TP", "Brak ceny bazowej (pozycja/cena). Najpierw 'Load Markets'.")
            return

        def _to_pct(s: str) -> Optional[float]:
            return float(s)/100.0 if s.strip() else None

        tp1_pct = _to_pct(self.tp1_pct_var.get() or "")
        tp2_pct = _to_pct(self.tp2_pct_var.get() or "")
        tp3_pct = _to_pct(self.tp3_pct_var.get() or "")

        def _to_part(s: str) -> Optional[float]:
            return float(s)/100.0 if s.strip() else None

        tp1_portion = _to_part(self.tp1_portion_pct_var.get() or "")
        tp2_portion = _to_part(self.tp2_portion_pct_var.get() or "")
        tp3_portion = _to_part(self.tp3_portion_pct_var.get() or "")

        if tp1_pct is None and tp2_pct is None and tp3_pct is None:
            self._error("Partial TP", "Podaj co najmniej jeden TP% (TP1/TP2/TP3).")
            return

        sum_portions = sum([p or 0.0 for p in (tp1_portion, tp2_portion, tp3_portion)])
        if sum_portions > 1.0 + 1e-9:
            self._error("Partial TP", "Suma udziałów TP przekracza 100%. Zmniejsz udziały.")
            return

        key: LevelsKey = (symbol, side_mode)
        d = self.levels.get(key, {})

        def _price_from_pct(pct: float) -> float:
            if side_mode == "LONG":
                return _fmt_float(base * (1.0 + pct), 8)
            else:
                return _fmt_float(base * (1.0 - pct), 8)

        d["tp1"] = {"price": _price_from_pct(tp1_pct), "portion": tp1_portion or 0.0, "done": False} if tp1_pct is not None else None
        d["tp2"] = {"price": _price_from_pct(tp2_pct), "portion": tp2_portion or 0.0, "done": False} if tp2_pct is not None else None
        d["tp3"] = {"price": _price_from_pct(tp3_pct), "portion": tp3_portion or 0.0, "done": False} if tp3_pct is not None else None

        self.levels[key] = d
        self.watch_symbols.add(symbol)

        self._log(f"[Paper] Ustawiono Partial TP dla {symbol} [{side_mode}]: TP1={d['tp1']}, TP2={d['tp2']}, TP3={d['tp3']}", "INFO")
        self._refresh_sl_label()

    def _clear_partial_tp(self):
        symbol = (self.sl_symbol_var.get() or "").strip().upper()
        side_mode = self.side_mode_var.get() or "LONG"
        key: LevelsKey = (symbol, side_mode)
        d = self.levels.get(key)
        if d:
            d["tp1"] = None; d["tp2"] = None; d["tp3"] = None
            self._log(f"[Paper] Wyczyszczono Partial TP dla {symbol} [{side_mode}]", "INFO")
        self._refresh_sl_label()

    # ---------- Trailing ----------

    def _set_trailing(self):
        symbol = (self.sl_symbol_var.get() or "").strip().upper()
        side_mode = self.side_mode_var.get() or "LONG"
        if not symbol:
            self._error("Trailing", "Podaj symbol.")
            return
        try:
            act = float((self.tr_activate_pct_var.get() or "").strip())/100.0
            trail = float((self.tr_trail_pct_var.get() or "").strip())/100.0
        except Exception:
            self._error("Trailing", "Podaj poprawne wartości %.")
            return
        if act <= 0 or trail <= 0:
            self._error("Trailing", "Wartości muszą być > 0%.")
            return

        key: LevelsKey = (symbol, side_mode)
        d = self.levels.get(key, {})
        d["trailing"] = {"activate_pct": act, "trail_pct": trail, "active": False,
                         "peak": None, "trough": None, "dir": side_mode}
        self.levels[key] = d
        self.watch_symbols.add(symbol)

        self._log(f"[Paper] Ustawiono Trailing dla {symbol} [{side_mode}]: act={act*100:.2f}%, trail={trail*100:.2f}%", "INFO")
        self._refresh_sl_label()

    def _clear_trailing(self):
        symbol = (self.sl_symbol_var.get() or "").strip().upper()
        side_mode = self.side_mode_var.get() or "LONG"
        key: LevelsKey = (symbol, side_mode)
        d = self.levels.get(key)
        if d:
            d["trailing"] = None
            self._log(f"[Paper] Wyczyszczono Trailing dla {symbol} [{side_mode}]", "INFO")
        self._refresh_sl_label()

    # ---------- Silnik symulacji ----------

    def _engine_tick(self):
        if not self._engine_running:
            return
        try:
            self._process_engine()
        except Exception as e:
            self._log(f"[Paper] Engine exception: {e}", "ERROR")
        self.after(ENGINE_TICK_MS, self._engine_tick)

    def _process_engine(self):
        symbols = set(self.watch_symbols)
        cur = (self.symbol_var.get() or "").strip().upper()
        if cur: symbols.add(cur)

        for sym in list(symbols):
            price = _get_last_price(self.app, sym)
            if price is None:
                continue

            # LIMIT-y
            for od in list(self.open_limit_orders):
                if od["symbol"] != sym or od["status"] != "OPEN":
                    continue
                side_click, qty, lim, intent = od["side"], float(od["qty"]), float(od["limit_price"]), od["intent"]

                should_fill = False
                fill_price = price

                if intent == "OPEN_LONG":
                    if price <= lim:
                        should_fill, fill_price = True, min(price, lim)
                elif intent == "CLOSE_LONG":
                    pos = self._get_position(sym, "LONG")
                    pos_qty = float(pos["quantity"]) if pos else 0.0
                    if pos_qty <= 0:
                        self._cancel_limit(od, reason="brak LONG")
                        continue
                    qty = min(qty, pos_qty)
                    if price >= lim:
                        should_fill, fill_price = True, max(price, lim)
                elif intent == "OPEN_SHORT":
                    if price >= lim:
                        should_fill, fill_price = True, max(price, lim)
                elif intent == "CLOSE_SHORT":
                    pos = self._get_position(sym, "SHORT")
                    pos_qty = float(pos["quantity"]) if pos else 0.0
                    if pos_qty <= 0:
                        self._cancel_limit(od, reason="brak SHORT")
                        continue
                    qty = min(qty, pos_qty)
                    if price <= lim:
                        should_fill, fill_price = True, min(price, lim)

                if should_fill:
                    self._fill_limit_order(od, sym, side_click, qty, fill_price, intent=intent)
                    self.open_limit_orders.remove(od)

            # Poziomy SL/TP/Trailing dla LONG i SHORT niezależnie
            for side_mode in ("LONG", "SHORT"):
                st = self.levels.get((sym, side_mode)) or {}
                pos = self._get_position(sym, side_mode)
                pos_qty = float(pos["quantity"]) if pos else 0.0
                avg = float(pos["avg_price"]) if pos else None

                if pos_qty <= 0:
                    continue

                # HARD SL
                sl_p = st.get("sl")
                if sl_p is not None:
                    if side_mode == "LONG" and price <= float(sl_p):
                        self._log(f"[Paper] SL trigger {sym} [{side_mode}] @ {price} (SL={sl_p}) – zamknięcie.", "INFO")
                        self._fill_now(sym, "SELL", _fmt_float(pos_qty, 8), price, intent="CLOSE_LONG")
                        self._clear_all_for(sym, side_mode)
                        continue
                    if side_mode == "SHORT" and price >= float(sl_p):
                        self._log(f"[Paper] SL trigger {sym} [{side_mode}] @ {price} (SL={sl_p}) – zamknięcie.", "INFO")
                        self._fill_now(sym, "BUY", _fmt_float(pos_qty, 8), price, intent="CLOSE_SHORT")
                        self._clear_all_for(sym, side_mode)
                        continue

                # TRAILING
                tr = st.get("trailing")
                if tr:
                    act_pct, trail_pct = tr.get("activate_pct"), tr.get("trail_pct")
                    active = tr.get("active", False)
                    base = avg if avg else price
                    if side_mode == "LONG":
                        peak = tr.get("peak")
                        if not active and act_pct and base and price >= base * (1.0 + act_pct):
                            tr["active"], tr["peak"] = True, price
                            self._log(f"[Paper] Trailing aktywowany {sym} [LONG] @ {price}", "INFO")
                            active, peak = True, price
                        if active:
                            if peak is None or price > peak:
                                tr["peak"] = price; peak = price
                            stop_price = peak * (1.0 - float(trail_pct or 0.0))
                            if price <= stop_price:
                                self._log(f"[Paper] Trailing stop {sym} [LONG] @ {price} (stop~{_fmt_float(stop_price,8)}) – zamknięcie.", "INFO")
                                self._fill_now(sym, "SELL", _fmt_float(pos_qty, 8), price, intent="CLOSE_LONG")
                                self._clear_all_for(sym, side_mode)
                                continue
                    else:
                        trough = tr.get("trough")
                        if not active and act_pct and base and price <= base * (1.0 - act_pct):
                            tr["active"], tr["trough"] = True, price
                            self._log(f"[Paper] Trailing aktywowany {sym} [SHORT] @ {price}", "INFO")
                            active, trough = True, price
                        if active:
                            if trough is None or price < trough:
                                tr["trough"] = price; trough = price
                            stop_price = trough * (1.0 + float(trail_pct or 0.0))
                            if price >= stop_price:
                                self._log(f"[Paper] Trailing stop {sym} [SHORT] @ {price} (stop~{_fmt_float(stop_price,8)}) – zamknięcie.", "INFO")
                                self._fill_now(sym, "BUY", _fmt_float(pos_qty, 8), price, intent="CLOSE_SHORT")
                                self._clear_all_for(sym, side_mode)
                                continue

                # PARTIAL TP
                for key in ("tp1", "tp2", "tp3"):
                    tp = st.get(key)
                    if pos_qty <= 0 or not tp or tp.get("done"):
                        continue
                    tp_price = float(tp.get("price"))
                    trigger = (price >= tp_price) if side_mode == "LONG" else (price <= tp_price)
                    if trigger:
                        portion = float(tp.get("portion", 0.0))
                        qty_exec = _fmt_float(pos_qty * max(0.0, min(1.0, portion)), 8)
                        if qty_exec > 0:
                            side_click = "SELL" if side_mode == "LONG" else "BUY"
                            intent = "CLOSE_LONG" if side_mode == "LONG" else "CLOSE_SHORT"
                            self._log(f"[Paper] {key.upper()} trigger {sym} [{side_mode}] @ {price} – zamknięcie {qty_exec}.", "INFO")
                            self._fill_now(sym, side_click, qty_exec, price, intent=intent)
                            tp["done"] = True
                            pos = self._get_position(sym, side_mode)
                            pos_qty = float(pos["quantity"]) if pos else 0.0

                # HARD TP
                tp_p = st.get("tp")
                if pos_qty > 0 and tp_p is not None:
                    if (side_mode == "LONG" and price >= float(tp_p)) or (side_mode == "SHORT" and price <= float(tp_p)):
                        side_click = "SELL" if side_mode == "LONG" else "BUY"
                        intent = "CLOSE_LONG" if side_mode == "LONG" else "CLOSE_SHORT"
                        self._log(f"[Paper] TP trigger {sym} [{side_mode}] @ {price} (TP={tp_p}) – pełne wyjście.", "INFO")
                        self._fill_now(sym, side_click, _fmt_float(pos_qty, 8), price, intent=intent)
                        self._clear_all_for(sym, side_mode)
                        continue

            self._refresh_views(light=True)

    # ---------- Pomoc / DB ----------

    def _log(self, msg: str, level: str = "INFO"):
        try: self.app._log(msg, level)
        except Exception: print(msg)

    def _get_position(self, symbol: str, side_mode: str) -> Optional[Dict[str, Any]]:
        try: positions = self.db.sync.get_open_positions(mode="paper")
        except Exception: return None
        for p in positions:
            if p.get("symbol") == symbol and p.get("side") == side_mode:
                return p
        return None

    def _get_base_price(self, symbol: str, side_mode: str) -> Optional[float]:
        pos = self._get_position(symbol, side_mode)
        if pos and float(pos.get("quantity", 0.0)) > 0:
            return float(pos["avg_price"])
        return _get_last_price(self.app, symbol)

    def _set_order_status(self, order_id: int, status: str):
        try: self.db.sync.update_order_status(order_id=order_id, status=status)
        except Exception: pass

    def _clear_all_for(self, symbol: str, side_mode: str):
        key: LevelsKey = (symbol, side_mode)
        d = self.levels.get(key)
        if not d: return
        d["sl"] = None; d["tp"] = None
        d["tp1"] = None; d["tp2"] = None; d["tp3"] = None
        d["trailing"] = None
        self._refresh_sl_label()

    def _cancel_limit(self, od: Dict[str, Any], reason: str = ""):
        self._set_order_status(od["oid"], "CANCELED")
        od["status"] = "CANCELED"
        self.open_limit_orders.remove(od)
        msg = f"[Paper] LIMIT {od.get('side')} {od.get('symbol')} anulowany ({reason})."
        self._log(msg, "ERROR")

    # ---------- Wypełnienia ----------

    def _fill_now(self, symbol: str, side: str, qty: float, price: float, *, intent: str):
        qty = _fmt_float(qty, 8); price = float(price)
        fee = _fmt_float(qty * price * FEE_RATE, 8)

        oid = self.db.sync.record_order({
            "symbol": symbol, "side": side, "type": "MARKET",
            "quantity": qty, "price": None, "mode": "paper",
            "client_order_id": intent
        })
        self._set_order_status(oid, "FILLED")

        self.db.sync.record_trade({
            "symbol": symbol, "side": side, "quantity": qty,
            "price": price, "fee": fee, "order_id": oid, "mode": "paper"
        })

        self._apply_position_change(symbol, qty, price, intent=intent)
        self._log(f"[Paper] FILLED {side} {symbol} qty={qty} @ {price} (fee≈{fee}) [{intent}]", "INFO")

    def _fill_limit_order(self, od: Dict[str, Any], symbol: str, side: str, qty: float, fill_price: float, *, intent: str):
        qty = _fmt_float(qty, 8); fill_price = float(fill_price)
        fee = _fmt_float(qty * fill_price * FEE_RATE, 8)
        self._set_order_status(od["oid"], "FILLED")

        self.db.sync.record_trade({
            "symbol": symbol, "side": side, "quantity": qty,
            "price": fill_price, "fee": fee, "order_id": od["oid"], "mode": "paper"
        })

        self._apply_position_change(symbol, qty, fill_price, intent=intent)
        self._log(f"[Paper] FILLED LIMIT {side} {symbol} qty={qty} @ {fill_price} [{intent}]", "INFO")

    def _apply_position_change(self, symbol: str, qty: float, price: float, *, intent: str):
        if intent in ("OPEN_LONG", "CLOSE_LONG"):
            side_mode = "LONG"
        else:
            side_mode = "SHORT"

        pos = self._get_position(symbol, side_mode)
        if intent in ("OPEN_LONG", "OPEN_SHORT"):
            if pos:
                old_qty = float(pos["quantity"]); old_avg = float(pos["avg_price"])
                new_qty = _fmt_float(old_qty + qty, 8)
                new_avg = _fmt_float((old_qty*old_avg + qty*price) / new_qty, 8) if new_qty > 0 else 0.0
            else:
                new_qty, new_avg = qty, price
            self.db.sync.upsert_position({
                "symbol": symbol, "side": side_mode, "quantity": new_qty,
                "avg_price": new_avg, "unrealized_pnl": 0.0, "mode": "paper"
            })
        else:
            if pos:
                old_qty = float(pos["quantity"])
                new_qty = _fmt_float(old_qty - qty, 8)
                if new_qty < 0: new_qty = 0.0
                new_avg = float(pos["avg_price"]) if new_qty > 0 else 0.0
                self.db.sync.upsert_position({
                    "symbol": symbol, "side": side_mode, "quantity": new_qty,
                    "avg_price": new_avg, "unrealized_pnl": 0.0, "mode": "paper"
                })

    # ---------- Widoki ----------

    def _refresh_views(self, light: bool = False):
        try: pos = self.db.sync.get_open_positions(mode="paper")
        except Exception: pos = []
        if not light:
            for i in self.tree_pos.get_children(): self.tree_pos.delete(i)
            for p in pos:
                self.tree_pos.insert("", "end", values=(
                    p.get("symbol"), p.get("side"),
                    _fmt_float(p.get("quantity", 0.0)), _fmt_float(p.get("avg_price", 0.0)),
                    _fmt_float(p.get("unrealized_pnl", 0.0))
                ))
            try: tr = self.db.sync.fetch_trades(mode="paper")
            except Exception: tr = []
            tr = sorted(tr, key=lambda x: x.get("id", 0), reverse=True)[:10]
            for i in self.tree_tr.get_children(): self.tree_tr.delete(i)
            for t in tr:
                self.tree_tr.insert("", "end", values=(
                    t.get("id"), t.get("ts"), t.get("symbol"), t.get("side"),
                    _fmt_float(t.get("quantity", 0.0)), _fmt_float(t.get("price", 0.0)),
                    _fmt_float(t.get("fee", 0.0))
                ))
            for i in self.tree_lims.get_children(): self.tree_lims.delete(i)
            for od in self.open_limit_orders:
                self.tree_lims.insert("", "end", values=(
                    od.get("oid"), od.get("symbol"), od.get("side"),
                    od.get("intent"),
                    _fmt_float(od.get("qty", 0.0)), _fmt_float(od.get("limit_price", 0.0)),
                    od.get("status", "OPEN")
                ))
        self._refresh_sl_label()

    def _refresh_sl_label(self):
        symbol = (self.sl_symbol_var.get() or "").strip().upper()
        side_mode = self.side_mode_var.get() or "LONG"
        d = self.levels.get((symbol, side_mode))
        if not d:
            self.lbl_sl_info.config(text=f"Brak aktywnych poziomów dla {symbol} [{side_mode}]."); return
        parts = []
        if d.get("sl") is not None: parts.append(f"SL={d.get('sl')}")
        if d.get("tp") is not None: parts.append(f"TP={d.get('tp')}")
        for key in ("tp1","tp2","tp3"):
            tp = d.get(key)
            if tp:
                parts.append(f"{key.upper()}={tp['price']} ({int(tp.get('portion',0)*100)}%) {'✓' if tp.get('done') else ''}")
        tr = d.get("trailing")
        if tr:
            status = "ON" if tr.get("active") else "ARMED"
            parts.append(f"TRAIL {status} [{tr.get('dir')}] (act={tr.get('activate_pct',0)*100:.2f}%, tr={tr.get('trail_pct',0)*100:.2f}%)")
        atrs = d.get("atr_snapshot")
        if atrs:
            parts.append(f"ATR(len={atrs.get('len')}, tf={atrs.get('tf')}, ≈{_fmt_float(atrs.get('atr',0.0),6)})")
        self.lbl_sl_info.config(text="; ".join(parts) if parts else f"Brak aktywnych poziomów dla {symbol} [{side_mode}].")

    # ---------- Anulowanie LIMIT ----------

    def _cancel_selected_limit(self):
        sel = self.tree_lims.selection()
        if not sel:
            self._warning("Cancel", "Zaznacz zlecenie LIMIT do anulowania.")
            return
        item = self.tree_lims.item(sel[0]); oid = item["values"][0]
        for od in list(self.open_limit_orders):
            if od.get("oid") == oid and od.get("status") == "OPEN":
                od["status"] = "CANCELED"
                self._set_order_status(od["oid"], "CANCELED")
                self.open_limit_orders.remove(od)
                self._log(f"[Paper] Canceled LIMIT order {oid}", "INFO")
                break
        self._refresh_views()

    # ---------- zamknięcie ----------

    def destroy(self):
        self._engine_running = False
        try:
            return super().destroy()
        finally:
            pass


# --- Executor stabilizujący komunikaty z mostka GUI -----------------

def _paper_trade_executor(gui: TradingGUI, symbol: str, side: str, mkt_price: float) -> None:
    try:
        gui.default_trade_executor(symbol, side, mkt_price)
    except Exception as exc:
        tb = traceback.format_exc()
        try:
            gui._log(f"AI Manager: failed in _bridge_execute_trade: {exc}\n{tb}", "ERROR")
        except Exception:
            print(f"[ERROR] _bridge_execute_trade: {exc}\n{tb}")
        # Jeśli istnieje okno QuickPaperTrade – użyj jego „topmost” i pokaż komunikat nad nim
        qp = getattr(QuickPaperTrade, "_last_instance", None)
        if isinstance(qp, QuickPaperTrade):
            qp._error("Paper", f"Nieoczekiwany błąd: {exc}\n{tb}")
        else:
            # awaryjnie nad głównym oknem
            try:
                gui.root.lift()
                gui.root.attributes("-topmost", True)
                messagebox.showerror(
                    "Paper", f"Nieoczekiwany błąd: {exc}\n{tb}", parent=gui.root
                )
            finally:
                try:
                    gui.root.attributes("-topmost", False)
                except Exception:
                    pass


# --- Start GUI + auto-okno ----------------------------------------------------

def _open_paper_panel_on_start(app: TradingGUI):
    app.root.after(800, lambda: QuickPaperTrade(app))

if __name__ == "__main__":
    root = tk.Tk()
    app = TradingGUI(root, trade_executor=_paper_trade_executor)
    apply_runtime_risk_context(
        app,
        entrypoint="trading_gui",
        config_path=getattr(app, "_core_config_path", None),
        default_notional=DEFAULT_NOTIONAL_USDT,
        logger=logger,
    )
    _open_paper_panel_on_start(app)
    root.mainloop()
