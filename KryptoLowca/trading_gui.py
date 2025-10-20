"""Warstwa zgodności eksportująca nowe komponenty GUI."""

from __future__ import annotations

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


import os
import re
import json
import time
import math
import gc
import threading
import tempfile
import webbrowser
import asyncio
import inspect
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional

# Tkinter
import tkinter as tk
from tkinter import ttk, messagebox
import tkinter.font as tkfont

# Data / Plot
import numpy as np
import pandas as pd
try:
    import plotly.graph_objects as go
    _PLOTLY = True
except Exception:
    _PLOTLY = False
try:
    from tkinterweb import HtmlFrame
    _TKHTML = True
except Exception:
    _TKHTML = False

# --- LOGGING ---
import logging
from KryptoLowca.logging_utils import (
    LOGS_DIR as GLOBAL_LOGS_DIR,
    DEFAULT_LOG_FILE,
    get_logger,
    setup_app_logging,
)

__all__ = [
    "AppState",
    "TradingGUI",
    "TradingSessionController",
    "TradingView",
    "main",
]

# --- ŚCIEŻKI APLIKACJI ---
APP_ROOT = Path(__file__).resolve().parent
LOGS_DIR = GLOBAL_LOGS_DIR
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TEXT_LOG_FILE = DEFAULT_LOG_FILE
DB_FILE = APP_ROOT / "trading_bot.db"
OPEN_POS_FILE = APP_ROOT / "open_positions.json"
FAV_FILE = APP_ROOT / "favorites.json"
PRESETS_DIR = APP_ROOT / "presets"; PRESETS_DIR.mkdir(exist_ok=True)
MODELS_DIR = APP_ROOT / "models"; MODELS_DIR.mkdir(exist_ok=True)
KEYS_FILE = APP_ROOT / "api_keys.enc"
SALT_FILE = APP_ROOT / "salt.bin"

# --- MENEDŻERY / CORE ---
from bot_core.exchanges.core import PositionDTO

from KryptoLowca.config_manager import ConfigManager
from KryptoLowca.exchange_manager import ExchangeManager
from KryptoLowca.security_manager import SecurityManager
from KryptoLowca.ai_manager import AIManager
from KryptoLowca.report_manager import ReportManager
from KryptoLowca.risk_manager import RiskManager
from KryptoLowca.core.trading_engine import TradingEngine
from KryptoLowca.risk_settings_loader import (
    DEFAULT_CORE_CONFIG_PATH,
    load_risk_settings_from_core,
)

# istniejące moduły w repo
from KryptoLowca.trading_strategies import TradingStrategies
from reporting import TradeInfo
from KryptoLowca.database_manager import DatabaseManager  # klasyczny (bezargumentowy) konstruktor
from KryptoLowca.ui.trading import view as trading_view

# =====================================
# Pomocnicze
# =====================================
def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def safe_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default

def human_money(x: float) -> str:
    try:
        return f"{x:,.2f}"
    except Exception:
        return str(x)

def clamp(v, lo, hi):
    try:
        return max(lo, min(hi, v))
    except Exception:
        return v

# --- Tooltip ---
class Tooltip:
    def __init__(self, widget, text: str, delay: int = 500):
        self.widget = widget
        self.text = text
        self.delay = delay
        self._id = None
        self.tip = None
        widget.bind("<Enter>", self._enter)
        widget.bind("<Leave>", self._leave)
    def _enter(self, *_): self._schedule()
    def _leave(self, *_): self._unschedule(); self._hide()
    def _schedule(self):
        self._unschedule()
        self._id = self.widget.after(self.delay, self._show)
    def _unschedule(self):
        if self._id:
            try: self.widget.after_cancel(self._id)
            except Exception: pass
            self._id = None
    def _show(self):
        if self.tip or not self.text: return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + 20
        self.tip = tw = tk.Toplevel(self.widget)
        try: tw.wm_overrideredirect(True)
        except Exception: pass
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, background="#ffffe0", relief="solid", borderwidth=1,
                         justify="left", padx=6, pady=4, wraplength=380)
        label.pack()
    def _hide(self):
        if self.tip:
            try: self.tip.destroy()
            except Exception: pass
            self.tip = None

# =====================================
# G U I   K L A S A
# =====================================
class TradingGUI:
    """
    TYLKO UI + event handling.
    Logika: TradingEngine + nowe moduły wysokiego poziomu z pakietu ``KryptoLowca``.
    """

    def __init__(
        self,
        root: Optional[tk.Tk] = None,
        *,
        enable_web_api: bool = True,
        event_bus: Any | None = None,
        core_config_path: str | Path | None = None,
        core_environment: Optional[str] = None,
    ):
        self.enable_web_api = enable_web_api
        self.event_bus = event_bus

        self.root = root or tk.Tk()
        self.root.title("Trading Bot — AI integrated")
        self.root.geometry("1400x900")
        self.style = ttk.Style(self.root)
        try: self.style.theme_use("clam")
        except Exception: pass
        base_font = tkfont.Font(family="Segoe UI" if sys.platform.startswith("win") else "Helvetica", size=11)
        self.root.option_add("*Font", base_font)

        # ====== STAN / ZMIENNE UI ======
        self.network_var = tk.StringVar(value="Testnet")
        self.mode_var = tk.StringVar(value="Spot")
        self.timeframe_var = tk.StringVar(value="1m")
        self.fraction_var = tk.DoubleVar(value=0.05)
        self.paper_capital = tk.DoubleVar(value=10000.0)
        self.paper_balance = self.paper_capital.get()
        self.paper_balance_var = tk.StringVar(value=f"{self.paper_balance:,.2f}")
        self.account_balance_var = tk.StringVar(value="— (Live only)")
        self._live_trading_confirmed = False

        # AI
        self.enable_ai_var = tk.BooleanVar(value=False)
        self.model_var = tk.StringVar(value="lstm")
        self.auto_model_var = tk.BooleanVar(value=True)
        self.ai_epochs_var = tk.IntVar(value=30)
        self.ai_batch_var = tk.IntVar(value=64)
        self.seq_len_var = tk.IntVar(value=40)
        self.train_progress_var = tk.DoubleVar(value=0.0)
        self.ai_threshold_var = tk.DoubleVar(value=5.0)  # bps

        self.retrain_every_min_var = tk.IntVar(value=30)
        self.train_window_bars_var = tk.IntVar(value=600)
        self.valid_window_bars_var = tk.IntVar(value=200)
        self.train_all_var = tk.BooleanVar(value=True)

        self.model_types = [
            "lstm","gru","mlp","transformer","lstm_transformer",
            "lightgbm","xgboost","svr","random_forest"
        ]
        self.model_progress: Dict[str, tk.DoubleVar] = {m: tk.DoubleVar(value=0.0) for m in self.model_types}
        self.model_score: Dict[str, tk.StringVar] = {m: tk.StringVar(value="—") for m in self.model_types}

        # Guards & Risk
        self.onebar_var = tk.BooleanVar(value=True)
        self.cooldown_var = tk.IntVar(value=30)
        self.minmove_var = tk.DoubleVar(value=0.15)
        self.max_daily_loss_pct = 0.05
        self.max_daily_loss_var = tk.DoubleVar(value=self.max_daily_loss_pct * 100.0)
        self._max_daily_loss_trace = self.max_daily_loss_var.trace_add(
            "write", lambda *_: self._on_max_daily_loss_changed()
        )
        self.soft_halt_losses = 3
        self.trade_cooldown_on_error = 30
        self.risk_per_trade = tk.DoubleVar(value=0.01)
        self.portfolio_risk = tk.DoubleVar(value=0.20)

        # DCA/Trailing
        self.use_trailing = tk.BooleanVar(value=True)
        self.atr_period_var = tk.IntVar(value=14)
        self.trail_atr_mult_var = tk.DoubleVar(value=2.0)
        self.take_atr_mult_var = tk.DoubleVar(value=3.0)
        self.dca_enabled_var = tk.BooleanVar(value=False)
        self.dca_max_adds_var = tk.IntVar(value=0)
        self.dca_step_atr_var = tk.DoubleVar(value=2.0)

        self.use_orderbook_vwap = tk.BooleanVar(value=True)
        self.slippage_bps_default = 5.0
        self.slippage_bps_var = tk.DoubleVar(value=self.slippage_bps_default)

        # Auto Market Picker
        self.auto_pick_var = tk.BooleanVar(value=False)
        self.auto_pick_topn_var = tk.IntVar(value=3)
        self.auto_pick_interval_min_var = tk.IntVar(value=5)
        self.auto_pick_min_vol_usd = tk.DoubleVar(value=5_000_000.0)
        self.auto_pick_min_price = tk.DoubleVar(value=0.05)
        self.auto_pick_exclude_lev = tk.BooleanVar(value=True)
        self.auto_pick_min_bars = tk.IntVar(value=200)

        # API i hasło do klucza
        self.testnet_key = tk.StringVar(value="")
        self.testnet_secret = tk.StringVar(value="")
        self.live_key = tk.StringVar(value="")
        self.live_secret = tk.StringVar(value="")
        self.password_var = tk.StringVar(value="")

        # Risk profile presentation state
        self.state = SimpleNamespace(risk_profile_name=None)
        self._risk_section: Optional[trading_view.RiskProfileSection] = None
        self.risk_profile_display_var = tk.StringVar(value="—")
        self.risk_limits_display_var = tk.StringVar(value="—")
        self._risk_manager_settings = self._current_risk_manager_settings()

        # wewn.
        self.selected_symbols: List[str] = []
        self.symbol_vars: Dict[str, tk.BooleanVar] = {}
        self.favorites: Dict[str, bool] = self._load_favorites()

        self.chart_df: Optional[pd.DataFrame] = None
        self.chart_symbol: Optional[str] = None
        self.ai_series: Optional[pd.Series] = None
        self.last_redraw_ts = 0.0
        self.redraw_min_interval = 0.7

        self._run_flag = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None
        self._last_retrain_ts: float = 0.0
        self._last_trade_ts_per_symbol: Dict[str, float] = {}
        self._open_positions: Dict[str, Dict[str, Any]] = {}
        self._market_data: Dict[str, pd.DataFrame] = {}

        # Risk profile (core.yaml)
        config_hint = core_config_path or os.environ.get(
            "KRYPTLOWCA_CORE_CONFIG", DEFAULT_CORE_CONFIG_PATH
        )
        self.core_config_path: Path = Path(config_hint).expanduser()
        self.core_environment: Optional[str] = core_environment
        self.risk_profile_name: Optional[str] = None
        self.risk_manager_settings: Dict[str, Any] = {}
        self.risk_manager_config: Optional[Any] = None
        self.risk_profile_var = tk.StringVar(value="—")

        # ====== MENEDŻERY ======
        self.db = DatabaseManager()  # bezargumentowy
        self.sec = SecurityManager(KEYS_FILE, SALT_FILE)
        self.cfg = ConfigManager(PRESETS_DIR)
        self.reporter = ReportManager(str(DB_FILE))
        self.risk_mgr = RiskManager(
            config={
                "max_risk_per_trade": float(self.risk_per_trade.get()),
                "max_portfolio_risk": float(self.portfolio_risk.get()),
                "max_daily_loss_pct": float(self.max_daily_loss_pct),
            }
        )
        # --- AIManager: zgodność z różnymi sygnaturami konstruktora
        try:
            self.ai_mgr = AIManager(models_dir=MODELS_DIR, logger_=logger)
        except TypeError:
            try:
                self.ai_mgr = AIManager(MODELS_DIR, logger)
            except TypeError:
                try:
                    self.ai_mgr = AIManager(MODELS_DIR)
                except Exception:
                    self.ai_mgr = AIManager()
        try:
            setattr(self.ai_mgr, "ai_threshold_bps", float(self.ai_threshold_var.get()))
        except Exception:
            setattr(self.ai_mgr, "ai_threshold_bps", 5.0)

        self.ex_mgr: Optional[ExchangeManager] = None

        # === TradingEngine: bez argumentu 'reporter' ===
        try:
            self.engine = TradingEngine()
        except TypeError:
            self.engine = TradingEngine()
        if hasattr(self.engine, "set_report_manager"):
            try:
                self.engine.set_report_manager(self.reporter)
            except Exception:
                pass
        if hasattr(self.engine, "on_event"):
            try:
                self.engine.on_event(self._handle_engine_event)
            except Exception:
                pass

        # ====== UI ======
        self._build_ui()
        try:
            self.reload_risk_manager_settings()
        except Exception:
            self._log("Failed to load risk settings from core.yaml", "WARNING")
        self.network_var.trace_add("write", lambda *_: self._on_network_changed())
        self._on_network_changed()
        self.ai_threshold_var.trace_add("write", lambda *_: self._on_ai_threshold_changed())
        self._log("GUI ready", "INFO")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._on_risk_limit_changed()

    def _current_risk_manager_settings(self) -> Dict[str, Any]:
        return {
            "risk_per_trade": float(self.risk_per_trade.get()),
            "portfolio_risk": float(self.portfolio_risk.get()),
            "max_daily_loss_pct": float(self.max_daily_loss_pct),
        }

    @property
    def risk_manager_settings(self) -> Dict[str, Any]:
        return dict(self._risk_manager_settings)

    @risk_manager_settings.setter
    def risk_manager_settings(self, value: Mapping[str, Any]):
        self._risk_manager_settings = dict(value or {})
        if self._risk_section is not None:
            self._risk_section.update(
                profile_name=self.state.risk_profile_name,
                settings=self._risk_manager_settings,
            )
            self.risk_profile_display_var = self._risk_section.profile_var
            self.risk_limits_display_var = self._risk_section.limits_var

    def set_risk_profile_context(
        self,
        name: Optional[str] = None,
        settings: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if name is not None:
            self.state.risk_profile_name = name
        if settings is not None:
            self.risk_manager_settings = settings
        elif self._risk_section is not None:
            self._risk_section.update(
                profile_name=self.state.risk_profile_name,
                settings=self._risk_manager_settings,
            )

    def _on_risk_limit_changed(self, *_: Any) -> None:
        self._risk_manager_settings.update(self._current_risk_manager_settings())
        if self._risk_section is not None:
            self._risk_section.update(
                profile_name=self.state.risk_profile_name,
                settings=self._risk_manager_settings,
            )

    # ===================== UI budowa =====================
    def _build_ui(self):
        nb = ttk.Notebook(self.root); nb.pack(fill="both", expand=True)
        t1 = ttk.Frame(nb); t2 = ttk.Frame(nb); t3 = ttk.Frame(nb); t4 = ttk.Frame(nb); t6 = ttk.Frame(nb); t5 = ttk.Frame(nb)
        nb.add(t1, text="Trading"); nb.add(t2, text="History"); nb.add(t3, text="AI"); nb.add(t4, text="Settings"); nb.add(t6, text="Advanced"); nb.add(t5, text="Logs")

        # top bar
        top = ttk.Frame(t1); top.pack(side="top", fill="x")
        ttk.Label(top, text="Network").pack(side="left", padx=6)
        net_cmb = ttk.Combobox(top, textvariable=self.network_var, values=["Testnet","Live"], width=8, state="readonly")
        net_cmb.pack(side="left"); Tooltip(net_cmb, "Wybór środowiska. Testnet = wbudowany paper trading. Live = realne zlecenia (upewnij się, że masz klucze API).")
        ttk.Label(top, text="Mode").pack(side="left", padx=6)
        mode_cmb = ttk.Combobox(top, textvariable=self.mode_var, values=["Spot","Futures"], width=8, state="readonly")
        mode_cmb.pack(side="left"); Tooltip(mode_cmb, "Tryb rynku: Spot/Futures. W Futures dostępna dźwignia (niezalecane dla początkujących).")
        ttk.Label(top, text="TF").pack(side="left", padx=6)
        tf_cmb = ttk.Combobox(top, textvariable=self.timeframe_var, width=6, state="readonly",
                               values=["1m","3m","5m","15m","1h","4h","1d"])
        tf_cmb.pack(side="left"); Tooltip(tf_cmb, "Interwał świec dla danych OHLCV (wykres i AI).")
        ttk.Label(top, text="Fraction").pack(side="left", padx=6)
        frac_sp = ttk.Spinbox(top, textvariable=self.fraction_var, from_=0.01, to=1.0, increment=0.01, width=6)
        frac_sp.pack(side="left"); Tooltip(frac_sp, "Udział kapitału w pojedynczym trade (w trybie Live). W paper trading ogranicza rozmiar pozycji.")
        ttk.Label(top, text="AI Threshold (bp)").pack(side="left", padx=(6,0))
        thr_sp = ttk.Spinbox(top, textvariable=self.ai_threshold_var, from_=0.0, to=100.0, increment=0.5, width=6)
        thr_sp.pack(side="left"); Tooltip(thr_sp, "Minimalna prognozowana stopa zwrotu (w bps = 1/100 procenta) wymagana do wejścia w pozycję.")
        ttk.Label(top, text="Account:").pack(side="left", padx=(12,4))
        ttk.Label(top, textvariable=self.account_balance_var).pack(side="left")
        ttk.Label(top, text="Paper:").pack(side="left", padx=(12,4))
        ttk.Label(top, textvariable=self.paper_balance_var).pack(side="left")
        self._risk_section = trading_view.build_risk_profile_section(
            top,
            self.state,
            self._risk_manager_settings,
        )
        self._risk_section.container.pack(side="left", padx=(12, 0))
        self.risk_profile_display_var = self._risk_section.profile_var
        self.risk_limits_display_var = self._risk_section.limits_var
        exp_btn = ttk.Button(top, text="Export PDF", command=self._export_pdf_report)
        exp_btn.pack(side="left", padx=8); Tooltip(exp_btn, "Eksportuj raport do PDF: zestawienie transakcji i metryk.")
        self.btn_start = tk.Button(top, text="Start trading", command=self._on_start)
        self.btn_start.pack(side="right", padx=8); Tooltip(self.btn_start, "Uruchamia pętlę handlową. Bot zaczyna pobierać dane i podejmować decyzje.")
        self.btn_stop  = tk.Button(top, text="Stop", command=self._on_stop)
        self.btn_stop.pack(side="right"); Tooltip(self.btn_stop, "Zatrzymuje pętlę handlową (bez zamykania aplikacji).")
        self._update_run_buttons(False)

        # split
        mid = ttk.Panedwindow(t1, orient="horizontal"); mid.pack(fill="both", expand=True)
        left = ttk.Frame(mid, width=320); right = ttk.Frame(mid)
        mid.add(left, weight=1); mid.add(right, weight=4)

        # left head
        head = ttk.Frame(left); head.pack(fill="x", pady=(6,0))
        lm_btn = ttk.Button(head, text="Load markets", command=self._load_markets)
        lm_btn.pack(side="left", padx=6); Tooltip(lm_btn, "Pobierz listę rynków z giełdy (np. Binance).")
        self.sym_search_var = tk.StringVar(value="")
        ent_search = ttk.Entry(head, textvariable=self.sym_search_var)
        ent_search.pack(side="left", fill="x", expand=True, padx=6); Tooltip(ent_search, "Szukaj po nazwie symbolu (np. SOL/USDT). Wielkość liter bez znaczenia.")
        sb = ttk.Button(head, text="Search", command=self._filter_symbols)
        sb.pack(side="left", padx=4); Tooltip(sb, "Zastosuj filtr nazwy do listy symboli poniżej.")

        # left list
        btns = ttk.Frame(left); btns.pack(fill="x", pady=(4,0))
        b1 = ttk.Button(btns, text="Select all", command=self._select_all_symbols); b1.pack(side="left", padx=4); Tooltip(b1, "Zaznacz wszystkie symbole na liście.")
        b2 = ttk.Button(btns, text="None", command=self._deselect_all_symbols); b2.pack(side="left"); Tooltip(b2, "Odznacz wszystkie symbole.")
        b3 = ttk.Button(btns, text="Apply selection", command=self._apply_symbol_selection); b3.pack(side="right", padx=6); Tooltip(b3, "Ustaw zaznaczone symbole jako aktywne do handlu.")
        self.symbols_canvas = tk.Canvas(left, highlightthickness=0)
        self.symbols_scroll = ttk.Scrollbar(left, orient="vertical", command=self.symbols_canvas.yview)
        self.symbols_canvas.configure(yscrollcommand=self.symbols_scroll.set)
        self.symbols_canvas.pack(side="left", fill="both", expand=True, pady=(4,0)); self.symbols_scroll.pack(side="right", fill="y")
        self.symbols_inner = ttk.Frame(self.symbols_canvas)
        self.symbols_canvas.create_window((0,0), window=self.symbols_inner, anchor="nw")
        self.symbols_inner.bind("<Configure>", lambda e: self.symbols_canvas.configure(scrollregion=self.symbols_canvas.bbox("all")))
        self.symbols_canvas.bind_all("<MouseWheel>", lambda e: self.symbols_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # right plot
        self.chart_frame = ttk.Frame(right); self.chart_frame.pack(fill="both", expand=True)
        self._init_plotly_chart(self.chart_frame)

        # bottom tables
        bottom = ttk.Panedwindow(t1, orient="horizontal"); bottom.pack(fill="x")
        lf_open = ttk.Labelframe(bottom, text="Open positions")
        cols = ("symbol","side","qty","entry","pnl")
        self.open_tv = ttk.Treeview(lf_open, columns=cols, show="headings", height=6)
        for c in cols: self.open_tv.heading(c, text=c.capitalize()); self.open_tv.column(c, width=100, anchor="center")
        self.open_tv.pack(fill="both", expand=True); bottom.add(lf_open, weight=1)
        Tooltip(self.open_tv, "Bieżące otwarte pozycje. PNL aktualizuje się przy zamknięciu transakcji.")

        lf_closed = ttk.Labelframe(bottom, text="Closed trades")
        cols2 = ("ts","symbol","side","qty","entry","exit","pnl")
        self.closed_tv = ttk.Treeview(lf_closed, columns=cols2, show="headings", height=6)
        for c in cols2: self.closed_tv.heading(c, text=c.upper()); self.closed_tv.column(c, width=100, anchor="center")
        self.closed_tv.pack(fill="both", expand=True); bottom.add(lf_closed, weight=1)
        Tooltip(self.closed_tv, "Historia zamkniętych transakcji (czas, kierunek, wolumen, ceny i wynik).")

        # History
        hist_top = ttk.Frame(t2); hist_top.pack(fill="x")
        ttk.Button(hist_top, text="Run backtest", command=self._run_backtest).pack(side="right", padx=8, pady=4)
        self.history_out = tk.Text(t2, height=20); self.history_out.pack(fill="both", expand=True, padx=6, pady=6)
        Tooltip(self.history_out, "Wyniki testów historycznych i raporty z backtestów.")

        # AI
        self._build_ai_tab(t3)

        # Settings
        self._build_settings_tab(t4)

        # Advanced (TradingStrategies)
        self._build_advanced_tab(t6)

        # Logs
        self.log_text = tk.Text(t5, height=24); self.log_text.pack(fill="both", expand=True, padx=6, pady=6)
        Tooltip(self.log_text, "Dziennik zdarzeń bota. Przydatny przy diagnozowaniu błędów.")

    # --------------- Plotly ---------------
    def _init_plotly_chart(self, parent: ttk.Frame):
        for w in parent.winfo_children(): w.destroy()
        if _PLOTLY and _TKHTML:
            self.chart_html = HtmlFrame(parent, messages_enabled=False)
            self.chart_html.pack(fill="both", expand=True)
            self.chart_html.load_html("<html><body><div style='padding:10px;color:#666;font-family:sans-serif'>Waiting for data…</div></body></html>")
        else:
            ttk.Label(parent, text="Plotly/tkinterweb niedostępne — podgląd wykresu ograniczony.").pack(fill="both", expand=True)

    def _render_plotly(self, df: pd.DataFrame, predictions: Optional[pd.Series] = None):
        if not (_PLOTLY and _TKHTML): return
        if df is None or df.empty: return
        now = time.time()
        if now - self.last_redraw_ts < self.redraw_min_interval: return
        self.last_redraw_ts = now
        dfx = df.copy()
        if not np.issubdtype(dfx["timestamp"].dtype, np.datetime64):
            dfx["timestamp"] = pd.to_datetime(dfx["timestamp"], unit="ms", utc=True)
        dfx["timestamp"] = dfx["timestamp"].dt.tz_convert(None)
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=dfx["timestamp"], open=dfx["open"], high=dfx["high"], low=dfx["low"], close=dfx["close"],
            increasing_line_color="#2fbf71", decreasing_line_color="#e74c3c",
            increasing_fillcolor="#2fbf71", decreasing_fillcolor="#e74c3c", name="Price"
        ))
        if predictions is not None and len(predictions.dropna()) >= 1:
            close = dfx["close"].values; proj = [None]
            preds = predictions.fillna(0.0).values
            for i in range(1, len(dfx)):
                rv = float(preds[i-1]) if i-1 < len(preds) else 0.0
                proj.append(float(close[i-1]*(1.0+rv)))
            fig.add_trace(go.Scatter(x=dfx["timestamp"], y=proj, mode="lines", name="AI proj", opacity=0.85))
        fig.update_layout(title=f"{self.timeframe_var.get()}",
                          xaxis_rangeslider_visible=False, template="plotly_white",
                          dragmode="pan", hovermode="x unified", margin=dict(l=40,r=20,t=40,b=40), height=780)
        html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        try: self.chart_html.load_html(html)
        except Exception:
            tmp = Path(tempfile.gettempdir()) / f"chart_{int(time.time())}.html"
            tmp.write_text(html, encoding="utf-8"); webbrowser.open(tmp.as_posix())

    # --------------- LOGS ---------------
    def _log(self, msg: str, level: str = "INFO"):
        level_name = (level or "INFO").upper()
        log_level = getattr(logging, level_name, logging.INFO)
        logger.log(log_level, msg)

        # 1) UI log window
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {level_name}: {msg}\n"
        try: self.log_text.insert("end", line); self.log_text.see("end")
        except Exception: pass

        # 2) DB (obsługa sync/async + różnych sygnatur)
        try:
            if not hasattr(self, "db") or not hasattr(self.db, "log"):
                return

            fn = self.db.log

            async def _run_async_call():
                try:
                    await fn(level, msg); return
                except TypeError:
                    pass
                try:
                    await fn(level, msg, "gui"); return
                except TypeError:
                    pass
                await fn(1, level, msg, "gui")

            def _run_sync_call():
                try:
                    fn(level, msg); return
                except TypeError:
                    pass
                try:
                    fn(level, msg, "gui"); return
                except TypeError:
                    pass
                fn(1, level, msg, "gui")

            if inspect.iscoroutinefunction(fn):
                asyncio.run(_run_async_call())
            else:
                try:
                    result = fn(level, msg)
                    if inspect.isawaitable(result):
                        asyncio.run(result)
                    else:
                        return
                except TypeError:
                    try:
                        result = fn(level, msg, "gui")
                        if inspect.isawaitable(result):
                            asyncio.run(result)
                        else:
                            return
                    except TypeError:
                        result = fn(1, level, msg, "gui")
                        if inspect.isawaitable(result):
                            asyncio.run(result)
                        else:
                            return
                except Exception:
                    _run_sync_call()
        except Exception:
            pass  # nie blokuj UI błędami logowania

    # --------------- AI TAB ---------------
    def _build_ai_tab(self, parent: ttk.Frame):
        f = ttk.Frame(parent); f.pack(fill="both", expand=True, padx=8, pady=8)
        chk_ai = ttk.Checkbutton(f, text="Enable AI", variable=self.enable_ai_var)
        chk_ai.grid(row=0, column=0, sticky="w", pady=2); Tooltip(chk_ai, "Włącz/wyłącz silnik AI. Gdy wyłączony, decyzje oparte są na parametrach strategii/guardach.")
        ttk.Label(f, text="Seq len").grid(row=0, column=1, sticky="e")
        sp_seq = ttk.Spinbox(f, textvariable=self.seq_len_var, from_=10, to=200, increment=1, width=6)
        sp_seq.grid(row=0, column=2, sticky="w"); Tooltip(sp_seq, "Długość sekwencji (liczba świec) używana jako wejście dla modeli sekwencyjnych (LSTM/GRU).")
        ttk.Label(f, text="Epochs").grid(row=0, column=3, sticky="e")
        sp_ep = ttk.Spinbox(f, textvariable=self.ai_epochs_var, from_=5, to=500, increment=5, width=6)
        sp_ep.grid(row=0, column=4, sticky="w"); Tooltip(sp_ep, "Liczba przejść po danych treningowych. Większa wartość = dłuższy trening.")
        ttk.Label(f, text="Batch").grid(row=0, column=5, sticky="e")
        sp_bs = ttk.Spinbox(f, textvariable=self.ai_batch_var, from_=8, to=512, increment=8, width=6)
        sp_bs.grid(row=0, column=6, sticky="w"); Tooltip(sp_bs, "Wielkość minibatcha. Większy batch może przyspieszyć trening (zależnie od modelu).")
        chk_all = ttk.Checkbutton(f, text="Train ALL models", variable=self.train_all_var)
        chk_all.grid(row=0, column=7, sticky="w", padx=12); Tooltip(chk_all, "Jeśli zaznaczone, trenuje zestaw modeli i wybiera najlepszy po hit-rate.")
        btn_train = ttk.Button(f, text="Train all now", command=self._train_all_now)
        btn_train.grid(row=0, column=8, padx=8, sticky="w"); Tooltip(btn_train, "Rozpocznij trening na danych z bieżącego wykresu (symbol po prawej).")

        ttk.Label(f, text="Retrain every [min]").grid(row=1, column=0, sticky="e", pady=(6,2))
        sp_rt = ttk.Spinbox(f, textvariable=self.retrain_every_min_var, from_=5, to=240, increment=5, width=6)
        sp_rt.grid(row=1, column=1, sticky="w", pady=(6,2)); Tooltip(sp_rt, "Automatyczne ponowne trenowanie co N minut (gdy AI włączone).")
        ttk.Label(f, text="Train window [bars]").grid(row=1, column=2, sticky="e", pady=(6,2))
        sp_tw = ttk.Spinbox(f, textvariable=self.train_window_bars_var, from_=100, to=3000, increment=50, width=8)
        sp_tw.grid(row=1, column=3, sticky="w", pady=(6,2)); Tooltip(sp_tw, "Ile ostatnich świec użyć do treningu (większe okno = więcej danych).")
        ttk.Label(f, text="Valid window [bars]").grid(row=1, column=4, sticky="e", pady=(6,2))
        sp_vw = ttk.Spinbox(f, textvariable=self.valid_window_bars_var, from_=50, to=1000, increment=10, width=8)
        sp_vw.grid(row=1, column=5, sticky="w", pady=(6,2)); Tooltip(sp_vw, "Rozmiar okna walidacyjnego do oceny trafności (hit-rate).")

        lf = ttk.Labelframe(f, text="Model quality / progress"); lf.grid(row=2, column=0, columnspan=9, sticky="nsew", pady=(8,0))
        cols = ("model","progress","hit_rate")
        self.ai_tv = ttk.Treeview(lf, columns=cols, show="headings", height=6)
        for c in cols:
            self.ai_tv.heading(c, text=c.upper())
            self.ai_tv.column(c, anchor="center", width=140 if c!="model" else 200)
        self.ai_tv.pack(fill="x", expand=True)
        for m in self.model_types: self.ai_tv.insert("", "end", iid=m, values=(m, "0%", "—"))
        Tooltip(self.ai_tv, "Postęp i trafność kierunku (hit-rate) z ostatniego okna walidacyjnego. Po treningu najlepszy model jest zapisywany na dysku.")

        # Blok progres barów
        self.ai_progress_frame = ttk.Frame(lf)
        self.ai_progress_frame.pack(fill="x", expand=False, padx=6, pady=(4, 8))
        self.ai_progressbars: Dict[str, ttk.Progressbar] = {}
        self.ai_progress_lbls: Dict[str, tk.StringVar] = {}
        self.ai_hitrate_lbls: Dict[str, tk.StringVar] = {}
        for i, m in enumerate(self.model_types):
            rowf = ttk.Frame(self.ai_progress_frame)
            rowf.grid(row=i, column=0, sticky="ew", pady=2)
            ttk.Label(rowf, text=m.upper(), width=18, anchor="w").pack(side="left")
            pb = ttk.Progressbar(rowf, orient="horizontal", mode="determinate", length=320, maximum=100)
            pb.pack(side="left", padx=(6,6))
            self.ai_progressbars[m] = pb
            sv = tk.StringVar(value="0%"); self.ai_progress_lbls[m] = sv
            ttk.Label(rowf, textvariable=sv, width=6).pack(side="left")
            hv = tk.StringVar(value="—"); self.ai_hitrate_lbls[m] = hv
            ttk.Label(rowf, textvariable=hv, width=10).pack(side="left")
        for i in range(1): self.ai_progress_frame.columnconfigure(i, weight=1)

        for i in range(9): f.columnconfigure(i, weight=1)

    # --------------- SETTINGS TAB ---------------
    def _build_settings_tab(self, parent: ttk.Frame):
        lf1 = ttk.Labelframe(parent, text="API keys"); lf1.pack(fill="x", padx=8, pady=6)
        ttk.Label(lf1, text="Password for key file").grid(row=0, column=0, padx=6, pady=4, sticky="e")
        e_pwd = ttk.Entry(lf1, textvariable=self.password_var, show="*"); e_pwd.grid(row=0, column=1, padx=6, pady=4, sticky="w")
        Tooltip(e_pwd, "Hasło do lokalnego pliku z kluczami API (szyfrowany). Użyj silnego hasła.")
        ttk.Label(lf1, text="Testnet key").grid(row=1, column=0, sticky="e", padx=6)
        e_tk = ttk.Entry(lf1, textvariable=self.testnet_key, width=36); e_tk.grid(row=1, column=1, sticky="w")
        Tooltip(e_tk, "Klucz API dla konta testnet (jeśli używasz Live, wprowadź klucz w sekcji Live).")
        ttk.Label(lf1, text="Testnet secret").grid(row=1, column=2, sticky="e", padx=6)
        e_ts = ttk.Entry(lf1, textvariable=self.testnet_secret, width=36); e_ts.grid(row=1, column=3, sticky="w")
        Tooltip(e_ts, "Sekretny klucz API dla testnetu. Trzymaj w tajemnicy.")
        ttk.Label(lf1, text="Live key").grid(row=2, column=0, sticky="e", padx=6)
        e_lk = ttk.Entry(lf1, textvariable=self.live_key, width=36); e_lk.grid(row=2, column=1, sticky="w")
        Tooltip(e_lk, "Klucz API dla konta Live (realne środki).")
        ttk.Label(lf1, text="Live secret").grid(row=2, column=2, sticky="e", padx=6)
        e_ls = ttk.Entry(lf1, textvariable=self.live_secret, width=36); e_ls.grid(row=2, column=3, sticky="w")
        Tooltip(e_ls, "Sekretny klucz API dla konta Live.")
        ttk.Button(lf1, text="Save keys", command=self._save_keys).grid(row=3, column=1, sticky="w", pady=6)
        ttk.Button(lf1, text="Load keys", command=self._load_keys).grid(row=3, column=2, sticky="w", pady=6)
        for i in range(4): lf1.columnconfigure(i, weight=1)

        lfAM = ttk.Labelframe(parent, text="Auto Market Picker (AI-driven)"); lfAM.pack(fill="x", padx=8, pady=6)
        cb_ap = ttk.Checkbutton(lfAM, text="Enable auto-pick", variable=self.auto_pick_var); cb_ap.grid(row=0, column=0, sticky="w", padx=6)
        Tooltip(cb_ap, "Włącz automatyczny wybór rynków: z zaznaczonych wybiera TOP-N do handlu.")
        ttk.Label(lfAM, text="Top-N").grid(row=0, column=1, sticky="e", padx=6)
        sp_topn = ttk.Spinbox(lfAM, textvariable=self.auto_pick_topn_var, from_=1, to=20, increment=1, width=6)
        sp_topn.grid(row=0, column=2, sticky="w"); Tooltip(sp_topn, "Ile rynków jednocześnie aktywować (z listy zaznaczonych).")
        ttk.Label(lfAM, text="Refresh [min]").grid(row=0, column=3, sticky="e", padx=6)
        sp_ref = ttk.Spinbox(lfAM, textvariable=self.auto_pick_interval_min_var, from_=1, to=120, increment=1, width=6)
        sp_ref.grid(row=0, column=4, sticky="w"); Tooltip(sp_ref, "Jak często odświeżać wybór rynków (minuty).")
        ttk.Label(lfAM, text="Min 24h quoteVol [USD]").grid(row=1, column=0, sticky="e", padx=6, pady=4)
        sp_mv = ttk.Spinbox(lfAM, textvariable=self.auto_pick_min_vol_usd, from_=0.0, to=1e9, increment=100000.0, width=12)
        sp_mv.grid(row=1, column=1, sticky="w"); Tooltip(sp_mv, "Minimalny 24h wolumen (w USD), aby rynek przeszedł wstępny filtr.")
        ttk.Label(lfAM, text="Min price [USDT]").grid(row=1, column=2, sticky="e", padx=6)
        sp_mp = ttk.Spinbox(lfAM, textvariable=self.auto_pick_min_price, from_=0.0, to=10000.0, increment=0.01, width=10)
        sp_mp.grid(row=1, column=3, sticky="w"); Tooltip(sp_mp, "Minimalna cena tokena (filtruje bardzo tanie monety/podejrzane TKN).")
        cb_lv = ttk.Checkbutton(lfAM, text="Exclude leveraged tokens", variable=self.auto_pick_exclude_lev)
        cb_lv.grid(row=1, column=4, sticky="w", padx=6); Tooltip(cb_lv, "Wyklucz monety typu UP/DOWN/BULL/BEAR (z dźwignią syntetyczną).")
        ttk.Label(lfAM, text="Min OHLCV bars for refine").grid(row=2, column=0, sticky="e", padx=6)
        sp_mb = ttk.Spinbox(lfAM, textvariable=self.auto_pick_min_bars, from_=50, to=1000, increment=10, width=10)
        sp_mb.grid(row=2, column=1, sticky="w"); Tooltip(sp_mb, "Ile świec pobrać do wstępnej analizy AI przy selekcji rynków.")
        for i in range(5): lfAM.columnconfigure(i, weight=1)

        lfE = ttk.Labelframe(parent, text="Execution & Slippage"); lfE.pack(fill="x", padx=8, pady=6)
        cb_vwap = ttk.Checkbutton(lfE, text="Use order-book VWAP slippage", variable=self.use_orderbook_vwap)
        cb_vwap.grid(row=0, column=0, sticky="w", padx=6, pady=4); Tooltip(cb_vwap, "Szacuj poślizg na podstawie VWAP z głębokości zleceń. Gdy brak danych, użyj stałego bps.")
        ttk.Label(lfE, text="Fallback slippage [bps]").grid(row=0, column=1, sticky="e", padx=6)
        sp_sl = ttk.Spinbox(lfE, textvariable=self.slippage_bps_var, from_=0.0, to=100.0, increment=0.5, width=8)
        sp_sl.grid(row=0, column=2, sticky="w"); Tooltip(sp_sl, "Stały poślizg w punktach bazowych, gdy VWAP niedostępny.")
        for i in range(3): lfE.columnconfigure(i, weight=1)

        lfR = ttk.Labelframe(parent, text="Risk & Safeguards"); lfR.pack(fill="x", padx=8, pady=6)
        ttk.Label(lfR, text="Max daily loss [%]").grid(row=0, column=0, sticky="e", padx=6)
        sp_mdl = ttk.Spinbox(
            lfR,
            textvariable=self.max_daily_loss_var,
            from_=0.1,
            to=50.0,
            increment=0.1,
            width=8,
        )
        sp_mdl.grid(row=0, column=1, sticky="w")
        Tooltip(sp_mdl, "Maksymalny dzienny spadek wartości portfela, po którym bot przestaje handlować (soft stop).")
        ttk.Label(lfR, text="Soft-halt after losses").grid(row=0, column=2, sticky="e", padx=6)
        sp_soft = ttk.Spinbox(lfR, from_=0, to=10, increment=1, width=8); sp_soft.grid(row=0, column=3, sticky="w")
        sp_soft.delete(0,"end"); sp_soft.insert(0, str(self.soft_halt_losses))
        sp_soft.bind("<FocusOut>", lambda *_: setattr(self, "soft_halt_losses", int(sp_soft.get())))
        Tooltip(sp_soft, "Liczba stratnych transakcji z rzędu, po której bot robi przerwę.")
        ttk.Label(lfR, text="Cooldown on error [s]").grid(row=1, column=0, sticky="e", padx=6)
        e2 = ttk.Spinbox(lfR, from_=0, to=600, increment=5, width=8); e2.grid(row=1, column=1, sticky="w")
        e2.delete(0,"end"); e2.insert(0, str(self.trade_cooldown_on_error))
        e2.bind("<FocusOut>", lambda *_: setattr(self, "trade_cooldown_on_error", int(e2.get())))
        Tooltip(e2, "Przerwa (sekundy) po błędzie sieci/egzekucji przed kolejną próbą.")
        ttk.Label(lfR, text="Risk per trade [%]").grid(row=1, column=2, sticky="e", padx=6)
        sp_rpt = ttk.Spinbox(
            lfR,
            textvariable=self.risk_per_trade,
            from_=0.001,
            to=1.0,
            increment=0.001,
            width=8,
        )
        sp_rpt.grid(row=1, column=3, sticky="w"); Tooltip(sp_rpt, "Procent kapitału ryzykowany w pojedynczej transakcji (w paperze używany do wyliczenia wolumenu).")
        ttk.Label(lfR, text="Portfolio risk [%]").grid(row=2, column=0, sticky="e", padx=6)
        sp_pr = ttk.Spinbox(
            lfR,
            textvariable=self.portfolio_risk,
            from_=0.01,
            to=2.0,
            increment=0.01,
            width=8,
        )
        sp_pr.grid(row=2, column=1, sticky="w"); Tooltip(sp_pr, "Łączny dopuszczalny poziom ryzyka portfela (kontroluje łączną ekspozycję).")
        ttk.Label(lfR, text="Risk profile").grid(row=3, column=0, sticky="e", padx=6)
        ttk.Label(lfR, textvariable=self.risk_profile_var).grid(row=3, column=1, sticky="w")
        reload_btn = ttk.Button(lfR, text="Reload core.yaml", command=self.reload_risk_manager_settings)
        reload_btn.grid(row=3, column=3, sticky="w", padx=6)
        Tooltip(reload_btn, "Ponownie wczytaj limity ryzyka z pliku config/core.yaml bez restartu GUI.")
        for i in range(4): lfR.columnconfigure(i, weight=1)

    def _on_max_daily_loss_changed(self, *_):
        try:
            value = float(self.max_daily_loss_var.get())
        except (tk.TclError, ValueError):
            return
        self.max_daily_loss_pct = max(0.0, value / 100.0)

    def _map_network_to_environment(self) -> Optional[str]:
        try:
            network = (self.network_var.get() or "").strip().lower()
        except Exception:
            network = ""
        if network.startswith("testnet") or network.startswith("paper"):
            return "binance_paper"
        if network.startswith("live"):
            return "binance_live"
        return None

    def load_risk_manager_settings(
        self,
        *,
        config_path: Optional[Path | str] = None,
        environment: Optional[str] = None,
    ) -> tuple[str, Dict[str, Any], Optional[Any]]:
        if config_path is not None:
            target_path = Path(config_path).expanduser().resolve()
            self.core_config_path = target_path
        else:
            target_path = self.core_config_path.expanduser().resolve()
            self.core_config_path = target_path

        env_name = environment or self.core_environment
        profile_name, settings, profile_cfg, _core = load_risk_settings_from_core(
            target_path,
            environment=env_name,
        )
        if environment:
            self.core_environment = environment
        elif env_name is None and _core.environments:
            # Zapamiętaj środowisko wybrane domyślnie przez loader
            self.core_environment = next(iter(_core.environments))

        return profile_name, dict(settings), profile_cfg

    def reload_risk_manager_settings(
        self,
        *,
        config_path: Optional[Path | str] = None,
        environment: Optional[str] = None,
    ) -> tuple[str, Dict[str, Any], Optional[Any]]:
        env_hint = environment or self.core_environment or self._map_network_to_environment()
        profile_name, settings, profile_cfg = self.load_risk_manager_settings(
            config_path=config_path or self.core_config_path,
            environment=env_hint,
        )

        if not settings:
            self._log("Risk profile settings unavailable in core.yaml", "WARNING")
            return profile_name, settings, profile_cfg

        try:
            self.risk_mgr = RiskManager(config=dict(settings))
        except Exception as exc:
            self._log(f"Risk manager reload failed: {exc}", "ERROR")
            raise

        self.risk_profile_name = profile_name
        self.risk_manager_settings = dict(settings)
        self.risk_manager_config = profile_cfg
        self._apply_risk_settings_to_ui(settings, profile_name)
        self._log(f"Risk settings reloaded ({profile_name})", "INFO")
        return profile_name, dict(settings), profile_cfg

    def _apply_risk_settings_to_ui(
        self,
        settings: Mapping[str, Any],
        profile_name: Optional[str],
    ) -> None:
        if profile_name:
            self.risk_profile_var.set(str(profile_name))
        if not settings:
            return
        try:
            risk_per_trade = float(settings.get("max_risk_per_trade", self.risk_per_trade.get()))
        except Exception:
            risk_per_trade = self.risk_per_trade.get()
        try:
            portfolio_risk = float(settings.get("max_portfolio_risk", self.portfolio_risk.get()))
        except Exception:
            portfolio_risk = self.portfolio_risk.get()
        self.risk_per_trade.set(risk_per_trade)
        self.portfolio_risk.set(portfolio_risk)

        if "max_daily_loss_pct" in settings:
            try:
                self.max_daily_loss_pct = float(settings["max_daily_loss_pct"])
                self.max_daily_loss_var.set(self.max_daily_loss_pct * 100.0)
            except Exception:
                pass

        lfT = ttk.Labelframe(parent, text="DCA & Trailing (ATR)"); lfT.pack(fill="x", padx=8, pady=6)
        cb_tr = ttk.Checkbutton(lfT, text="Use trailing stops", variable=self.use_trailing)
        cb_tr.grid(row=0, column=0, sticky="w", padx=6, pady=4); Tooltip(cb_tr, "Włącz trailing stop-loss (na podstawie ATR).")
        ttk.Label(lfT, text="ATR period").grid(row=0, column=1, sticky="e", padx=6)
        sp_atrp = ttk.Spinbox(lfT, textvariable=self.atr_period_var, from_=5, to=100, increment=1, width=8)
        sp_atrp.grid(row=0, column=2, sticky="w"); Tooltip(sp_atrp, "Okres ATR — wskaźnik zmienności do SL/TP.")
        ttk.Label(lfT, text="Trail ATR ×").grid(row=1, column=0, sticky="e", padx=6)
        sp_ta = ttk.Spinbox(lfT, textvariable=self.trail_atr_mult_var, from_=0.5, to=10.0, increment=0.1, width=8)
        sp_ta.grid(row=1, column=1, sticky="w"); Tooltip(sp_ta, "Mnożnik ATR do trailingu SL (np. 2x ATR).")
        ttk.Label(lfT, text="Take ATR ×").grid(row=1, column=2, sticky="e", padx=6)
        sp_tk = ttk.Spinbox(lfT, textvariable=self.take_atr_mult_var, from_=0.5, to=10.0, increment=0.1, width=8)
        sp_tk.grid(row=1, column=3, sticky="w"); Tooltip(sp_tk, "Mnożnik ATR do realizacji zysku (TP).")
        cb_dca = ttk.Checkbutton(lfT, text="Enable DCA", variable=self.dca_enabled_var)
        cb_dca.grid(row=2, column=0, sticky="w", padx=6); Tooltip(cb_dca, "Włącz uśrednianie (dokupowanie) przy spadkach co N×ATR.")
        ttk.Label(lfT, text="DCA max adds").grid(row=2, column=1, sticky="e", padx=6)
        sp_dm = ttk.Spinbox(lfT, textvariable=self.dca_max_adds_var, from_=0, to=10, increment=1, width=8)
        sp_dm.grid(row=2, column=2, sticky="w"); Tooltip(sp_dm, "Maksymalna liczba dograń w dół (uśrednianie).")
        ttk.Label(lfT, text="DCA step ATR ×").grid(row=2, column=3, sticky="e", padx=6)
        sp_ds = ttk.Spinbox(lfT, textvariable=self.dca_step_atr_var, from_=0.5, to=10.0, increment=0.1, width=8)
        sp_ds.grid(row=2, column=4, sticky="w"); Tooltip(sp_ds, "Próg aktywacji kolejnego dogrania (mnożnik ATR).")
        for i in range(5): lfT.columnconfigure(i, weight=1)

        guard = ttk.Labelframe(parent, text="Guards"); guard.pack(fill="x", padx=8, pady=6)
        ttk.Label(guard, text="Cooldown [s]").grid(row=0, column=0, sticky="e", padx=6)
        ent_cd = ttk.Entry(guard, textvariable=self.cooldown_var, width=8); ent_cd.grid(row=0, column=1, sticky="w")
        Tooltip(ent_cd, "Minimalny czas (sekundy) między transakcjami na tym samym symbolu.")
        ttk.Label(guard, text="Min move [%]").grid(row=0, column=2, sticky="e", padx=6)
        ent_mm = ttk.Entry(guard, textvariable=self.minmove_var, width=8); ent_mm.grid(row=0, column=3, sticky="w")
        Tooltip(ent_mm, "Minimalna zmiana ceny (procent), by uznać ruch za istotny (redukcja szumu).")
        cb_ob = ttk.Checkbutton(guard, text="One trade per bar", variable=self.onebar_var)
        cb_ob.grid(row=0, column=4, sticky="w", padx=8); Tooltip(cb_ob, "Zmień tylko jedną pozycję na świecę (ogranicza nadmierne transakcje).")
        for i in range(5): guard.columnconfigure(i, weight=1)

        lf2 = ttk.Labelframe(parent, text="Paper"); lf2.pack(fill="x", padx=8, pady=6)
        ttk.Label(lf2, text="Paper capital").grid(row=0, column=0, sticky="e", padx=6)
        ent_pc = ttk.Entry(lf2, textvariable=self.paper_capital, width=12); ent_pc.grid(row=0, column=1, sticky="w")
        Tooltip(ent_pc, "Wirtualny kapitał do testów (paper trading). Nie dotyczy realnych środków.")
        ttk.Button(lf2, text="Apply paper capital", command=self._apply_paper_capital).grid(row=0, column=2, padx=6)

        lfP = ttk.Labelframe(parent, text="Presets"); lfP.pack(fill="x", padx=8, pady=8)
        self.preset_name = tk.StringVar(value="default")
        ttk.Label(lfP, text="Name").grid(row=0, column=0, sticky="e", padx=6)
        e_pn = ttk.Entry(lfP, textvariable=self.preset_name, width=24); e_pn.grid(row=0, column=1, sticky="w")
        Tooltip(e_pn, "Nazwa zestawu ustawień (zapisywana jako plik w folderze presets/).")
        ttk.Button(lfP, text="Save preset", command=self._save_preset).grid(row=0, column=2, padx=6)
        ttk.Button(lfP, text="Load preset", command=self._load_preset).grid(row=0, column=3, padx=6)
        ttk.Button(lfP, text="Delete preset", command=self._delete_preset).grid(row=0, column=4, padx=6)
        self.preset_list = tk.Listbox(lfP, height=5); self.preset_list.grid(row=1, column=0, columnspan=5, sticky="ew", padx=6, pady=6)
        self._refresh_presets_list()
        for i in range(5): lfP.columnconfigure(i, weight=1)

    # ============= Advanced (TradingStrategies) =============
    def _build_advanced_tab(self, parent: ttk.Frame):
        f = ttk.Frame(parent); f.pack(fill="both", expand=True, padx=8, pady=8)

        # Sekcja wskaźników i parametrów
        lfI = ttk.Labelframe(f, text="Indicators & Strategy"); lfI.pack(fill="x", padx=4, pady=4)
        self.adv_rsi_period = tk.IntVar(value=14)
        self.adv_ema_fast = tk.IntVar(value=12)
        self.adv_ema_slow = tk.IntVar(value=26)
        self.adv_atr_period = tk.IntVar(value=14)
        self.adv_rsi_buy = tk.IntVar(value=30)
        self.adv_rsi_sell = tk.IntVar(value=70)

        row = 0
        ttk.Label(lfI, text="RSI period").grid(row=row, column=0, sticky="e", padx=6, pady=4)
        sp_rsi = ttk.Spinbox(lfI, textvariable=self.adv_rsi_period, from_=2, to=100, width=8)
        sp_rsi.grid(row=row, column=1, sticky="w"); Tooltip(sp_rsi, "Okres RSI (niższy = szybszy, bardziej czuły). Typowo 14.")
        ttk.Label(lfI, text="EMA fast").grid(row=row, column=2, sticky="e", padx=6)
        sp_ef = ttk.Spinbox(lfI, textvariable=self.adv_ema_fast, from_=2, to=200, width=8)
        sp_ef.grid(row=row, column=3, sticky="w"); Tooltip(sp_ef, "Szybka średnia krocząca (EMA). Przykład: 12.")
        ttk.Label(lfI, text="EMA slow").grid(row=row, column=4, sticky="e", padx=6)
        sp_es = ttk.Spinbox(lfI, textvariable=self.adv_ema_slow, from_=2, to=400, width=8)
        sp_es.grid(row=row, column=5, sticky="w"); Tooltip(sp_es, "Wolna średnia krocząca (EMA). Przykład: 26.")

        row += 1
        ttk.Label(lfI, text="ATR period").grid(row=row, column=0, sticky="e", padx=6, pady=4)
        sp_atr = ttk.Spinbox(lfI, textvariable=self.adv_atr_period, from_=5, to=100, width=8)
        sp_atr.grid(row=row, column=1, sticky="w"); Tooltip(sp_atr, "Okres ATR (zmienność). Używany do SL/TP i DCA.")
        ttk.Label(lfI, text="RSI buy level").grid(row=row, column=2, sticky="e", padx=6)
        sp_rb = ttk.Spinbox(lfI, textvariable=self.adv_rsi_buy, from_=10, to=50, width=8)
        sp_rb.grid(row=row, column=3, sticky="w"); Tooltip(sp_rb, "Poziom RSI dla sygnału kupna (wyprzedanie). Niżej = agresywniej.")
        ttk.Label(lfI, text="RSI sell level").grid(row=row, column=4, sticky="e", padx=6)
        sp_rs = ttk.Spinbox(lfI, textvariable=self.adv_rsi_sell, from_=50, to=90, width=8)
        sp_rs.grid(row=row, column=5, sticky="w"); Tooltip(sp_rs, "Poziom RSI dla sygnału sprzedaży (wykupienie). Wyżej = agresywniej.")
        for i in range(6): lfI.columnconfigure(i, weight=1)

        # Akcje: backtest i optymalizacja
        lfB = ttk.Labelframe(f, text="Backtest & Optimization"); lfB.pack(fill="x", padx=4, pady=6)
        b_bt = ttk.Button(lfB, text="Backtest current strategy", command=self._adv_backtest)
        b_bt.grid(row=0, column=0, padx=6, pady=6, sticky="w"); Tooltip(b_bt, "Uruchom test na danych z bieżącego wykresu (zakładka Trading).")
        b_opt = ttk.Button(lfB, text="Optimize RSI/EMA", command=self._adv_optimize)
        b_opt.grid(row=0, column=1, padx=6, pady=6, sticky="w"); Tooltip(b_opt, "Szybka optymalizacja parametrów RSI/EMA (przeszukiwanie siatką).")

        info = ttk.Label(
            f,
            text="Advanced = elementy TradingStrategies (backtest/opt). UI bazowe pozostaje niezmienione."
        )
        info.pack(fill="x", padx=6, pady=4)
        Tooltip(info, "Ta sekcja agreguje zaawansowane funkcje strategii. Nie zmienia podstawowego workflow.")

    # ===================== Handlery =====================
    def _on_network_changed(self):
        net = self.network_var.get()
        self.account_balance_var.set("…" if net == "Live" else "— (Live only)")
        self._reset_live_confirmation()
        if net == "Live":
            self._log(
                "Tryb LIVE: przed uruchomieniem handlu wykonaj pełne testy na koncie demo/testnet i przygotuj plan zarządzania ryzykiem.",
                "WARNING",
            )

    def _on_ai_threshold_changed(self):
        try:
            if hasattr(self.ai_mgr, "ai_threshold_bps"):
                self.ai_mgr.ai_threshold_bps = float(self.ai_threshold_var.get())
        except Exception:
            pass

    def _update_run_buttons(self, running: bool):
        try:
            if running:
                self.btn_start.configure(bg="#2fbf71", fg="white", state="disabled", relief="sunken", cursor="arrow")
                self.btn_stop.configure(bg="#e74c3c", fg="white", state="normal", relief="raised", cursor="hand2")
            else:
                self.btn_start.configure(bg="#e74c3c", fg="white", state="normal", relief="raised", cursor="hand2")
                self.btn_stop.configure(bg="", fg="black", state="disabled", relief="raised", cursor="")
        except Exception: pass

    def _show_loader(self, text="Working..."):
        try: self.root.config(cursor="watch"); self.root.update_idletasks()
        except Exception: pass
        self._log(text, "INFO")

    def _hide_loader(self):
        try: self.root.config(cursor=""); self.root.update_idletasks()
        except Exception: pass

    # Markets / Symbols
    def _ensure_exchange(self):
        if self.ex_mgr is None:
            self.ex_mgr = ExchangeManager(exchange_id="binance", paper_initial_cash=float(self.paper_capital.get()))

        network = (self.network_var.get() or "testnet").strip().lower()
        mode = (self.mode_var.get() or "spot").strip().lower()
        api_key = self.testnet_key.get() if network == "testnet" else self.live_key.get()
        secret = self.testnet_secret.get() if network == "testnet" else self.live_secret.get()

        if network == "testnet" and not api_key and not secret:
            self.ex_mgr.set_mode(paper=True)
            self.ex_mgr.set_paper_balance(float(self.paper_capital.get()), asset="USDT")
            self.paper_balance = float(self.paper_capital.get())
            self.paper_balance_var.set(f"{self.paper_balance:,.2f}")
        else:
            futures = (mode == "futures")
            if network == "testnet":
                self.ex_mgr.set_mode(futures=futures, testnet=True)
            else:
                self.ex_mgr.set_mode(futures=futures, spot=not futures, testnet=False)
            self.ex_mgr.set_credentials(api_key or None, secret or None)

    def _load_markets(self):
        try:
            self._show_loader("Loading markets...")
            self._ensure_exchange()
            markets = self.ex_mgr.load_markets() or {}
            usdt = [m for m in markets.keys() if ("USDT" in m) and (m.count("/") == 1) and (":" not in m)]
            self._populate_symbols(sorted(usdt, key=lambda s: (0 if self.favorites.get(s, False) else 1, s)))
            self._log(f"Loaded {len(usdt)} USDT markets", "INFO")
        except Exception as e:
            self._log(f"Market load error: {e}", "ERROR")
            messagebox.showerror("Error", f"Market load error:\n{e}")
        finally:
            self._hide_loader()

    def _populate_symbols(self, symbols: List[str]):
        for w in self.symbols_inner.winfo_children(): w.destroy()
        self.symbol_vars.clear()
        for sym in symbols:
            var = tk.BooleanVar(value=self.favorites.get(sym, False))
            cb = ttk.Checkbutton(self.symbols_inner, text=sym, variable=var)
            cb.pack(anchor="w")
            self.symbol_vars[sym] = var
        self._log(f"Symbols populated ({len(symbols)})", "INFO")

    def _filter_symbols(self):
        q = self.sym_search_var.get().strip().upper()
        for w in self.symbols_inner.winfo_children(): w.destroy()
        for sym, var in self.symbol_vars.items():
            if q in sym.upper():
                cb = ttk.Checkbutton(self.symbols_inner, text=sym, variable=var)
                cb.pack(anchor="w")

    def _select_all_symbols(self):
        for var in self.symbol_vars.values(): var.set(True)

    def _deselect_all_symbols(self):
        for var in self.symbol_vars.values(): var.set(False)

    def _apply_symbol_selection(self):
        self.selected_symbols = [s for s,v in self.symbol_vars.items() if v.get()]
        self._log(f"Applied {len(self.selected_symbols)} symbols", "INFO")

    # Safety helpers -------------------------------------------------
    def is_demo_mode_active(self) -> bool:
        try:
            network = (self.network_var.get() or "").strip().lower()
            return network != "live"
        except Exception:
            return True

    def is_live_trading_allowed(self) -> bool:
        return bool(self._live_trading_confirmed)

    def _has_valid_live_keys(self) -> bool:
        key = (self.live_key.get() or "").strip()
        secret = (self.live_secret.get() or "").strip()
        if not key or not secret:
            return False
        try:
            return SecurityManager.validate_api_key(key) and SecurityManager.validate_api_key(secret)
        except Exception:
            return False

    def _reset_live_confirmation(self) -> None:
        self._live_trading_confirmed = False

    def _ensure_live_trading_prerequisites(self) -> bool:
        if self.is_demo_mode_active():
            return True

        if not self._has_valid_live_keys():
            messagebox.showwarning(
                "Klucze API",
                "Przed uruchomieniem handlu LIVE uzupełnij poprawne klucze API i upewnij się, "
                "że konto ma włączone limity bezpieczeństwa. Na tym etapie zalecamy kontynuację w trybie demo/testnet.",
            )
            self._log("Live trading blocked: missing or invalid API keys.", "WARNING")
            return False

        if not self._live_trading_confirmed:
            proceed = messagebox.askyesno(
                "Potwierdzenie handlu LIVE",
                "Handel na żywym rynku wiąże się z ryzykiem utraty środków. Bot jest w fazie rozwoju — uruchamiaj go wyłącznie po "
                "pełnym przetestowaniu na koncie demo/testnet i upewnij się, że spełniasz wymogi KYC/AML swojej giełdy.\n\nCzy na pewno chcesz kontynuować?",
            )
            if not proceed:
                self._log("Live trading start cancelled przez użytkownika.", "WARNING")
                return False
            self._live_trading_confirmed = True
            self._log("Live trading confirmed przez użytkownika. Zachowaj ostrożność i monitoruj transakcje.", "WARNING")

        return True

    # Start/Stop
    def _on_start(self):
        if self._worker_thread and self._worker_thread.is_alive(): return
        if not self.selected_symbols:
            messagebox.showwarning("No symbols", "Zaznacz symbole (po lewej) i kliknij Apply selection.")
            return
        if not self._ensure_live_trading_prerequisites():
            return
        self._run_flag.set()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()
        self._update_run_buttons(True)
        self._log("Trading loop started", "INFO")

    def _on_stop(self):
        self._run_flag.clear()
        self._log("Stop requested", "INFO")

    # Backtest
    def _run_backtest(self):
        if self.chart_df is None or self.chart_df.empty:
            self.history_out.insert("end", "No chart data; fetch first.\n"); self.history_out.see("end"); return
        try:
            res = self.engine.execute_backtest(
                self.chart_df.copy(),
                initial_capital=float(self.paper_capital.get()),
                fraction=float(self.fraction_var.get()),
                allow_short=False
            )
            out = "Backtest metrics:\n"
            for k, v in res.get("metrics", {}).items():
                out += f" - {k}: {v}\n"
            self.history_out.insert("end", out + "\n"); self.history_out.see("end")
        except Exception as e:
            self.history_out.insert("end", f"Backtest error: {e}\n"); self.history_out.see("end")

    # Export PDF
    def _export_pdf_report(self):
        try:
            fn = str(APP_ROOT / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
            self.reporter.export_pdf(fn)
            self._log(f"Report exported: {fn}", "INFO")
            messagebox.showinfo("Report", f"Saved report:\n{fn}")
        except Exception as e:
            self._log(f"Export error: {e}", "ERROR")
            messagebox.showerror("Export error", str(e))

    # Keys
    def _save_keys(self):
        try:
            pwd = (self.password_var.get() or "").strip()
            if not pwd:
                messagebox.showwarning("Password", "Podaj hasło do zaszyfrowania kluczy.")
                return

            # walidacja (nie blokujemy zapisu, ale ostrzegamy)
            def _warn_if_invalid(name, val):
                if not SecurityManager.validate_api_key(val):
                    self._log(f"Warning: {name} looks invalid (len<32 or bad charset).", "WARNING")

            tk_ = (self.testnet_key.get() or "").strip()
            ts_ = (self.testnet_secret.get() or "").strip()
            lk_ = (self.live_key.get() or "").strip()
            ls_ = (self.live_secret.get() or "").strip()
            if tk_: _warn_if_invalid("testnet_key", tk_)
            if ts_: _warn_if_invalid("testnet_secret", ts_)
            if lk_: _warn_if_invalid("live_key", lk_)
            if ls_: _warn_if_invalid("live_secret", ls_)

            # ZAPIS w *płaskim* schemacie, zgodnym z SecurityManager (bez zagnieżdżeń)
            data = {
                "testnet_key": tk_,
                "testnet_secret": ts_,
                "live_key": lk_,
                "live_secret": ls_,
            }
            self.sec.save_encrypted_keys(data, pwd)
            self._log("Saved encrypted keys", "INFO")
        except Exception as e:
            self._log(f"Save keys error: {e}", "ERROR")
            messagebox.showerror("Save keys error", str(e))

    def _load_keys(self):
        try:
            pwd = (self.password_var.get() or "").strip()
            if not pwd:
                messagebox.showwarning("Password", "Podaj hasło do odszyfrowania kluczy.")
                return
            data = self.sec.load_encrypted_keys(pwd) or {}

            # Oczekujemy płaskich kluczy. Jeżeli ktoś wcześniej zapisał inaczej,
            # nie próbujemy parsować stringów-dictów (SecurityManager zamienia na str).
            self.testnet_key.set(data.get("testnet_key", ""))
            self.testnet_secret.set(data.get("testnet_secret", ""))
            self.live_key.set(data.get("live_key", ""))
            self.live_secret.set(data.get("live_secret", ""))

            self._log("Loaded encrypted keys", "INFO")
        except Exception as e:
            self._log(f"Load keys error: {e}", "ERROR")
            messagebox.showerror("Load keys error", str(e))

    # Presets
    def _refresh_presets_list(self):
        try:
            names = self.cfg.list_presets()
            self.preset_list.delete(0, "end")
            for n in names: self.preset_list.insert("end", n)
        except Exception as e:
            self._log(f"Preset list error: {e}", "ERROR")

    def _save_preset(self):
        try:
            name = (self.preset_name.get() or "default").strip()
            d = self._collect_settings_dict()
            audit = self.cfg.audit_preset(d)
            if audit["issues"]:
                messagebox.showerror("Preset audit", "\n".join(audit["issues"]))
                return
            path = self.cfg.save_preset(name, d)
            self._refresh_presets_list()
            self._log(f"Preset saved: {Path(path).name}", "INFO")
            if audit["warnings"]:
                messagebox.showwarning("Preset warnings", "\n".join(audit["warnings"]))
        except Exception as e:
            self._log(f"Save preset error: {e}", "ERROR")
            messagebox.showerror("Save preset error", str(e))

    def _load_preset(self):
        try:
            idx = self.preset_list.curselection()
            if not idx:
                messagebox.showinfo("Preset", "Wybierz preset z listy.")
                return
            name = self.preset_list.get(idx[0])
            d = self.cfg.load_preset(name)
            self._apply_settings_dict(d)
            self._log(f"Preset applied: {name}", "INFO")
        except Exception as e:
            self._log(f"Apply preset error: {e}", "ERROR")
            messagebox.showerror("Apply preset error", str(e))

    def _delete_preset(self):
        try:
            idx = self.preset_list.curselection()
            if not idx: return
            name = self.preset_list.get(idx[0])
            self.cfg.delete_preset(name)
            self._refresh_presets_list()
            self._log(f"Preset deleted: {name}", "INFO")
        except Exception as e:
            self._log(f"Delete preset error: {e}", "ERROR")

    def _collect_settings_dict(self) -> Dict[str, Any]:
        return {
            "network": self.network_var.get(),
            "mode": self.mode_var.get(),
            "timeframe": self.timeframe_var.get(),
            "fraction": self.fraction_var.get(),
            "ai": {
                "enable": self.enable_ai_var.get(),
                "seq_len": self.seq_len_var.get(),
                "epochs": self.ai_epochs_var.get(),
                "batch": self.ai_batch_var.get(),
                "retrain_min": self.retrain_every_min_var.get(),
                "train_window": self.train_window_bars_var.get(),
                "valid_window": self.valid_window_bars_var.get(),
                "ai_threshold_bps": self.ai_threshold_var.get(),
                "train_all": self.train_all_var.get(),
            },
            "risk": {
                "max_daily_loss_pct": self.max_daily_loss_pct,
                "soft_halt_losses": self.soft_halt_losses,
                "trade_cooldown_on_error": self.trade_cooldown_on_error,
                "risk_per_trade": self.risk_per_trade.get(),
                "portfolio_risk": self.portfolio_risk.get(),
                "profile_name": self.state.risk_profile_name,
                "one_trade_per_bar": self.onebar_var.get(),
                "cooldown_s": self.cooldown_var.get(),
                "min_move_pct": self.minmove_var.get(),
            },
            "dca_trailing": {
                "use_trailing": self.use_trailing.get(),
                "atr_period": self.atr_period_var.get(),
                "trail_atr_mult": self.trail_atr_mult_var.get(),
                "take_atr_mult": self.take_atr_mult_var.get(),
                "dca_enabled": self.dca_enabled_var.get(),
                "dca_max_adds": self.dca_max_adds_var.get(),
                "dca_step_atr": self.dca_step_atr_var.get(),
            },
            "slippage": {
                "use_orderbook_vwap": self.use_orderbook_vwap.get(),
                "fallback_bps": self.slippage_bps_var.get(),
            },
            "advanced": {
                "rsi_period": self.adv_rsi_period.get(),
                "ema_fast": self.adv_ema_fast.get(),
                "ema_slow": self.adv_ema_slow.get(),
                "atr_period": self.adv_atr_period.get(),
                "rsi_buy": self.adv_rsi_buy.get(),
                "rsi_sell": self.adv_rsi_sell.get(),
            },
            "paper": {"capital": self.paper_capital.get()},
            "selected_symbols": self.selected_symbols
        }

    def _handle_security_audit(self, action: str, payload: Dict[str, Any]) -> None:
        try:
            record = {
                "action": action,
                "status": payload.get("status", "ok"),
                "detail": payload.get("detail", action),
                "metadata": payload.get("metadata"),
                "actor": payload.get("actor"),
            }
            self.db.sync.log_security_audit(record)
            self._log(f"Security audit: {action} ({record['status']})", "INFO")
        except Exception as exc:
            logger.exception("Failed to persist security audit event")
            self._log(f"Security audit log error: {exc}", "ERROR")

    def _apply_settings_dict(self, d: Dict[str, Any]):
        try:
            self.network_var.set(d.get("network", self.network_var.get()))
            self.mode_var.set(d.get("mode", self.mode_var.get()))
            self.timeframe_var.set(d.get("timeframe", self.timeframe_var.get()))
            self.fraction_var.set(float(d.get("fraction", self.fraction_var.get())))

            ai = d.get("ai", {})
            self.enable_ai_var.set(bool(ai.get("enable", self.enable_ai_var.get())))
            self.seq_len_var.set(int(ai.get("seq_len", self.seq_len_var.get())))
            self.ai_epochs_var.set(int(ai.get("epochs", self.ai_epochs_var.get())))
            self.ai_batch_var.set(int(ai.get("batch", self.ai_batch_var.get())))
            self.retrain_every_min_var.set(int(ai.get("retrain_min", self.retrain_every_min_var.get())))
            self.train_window_bars_var.set(int(ai.get("train_window", self.train_window_bars_var.get())))
            self.valid_window_bars_var.set(int(ai.get("valid_window", self.valid_window_bars_var.get())))
            self.ai_threshold_var.set(float(ai.get("ai_threshold_bps", self.ai_threshold_var.get())))
            self.train_all_var.set(bool(ai.get("train_all", self.train_all_var.get())))

            rk = d.get("risk", {})
            self.max_daily_loss_pct = float(rk.get("max_daily_loss_pct", self.max_daily_loss_pct))
            self.soft_halt_losses = int(rk.get("soft_halt_losses", self.soft_halt_losses))
            self.trade_cooldown_on_error = int(rk.get("trade_cooldown_on_error", self.trade_cooldown_on_error))
            self.risk_per_trade.set(float(rk.get("risk_per_trade", self.risk_per_trade.get())))
            self.portfolio_risk.set(float(rk.get("portfolio_risk", self.portfolio_risk.get())))
            profile_name = rk.get("profile_name")
            if profile_name is not None:
                self.state.risk_profile_name = profile_name
            self.onebar_var.set(bool(rk.get("one_trade_per_bar", self.onebar_var.get())))
            self.cooldown_var.set(int(rk.get("cooldown_s", self.cooldown_var.get())))
            self.minmove_var.set(float(rk.get("min_move_pct", self.minmove_var.get())))

            dt = d.get("dca_trailing", {})
            self.use_trailing.set(bool(dt.get("use_trailing", self.use_trailing.get())))
            self.atr_period_var.set(int(dt.get("atr_period", self.atr_period_var.get())))
            self.trail_atr_mult_var.set(float(dt.get("trail_atr_mult", self.trail_atr_mult_var.get())))
            self.take_atr_mult_var.set(float(dt.get("take_atr_mult", self.take_atr_mult_var.get())))
            self.dca_enabled_var.set(bool(dt.get("dca_enabled", self.dca_enabled_var.get())))
            self.dca_max_adds_var.set(int(dt.get("dca_max_adds", self.dca_max_adds_var.get())))
            self.dca_step_atr_var.set(float(dt.get("dca_step_atr", self.dca_step_atr_var.get())))

            sl = d.get("slippage", {})
            self.use_orderbook_vwap.set(bool(sl.get("use_orderbook_vwap", self.use_orderbook_vwap.get())))
            self.slippage_bps_var.set(float(sl.get("fallback_bps", self.slippage_bps_var.get())))

            adv = d.get("advanced", {})
            self.adv_rsi_period.set(int(adv.get("rsi_period", self.adv_rsi_period.get())))
            self.adv_ema_fast.set(int(adv.get("ema_fast", self.adv_ema_fast.get())))
            self.adv_ema_slow.set(int(adv.get("ema_slow", self.adv_ema_slow.get())))
            self.adv_atr_period.set(int(adv.get("atr_period", self.adv_atr_period.get())))
            self.adv_rsi_buy.set(int(adv.get("rsi_buy", self.adv_rsi_buy.get())))
            self.adv_rsi_sell.set(int(adv.get("rsi_sell", self.adv_rsi_sell.get())))

            paper = d.get("paper", {})
            self.paper_capital.set(float(paper.get("capital", self.paper_capital.get())))
            self.paper_balance = self.paper_capital.get()
            self.paper_balance_var.set(f"{self.paper_balance:,.2f}")

            sel = d.get("selected_symbols", None)
            if sel:
                for k in self.symbol_vars:
                    self.symbol_vars[k].set(k in sel)
                self.selected_symbols = list(sel)
        except Exception as e:
            self._log(f"Apply settings error: {e}", "ERROR")
        finally:
            self._on_risk_limit_changed()

    def _apply_paper_capital(self):
        try:
            self.paper_balance = float(self.paper_capital.get())
            self.paper_balance_var.set(f"{self.paper_balance:,.2f}")
            try:
                if self.ex_mgr:
                    mode_attr = getattr(self.ex_mgr, "mode", "")
                    mode_val = getattr(mode_attr, "value", mode_attr)
                    if isinstance(mode_val, str) and mode_val.lower() == "paper":
                        self.ex_mgr.set_paper_balance(self.paper_balance, asset="USDT")
            except Exception:
                pass
            self._log(f"Paper capital applied: {self.paper_balance:,.2f}", "INFO")
        except Exception as e:
            self._log(f"Paper capital error: {e}", "ERROR")

    def _load_favorites(self) -> Dict[str, bool]:
        try:
            if FAV_FILE.exists():
                return json.loads(FAV_FILE.read_text("utf-8"))
        except Exception: pass
        return {}

    # ===================== Worker =====================
    def _worker(self):
        try:
            self._ensure_exchange()
        except Exception as e:
            self._log(f"Exchange init error: {e}", "ERROR"); return

        last_price_log: Dict[str, float] = {}
        fetch_interval_s = 5

        while self._run_flag.is_set():
            t0 = time.time()
            try:
                syms = list(self.selected_symbols)
                if self.auto_pick_var.get():
                    syms = self._auto_pick_symbols()
                if not syms:
                    time.sleep(1.0); continue

                main_df = None
                symbol_for_chart = None
                tickers: Dict[str, Dict[str, Any]] = {}
                for sym in syms:
                    df = None
                    try:
                        raw = self.ex_mgr.fetch_ohlcv(
                            sym,
                            timeframe=self.timeframe_var.get(),
                            limit=max(300, self.train_window_bars_var.get())
                        ) or []
                        if raw:
                            df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
                            self._market_data[sym] = df
                    except Exception as exc:
                        df = self._market_data.get(sym)
                        self._log(f"{sym} fetch warn: {exc}", "WARNING")

                    if df is not None and main_df is None:
                        symbol_for_chart = sym
                        main_df = df

                    try:
                        ticker = self.ex_mgr.fetch_ticker(sym) or {}
                    except Exception as exc:
                        ticker = {}
                        self._log(f"{sym} ticker warn: {exc}", "WARNING")
                    if ticker:
                        tickers[sym] = ticker
                        last_price = ticker.get("last") or ticker.get("close")
                        if last_price and (sym not in last_price_log or (time.time() - last_price_log.get(sym, 0)) > 1.0):
                            self._log(f"{sym} price: {last_price}", "INFO")
                            last_price_log[sym] = time.time()

                preds = None
                if main_df is not None:
                    self.chart_df = main_df
                    self.chart_symbol = symbol_for_chart
                    if self.enable_ai_var.get():
                        try:
                            preds = self.ai_mgr.predict_series(
                                self.chart_df,
                                feature_cols=["open", "high", "low", "close", "volume"]
                            )
                        except Exception as e:
                            self._log(f"AI predict error: {e}", "ERROR")
                    self._render_plotly(self.chart_df, predictions=preds)

                for sym in syms:
                    ticker = tickers.get(sym) or {}
                    sym_preds = preds if sym == self.chart_symbol else None
                    self._bridge_decide_and_trade(sym, ticker, sym_preds)

                if self.enable_ai_var.get():
                    if (time.time() - self._last_retrain_ts) >= max(60, self.retrain_every_min_var.get()*60):
                        self._train_ai_background()

                if int(time.time()) % 20 == 0:
                    gc.collect()

            except Exception as e:
                self._log(f"Worker error: {e}", "ERROR")

            dt = time.time() - t0
            time.sleep(max(0.5, fetch_interval_s - dt))

        self._log("Worker ended", "INFO")
        self._update_run_buttons(False)

    def _auto_pick_symbols(self) -> List[str]:
        try:
            if not self.selected_symbols: return []
            topn = max(1, int(self.auto_pick_topn_var.get()))
            return self.selected_symbols[:topn]
        except Exception:
            return self.selected_symbols[:3]

    # ======== Public helpers (AutoTrader integration) ========
    def get_symbol_for_auto_trader(self) -> str:
        """Zwraca symbol, który powinien być używany przez moduł AutoTrader."""
        try:
            primary = (self.symbol_var.get() if hasattr(self, "symbol_var") else "")
        except Exception:
            primary = ""
        primary = (primary or "").strip()
        if primary:
            return primary

        picks = self._auto_pick_symbols()
        if picks:
            return picks[0]

        if getattr(self, "chart_symbol", None):
            return str(self.chart_symbol)

        if getattr(self, "selected_symbols", None):
            for sym in self.selected_symbols:
                if sym:
                    return sym
        return ""

    def get_symbols_for_auto_trader(self, limit: Optional[int] = None) -> List[str]:
        """Lista symboli w kolejności priorytetów dla automatycznego handlu."""
        symbols: List[str] = []
        symbols.extend(self._auto_pick_symbols())
        if getattr(self, "chart_symbol", None):
            symbols.append(str(self.chart_symbol))
        if getattr(self, "selected_symbols", None):
            symbols.extend([sym for sym in self.selected_symbols if sym])

        # usuń duplikaty zachowując kolejność
        seen = set()
        ordered: List[str] = []
        for sym in symbols:
            sym_norm = (sym or "").strip()
            if not sym_norm or sym_norm in seen:
                continue
            ordered.append(sym_norm)
            seen.add(sym_norm)

        if limit is not None and limit > 0:
            return ordered[:limit]
        return ordered

    # ============== BRIDGE: AI → Risk → Execution ==============
    def _bridge_decide_and_trade(self, symbol: str, ticker: Dict[str, Any], preds: Optional[pd.Series]):
        try:
            if not self.enable_ai_var.get() or preds is None or len(preds.dropna()) == 0:
                return
            price = safe_float(ticker.get("last") or ticker.get("close") or 0.0, 0.0)
            if price <= 0: return

            threshold = float(self.ai_threshold_var.get()) / 10000.0  # bps
            predicted_ret = float(preds.dropna().iloc[-1]) if len(preds.dropna()) else 0.0

            nowt = time.time()
            last_t = self._last_trade_ts_per_symbol.get(symbol, 0.0)
            if (nowt - last_t) < max(1, int(self.cooldown_var.get())):
                return

            pos = self._open_positions.get(symbol)
            if predicted_ret >= threshold:
                if not pos or pos["side"] != "buy":
                    self._bridge_execute_trade(symbol, "buy", price)
            elif predicted_ret <= -threshold:
                if pos and pos["side"] == "buy":
                    self._bridge_execute_trade(symbol, "sell", price)

        except Exception as e:
            self._log(f"Bridge decision error: {e}", "ERROR")

    def _bridge_execute_trade(self, symbol: str, side: str, mkt_price: float):
        try:
            self._ensure_exchange()
            slip_bps = float(self.slippage_bps_var.get())
            vwap, slip = self.ex_mgr.simulate_vwap_price(symbol, "buy" if side == "buy" else "sell", amount=None, fallback_bps=slip_bps)
            exec_price = safe_float(vwap, mkt_price)

            market_df = self._market_data.get(symbol)
            signal_payload = {
                "symbol": symbol,
                "direction": "LONG" if side == "buy" else "SHORT",
                "strength": 1.0,
                "confidence": 1.0,
                "prediction": 1.0 if side == "buy" else -1.0,
            }

            try:
                fraction = float(self.risk_mgr.calculate_position_size(
                    symbol=symbol,
                    signal=signal_payload,
                    market_data=market_df if market_df is not None else {"price": exec_price},
                    portfolio={"cash": self.paper_balance, "positions": self._open_positions},
                ))
            except Exception as exc:
                self._log(f"Risk sizing error: {exc}", "WARNING")
                fraction = 0.0

            capital = float(self.paper_balance)
            if fraction <= 0:
                fraction = float(self.fraction_var.get())
            notional = max(0.0, capital * min(1.0, fraction))
            qty = self.ex_mgr.quantize_amount(symbol, notional / max(exec_price, 1e-8))
            if qty <= 0: return

            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            pos = self._open_positions.get(symbol)

            if side == "buy":
                if pos and pos["side"] == "buy": return
                cost = qty * exec_price
                if cost > self.paper_balance + 1e-8:
                    self._log("Insufficient paper balance for BUY", "WARNING")
                    return
                self.paper_balance = max(0.0, self.paper_balance - cost)
                self.paper_balance_var.set(f"{self.paper_balance:,.2f}")
                self._open_positions[symbol] = {"side":"buy","qty":qty,"entry":exec_price,"ts":ts}
                self._append_open_position(symbol, "buy", qty, exec_price, 0.0)
                self._log(f"EXEC BUY {symbol} qty={qty} @ {exec_price} (paper)", "INFO")
            else:
                if pos and pos["side"] == "buy":
                    qty = float(pos["qty"])
                    pnl = (exec_price - pos["entry"]) * qty
                    self._append_closed_trade_row(ts, symbol, "sell", qty, pos["entry"], exec_price, pnl)
                    self.paper_balance += qty * exec_price
                    self.paper_balance_var.set(f"{self.paper_balance:,.2f}")
                    self._remove_open_position(symbol)
                    self._log(f"CLOSE LONG {symbol} qty={pos['qty']} @ {exec_price} pnl={pnl:.2f}", "INFO")
                else:
                    return

            self._last_trade_ts_per_symbol[symbol] = time.time()
            if hasattr(self.engine, "_emit"):
                try:
                    self.engine._emit("trade_exec", symbol=symbol, side=side, price=exec_price, qty=qty, paper=True, slip_bps=slip)
                except Exception:
                    pass

            try:
                if hasattr(self.db, "log_trade"):
                    ti = TradeInfo(ts=ts, symbol=symbol, side=side, qty=qty,
                                   entry=exec_price if side=="buy" else (self._open_positions.get(symbol,{}).get("entry", exec_price)),
                                   exit=exec_price if side=="sell" else None, pnl=None)
                    try: self.db.log_trade(ti)
                    except TypeError:
                        try: self.db.log_trade(1, ti)
                        except Exception: pass
            except Exception: pass

        except Exception as e:
            self._log(f"Bridge exec error: {e}", "ERROR")

    def _sync_positions_from_service(self) -> List[PositionDTO]:
        """Pobiera pozycje z ExchangeManagera i odświeża tabelę GUI."""
        try:
            self._ensure_exchange()
        except Exception as exc:
            self._log(f"Position sync exchange error: {exc}", "ERROR")
            return []

        if not self.ex_mgr:
            return []

        try:
            positions = self.ex_mgr.fetch_positions()
        except Exception as exc:
            self._log(f"Position sync fetch error: {exc}", "ERROR")
            return []

        try:
            self._open_positions.clear()
        except Exception:
            self._open_positions = {}

        try:
            for iid in self.open_tv.get_children():
                self.open_tv.delete(iid)
        except Exception:
            pass

        for pos in positions:
            try:
                side_raw = str(getattr(pos, "side", "LONG") or "LONG").upper()
            except Exception:
                side_raw = "LONG"
            side_ui = "buy" if side_raw in {"LONG", "BUY"} else "sell"
            try:
                qty = float(getattr(pos, "quantity", 0.0) or 0.0)
            except Exception:
                qty = 0.0
            try:
                entry = float(getattr(pos, "avg_price", 0.0) or 0.0)
            except Exception:
                entry = 0.0
            try:
                pnl_val = float(getattr(pos, "unrealized_pnl", 0.0) or 0.0)
            except Exception:
                pnl_val = 0.0

            self._open_positions[pos.symbol] = {
                "side": side_ui,
                "qty": qty,
                "entry": entry,
            }

            try:
                self.open_tv.insert("", "end", values=(pos.symbol, side_ui, qty, human_money(entry), human_money(pnl_val)))
            except Exception:
                pass

        return positions

    def _append_open_position(self, symbol: str, side: str, qty: float, entry: float, pnl: float):
        try:
            for iid in self.open_tv.get_children():
                vals = self.open_tv.item(iid, "values")
                if vals and vals[0] == symbol:
                    self.open_tv.delete(iid)
            self.open_tv.insert("", "end", values=(symbol, side, qty, human_money(entry), human_money(pnl)))
        except Exception: pass

    def _remove_open_position(self, symbol: str):
        try:
            if symbol in self._open_positions: del self._open_positions[symbol]
            for iid in self.open_tv.get_children():
                vals = self.open_tv.item(iid, "values")
                if vals and vals[0] == symbol:
                    self.open_tv.delete(iid)
        except Exception: pass

    def _append_closed_trade_row(self, ts: str, symbol: str, side: str, qty: float, entry: float, exitp: float, pnl: float):
        try:
            self.closed_tv.insert("", "end", values=(ts, symbol, side, qty, human_money(entry), human_money(exitp), human_money(pnl)))
        except Exception: pass

    # ===================== AI =====================
    def _on_ai_progress(self, model_type: str, progress: float, hitrate: Optional[float]=None):
        try:
            pv = int(clamp(round(progress * 100), 0, 100))
            if model_type in self.ai_progressbars:
                self.ai_progressbars[model_type]["value"] = pv
            if model_type in self.model_progress:
                self.model_progress[model_type].set(progress)
            if model_type in self.ai_progress_lbls:
                self.ai_progress_lbls[model_type].set(f"{pv}%")
            if hitrate is not None and model_type in self.ai_hitrate_lbls:
                self.ai_hitrate_lbls[model_type].set(f"{hitrate:.2%}")
        except Exception:
            pass

    def _train_all_now(self):
        if self.chart_df is None or self.chart_df.empty or not self.chart_symbol:
            messagebox.showinfo("AI", "Brak danych wykresu do uczenia. Załaduj rynki, włącz worker i poczekaj na dane.")
            return
        self._train_ai_background(force=True)

    def _train_ai_background(self, force: bool=False):
        if not force and (time.time() - self._last_retrain_ts) < 5.0: return
        self._last_retrain_ts = time.time()
        df = self.chart_df.copy() if self.chart_df is not None else None
        symbol = self.chart_symbol or (self.selected_symbols[0] if self.selected_symbols else None)
        if df is None or df.empty or not symbol:
            self._log("AI training skipped: no data/symbol", "WARNING"); return

        for m in self.model_types:
            try:
                self.model_progress[m].set(0.0)
                self.ai_progressbars[m]["value"] = 0.0
                self.ai_progress_lbls[m].set("0%")
                self.ai_hitrate_lbls[m].set("—")
            except Exception: pass

        def runner():
            self._log("AI training started", "INFO")
            try:
                seq = int(self.seq_len_var.get())
                ep = int(self.ai_epochs_var.get())
                bs = int(self.ai_batch_var.get())
                types = self.model_types if self.train_all_var.get() else [self.model_var.get()]

                def cb(model_name: str, progress: float):
                    self._on_ai_progress(model_name, progress)

                best = self.ai_mgr.train_all_models(
                    symbol=symbol,
                    df=df.tail(max(300, self.train_window_bars_var.get())),
                    feature_cols=["open","high","low","close","volume"],
                    seq_len=seq, epochs=ep, batch_size=bs,
                    progress_callback=cb
                )
                if best and hasattr(best, "model_type"):
                    try:
                        self.ai_hitrate_lbls.get(best.model_type, tk.StringVar()).set(f"{best.hit_rate:.2%}")
                        if best.model_type in self.ai_tv.get_children():
                            self.ai_tv.set(best.model_type, "hit_rate", f"{best.hit_rate:.2%}")
                            self.ai_tv.set(best.model_type, "progress", "100%")
                    except Exception:
                        pass
                    self._log(f"Model saved: {getattr(best, 'model_path', 'N/A')}", "INFO")
            except Exception as e:
                self._log(f"AI training failed: {e}", "ERROR")
            finally:
                self._log("AI training finished", "INFO")

        threading.Thread(target=runner, daemon=True).start()

    # ============== Advanced actions ==============
    def _adv_backtest(self):
        if self.chart_df is None or self.chart_df.empty:
            messagebox.showinfo("Advanced", "Brak danych wykresu."); return
        try:
            rsi_p = int(clamp(self.adv_rsi_period.get(), 2, 100))
            ema_f = int(clamp(self.adv_ema_fast.get(), 2, 200))
            ema_s = int(clamp(self.adv_ema_slow.get(), 2, 400))
            atr_p = int(clamp(self.adv_atr_period.get(), 5, 100))
            rsi_b = int(clamp(self.adv_rsi_buy.get(), 10, 50))
            rsi_s = int(clamp(self.adv_rsi_sell.get(), 50, 90))
            if ema_f >= ema_s:
                messagebox.showwarning("Advanced", "EMA fast musi być mniejsza niż EMA slow."); return

            params = {
                "rsi_period": rsi_p,
                "ema_fast": ema_f,
                "ema_slow": ema_s,
                "atr_period": atr_p,
                "rsi_buy": rsi_b,
                "rsi_sell": rsi_s,
            }
            res = TradingStrategies.backtest(self.chart_df.copy(), params)
            msg = "Advanced Backtest metrics:\n" + "\n".join(f" - {k}: {v}" for k,v in res.get("metrics", {}).items())
            messagebox.showinfo("Advanced backtest", msg)
            self._log("Advanced backtest done", "INFO")
        except Exception as e:
            self._log(f"Advanced backtest error: {e}", "ERROR")
            messagebox.showerror("Advanced backtest error", str(e))

    def _adv_optimize(self):
        if self.chart_df is None or self.chart_df.empty:
            messagebox.showinfo("Advanced", "Brak danych wykresu."); return
        try:
            res = TradingStrategies.quick_optimize(self.chart_df.copy())
            best = res.get("best_params", {})
            self.adv_rsi_period.set(int(best.get("rsi_period", self.adv_rsi_period.get())))
            self.adv_ema_fast.set(int(best.get("ema_fast", self.adv_ema_fast.get())))
            self.adv_ema_slow.set(int(best.get("ema_slow", self.adv_ema_slow.get())))
            self.adv_atr_period.set(int(best.get("atr_period", self.adv_atr_period.get())))
            self.adv_rsi_buy.set(int(best.get("rsi_buy", self.adv_rsi_buy.get())))
            self.adv_rsi_sell.set(int(best.get("rsi_sell", self.adv_rsi_sell.get())))
            messagebox.showinfo("Optimization", f"Zastosowano parametry:\n{json.dumps(best, indent=2)}")
            self._log("Advanced optimize applied", "INFO")
        except Exception as e:
            self._log(f"Advanced optimize error: {e}", "ERROR")
            messagebox.showerror("Advanced optimize error", str(e))

    # ============== Engine event handler (opcjonalny) ==============
    def _handle_engine_event(self, evt: Dict[str, Any]):
        try:
            kind = evt.get("type", "")
            if kind == "trade_closed":
                ti: TradeInfo = evt.get("trade_info")
                self._append_closed_trade(ti)
        except Exception as e:
            self._log(f"Engine event error: {e}", "ERROR")

    def _append_closed_trade(self, ti: TradeInfo):
        try:
            self.closed_tv.insert("", "end",
                values=(ti.ts, ti.symbol, ti.side, ti.qty, human_money(ti.entry), human_money(ti.exit), human_money(ti.pnl)))
        except Exception: pass

    # ============== Zamknięcie ==============
    def _on_close(self):
        try: self._run_flag.clear()
        except Exception: pass
        try:
            if hasattr(self, "ex_mgr"):
                self.ex_mgr = None
        except Exception:
            pass
        self.root.after(150, self.root.destroy)

# ===== MAIN =====
if __name__ == "__main__":
    if not _PLOTLY:
        print("Matplotlib not available. Chart generation will be limited.", file=sys.stderr)
    root = tk.Tk()
    app = TradingGUI(root)
    root.mainloop()
