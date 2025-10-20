# backtest/engine.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

from KryptoLowca.exchange_manager import ExchangeManager
from archive.legacy_bot.managers.scanner import _compute_atr


MIN_SL_PCT = 0.0005  # min 0.05% – anty-mikro
DEFAULT_FEE = 0.001  # 0.1% na wejściu i 0.1% na wyjściu (spot paper parytet)


def _ema(values: List[float], period: int) -> List[float]:
    """EMA bez numpy; zwraca listę float o tej samej długości."""
    if period <= 1 or not values:
        return [float(v) for v in values]
    k = 2.0 / (period + 1.0)
    out = [float("nan")] * len(values)
    # seed jako SMA(period)
    s = 0.0
    for i in range(len(values)):
        s += values[i]
        if i + 1 == period:
            sma = s / period
            out[i] = sma
            prev = sma
            for j in range(i + 1, len(values)):
                prev = (values[j] - prev) * k + prev
                out[j] = prev
            break
    # uzupełnij początkowe NaN ostatnią znaną wartością
    last = None
    for i in range(len(out)):
        if not math.isnan(out[i]):
            last = out[i]
        else:
            out[i] = float(last) if last is not None else float("nan")
    return [float(x) for x in out]


@dataclass
class EntryParams:
    capital_usdt: float = 10_000.0
    risk_pct: float = 0.01           # 1% kapitału
    k_sl_atr: float = 1.5
    fee_rate: float = DEFAULT_FEE     # 0.1% na trade leg
    allow_short: bool = False         # Faza 1.0: tylko LONG


@dataclass
class ExitParams:
    k_tp1_atr: float = 1.0
    k_tp2_atr: float = 2.0
    k_tp3_atr: float = 3.0
    p1: float = 0.50                  # 50%
    p2: float = 0.30
    p3: float = 0.20
    move_sl_to_be_after_tp1: bool = True
    trail_activate_pct: float = 0.008 # 0.8%
    trail_dist_pct: float = 0.003     # 0.3%


@dataclass
class StrategyParams:
    timeframe: str = "15m"
    ema_fast: int = 50
    ema_slow: int = 200
    atr_len: int = 14
    min_atr_pct: float = 0.5          # min ATR% (filtr wejścia)
    execution: str = "next_open"      # "next_open" lub "close"


@dataclass
class BacktestConfig:
    symbols: List[str]
    strategy: StrategyParams = field(default_factory=StrategyParams)
    entry: EntryParams = field(default_factory=EntryParams)
    exit: ExitParams = field(default_factory=ExitParams)
    start_index_pad: int = 600        # od którego indeksu świec startujemy (po to by wskaźniki się ustabilizowały)
    max_bars: Optional[int] = None    # jeśli chcesz ograniczyć liczbę świec
    save_dir: str = "backtests"


@dataclass
class TradeFill:
    ts: int
    price: float
    qty: float
    tag: str  # ENTRY / TP1 / TP2 / TP3 / SL / TRAIL / EXIT


@dataclass
class TradeRecord:
    symbol: str
    timeframe: str
    entry_ts: int
    entry_price: float
    entry_qty: float
    fills: List[TradeFill] = field(default_factory=list)
    exit_ts: Optional[int] = None
    exit_price_wap: Optional[float] = None
    pnl_usdt: Optional[float] = None
    pnl_pct: Optional[float] = None
    risk_usdt: Optional[float] = None
    r_multiple: Optional[float] = None


class BacktestEngine:
    """
    Backtester 1.0:
      - LONG-only, sygnał: przecięcie EMA50/200 + close>EMA200,
      - wejście na następnym otwarciu (domyślnie) lub na close,
      - sizing: risk_usdt / SL%, gdzie SL=k_sl*ATR,
      - wyjścia: TP1-3 (częściowe), SL, trailing (po aktywacji), opcjonalnie SL->BE po TP1.
    """

    def __init__(self, ex: Optional[ExchangeManager] = None):
        self.ex = ex or ExchangeManager(exchange_id="binance")
        # Upewnij się, że mamy rynki (precision/minNotional) – potrzebne do kwantyzacji
        self.ex.load_markets()

    # ---------- Dane ----------

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 3000) -> List[List[float]]:
        """Pobiera OHLCV; zwraca listę [ts, o, h, l, c, v]."""
        return self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit) or []

    # ---------- Wspólne obliczenia ----------

    @staticmethod
    def _signals_ema_cross(closes: List[float], ema_fast: int, ema_slow: int) -> List[bool]:
        ef = _ema(closes, ema_fast)
        es = _ema(closes, ema_slow)
        sig = [False] * len(closes)
        # sygnał LONG: ef przecina es w górę na zamkniętej świecy i close>es
        for i in range(1, len(closes)):
            crossed = (ef[i - 1] <= es[i - 1]) and (ef[i] > es[i]) and (closes[i] > es[i])
            sig[i] = crossed
        return sig

    # ---------- Wejście/wyjście ----------

    def _risk_size_long(
        self, symbol: str, price: float, atr: float, cfg: BacktestConfig
    ) -> Tuple[float, float, float]:
        """
        Zwraca: qty, sl_price, sl_pct. Uwzględnia minNotional i kwantyzację.
        """
        sl_pct = max((cfg.entry.k_sl_atr * atr) / price, MIN_SL_PCT)
        risk_usdt = cfg.entry.capital_usdt * cfg.entry.risk_pct
        req_notional = risk_usdt / sl_pct
        qty = req_notional / price

        # kwantyzacja + minNotional
        qty = self.ex.quantize_amount(symbol, qty)
        if qty <= 0:
            return 0.0, price * (1.0 - sl_pct), sl_pct

        mn = self.ex.min_notional(symbol)
        if mn and (qty * price) < mn:
            qty = self.ex.quantize_amount(symbol, (mn / price) * 1.001)
            if qty <= 0:
                return 0.0, price * (1.0 - sl_pct), sl_pct

        sl_price = price * (1.0 - sl_pct)
        return float(qty), float(sl_price), float(sl_pct)

    # ---------- Symulacja jednego symbolu ----------

    def run_symbol(self, symbol: str, cfg: BacktestConfig) -> Tuple[List[TradeRecord], Dict[str, Any]]:
        s = cfg.strategy
        bars = self.fetch_ohlcv(symbol, s.timeframe, limit=cfg.max_bars or 3000)
        if not bars or len(bars) < max(s.ema_slow, s.atr_len) + 50:
            return [], {"symbol": symbol, "note": "Za mało danych"}

        ts = [int(x[0]) for x in bars]
        opens = [float(x[1]) for x in bars]
        highs = [float(x[2]) for x in bars]
        lows  = [float(x[3]) for x in bars]
        closes= [float(x[4]) for x in bars]

        # ATR w każdym punkcie (Wilder). Wykorzystamy atr[i] do wejścia/wyjść.
        atr_seq: List[Optional[float]] = []
        prev = None
        # przygotuj TR serie:
        trs: List[float] = []
        prev_close = closes[0]
        for i in range(1, len(bars)):
            h = highs[i]; l = lows[i]; c_prev = prev_close
            tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
            trs.append(tr)
            prev_close = closes[i]
        # ATR start
        if len(trs) < s.atr_len:
            return [], {"symbol": symbol, "note": "Za mało danych do ATR"}
        atr = sum(trs[:s.atr_len]) / float(s.atr_len)
        atr_seq = [None] * (s.atr_len + 1) + [atr]  # align do indeksów close
        for tr in trs[s.atr_len:]:
            atr = (atr * (s.atr_len - 1) + tr) / float(s.atr_len)
            atr_seq.append(atr)
        # uzupełnij brakujące początki
        while len(atr_seq) < len(closes):
            atr_seq.insert(0, None)
        atr_seq = atr_seq[:len(closes)]

        # sygnały EMA
        sig_long = self._signals_ema_cross(closes, s.ema_fast, s.ema_slow)

        trades: List[TradeRecord] = []
        position: Optional[Dict[str, Any]] = None  # aktualna pozycja

        start_i = max(cfg.start_index_pad, s.ema_slow + 5, s.atr_len + 5)
        for i in range(start_i, len(closes) - 1):  # -1 bo używamy next_open przy wejściu
            # zarządzanie pozycją (na CLOSE bara i)
            close_i = closes[i]
            ts_i = ts[i]

            if position:
                # trailing: update peak
                if position["trail_on"]:
                    position["trail_peak"] = max(position["trail_peak"], close_i)

                # SL pełny?
                if close_i <= position["sl"]:
                    # zamknij całość przy sl (użyj sl price jako fill)
                    qty = position["qty_remain"]
                    price = position["sl"]
                    trades[-1].fills.append(TradeFill(ts=ts_i, price=price, qty=qty, tag="SL"))
                    trades[-1].exit_ts = ts_i
                    # WAP wyjścia:
                    wap = _weighted_avg_exit(trades[-1])
                    trades[-1].exit_price_wap = wap
                    # PnL:
                    pnl = (wap - position["entry"]) * position["qty_total"] - _fees_total(trades[-1], cfg.entry.fee_rate)
                    trades[-1].pnl_usdt = pnl
                    trades[-1].pnl_pct = pnl / (position["entry"] * position["qty_total"]) * 100.0
                    trades[-1].r_multiple = pnl / position["risk_usdt"] if position["risk_usdt"] > 0 else None
                    position = None
                    continue

                # TP1/2/3 (częściowe, jeśli jeszcze nie zrobione)
                # używamy ceny docelowej jako fill (bardziej realistyczne niż close)
                if (not position["tp1_done"]) and close_i >= position["tp1"] and position["qty_remain"] > 0:
                    q = min(position["q1"], position["qty_remain"])
                    if q > 0:
                        trades[-1].fills.append(TradeFill(ts=ts_i, price=position["tp1"], qty=q, tag="TP1"))
                        position["qty_remain"] -= q
                        position["tp1_done"] = True
                        if position["move_be"]:
                            be = max(position["entry"], position["entry"] * (1 + cfg.entry.fee_rate * 2))
                            position["sl"] = max(position["sl"], be)
                if (not position["tp2_done"]) and close_i >= position["tp2"] and position["qty_remain"] > 0:
                    q = min(position["q2"], position["qty_remain"])
                    if q > 0:
                        trades[-1].fills.append(TradeFill(ts=ts_i, price=position["tp2"], qty=q, tag="TP2"))
                        position["qty_remain"] -= q
                        position["tp2_done"] = True
                if (not position["tp3_done"]) and close_i >= position["tp3"] and position["qty_remain"] > 0:
                    q = min(position["q3"], position["qty_remain"])
                    if q > 0:
                        trades[-1].fills.append(TradeFill(ts=ts_i, price=position["tp3"], qty=q, tag="TP3"))
                        position["qty_remain"] -= q
                        position["tp3_done"] = True

                # trailing – aktywacja / realizacja
                if (not position["trail_on"]) and position["trail_activate"] > 0.0:
                    if close_i >= position["entry"] * (1.0 + position["trail_activate"]):
                        position["trail_on"] = True
                        position["trail_peak"] = close_i
                if position["trail_on"] and position["trail_dist"] > 0.0 and position["qty_remain"] > 0:
                    stop_trail = position["trail_peak"] * (1.0 - position["trail_dist"])
                    if close_i <= stop_trail:
                        q = position["qty_remain"]
                        trades[-1].fills.append(TradeFill(ts=ts_i, price=stop_trail, qty=q, tag="TRAIL"))
                        trades[-1].exit_ts = ts_i
                        wap = _weighted_avg_exit(trades[-1])
                        trades[-1].exit_price_wap = wap
                        pnl = (wap - position["entry"]) * position["qty_total"] - _fees_total(trades[-1], cfg.entry.fee_rate)
                        trades[-1].pnl_usdt = pnl
                        trades[-1].pnl_pct = pnl / (position["entry"] * position["qty_total"]) * 100.0
                        trades[-1].r_multiple = pnl / position["risk_usdt"] if position["risk_usdt"] > 0 else None
                        position = None
                        continue

                # jeśli wszystko sprzedane TP1-3 (qty_remain≈0), zamknij trade:
                if position and position["qty_remain"] <= 1e-15:
                    trades[-1].exit_ts = ts_i
                    wap = _weighted_avg_exit(trades[-1])
                    trades[-1].exit_price_wap = wap
                    pnl = (wap - position["entry"]) * position["qty_total"] - _fees_total(trades[-1], cfg.entry.fee_rate)
                    trades[-1].pnl_usdt = pnl
                    trades[-1].pnl_pct = pnl / (position["entry"] * position["qty_total"]) * 100.0
                    trades[-1].r_multiple = pnl / position["risk_usdt"] if position["risk_usdt"] > 0 else None
                    position = None

            # sprawdzamy sygnał wejścia dla następnego bara
            if position is None:
                atr_now = atr_seq[i]
                if atr_now is None or atr_now <= 0:
                    continue
                atr_pct = atr_now / close_i * 100.0
                if atr_pct < cfg.strategy.min_atr_pct:
                    continue
                if sig_long[i]:
                    # wybór ceny wejścia
                    if cfg.strategy.execution == "next_open":
                        j = i + 1
                        entry_price = float(opens[j])
                        entry_ts = int(ts[j])
                    else:  # "close"
                        entry_price = float(close_i)
                        entry_ts = int(ts_i)

                    qty, sl_price, sl_pct = self._risk_size_long(symbol, entry_price, atr_now, cfg)
                    if qty <= 0:
                        continue

                    # załóż trade
                    tr = TradeRecord(
                        symbol=symbol, timeframe=s.timeframe,
                        entry_ts=entry_ts, entry_price=entry_price, entry_qty=qty,
                    )
                    tr.fills.append(TradeFill(ts=entry_ts, price=entry_price, qty=qty, tag="ENTRY"))
                    trades.append(tr)

                    # plan wyjść
                    q1 = qty * cfg.exit.p1
                    q2 = qty * cfg.exit.p2
                    q3 = qty * cfg.exit.p3
                    position = {
                        "entry": entry_price,
                        "risk_usdt": cfg.entry.capital_usdt * cfg.entry.risk_pct,
                        "qty_total": qty,
                        "qty_remain": qty,
                        "sl": sl_price,
                        "tp1": entry_price + cfg.exit.k_tp1_atr * atr_now,
                        "tp2": entry_price + cfg.exit.k_tp2_atr * atr_now,
                        "tp3": entry_price + cfg.exit.k_tp3_atr * atr_now,
                        "q1": q1, "q2": q2, "q3": q3,
                        "tp1_done": False, "tp2_done": False, "tp3_done": False,
                        "move_be": cfg.exit.move_sl_to_be_after_tp1,
                        "trail_activate": cfg.exit.trail_activate_pct,
                        "trail_dist": cfg.exit.trail_dist_pct,
                        "trail_on": False,
                        "trail_peak": entry_price,
                    }

        # jeśli na końcu pozycja nadal otwarta – zamknij na ostatnim close
        if position is not None and trades:
            last_i = len(closes) - 1
            q = position["qty_remain"]
            price = closes[last_i]
            trades[-1].fills.append(TradeFill(ts=ts[last_i], price=price, qty=q, tag="EXIT"))
            trades[-1].exit_ts = ts[last_i]
            wap = _weighted_avg_exit(trades[-1])
            trades[-1].exit_price_wap = wap
            pnl = (wap - position["entry"]) * position["qty_total"] - _fees_total(trades[-1], cfg.entry.fee_rate)
            trades[-1].pnl_usdt = pnl
            trades[-1].pnl_pct = pnl / (position["entry"] * position["qty_total"]) * 100.0
            trades[-1].r_multiple = pnl / position["risk_usdt"] if position["risk_usdt"] > 0 else None

        summary = {"symbol": symbol, "trades": len(trades)}
        return trades, summary


# ---------- Pomocnicze obliczenia ----------

def _weighted_avg_exit(tr: TradeRecord) -> float:
    total_qty = 0.0
    total_cash = 0.0
    for f in tr.fills:
        if f.tag in ("TP1", "TP2", "TP3", "SL", "TRAIL", "EXIT"):
            total_qty += f.qty
            total_cash += f.price * f.qty
    if total_qty <= 0:
        return tr.entry_price
    return total_cash / total_qty


def _fees_total(tr: TradeRecord, fee_rate: float) -> float:
    """
    Fee liczymy per leg (ENTRY + każde wyjście). ENTRY: price*qty*fee_rate,
    dla wyjść analogicznie.
    """
    total = tr.entry_price * tr.entry_qty * fee_rate
    for f in tr.fills:
        if f.tag in ("TP1", "TP2", "TP3", "SL", "TRAIL", "EXIT"):
            total += f.price * f.qty * fee_rate
    return total
