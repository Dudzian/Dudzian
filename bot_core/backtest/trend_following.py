"""Trend-following backtest engine niezależny od warstwy archiwalnej."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple


__all__ = [
    "MIN_SL_PCT",
    "DEFAULT_FEE",
    "ExchangeLike",
    "EntryParams",
    "ExitParams",
    "StrategyParams",
    "BacktestConfig",
    "TradeFill",
    "TradeRecord",
    "TrendBacktestEngine",
]


MIN_SL_PCT = 0.0005  # minimalny SL% (0.05%) chroniący przed mikropozycjami
DEFAULT_FEE = 0.001  # 0.1% taker fee na wejściu i wyjściu (spot parity)


class ExchangeLike(Protocol):
    """Minimalny interfejs wymagany przez silnik backtestu."""

    def load_markets(self) -> None:
        """Ładuje metadane rynku (precision/minNotional)."""

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, *, limit: int = 3000
    ) -> Sequence[Sequence[float]]:
        """Zwraca listę świec [ts, open, high, low, close, volume]."""

    def quantize_amount(self, symbol: str, amount: float) -> float:
        """Przycina wartość do dozwolonej precyzji wolumenu."""

    def min_notional(self, symbol: str) -> float | None:
        """Zwraca minimalną wartość nominalną dla symbolu (jeśli dostępna)."""


def _ema(values: Sequence[float], period: int) -> List[float]:
    if period <= 1 or not values:
        return [float(v) for v in values]
    k = 2.0 / (period + 1.0)
    out = [float("nan")] * len(values)
    cumulative = 0.0
    for idx, value in enumerate(values):
        cumulative += value
        if idx + 1 == period:
            sma = cumulative / period
            out[idx] = sma
            prev = sma
            for jdx in range(idx + 1, len(values)):
                prev = (values[jdx] - prev) * k + prev
                out[jdx] = prev
            break
    last = None
    for idx, value in enumerate(out):
        if not math.isnan(value):
            last = value
        else:
            out[idx] = float(last) if last is not None else float("nan")
    return [float(v) for v in out]


@dataclass(slots=True)
class EntryParams:
    capital_usdt: float = 10_000.0
    risk_pct: float = 0.01
    k_sl_atr: float = 1.5
    fee_rate: float = DEFAULT_FEE
    allow_short: bool = False


@dataclass(slots=True)
class ExitParams:
    k_tp1_atr: float = 1.0
    k_tp2_atr: float = 2.0
    k_tp3_atr: float = 3.0
    p1: float = 0.50
    p2: float = 0.30
    p3: float = 0.20
    move_sl_to_be_after_tp1: bool = True
    trail_activate_pct: float = 0.008
    trail_dist_pct: float = 0.003


@dataclass(slots=True)
class StrategyParams:
    timeframe: str = "15m"
    ema_fast: int = 50
    ema_slow: int = 200
    atr_len: int = 14
    min_atr_pct: float = 0.5
    execution: str = "next_open"


@dataclass(slots=True)
class BacktestConfig:
    symbols: List[str]
    strategy: StrategyParams = field(default_factory=StrategyParams)
    entry: EntryParams = field(default_factory=EntryParams)
    exit: ExitParams = field(default_factory=ExitParams)
    start_index_pad: int = 600
    max_bars: Optional[int] = None
    save_dir: str = "backtests"


@dataclass(slots=True)
class TradeFill:
    ts: int
    price: float
    qty: float
    tag: str


@dataclass(slots=True)
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


def _weighted_avg_exit(trade: TradeRecord) -> float:
    total_qty = 0.0
    total_cash = 0.0
    for fill in trade.fills:
        if fill.tag in {"TP1", "TP2", "TP3", "SL", "TRAIL", "EXIT"}:
            total_qty += fill.qty
            total_cash += fill.price * fill.qty
    if total_qty <= 0:
        return trade.entry_price
    return total_cash / total_qty


def _fees_total(trade: TradeRecord, fee_rate: float) -> float:
    total = trade.entry_price * trade.entry_qty * fee_rate
    for fill in trade.fills:
        if fill.tag in {"TP1", "TP2", "TP3", "SL", "TRAIL", "EXIT"}:
            total += fill.price * fill.qty * fee_rate
    return total


class TrendBacktestEngine:
    """Trend-following backtest engine wykorzystujący ATR i sygnał EMA."""

    def __init__(self, exchange: ExchangeLike) -> None:
        self._exchange = exchange
        self._exchange.load_markets()

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 3000) -> List[List[float]]:
        candles = self._exchange.fetch_ohlcv(symbol, timeframe, limit=limit) or []
        return [list(map(float, bar)) for bar in candles]

    @staticmethod
    def _signals_ema_cross(closes: Sequence[float], ema_fast: int, ema_slow: int) -> List[bool]:
        ef = _ema(closes, ema_fast)
        es = _ema(closes, ema_slow)
        signals = [False] * len(closes)
        for idx in range(1, len(closes)):
            crossed = ef[idx - 1] <= es[idx - 1] and ef[idx] > es[idx] and closes[idx] > es[idx]
            signals[idx] = crossed
        return signals

    def _risk_size_long(
        self, symbol: str, price: float, atr: float, cfg: BacktestConfig
    ) -> Tuple[float, float, float]:
        sl_pct = max((cfg.entry.k_sl_atr * atr) / price, MIN_SL_PCT)
        risk_usdt = cfg.entry.capital_usdt * cfg.entry.risk_pct
        required_notional = risk_usdt / sl_pct if sl_pct > 0 else 0.0
        qty = required_notional / price if price > 0 else 0.0
        qty = self._exchange.quantize_amount(symbol, qty)
        if qty <= 0:
            return 0.0, price * (1.0 - sl_pct), sl_pct
        min_notional = self._exchange.min_notional(symbol)
        if min_notional and price * qty < min_notional:
            qty = self._exchange.quantize_amount(symbol, (min_notional / price) * 1.001)
            if qty <= 0:
                return 0.0, price * (1.0 - sl_pct), sl_pct
        sl_price = price * (1.0 - sl_pct)
        return float(qty), float(sl_price), float(sl_pct)

    def run_symbol(self, symbol: str, cfg: BacktestConfig) -> Tuple[List[TradeRecord], Dict[str, Any]]:
        strategy = cfg.strategy
        bars = self.fetch_ohlcv(symbol, strategy.timeframe, limit=cfg.max_bars or 3000)
        if not bars or len(bars) < max(strategy.ema_slow, strategy.atr_len) + 50:
            return [], {"symbol": symbol, "note": "Za mało danych"}

        timestamps = [int(row[0]) for row in bars]
        opens = [float(row[1]) for row in bars]
        highs = [float(row[2]) for row in bars]
        lows = [float(row[3]) for row in bars]
        closes = [float(row[4]) for row in bars]

        true_ranges: List[float] = []
        prev_close = closes[0]
        for idx in range(1, len(bars)):
            high = highs[idx]
            low = lows[idx]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)
            prev_close = closes[idx]
        if len(true_ranges) < strategy.atr_len:
            return [], {"symbol": symbol, "note": "Za mało danych do ATR"}
        atr = sum(true_ranges[: strategy.atr_len]) / float(strategy.atr_len)
        atr_seq: List[Optional[float]] = [None] * (strategy.atr_len + 1) + [atr]
        for tr in true_ranges[strategy.atr_len :]:
            atr = (atr * (strategy.atr_len - 1) + tr) / float(strategy.atr_len)
            atr_seq.append(atr)
        while len(atr_seq) < len(closes):
            atr_seq.insert(0, None)
        atr_seq = atr_seq[: len(closes)]

        long_signals = self._signals_ema_cross(closes, strategy.ema_fast, strategy.ema_slow)

        trades: List[TradeRecord] = []
        position: Optional[Dict[str, float | bool]] = None

        start_index = max(cfg.start_index_pad, strategy.ema_slow + 5, strategy.atr_len + 5)
        for idx in range(start_index, len(closes) - 1):
            close_price = closes[idx]
            ts = timestamps[idx]

            if position:
                if position["trail_on"]:
                    position["trail_peak"] = max(position["trail_peak"], close_price)

                if close_price <= position["sl"]:
                    qty = position["qty_remain"]
                    price = position["sl"]
                    trades[-1].fills.append(TradeFill(ts=ts, price=price, qty=qty, tag="SL"))
                    trades[-1].exit_ts = ts
                    wap = _weighted_avg_exit(trades[-1])
                    trades[-1].exit_price_wap = wap
                    pnl = (wap - position["entry"]) * position["qty_total"]
                    pnl -= _fees_total(trades[-1], cfg.entry.fee_rate)
                    trades[-1].pnl_usdt = pnl
                    trades[-1].pnl_pct = pnl / (position["entry"] * position["qty_total"]) * 100.0
                    trades[-1].r_multiple = pnl / position["risk_usdt"] if position["risk_usdt"] else None
                    position = None
                    continue

                if (not position["tp1_done"]) and close_price >= position["tp1"] and position["qty_remain"] > 0:
                    qty = min(position["q1"], position["qty_remain"])
                    if qty > 0:
                        trades[-1].fills.append(
                            TradeFill(ts=ts, price=position["tp1"], qty=qty, tag="TP1")
                        )
                        position["qty_remain"] -= qty
                        position["tp1_done"] = True
                        if position["move_be"]:
                            breakeven = max(
                                position["entry"], position["entry"] * (1 + cfg.entry.fee_rate * 2)
                            )
                            position["sl"] = max(position["sl"], breakeven)

                if (not position["tp2_done"]) and close_price >= position["tp2"] and position["qty_remain"] > 0:
                    qty = min(position["q2"], position["qty_remain"])
                    if qty > 0:
                        trades[-1].fills.append(
                            TradeFill(ts=ts, price=position["tp2"], qty=qty, tag="TP2")
                        )
                        position["qty_remain"] -= qty
                        position["tp2_done"] = True

                if (not position["tp3_done"]) and close_price >= position["tp3"] and position["qty_remain"] > 0:
                    qty = min(position["q3"], position["qty_remain"])
                    if qty > 0:
                        trades[-1].fills.append(
                            TradeFill(ts=ts, price=position["tp3"], qty=qty, tag="TP3")
                        )
                        position["qty_remain"] -= qty
                        position["tp3_done"] = True

                if (not position["trail_on"]) and position["trail_activate"] > 0.0:
                    if close_price >= position["entry"] * (1.0 + position["trail_activate"]):
                        position["trail_on"] = True
                        position["trail_peak"] = close_price

                if position["trail_on"] and position["trail_dist"] > 0.0 and position["qty_remain"] > 0:
                    stop_trail = position["trail_peak"] * (1.0 - position["trail_dist"])
                    if close_price <= stop_trail:
                        qty = position["qty_remain"]
                        trades[-1].fills.append(
                            TradeFill(ts=ts, price=stop_trail, qty=qty, tag="TRAIL")
                        )
                        trades[-1].exit_ts = ts
                        wap = _weighted_avg_exit(trades[-1])
                        trades[-1].exit_price_wap = wap
                        pnl = (wap - position["entry"]) * position["qty_total"]
                        pnl -= _fees_total(trades[-1], cfg.entry.fee_rate)
                        trades[-1].pnl_usdt = pnl
                        trades[-1].pnl_pct = pnl / (position["entry"] * position["qty_total"]) * 100.0
                        trades[-1].r_multiple = pnl / position["risk_usdt"] if position["risk_usdt"] else None
                        position = None
                        continue

                if position and position["qty_remain"] <= 1e-15:
                    trades[-1].exit_ts = ts
                    wap = _weighted_avg_exit(trades[-1])
                    trades[-1].exit_price_wap = wap
                    pnl = (wap - position["entry"]) * position["qty_total"]
                    pnl -= _fees_total(trades[-1], cfg.entry.fee_rate)
                    trades[-1].pnl_usdt = pnl
                    trades[-1].pnl_pct = pnl / (position["entry"] * position["qty_total"]) * 100.0
                    trades[-1].r_multiple = pnl / position["risk_usdt"] if position["risk_usdt"] else None
                    position = None

            if position is None:
                atr_now = atr_seq[idx]
                if atr_now is None or atr_now <= 0:
                    continue
                atr_pct = atr_now / close_price * 100.0
                if atr_pct < cfg.strategy.min_atr_pct:
                    continue
                if long_signals[idx]:
                    if strategy.execution == "next_open":
                        jdx = idx + 1
                        entry_price = float(opens[jdx])
                        entry_ts = int(timestamps[jdx])
                    else:
                        entry_price = float(close_price)
                        entry_ts = int(ts)

                    qty, sl_price, sl_pct = self._risk_size_long(symbol, entry_price, atr_now, cfg)
                    if qty <= 0:
                        continue

                    trade = TradeRecord(
                        symbol=symbol,
                        timeframe=strategy.timeframe,
                        entry_ts=entry_ts,
                        entry_price=entry_price,
                        entry_qty=qty,
                    )
                    trade.fills.append(TradeFill(ts=entry_ts, price=entry_price, qty=qty, tag="ENTRY"))
                    trades.append(trade)

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
                        "q1": q1,
                        "q2": q2,
                        "q3": q3,
                        "tp1_done": False,
                        "tp2_done": False,
                        "tp3_done": False,
                        "move_be": cfg.exit.move_sl_to_be_after_tp1,
                        "trail_activate": cfg.exit.trail_activate_pct,
                        "trail_dist": cfg.exit.trail_dist_pct,
                        "trail_on": False,
                        "trail_peak": entry_price,
                    }

        if position is not None and trades:
            last_idx = len(closes) - 1
            qty = position["qty_remain"]
            price = closes[last_idx]
            trades[-1].fills.append(
                TradeFill(ts=timestamps[last_idx], price=price, qty=qty, tag="EXIT")
            )
            trades[-1].exit_ts = timestamps[last_idx]
            wap = _weighted_avg_exit(trades[-1])
            trades[-1].exit_price_wap = wap
            pnl = (wap - position["entry"]) * position["qty_total"]
            pnl -= _fees_total(trades[-1], cfg.entry.fee_rate)
            trades[-1].pnl_usdt = pnl
            trades[-1].pnl_pct = pnl / (position["entry"] * position["qty_total"]) * 100.0
            trades[-1].r_multiple = pnl / position["risk_usdt"] if position["risk_usdt"] else None

        summary = {"symbol": symbol, "trades": len(trades)}
        return trades, summary

