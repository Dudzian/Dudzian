# backtest/strategy_ma.py
from __future__ import annotations

from typing import Any, Dict, List

Bar = Dict[str, float]
Trade = Dict[str, Any]


def _sma(series: List[float], n: int) -> List[float]:
    if n <= 0:
        raise ValueError("n must be > 0")
    out: List[float] = []
    s = 0.0
    for i, x in enumerate(series):
        s += x
        if i >= n:
            s -= series[i - n]
        out.append(s / n if i + 1 >= n else float("nan"))
    return out


def _signals_ma_cross(closes: List[float], fast: int, slow: int) -> List[int]:
    """Zwraca: +1 kup, -1 sprzedaj, 0 brak zmiany (na zamknięciu baru)."""
    if fast >= slow:
        raise ValueError("fast must be < slow")
    f = _sma(closes, fast)
    s = _sma(closes, slow)
    pos = 0
    sigs: List[int] = [0] * len(closes)
    for i in range(1, len(closes)):
        if not (f[i] == f[i] and s[i] == s[i] and f[i - 1] == f[i - 1] and s[i - 1] == s[i - 1]):  # NaN check
            continue
        cross_up = f[i] > s[i] and f[i - 1] <= s[i - 1]
        cross_dn = f[i] < s[i] and f[i - 1] >= s[i - 1]
        if cross_up and pos <= 0:
            sigs[i] = +1
            pos = 1
        elif cross_dn and pos >= 0:
            sigs[i] = -1
            pos = -1
        else:
            sigs[i] = 0
    return sigs


def simulate_trades_ma(bars: List[Bar], params: Dict[str, Any]) -> List[Trade]:
    """
    Prosty backtest MA cross (long/short) bez prowizji/slippage (dla WFO).
    params: {'fast': int, 'slow': int}
    """
    fast = int(params.get("fast", 10))
    slow = int(params.get("slow", 50))
    if fast < 2 or slow < 3 or fast >= slow:
        return []

    closes = [float(b.get("close", 0.0)) for b in bars]
    sigs = _signals_ma_cross(closes, fast, slow)

    pos = 0
    entry_px = 0.0
    trades: List[Trade] = []

    for i in range(len(closes)):
        px = closes[i]
        sig = sigs[i]
        if sig == +1:
            # zamknij short
            if pos < 0:
                pnl = entry_px - px
                trades.append({"pnl": pnl, "ts_close": bars[i].get("ts"), "side": "buy_to_cover"})
                pos = 0
            # otwórz long
            if pos == 0:
                pos = +1
                entry_px = px
        elif sig == -1:
            # zamknij long
            if pos > 0:
                pnl = px - entry_px
                trades.append({"pnl": pnl, "ts_close": bars[i].get("ts"), "side": "sell_to_close"})
                pos = 0
            # otwórz short
            if pos == 0:
                pos = -1
                entry_px = px

    # bez wymuszania zamknięcia pozycji na końcu — to OOS może ocenić inaczej
    return trades


def param_grid_fast_slow(fast_min=5, fast_max=30, slow_min=20, slow_max=120, step=5) -> List[Dict[str, int]]:
    grid: List[Dict[str, int]] = []
    for f in range(fast_min, fast_max + 1, step):
        for s in range(max(slow_min, f + 5), slow_max + 1, step):
            grid.append({"fast": f, "slow": s})
    return grid
