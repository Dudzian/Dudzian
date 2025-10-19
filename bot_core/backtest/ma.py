"""Proste narzędzia MA-cross wykorzystywane przez testy i adaptery."""
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
    if fast >= slow:
        raise ValueError("fast must be < slow")
    f = _sma(closes, fast)
    s = _sma(closes, slow)
    pos = 0
    sigs: List[int] = [0] * len(closes)
    for i in range(1, len(closes)):
        if not (f[i] == f[i] and s[i] == s[i] and f[i - 1] == f[i - 1] and s[i - 1] == s[i - 1]):
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
            if pos < 0:
                pnl = entry_px - px
                trades.append({"pnl": pnl, "ts_close": bars[i].get("ts"), "side": "buy_to_cover"})
                pos = 0
            if pos == 0:
                pos = +1
                entry_px = px
        elif sig == -1:
            if pos > 0:
                pnl = px - entry_px
                trades.append({"pnl": pnl, "ts_close": bars[i].get("ts"), "side": "sell_to_close"})
                pos = 0
            if pos == 0:
                pos = -1
                entry_px = px
    return trades


def param_grid_fast_slow(
    fast_min: int = 5,
    fast_max: int = 30,
    slow_min: int = 20,
    slow_max: int = 120,
    step: int = 5,
) -> List[Dict[str, int]]:
    grid: List[Dict[str, int]] = []
    for f in range(fast_min, fast_max + 1, step):
        for s in range(max(slow_min, f + 5), slow_max + 1, step):
            grid.append({"fast": f, "slow": s})
    return grid


__all__ = [
    "Bar",
    "Trade",
    "simulate_trades_ma",
    "param_grid_fast_slow",
]
