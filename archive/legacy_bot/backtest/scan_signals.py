# backtest/scan_signals.py
"""
Skaner sygnałów zgodny z logiką backtestu.
- Pobiera OHLCV przez ExchangeManager
- Liczy EMA(slow) i ATR%
- Sygnał = close > EMA_slow AND ATR% >= min_atr_pct AND (cross z dołu)
- Wypisuje liczbę sygnałów i ostatnie timestampy (UTC)

Uruchom:
  python -m backtest.scan_signals --symbols "ETH/USDT,SOL/USDT" --timeframe 5m --max_bars 5000 --ema_slow 150 --min_atr_pct 0.10
"""

from __future__ import annotations
import argparse
import math
from datetime import datetime, timezone
from typing import List, Sequence

from KryptoLowca.exchange_manager import ExchangeManager


def ts_to_utc_str(ms: int) -> str:
    """Zamiana milisekund epochi na string UTC yyyy-mm-dd HH:MM:SS."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _ema(values: Sequence[float], period: int):
    import numpy as np
    v = np.asarray(values, dtype=float)
    if period <= 1 or v.size == 0:
        return v.copy()
    alpha = 2.0 / (period + 1.0)
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, v.size):
        out[i] = alpha * v[i] + (1.0 - alpha) * out[i - 1]
    return out


def _atr_pct(high: Sequence[float], low: Sequence[float], close: Sequence[float], period: int = 14):
    import numpy as np
    h = np.asarray(high, dtype=float)
    l = np.asarray(low, dtype=float)
    c = np.asarray(close, dtype=float)
    n = c.size
    tr = np.full(n, math.nan, dtype=float)
    for i in range(1, n):
        hl = h[i] - l[i]
        hc = abs(h[i] - c[i - 1])
        lc = abs(l[i] - c[i - 1])
        tr[i] = max(hl, hc, lc)
    atr = np.full(n, math.nan, dtype=float)
    if n > period:
        atr[period] = np.nanmean(tr[1 : period + 1])
        for i in range(period + 1, n):
            atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    atr_pct = (atr / c) * 100.0
    return atr_pct


def default_signal_detector(ohlcv: List[List[float]], ema_slow: int, min_atr_pct: float) -> List[int]:
    """
    Fallback sygnałów (gdy nie mamy helpera z engine):
    - trend: close > EMA_slow
    - zmienność: ATR% >= min_atr_pct
    - trigger: cross z dołu (close poprzednie <= EMA_slow poprzednie i teraz >)
    Zwraca listę timestampów (ms) barów, na których wystąpił sygnał.
    """
    import numpy as np

    if not ohlcv or len(ohlcv[0]) < 6:
        return []

    arr = np.asarray(ohlcv, dtype=float)
    ts = arr[:, 0].astype(np.int64)
    high = arr[:, 2]
    low = arr[:, 3]
    close = arr[:, 4]

    ema_s = _ema(close, ema_slow)
    atr_p = _atr_pct(high, low, close, period=14)

    sig_ts: List[int] = []
    for i in range(1, len(close)):
        if math.isnan(ema_s[i]) or math.isnan(atr_p[i]):
            continue
        cond_trend = close[i] > ema_s[i]
        cond_vol = atr_p[i] >= min_atr_pct
        cond_cross = (close[i - 1] <= ema_s[i - 1]) and (close[i] > ema_s[i])
        if cond_trend and cond_vol and cond_cross:
            sig_ts.append(int(ts[i]))
    return sig_ts


def detect_signals_compatible(ohlcv: List[List[float]], ema_slow: int, min_atr_pct: float) -> List[int]:
    """
    Próbuje użyć tej samej logiki co backtest.engine (jeśli dostępna),
    a jeśli nie – spada do default_signal_detector.
    """
    try:
        from backtest.engine import find_entry_signals  # opcjonalne
        return list(find_entry_signals(ohlcv, ema_slow=ema_slow, min_atr_pct=min_atr_pct))
    except Exception:
        return default_signal_detector(ohlcv, ema_slow, min_atr_pct)


def _safe_load_markets(ex: ExchangeManager):
    """Obsługa różnych wariantów API ExchangeManager."""
    try:
        if hasattr(ex, "load_markets_public"):
            return ex.load_markets_public()
        if hasattr(ex, "load_markets"):
            return ex.load_markets()
        # ostatnia próba – niektóre implementacje ładują w __init__/set_mode
        return None
    except Exception:
        # jeśli jedna metoda zawiedzie, spróbuj drugiej
        if hasattr(ex, "load_markets"):
            return ex.load_markets()
        raise


def main():
    parser = argparse.ArgumentParser(description="Skaner sygnałów zgodny z logiką backtestu.")
    parser.add_argument("--symbols", type=str, required=True,
                        help='Lista symboli rozdzielona przecinkami, np. "ETH/USDT,SOL/USDT"')
    parser.add_argument("--timeframe", type=str, default="5m")
    parser.add_argument("--max_bars", type=int, default=5000)
    parser.add_argument("--ema_slow", type=int, default=150)
    parser.add_argument("--min_atr_pct", type=float, default=0.10)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tf = args.timeframe
    limit = args.max_bars
    ema_slow = args.ema_slow
    min_atr_pct = args.min_atr_pct

    ex = ExchangeManager()
    # Publiczny tryb (bez kluczy) – wystarczy do OHLCV
    ex.set_mode(paper=True, futures=False, testnet=False)
    _safe_load_markets(ex)

    for sym in symbols:
        try:
            ohlcv = ex.fetch_ohlcv(sym, tf, limit=limit)
        except Exception as e:
            print(f"{sym} | TF={tf} | błąd fetch_ohlcv: {e}")
            continue

        sig_ts = detect_signals_compatible(ohlcv, ema_slow=ema_slow, min_atr_pct=min_atr_pct)
        print(f"{sym} | TF={tf} | sygnałów={len(sig_ts)}")
        if not sig_ts:
            print("  Brak sygnałów w oknie danych.")
            continue

        print("  Ostatnie sygnały:")
        for t in sig_ts[-10:]:
            print(f"   - {ts_to_utc_str(t)} UTC")


if __name__ == "__main__":
    main()
