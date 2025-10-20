# strategies/rules_v0.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import List, Dict, Any, Optional

from KryptoLowca.exchange_manager import ExchangeManager
from bot_core.exchanges.core import SignalDTO


def _ema(values: List[float], period: int) -> List[float]:
    """Prosta EMA (α=2/(n+1)), zwraca listę tej samej długości, z None na początku."""
    if period <= 1 or not values:
        return values[:]
    k = 2.0 / (period + 1.0)
    out: List[Optional[float]] = [None] * len(values)
    s = 0.0
    n = 0
    # seed – zwykła SMA z pierwszych 'period'
    for i, v in enumerate(values):
        s += v
        n += 1
        if n == period:
            sma = s / period
            out[i] = sma
            # kontynuuj EMA
            prev = sma
            for j in range(i + 1, len(values)):
                prev = (values[j] - prev) * k + prev
                out[j] = prev
            break
    # uzupełnij ewentualne None ostatnią znaną wartością
    last = None
    out2: List[float] = []
    for v in out:
        if v is not None:
            last = v
        out2.append(float(last) if last is not None else float("nan"))
    return out2


class RulesV0Strategy:
    """
    Regułowa strategia v0:
      - trend filter: EMA200 (LONG tylko gdy close > EMA200; SHORT gdy close < EMA200),
      - wejście: przecięcie EMA50 / EMA200 (z *zamkniętej* świecy),
      - filtr ATR% (min),
      - domyślnie SHORT wyłączony w paper (opcjonalnie można włączyć).
    """

    def __init__(self, ex_mgr: ExchangeManager):
        self.ex = ex_mgr

    def _load_closes(self, symbol: str, timeframe: str, limit: int = 400) -> List[float]:
        ohlcv = self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        if not ohlcv or len(ohlcv) < 250:
            return []
        return [float(x[4]) for x in ohlcv]

    def _atr_pct(self, symbol: str, timeframe: str, atr_len: int = 14) -> Optional[float]:
        ohlcv = self.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=atr_len + 100)
        if not ohlcv or len(ohlcv) < atr_len + 1:
            return None
        # ATR
        from archive.legacy_bot.managers.scanner import _compute_atr  # reuse
        atr = _compute_atr(ohlcv, atr_len)
        if not atr:
            return None
        last = float(ohlcv[-1][4])
        return float(atr) / last * 100.0

    def check_signal(
        self,
        *,
        symbol: str,
        timeframe: str = "15m",
        ema_fast: int = 50,
        ema_slow: int = 200,
        min_atr_pct: float = 0.5,
        allow_short: bool = False,
    ) -> Optional[SignalDTO]:
        closes = self._load_closes(symbol, timeframe=timeframe, limit=max(ema_slow * 2, 400))
        if len(closes) < ema_slow + 5:
            return None

        ema_f = _ema(closes, ema_fast)
        ema_s = _ema(closes, ema_slow)

        # używamy *zamkniętej* świecy: indeks -2 (ostatnia jest "in-progress")
        c_prev = closes[-2]
        ef_prev = ema_f[-2]
        es_prev = ema_s[-2]
        c_curr = closes[-1]
        ef_curr = ema_f[-1]
        es_curr = ema_s[-1]

        # ATR filter
        atrp = self._atr_pct(symbol, timeframe=timeframe, atr_len=14)
        if atrp is None or atrp < float(min_atr_pct):
            return None

        # LONG sygnał: przejście EMA50 ponad EMA200 i close > EMA200
        crossed_long = (ef_prev <= es_prev) and (ef_curr > es_curr) and (c_curr > es_curr)
        if crossed_long:
            return SignalDTO(symbol=symbol, direction="LONG", confidence=1.0, extra={
                "ema_fast": ef_curr, "ema_slow": es_curr, "atr_pct": atrp
            })

        # SHORT sygnał (opcjonalnie): EMA50 pod EMA200 i close < EMA200
        if allow_short:
            crossed_short = (ef_prev >= es_prev) and (ef_curr < es_curr) and (c_curr < es_curr)
            if crossed_short:
                return SignalDTO(symbol=symbol, direction="SHORT", confidence=1.0, extra={
                    "ema_fast": ef_curr, "ema_slow": es_curr, "atr_pct": atrp
                })

        return None

    def find_signals_for_list(
        self,
        symbols: List[str],
        timeframe: str = "15m",
        min_atr_pct: float = 0.5,
        allow_short: bool = False,
    ) -> List[SignalDTO]:
        out: List[SignalDTO] = []
        for s in symbols:
            sig = self.check_signal(symbol=s, timeframe=timeframe, min_atr_pct=min_atr_pct, allow_short=allow_short)
            if sig:
                out.append(sig)
        return out
