# managers/scanner.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from typing import Dict, Any, List, Optional

from KryptoLowca.managers.exchange_manager import ExchangeManager


def _compute_atr(ohlcv: List[List[float]], length: int) -> Optional[float]:
    """
    Wilder ATR – initial SMA(TR[0:N]), dalej wygładzanie.
    OHLCV: [ts, open, high, low, close, volume] – rosnąco.
    """
    if not ohlcv or len(ohlcv) < length + 1:
        return None
    trs: List[float] = []
    prev_close = float(ohlcv[0][4])
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


class MarketScanner:
    """
    Skaner rynku:
      - wybiera pary kończące się na /USDT,
      - liczy spread%, ATR% (na wskazanym TF) oraz szacuje wolumen w USDT,
      - filtruje i zwraca TOP-N w postaci listy słowników:
        {"symbol","last","atr","atr_pct","spread_pct","volume"}.
    """

    def __init__(self, ex_mgr: ExchangeManager):
        self.ex = ex_mgr
        self._last_rules: Dict[str, Any] = {}

    def ensure_markets(self) -> None:
        if not self._last_rules:
            self._last_rules = self.ex.load_markets()

    def _list_usdt_symbols(self) -> List[str]:
        self.ensure_markets()
        return [s for s in self._last_rules.keys() if s.endswith("/USDT")]

    def scan_usdt_markets(
        self,
        *,
        timeframe: str = "15m",
        atr_len: int = 14,
        min_atr_pct: float = 0.5,
        max_spread_pct: float = 0.1,
        min_quote_volume: float = 1_000_000.0,
        top_n: int = 10,
        hard_limit_symbols: Optional[int] = 200,
        sleep_between: float = 0.0,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        syms = self._list_usdt_symbols()
        if hard_limit_symbols and len(syms) > hard_limit_symbols:
            syms = syms[:hard_limit_symbols]

        for sym in syms:
            try:
                t = self.ex.fetch_ticker(sym) or {}
                bid = float(t.get("bid") or 0.0) or None
                ask = float(t.get("ask") or 0.0) or None
                last = float(t.get("last") or t.get("close") or 0.0)
                if last <= 0:
                    continue

                spread_pct = 0.0
                if bid and ask and ask > 0:
                    spread_pct = (ask - bid) / ask * 100.0

                # wolumen w USDT (jeśli mamy quoteVolume, bierzemy je; jeśli nie – baseVolume*last)
                vol_quote = t.get("quoteVolume")
                if vol_quote is not None:
                    volume_usdt = float(vol_quote)
                else:
                    base_vol = t.get("baseVolume")
                    volume_usdt = float(base_vol) * last if base_vol is not None else -1.0

                ohlcv = self.ex.fetch_ohlcv(sym, timeframe=timeframe, limit=atr_len + 100)
                if not ohlcv or len(ohlcv) < atr_len + 1:
                    continue

                atr = _compute_atr(ohlcv, atr_len) or 0.0
                if atr <= 0:
                    continue
                atr_pct = atr / last * 100.0

                # filtry
                if atr_pct < float(min_atr_pct):
                    continue
                if spread_pct > float(max_spread_pct):
                    continue
                if volume_usdt >= 0 and volume_usdt < float(min_quote_volume):
                    continue

                out.append({
                    "symbol": sym,
                    "last": last,
                    "atr": atr,
                    "atr_pct": atr_pct,
                    "spread_pct": spread_pct,
                    "volume": float(volume_usdt),  # <— UJEDNOLICONY KLUCZ
                })

                if sleep_between > 0:
                    time.sleep(sleep_between)
            except Exception:
                # ignoruj pojedyncze błędy
                continue

        # sortuj po atr_pct malejąco, potem po volume
        out.sort(key=lambda x: (x["atr_pct"], x["volume"]), reverse=True)
        return out[:max(1, int(top_n))]
