# backtest/walkforward.py
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple


Bar = Dict[str, float]          # oczekiwane klucze: 'ts','open','high','low','close' (min. 'close')
Trade = Dict[str, Any]          # oczekiwane klucze: 'pnl' (float, +/−), opcj.: 'risk','ts_open','ts_close','side'


# ============================================
# Konfiguracja i wagi celu
# ============================================

@dataclass
class WalkForwardConfig:
    train_ratio: float = 0.7          # część IS (reszta OOS) w pojedynczym oknie
    step: int = 1000                   # przesunięcie okna (w barach)
    min_bars: int = 2000               # minimalna liczba barów do pierwszego okna
    min_trades: int = 10               # minimalna liczba transakcji wymagana w IS
    require_profitable_is: bool = True # wymagać PF>1 w IS
    windows_limit: Optional[int] = None  # maks. liczba okien (None=bez limitu)


@dataclass
class ObjectiveWeights:
    w_pf: float = 1.0
    w_expectancy: float = 1.0
    w_sharpe: float = 0.2
    w_maxdd: float = 0.5               # karzemy DD (odejmowane)
    pf_cap: float = 3.0                # ograniczamy wpływ bardzo wysokiego PF


# ============================================
# Wyniki i struktury
# ============================================

@dataclass
class WindowResult:
    i0: int
    i1: int
    j0: int
    j1: int
    params_best: Dict[str, Any]
    is_metrics: Dict[str, float]
    oos_metrics: Dict[str, float]
    score_is: float
    score_oos: float


@dataclass
class WalkForwardResult:
    windows: List[WindowResult] = field(default_factory=list)

    @property
    def best_params_global(self) -> Optional[Dict[str, Any]]:
        """Wybór globalny: parametry z najlepiej sprawdzającego się OOS okna (score_oos)."""
        if not self.windows:
            return None
        best = max(self.windows, key=lambda w: w.score_oos)
        return best.params_best

    def oos_summary(self) -> Dict[str, float]:
        """Średnie OOS: PF, Exp, Sharpe, MaxDD, Trades."""
        if not self.windows:
            return {}
        ks = ("pf", "expectancy", "sharpe", "max_dd", "trades")
        agg = {k: 0.0 for k in ks}
        for w in self.windows:
            for k in ks:
                agg[k] += float(w.oos_metrics.get(k, 0.0))
        n = float(len(self.windows))
        return {k: (agg[k] / n) for k in ks}


# ============================================
# Funkcje pomocnicze
# ============================================

def slice_windows(bars: List[Bar], cfg: WalkForwardConfig) -> List[Tuple[int, int, int, int]]:
    """
    Zwraca listę okien (i0,i1,j0,j1): IS = [i0,i1), OOS = [j0,j1).
    """
    n = len(bars)
    if n < cfg.min_bars:
        return []
    win: List[Tuple[int, int, int, int]] = []
    start = 0
    made = 0
    while True:
        end = min(n, start + cfg.step + cfg.step)  # przynajmniej 2*step do podziału
        if end - start < cfg.min_bars:
            break
        cut = int(start + cfg.train_ratio * (end - start))
        i0, i1 = start, cut
        j0, j1 = cut, end
        if (j1 - j0) <= 0 or (i1 - i0) <= 0:
            break
        win.append((i0, i1, j0, j1))
        made += 1
        if cfg.windows_limit is not None and made >= cfg.windows_limit:
            break
        start += cfg.step
        if end >= n:
            break
    return win


def pf_from_trades(trades: Iterable[Trade]) -> float:
    gross_p = 0.0
    gross_l = 0.0
    for t in trades:
        pnl = float(t.get("pnl", 0.0))
        if pnl >= 0:
            gross_p += pnl
        else:
            gross_l += -pnl
    if gross_l <= 0:
        return float("inf") if gross_p > 0 else 0.0
    return gross_p / gross_l


def expectancy_from_trades(trades: Iterable[Trade]) -> float:
    xs = [float(t.get("pnl", 0.0)) for t in trades]
    if not xs:
        return 0.0
    return sum(xs) / float(len(xs))


def sharpe_from_trades(trades: Iterable[Trade]) -> float:
    xs = [float(t.get("pnl", 0.0)) for t in trades]
    if len(xs) < 2:
        return 0.0
    mu = statistics.mean(xs)
    sd = statistics.pstdev(xs) or 1e-9
    return mu / sd


def max_dd_from_trades(trades: Iterable[Trade]) -> float:
    """Max Drawdown liczony po krzywej equity (cumsum pnl)."""
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        eq += float(t.get("pnl", 0.0))
        peak = max(peak, eq)
        max_dd = min(max_dd, eq - peak)  # ujemne
    return abs(max_dd)


def summarize_trades(trades: List[Trade]) -> Dict[str, float]:
    return {
        "pf": pf_from_trades(trades),
        "expectancy": expectancy_from_trades(trades),
        "sharpe": sharpe_from_trades(trades),
        "max_dd": max_dd_from_trades(trades),
        "trades": float(len(trades)),
    }


def score(metrics: Dict[str, float], w: ObjectiveWeights) -> float:
    pf = min(metrics.get("pf", 0.0), w.pf_cap)
    exp = metrics.get("expectancy", 0.0)
    shr = metrics.get("sharpe", 0.0)
    dd = metrics.get("max_dd", 0.0)
    return w.w_pf * pf + w.w_expectancy * exp + w.w_sharpe * shr - w.w_maxdd * dd


# ============================================
# Główne API WFO
# ============================================

StrategyFn = Callable[[List[Bar], Dict[str, Any]], List[Trade]]


def run_walkforward(
    bars: List[Bar],
    param_grid: List[Dict[str, Any]],
    strategy_fn: StrategyFn,
    cfg: Optional[WalkForwardConfig] = None,
    weights: Optional[ObjectiveWeights] = None,
) -> WalkForwardResult:
    """
    Minimalny, samowystarczalny WFO:
      - dzieli dane na okna IS/OOS
      - dla każdego IS wybiera parametry maksymalizujące funkcję celu
      - ocenia zwycięzcę OOS
      - zwraca listę wyników plus wybór globalny (best_params_global)
    Wymagane: strategy_fn zwraca listę transakcji (dict z kluczem 'pnl').
    """
    cfg = cfg or WalkForwardConfig()
    w = weights or ObjectiveWeights()
    windows = slice_windows(bars, cfg)
    out = WalkForwardResult()

    if not windows:
        return out

    for (i0, i1, j0, j1) in windows:
        is_data = bars[i0:i1]
        oos_data = bars[j0:j1]

        best_params: Optional[Dict[str, Any]] = None
        best_is_score = -float("inf")
        best_is_metrics: Dict[str, float] = {}

        for params in param_grid:
            try:
                is_trades = strategy_fn(is_data, params)
            except Exception:
                # jeśli strategia padła dla paramów – pomiń
                continue

            if not isinstance(is_trades, list):
                raise TypeError("strategy_fn musi zwracać listę transakcji (list[Trade]).")

            m_is = summarize_trades(is_trades)

            if m_is.get("trades", 0.0) < cfg.min_trades:
                continue
            if cfg.require_profitable_is and m_is.get("pf", 0.0) <= 1.0:
                continue

            s_is = score(m_is, w)
            if s_is > best_is_score:
                best_is_score = s_is
                best_is_metrics = m_is
                best_params = dict(params)

        # jeśli nie wybrano nic sensownego – przeskocz okno
        if best_params is None:
            continue

        # OOS ocena zwycięzcy
        try:
            oos_trades = strategy_fn(oos_data, best_params)
        except Exception:
            # OOS padł – uznaj jako słabe okno
            oos_trades = []

        m_oos = summarize_trades(oos_trades)
        s_oos = score(m_oos, w)

        out.windows.append(
            WindowResult(
                i0=i0, i1=i1, j0=j0, j1=j1,
                params_best=best_params,
                is_metrics=best_is_metrics,
                oos_metrics=m_oos,
                score_is=best_is_score,
                score_oos=s_oos,
            )
        )

    return out
