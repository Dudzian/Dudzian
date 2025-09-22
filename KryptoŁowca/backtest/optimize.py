# backtest/optimize.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import csv
import json
import time
import math
import argparse
from typing import List, Dict, Any, Tuple

# --- FIX ŚCIEŻKI (jak w runner.py) ---
PKG_DIR = os.path.abspath(os.path.dirname(__file__))               # ...\KryptoŁowca\backtest
PROJECT_ROOT = os.path.abspath(os.path.join(PKG_DIR, ".."))        # ...\KryptoŁowca
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from managers.exchange_manager import ExchangeManager
from backtest.engine import BacktestEngine, BacktestConfig, StrategyParams, EntryParams, ExitParams, TradeRecord
from backtest.metrics import compute_metrics, to_dict


DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtester 1.0 – Grid Search (EMA50/200 + ATR exits)")
    p.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
                   help="Lista symboli po przecinku (np. BTC/USDT,ETH/USDT)")
    p.add_argument("--timeframe", type=str, default="15m")
    p.add_argument("--ema_fast", type=int, default=50)
    p.add_argument("--ema_slow", type=int, default=200)
    p.add_argument("--atr_len", type=int, default=14)
    p.add_argument("--execution", type=str, default="next_open", choices=["next_open","close"])

    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--risk_pct", type=float, default=1.0)   # %
    p.add_argument("--fee", type=float, default=0.1)        # %

    # siatki (możesz nadpisać parametrami, np. --grid_k_sl "1.0,1.5,2.0")
    p.add_argument("--grid_min_atr_pct", type=str, default="0.3,0.5,0.8")
    p.add_argument("--grid_k_sl", type=str, default="1.0,1.5,2.0")
    p.add_argument("--grid_k_tp1", type=str, default="0.8,1.0")
    p.add_argument("--grid_k_tp2", type=str, default="1.5,2.0,3.0")
    p.add_argument("--grid_k_tp3", type=str, default="2.5,3.0,4.0")
    p.add_argument("--grid_p1", type=str, default="50,40")  # %
    p.add_argument("--grid_p2", type=str, default="30,30")  # %
    p.add_argument("--grid_p3", type=str, default="20,30")  # %
    p.add_argument("--grid_trail_act_pct", type=str, default="0.5,0.8,1.2")  # %
    p.add_argument("--grid_trail_dist_pct", type=str, default="0.2,0.3,0.5") # %

    p.add_argument("--start_pad", type=int, default=600)
    p.add_argument("--max_bars", type=int, default=3000)
    p.add_argument("--min_trades", type=int, default=5, help="Minimalna liczba transakcji do rozważenia wyniku")
    p.add_argument("--out", type=str, default="backtests")
    return p.parse_args()


def _parse_list_floats(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _parse_list_ints(s: str) -> List[int]:
    return [int(round(float(x.strip()))) for x in s.split(",") if x.strip()]


def _score(metrics: Dict[str, Any], min_trades: int) -> float:
    """
    Funkcja celu (wyższe = lepsze):
      - podstawowo: profit_factor * net_profit,
      - kara za DD: / (1 + DD/|net_profit|) jeśli net_profit>0,
      - kara za małą liczbę transakcji: mnożnik (n_trades / (n_trades + min_trades)).
    """
    n = int(metrics.get("n_trades", 0))
    if n < max(1, min_trades):
        return -1e9  # odrzuć

    pf = metrics.get("profit_factor", 0.0)
    if pf == "inf":
        pf = 10.0  # traktuj nieskończoność jako duży mnożnik, ale skończony

    netp = float(metrics.get("net_profit", 0.0))
    dd = float(metrics.get("max_drawdown_usdt", 0.0))

    base = pf * netp
    if netp > 0:
        base = base / (1.0 + (dd / max(1.0, abs(netp))))

    # miękka kara za mało transakcji
    freq_penalty = (n / (n + float(min_trades)))
    return float(base * freq_penalty)


def main():
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    # Exchange i silnik backtestu (wspólne)
    ex = ExchangeManager(exchange_id="binance")
    engine = BacktestEngine(ex)

    # ---- Cache OHLCV (przyspieszenie) ----
    ohlcv_cache: Dict[Tuple[str, str, int], List[List[float]]] = {}

    def cached_fetch(symbol: str, timeframe: str, limit: int = 3000):
        key = (symbol, timeframe, int(limit))
        if key not in ohlcv_cache:
            ohlcv_cache[key] = engine.ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit) or []
        return ohlcv_cache[key]

    # podmień metodę w instancji (tylko na czas tej optymalizacji)
    engine.fetch_ohlcv = cached_fetch  # type: ignore

    # ---- Przygotuj siatkę ----
    grid_min_atr = _parse_list_floats(args.grid_min_atr_pct)
    grid_k_sl = _parse_list_floats(args.grid_k_sl)
    grid_k_tp1 = _parse_list_floats(args.grid_k_tp1)
    grid_k_tp2 = _parse_list_floats(args.grid_k_tp2)
    grid_k_tp3 = _parse_list_floats(args.grid_k_tp3)
    grid_p1 = _parse_list_ints(args.grid_p1)   # %
    grid_p2 = _parse_list_ints(args.grid_p2)   # %
    grid_p3 = _parse_list_ints(args.grid_p3)   # %
    grid_trail_act = _parse_list_floats(args.grid_trail_act_pct)
    grid_trail_dist = _parse_list_floats(args.grid_trail_dist_pct)

    # sanity: przefiltruj te preset-y, w których p1+p2+p3 > 100
    ppresets: List[Tuple[float,float,float]] = []
    for p1 in grid_p1:
        for p2 in grid_p2:
            for p3 in grid_p3:
                if (p1 + p2 + p3) <= 100:
                    ppresets.append((p1/100.0, p2/100.0, p3/100.0))
    if not ppresets:
        ppresets = [(0.5, 0.3, 0.2)]

    # ---- Uruchom grid ----
    results: List[Dict[str, Any]] = []
    total_runs = len(grid_min_atr) * len(grid_k_sl) * len(grid_k_tp1) * len(grid_k_tp2) * len(grid_k_tp3) * len(ppresets) * len(grid_trail_act) * len(grid_trail_dist)
    run_idx = 0

    for min_atr in grid_min_atr:
        for k_sl in grid_k_sl:
            for k1 in grid_k_tp1:
                for k2 in grid_k_tp2:
                    if k2 <= k1:  # sensowny porządek TP
                        continue
                    for k3 in grid_k_tp3:
                        if k3 <= k2:
                            continue
                        for (p1, p2, p3) in ppresets:
                            for ta in grid_trail_act:
                                for td in grid_trail_dist:
                                    run_idx += 1
                                    tag = f"{run_idx}/{total_runs}"
                                    # konfiguracja
                                    cfg = BacktestConfig(
                                        symbols=symbols,
                                        strategy=StrategyParams(
                                            timeframe=args.timeframe,
                                            ema_fast=args.ema_fast,
                                            ema_slow=args.ema_slow,
                                            atr_len=args.atr_len,
                                            min_atr_pct=min_atr,
                                            execution=args.execution
                                        ),
                                        entry=EntryParams(
                                            capital_usdt=args.capital,
                                            risk_pct=args.risk_pct / 100.0,
                                            k_sl_atr=k_sl,
                                            fee_rate=args.fee / 100.0,
                                        ),
                                        exit=ExitParams(
                                            k_tp1_atr=k1, k_tp2_atr=k2, k_tp3_atr=k3,
                                            p1=p1, p2=p2, p3=p3,
                                            move_sl_to_be_after_tp1=True,
                                            trail_activate_pct=ta / 100.0,
                                            trail_dist_pct=td / 100.0
                                        ),
                                        start_index_pad=max(args.start_pad, args.ema_slow + 10, args.atr_len + 10),
                                        max_bars=args.max_bars,
                                        save_dir=args.out
                                    )

                                    # backtest wielu symboli
                                    all_trades: List[TradeRecord] = []
                                    per_symbol: Dict[str, Any] = {}
                                    for sym in symbols:
                                        trades, note = engine.run_symbol(sym, cfg)
                                        all_trades.extend(trades)
                                        per_symbol[sym] = note

                                    m = compute_metrics(all_trades)
                                    md = to_dict(m)
                                    sc = _score(md, args.min_trades)

                                    rec = {
                                        "score": round(sc, 6),
                                        "params": {
                                            "min_atr_pct": min_atr,
                                            "k_sl": k_sl,
                                            "k_tp1": k1, "k_tp2": k2, "k_tp3": k3,
                                            "p1": p1, "p2": p2, "p3": p3,
                                            "trail_act_pct": ta, "trail_dist_pct": td,
                                            "timeframe": args.timeframe,
                                            "ema_fast": args.ema_fast,
                                            "ema_slow": args.ema_slow,
                                            "capital": args.capital,
                                            "risk_pct": args.risk_pct,
                                            "fee": args.fee,
                                            "symbols": symbols,
                                        },
                                        "metrics": md,
                                    }
                                    results.append(rec)

                                    # proste info w konsoli
                                    if run_idx % 10 == 0:
                                        print(f"[{tag}] score={rec['score']} net={md['net_profit']} PF={md['profit_factor']} n={md['n_trades']}")

    # ---- Zapis wyników ----
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(args.out, f"{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    # CSV
    csv_path = os.path.join(out_dir, "grid_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "score","n_trades","net_profit","profit_factor","win_rate_%","max_drawdown_usdt",
            "timeframe","ema_fast","ema_slow","min_atr_pct",
            "k_sl","k_tp1","k_tp2","k_tp3","p1","p2","p3","trail_act_%","trail_dist_%","symbols"
        ])
        for r in sorted(results, key=lambda x: x["score"], reverse=True):
            md = r["metrics"]; pm = r["params"]
            w.writerow([
                r["score"], md["n_trades"], md["net_profit"], md["profit_factor"], md["win_rate_%"], md["max_drawdown_usdt"],
                pm["timeframe"], pm["ema_fast"], pm["ema_slow"], pm["min_atr_pct"],
                pm["k_sl"], pm["k_tp1"], pm["k_tp2"], pm["k_tp3"],
                pm["p1"], pm["p2"], pm["p3"], pm["trail_act_pct"], pm["trail_dist_pct"],
                ";".join(pm["symbols"])
            ])

    # Najlepszy preset
    best = max(results, key=lambda x: x["score"]) if results else None
    if best:
        best_path = os.path.join(out_dir, "best_preset.json")
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best, f, ensure_ascii=False, indent=2)
        print(f"[Optimize] TOP preset zapisany: {best_path}")
    else:
        print("[Optimize] Brak wyników (sprawdź siatkę lub dane).")

    print(f"[Optimize] Wyniki siatki: {csv_path}")
    print("[Optimize] TOP 5:")
    for r in sorted(results, key=lambda x: x["score"], reverse=True)[:5]:
        md = r["metrics"]; pm = r["params"]
        print(f"  score={r['score']} net={md['net_profit']} PF={md['profit_factor']} n={md['n_trades']}  "
              f"SL={pm['k_sl']} TP=({pm['k_tp1']},{pm['k_tp2']},{pm['k_tp3']}) "
              f"P=({int(pm['p1']*100)}/{int(pm['p2']*100)}/{int(pm['p3']*100)})  "
              f"ATR%>={pm['min_atr_pct']} Trail={pm['trail_act_pct']}%/{pm['trail_dist_pct']}%  TF={pm['timeframe']}")


if __name__ == "__main__":
    main()
