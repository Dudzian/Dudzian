# backtest/runner.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import sys
import csv
import json
import time
import argparse
from typing import List, Dict, Any

# --- FIX ŚCIEŻKI ---
# Dodaj katalog projektu do sys.path, aby działały importy 'managers' oraz 'backtest.*'
PKG_DIR = os.path.abspath(os.path.dirname(__file__))               # ...\KryptoŁowca\backtest
PROJECT_ROOT = os.path.abspath(os.path.join(PKG_DIR, ".."))        # ...\KryptoŁowca
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Importy – działają zarówno przy `python -m backtest.runner`, jak i `python backtest\runner.py`
try:
    from KryptoLowca.exchange_manager import ExchangeManager
except ImportError as e:
    raise SystemExit(
        "[ImportError] Nie znaleziono modułu 'KryptoLowca.exchange_manager'. "
        "Upewnij się, że uruchamiasz z katalogu projektu. Szczegóły: {e}"
    )

try:
    from backtest.engine import BacktestEngine, BacktestConfig, StrategyParams, EntryParams, ExitParams, TradeRecord
    from backtest.metrics import compute_metrics, to_dict
except ImportError:
    # Fallback, gdy uruchamiasz bez pakietu (np. bez __init__.py) – import bez prefiksu
    from engine import BacktestEngine, BacktestConfig, StrategyParams, EntryParams, ExitParams, TradeRecord
    from metrics import compute_metrics, to_dict


DEFAULT_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]


def _save_trades_csv(path: str, rows: List[TradeRecord]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "symbol","timeframe","entry_ts","entry_price","entry_qty",
            "exit_ts","exit_price_wap","pnl_usdt","pnl_pct","r_multiple",
            "fills_json"
        ])
        for t in rows:
            fills = [{"ts": fl.ts, "price": fl.price, "qty": fl.qty, "tag": fl.tag} for fl in t.fills]
            w.writerow([
                t.symbol, t.timeframe, t.entry_ts, f"{t.entry_price:.8f}", f"{t.entry_qty:.8f}",
                t.exit_ts if t.exit_ts is not None else "",
                f"{t.exit_price_wap:.8f}" if t.exit_price_wap is not None else "",
                f"{t.pnl_usdt:.8f}" if t.pnl_usdt is not None else "",
                f"{t.pnl_pct:.4f}" if t.pnl_pct is not None else "",
                f"{t.r_multiple:.4f}" if t.r_multiple is not None else "",
                json.dumps(fills, ensure_ascii=False),
            ])


def _save_summary_json(path: str, summary: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtester 1.0 – EMA50/200 + ATR exits")
    p.add_argument("--symbols", type=str, default=",".join(DEFAULT_SYMBOLS),
                   help="Lista symboli po przecinku (np. BTC/USDT,ETH/USDT)")
    p.add_argument("--timeframe", type=str, default="15m")
    p.add_argument("--ema_fast", type=int, default=50)
    p.add_argument("--ema_slow", type=int, default=200)
    p.add_argument("--atr_len", type=int, default=14)
    p.add_argument("--min_atr_pct", type=float, default=0.5)
    p.add_argument("--execution", type=str, default="next_open", choices=["next_open","close"])

    p.add_argument("--capital", type=float, default=10000.0)
    p.add_argument("--risk_pct", type=float, default=1.0)   # %
    p.add_argument("--k_sl", type=float, default=1.5)
    p.add_argument("--fee", type=float, default=0.1)        # %

    p.add_argument("--k_tp1", type=float, default=1.0)
    p.add_argument("--k_tp2", type=float, default=2.0)
    p.add_argument("--k_tp3", type=float, default=3.0)
    p.add_argument("--p1", type=float, default=50.0)        # %
    p.add_argument("--p2", type=float, default=30.0)
    p.add_argument("--p3", type=float, default=20.0)
    p.add_argument("--move_be_after_tp1", action="store_true", default=True)
    p.add_argument("--trail_act_pct", type=float, default=0.8)  # %
    p.add_argument("--trail_dist_pct", type=float, default=0.3) # %

    p.add_argument("--start_pad", type=int, default=600)
    p.add_argument("--max_bars", type=int, default=3000)
    p.add_argument("--out", type=str, default="backtests")
    return p.parse_args()


def main():
    args = parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]

    ex = ExchangeManager(exchange_id="binance")
    engine = BacktestEngine(ex)

    cfg = BacktestConfig(
        symbols=symbols,
        strategy=StrategyParams(
            timeframe=args.timeframe,
            ema_fast=args.ema_fast,
            ema_slow=args.ema_slow,
            atr_len=args.atr_len,
            min_atr_pct=args.min_atr_pct,
            execution=args.execution
        ),
        entry=EntryParams(
            capital_usdt=args.capital,
            risk_pct=args.risk_pct / 100.0,
            k_sl_atr=args.k_sl,
            fee_rate=args.fee / 100.0,
        ),
        exit=ExitParams(
            k_tp1_atr=args.k_tp1, k_tp2_atr=args.k_tp2, k_tp3_atr=args.k_tp3,
            p1=args.p1 / 100.0, p2=args.p2 / 100.0, p3=args.p3 / 100.0,
            move_sl_to_be_after_tp1=args.move_be_after_tp1,
            trail_activate_pct=args.trail_act_pct / 100.0,
            trail_dist_pct=args.trail_dist_pct / 100.0
        ),
        start_index_pad=max(args.start_pad, args.ema_slow + 10, args.atr_len + 10),
        max_bars=args.max_bars,
        save_dir=args.out
    )

    all_trades: List[TradeRecord] = []
    per_symbol_summary: Dict[str, Any] = {}

    for sym in symbols:
        trades, note = engine.run_symbol(sym, cfg)
        all_trades.extend(trades)
        per_symbol_summary[sym] = note

    # metryki zagregowane
    from backtest.metrics import compute_metrics, to_dict  # pewność, że import się złapie (po sys.path fix)
    m = compute_metrics(all_trades)
    summary = {
        "params": {
            "symbols": symbols,
            "timeframe": cfg.strategy.timeframe,
            "ema_fast": cfg.strategy.ema_fast,
            "ema_slow": cfg.strategy.ema_slow,
            "atr_len": cfg.strategy.atr_len,
            "min_atr_pct": cfg.strategy.min_atr_pct,
            "execution": cfg.strategy.execution,
            "capital": cfg.entry.capital_usdt,
            "risk_pct": cfg.entry.risk_pct,
            "k_sl": cfg.entry.k_sl_atr,
            "fee_rate": cfg.entry.fee_rate,
            "tp": {
                "k_tp1": cfg.exit.k_tp1_atr,
                "k_tp2": cfg.exit.k_tp2_atr,
                "k_tp3": cfg.exit.k_tp3_atr,
                "p1": cfg.exit.p1,
                "p2": cfg.exit.p2,
                "p3": cfg.exit.p3,
                "move_be_after_tp1": cfg.exit.move_sl_to_be_after_tp1,
                "trail_act_pct": cfg.exit.trail_activate_pct,
                "trail_dist_pct": cfg.exit.trail_dist_pct,
            }
        },
        "per_symbol": per_symbol_summary,
        "metrics": to_dict(m),
    }

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg.save_dir, f"{stamp}")
    os.makedirs(out_dir, exist_ok=True)

    trades_csv = os.path.join(out_dir, "trades.csv")
    summary_json = os.path.join(out_dir, "summary.json")

    _save_trades_csv(trades_csv, all_trades)
    _save_summary_json(summary_json, summary)

    print(f"[Backtest] Zapisano: {trades_csv}")
    print(f"[Backtest] Zapisano: {summary_json}")
    import pprint; pprint.pp(summary["metrics"])


if __name__ == "__main__":
    main()
