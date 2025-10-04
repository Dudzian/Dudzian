# -*- coding: utf-8 -*-
"""
Autotune: per-symbol uruchamia backtest.optimize, zrzuca best_preset.json do presets/live/<SYMBOL>__<TF>.json,
opcjonalnie odpala backtest.runner na tych parametrach i generuje prosty raport (equity/drawdown).

Przykład:
python -m backtest.autotune --symbols "ETH/USDT,BNB/USDT" --timeframe 15m --max_bars 12000 --run_after --report
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List
import subprocess

# ------- Pomocnicze -------

def _safe_symbol(sym: str) -> str:
    return sym.replace("/", "_").replace(":", "_")

def _print_box(title: str):
    line = "─" * max(10, len(title))
    print(f"\n┌{line}\n│ {title}\n└{line}")

def _run(cmd: List[str]) -> str:
    """Uruchom proces i zwróć stdout jako tekst (UTF-8)."""
    res = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    out = (res.stdout or "") + ("\n" + res.stderr if res.stderr else "")
    return out

def _find_best_preset_path(stdout_text: str) -> Path | None:
    """
    Szuka ścieżki do best_preset.json w logach optimize.
    Linie zwykle wyglądają tak:
    [Optimize] TOP preset zapisany: backtests\YYYYMMDD_HHMMSS\best_preset.json
    """
    m = re.search(r"best_preset\.json[^\r\n]*", stdout_text, re.IGNORECASE)
    if not m:
        return None
    # wyciągamy pełną ścieżkę z końcówką best_preset.json
    line = stdout_text[m.start(): m.end()]
    # znajdź początek ścieżki "backtests\..."
    p = re.search(r"(backtests[\\/][^\r\n]+best_preset\.json)", line, re.IGNORECASE)
    if not p:
        return None
    return Path(p.group(1)).resolve()

def _newest_best_preset(after_ts: float | None = None) -> Path | None:
    """Awaryjnie: znajdź najnowszy backtests/*/best_preset.json (opcjonalnie tylko nowsze niż after_ts)."""
    candidates = list(Path("backtests").glob("*/best_preset.json"))
    if not candidates:
        return None
    if after_ts is not None:
        candidates = [c for c in candidates if c.stat().st_mtime >= after_ts - 2]
        if not candidates:
            return None
    return max(candidates, key=lambda p: p.stat().st_mtime).resolve()

def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        from typing import Dict, Any, cast
        return cast(Dict[str, Any], json.load(f))
def _save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _build_runner_args(symbol: str, timeframe: str, max_bars: int, capital: float, risk_pct: float, preset: Dict[str, Any]) -> List[str]:
    """
    Buduje argumenty dla backtest.runner na bazie preset.json.
    Działa tolerancyjnie na nazwy kluczy.
    """
    def g(*keys, default=None):
        for k in keys:
            if k in preset: return preset[k]
        return default

    args = [
        sys.executable, "-m", "backtest.runner",
        "--symbols", symbol,
        "--timeframe", timeframe,
        "--max_bars", str(max_bars),
        "--capital", str(capital),
        "--risk_pct", str(risk_pct),
    ]
    # typowe pola z optymalizacji
    map_simple = [
        (("ema_slow",), "ema_slow"),
        (("min_atr_pct", "grid_min_atr_pct", "atr_min_pct"), "min_atr_pct"),
        (("k_sl", "grid_k_sl"), "k_sl"),
        (("k_tp1", "grid_k_tp1"), "k_tp1"),
        (("k_tp2", "grid_k_tp2"), "k_tp2"),
        (("k_tp3", "grid_k_tp3"), "k_tp3"),
        (("p1", "grid_p1"), "p1"),
        (("p2", "grid_p2"), "p2"),
        (("p3", "grid_p3"), "p3"),
        (("trail_act_pct", "grid_trail_act_pct"), "trail_act_pct"),
        (("trail_dist_pct", "grid_trail_dist_pct"), "trail_dist_pct"),
    ]
    for keys, flag in map_simple:
        val = g(*keys)
        if val is not None:
            args += [f"--{flag}", str(val)]
    return args

def _make_report(backtests_dir: Path) -> None:
    """
    Prosty raport (equity i drawdown) na bazie najnowszego trades.csv w podanym folderze backtests/<stamp>/
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[report] Pomijam wykresy (brak pandas/matplotlib?): {e}")
        return

    try:
        trades_csv = max(backtests_dir.glob("*/trades.csv"), key=lambda p: p.stat().st_mtime)
    except ValueError:
        # gdy odpalamy per folder (już wskazaliśmy właściwy znacznik czasu)
        trades_csv = backtests_dir / "trades.csv"

    if not trades_csv.exists():
        print("[report] Brak trades.csv — nie wygeneruję wykresów.")
        return

    df = pd.read_csv(trades_csv)
    if "exit_ts" not in df or "pnl_usdt" not in df:
        print("[report] trades.csv nie zawiera potrzebnych kolumn.")
        return

    # konwersja czasu
    def ts_to_dt(s):
        s = pd.to_numeric(s, errors="coerce")
        return pd.to_datetime(s, unit="ms", utc=True)

    df["exit_time_utc"] = ts_to_dt(df["exit_ts"])
    eq = df.sort_values("exit_time_utc").copy()
    eq["cum_pnl"] = eq["pnl_usdt"].cumsum()
    eq["run_max"] = eq["cum_pnl"].cummax()
    eq["dd"] = eq["cum_pnl"] - eq["run_max"]

    png_eq = trades_csv.parent / "equity_curve.png"
    png_dd = trades_csv.parent / "drawdown.png"

    plt.figure()
    plt.plot(eq["exit_time_utc"], eq["cum_pnl"])
    plt.title("Equity curve (USDT)")
    plt.xlabel("Exit time (UTC)")
    plt.ylabel("Cum PnL")
    plt.tight_layout()
    plt.savefig(png_eq)

    plt.figure()
    plt.plot(eq["exit_time_utc"], eq["dd"])
    plt.title("Drawdown (USDT)")
    plt.xlabel("Exit time (UTC)")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(png_dd)

    print(f"[report] Wykresy zapisane:\n - {png_eq}\n - {png_dd}")

# ------- Główny przebieg -------

def main():
    ap = argparse.ArgumentParser(description="Autotune parametrów i (opcjonalnie) uruchomienie backtest.runner.")
    ap.add_argument("--symbols", required=True, help="Lista symboli rozdzielona przecinkami, np. 'ETH/USDT,BNB/USDT'")
    ap.add_argument("--timeframe", required=True, help="Npz. 5m, 15m, 1h")
    ap.add_argument("--max_bars", type=int, default=10000)
    ap.add_argument("--ema_slow", type=int, default=150)
    ap.add_argument("--min_trades", type=int, default=3)

    # Grid dla optimize (domyślne rozsądne; można nadpisać flagami)
    ap.add_argument("--grid_min_atr_pct", default="0.10,0.12,0.15")
    ap.add_argument("--grid_k_sl", default="1.0,1.2,1.5")
    ap.add_argument("--grid_k_tp1", default="0.8,1.0,1.2")
    ap.add_argument("--grid_k_tp2", default="1.5,2.0,2.5")
    ap.add_argument("--grid_k_tp3", default="2.5,3.0,3.5")
    ap.add_argument("--grid_p1", default="40,50")
    ap.add_argument("--grid_p2", default="30,40")
    ap.add_argument("--grid_p3", default="20,30")
    ap.add_argument("--grid_trail_act_pct", default="0.5,0.8,1.0")
    ap.add_argument("--grid_trail_dist_pct", default="0.2,0.3,0.4")

    # Runner + raport
    ap.add_argument("--capital", type=float, default=10000)
    ap.add_argument("--risk_pct", type=float, default=0.5)
    ap.add_argument("--run_after", action="store_true", help="Po optymalizacji uruchom backtest.runner na najlepszych parametrach.")
    ap.add_argument("--report", action="store_true", help="Po uruchomieniu runner wygeneruj equity/drawdown.")
    args = ap.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    tf = args.timeframe

    presets_dir = Path("presets") / "live"
    presets_dir.mkdir(parents=True, exist_ok=True)

    for sym in symbols:
        _print_box(f"Autotune: {sym} @ {tf}")

        start_ts = time.time()

        # zbuduj komendę optimize (po 1 symbolu)
        optimize_cmd = [
            sys.executable, "-m", "backtest.optimize",
            "--symbols", sym,
            "--timeframe", tf,
            "--max_bars", str(args.max_bars),
            "--min_trades", str(args.min_trades),
            "--ema_slow", str(args.ema_slow),
            "--grid_min_atr_pct", args.grid_min_atr_pct,
            "--grid_k_sl", args.grid_k_sl,
            "--grid_k_tp1", args.grid_k_tp1,
            "--grid_k_tp2", args.grid_k_tp2,
            "--grid_k_tp3", args.grid_k_tp3,
            "--grid_p1", args.grid_p1,
            "--grid_p2", args.grid_p2,
            "--grid_p3", args.grid_p3,
            "--grid_trail_act_pct", args.grid_trail_act_pct,
            "--grid_trail_dist_pct", args.grid_trail_dist_pct,
        ]

        print("[opt] Odpalam:", " ".join(optimize_cmd))
        out = _run(optimize_cmd)
        print(out)

        best_path = _find_best_preset_path(out)
        if not best_path:
            # fallback: najnowszy best_preset.json po czasie startu
            best_path = _newest_best_preset(after_ts=start_ts)
        if not best_path or not best_path.exists():
            print("[opt] NIE znaleziono best_preset.json — pomijam symbol.")
            continue

        # wczytaj preset i rozszerz metadane
        params = _load_json(best_path)
        params.setdefault("symbol", sym)
        params.setdefault("timeframe", tf)
        # zapisz do presets/live
        target = presets_dir / f"{_safe_symbol(sym)}__{tf}.json"
        _save_json(target, params)
        print(f"[opt] Zapisano preset live: {target}")

        # po optymalizacji – uruchom runner?
        if args.run_after:
            runner_args = _build_runner_args(
                symbol=sym,
                timeframe=tf,
                max_bars=args.max_bars,
                capital=args.capital,
                risk_pct=args.risk_pct,
                preset=params,
            )
            print("[run] Odpalam:", " ".join(runner_args))
            run_out = _run(runner_args)
            print(run_out)

            # wydobądź folder backtests/<stamp> z logu runnera
            m = re.findall(r"Zapisano:\s*(backtests[\\/][^\\/\r\n]+)[\\/]trades\.csv", run_out, flags=re.IGNORECASE)
            backtests_dir = None
            if m:
                backtests_dir = Path(m[-1]).resolve()
            else:
                # fallback: najnowszy folder z trades.csv
                try:
                    trades = max(Path("backtests").glob("*/trades.csv"), key=lambda p: p.stat().st_mtime)
                    backtests_dir = trades.parent.resolve()
                except ValueError:
                    backtests_dir = None

            if args.report and backtests_dir:
                _make_report(backtests_dir)

    print("\n[done] Autotune zakończony.")

if __name__ == "__main__":
    main()

