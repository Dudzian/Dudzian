# backtest/report.py
import argparse, pathlib
import pandas as pd

try:
    import matplotlib.pyplot as plt
    MATPLOT = True
except Exception:
    MATPLOT = False

from bot_core.backtest.trade_loader import load_trades

def find_latest(parent: pathlib.Path, name: str):
    paths = list(parent.glob(f"*/{name}"))
    if not paths:
        raise FileNotFoundError(f"Nie znaleziono żadnego pliku {name} w {parent}")
    return max(paths, key=lambda p: p.stat().st_mtime)

def summaries(df: pd.DataFrame):
    out = {}
    out["per_symbol_pnl"] = df.groupby("symbol")["pnl_usdt"].sum().sort_values()
    out["per_symbol_winrate"] = (df.assign(win=df["pnl_usdt"] > 0)
                                   .groupby("symbol")["win"].mean()
                                   .mul(100).round(1))
    out["per_symbol_R"] = pd.DataFrame({
        "R_avg": df.groupby("symbol")["r_multiple"].mean().round(3),
        "R_med": df.groupby("symbol")["r_multiple"].median().round(3),
    })
    out["R_mean"] = float(df["r_multiple"].mean())
    out["R_median"] = float(df["r_multiple"].median())
    return out

def make_plots(df: pd.DataFrame, outdir: pathlib.Path):
    if not MATPLOT:
        return []

    out = []

    # equity + dd
    eq = df.sort_values("exit_time_utc").copy()
    eq["cum_pnl"] = eq["pnl_usdt"].cumsum()
    eq["run_max"] = eq["cum_pnl"].cummax()
    eq["dd"] = eq["cum_pnl"] - eq["run_max"]

    png_eq = outdir / "equity_curve.png"
    png_dd = outdir / "drawdown.png"

    plt.figure()
    plt.plot(eq["exit_time_utc"], eq["cum_pnl"])
    plt.title("Equity curve (USDT)")
    plt.xlabel("Exit time (UTC)")
    plt.ylabel("Cum PnL")
    plt.tight_layout()
    plt.savefig(png_eq)
    out.append(png_eq)

    plt.figure()
    plt.plot(eq["exit_time_utc"], eq["dd"])
    plt.title("Drawdown (USDT)")
    plt.xlabel("Exit time (UTC)")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    plt.savefig(png_dd)
    out.append(png_dd)

    # histogram R
    if "r_multiple" in df:
        png_hist_r = outdir / "hist_r.png"
        plt.figure()
        df["r_multiple"].plot(kind="hist", bins=20)
        plt.title("Histogram R")
        plt.xlabel("R multiple")
        plt.tight_layout()
        plt.savefig(png_hist_r)
        out.append(png_hist_r)

    return out

def make_html_report(df: pd.DataFrame, sums: dict, outdir: pathlib.Path):
    html = outdir / "report.html"

    parts = []
    parts.append("<h2>Ostatnie transakcje</h2>")
    cols = [c for c in ["entry_time_utc","exit_time_utc","symbol","timeframe",
                        "entry_price","exit_price_wap","pnl_usdt","pnl_pct",
                        "r_multiple","hold_minutes","exit_reason"] if c in df.columns]
    parts.append(df[cols].tail(50).to_html(index=False))

    parts.append("<h2>Wynik per symbol (USDT)</h2>")
    parts.append(sums["per_symbol_pnl"].to_frame("pnl_usdt").to_html())

    parts.append("<h2>Winrate per symbol (%)</h2>")
    parts.append(sums["per_symbol_winrate"].to_frame("winrate_%").to_html())

    parts.append("<h2>R multiple — średnia / mediana per symbol</h2>")
    parts.append(sums["per_symbol_R"].to_html())

    parts.append(f"<h2>R (całość): mean={sums['R_mean']:.4f} / median={sums['R_median']:.4f}</h2>")

    # obrazki jeśli są
    for name in ["equity_curve.png", "drawdown.png", "hist_r.png"]:
        p = outdir / name
        if p.exists():
            parts.append(f'<h3>{name}</h3><img src="{name}" style="max-width:100%;height:auto;">')

    html.write_text("""<html><meta charset="utf-8"><body style="font-family:Arial, sans-serif;">"""
                    + "\n".join(parts) + "</body></html>", encoding="utf-8")
    return html

def main():
    ap = argparse.ArgumentParser(description="Raport z najnowszego backtestu")
    ap.add_argument("--input", default=None, help="Folder z backtestem (zawiera trades.csv). Jeśli brak: użyj --last.")
    ap.add_argument("--last", action="store_true", help="Użyj najnowszego folderu w backtests")
    ap.add_argument("--html", action="store_true", help="Zapisz report.html")
    ap.add_argument("--plots", action="store_true", help="Zapisz wykresy (wymaga matplotlib)")
    args = ap.parse_args()

    root = pathlib.Path(".").resolve()
    backtests = root / "backtests"

    if args.input:
        input_dir = pathlib.Path(args.input)
    elif args.last:
        # bierzemy folder po obecności trades.csv
        latest_trades = find_latest(backtests, "trades.csv")
        input_dir = latest_trades.parent
    else:
        ap.error("Podaj --input <folder> albo --last")

    df, csv_path = load_trades(input_dir)
    print(f"[Report] Plik: {csv_path}")

    sums = summaries(df)

    # Wydruk skrótów w konsoli
    print("\n[Per symbol PnL]")
    print(sums["per_symbol_pnl"].to_string())
    print("\n[Winrate %]")
    print(sums["per_symbol_winrate"].to_string())
    print("\n[R avg/med per symbol]")
    print(sums["per_symbol_R"].to_string())
    print(f"\n[R całość] mean={sums['R_mean']:.4f} | median={sums['R_median']:.4f}")

    if args.plots:
        if not MATPLOT:
            print("\n[Plots] matplotlib niedostępny – pomijam.")
        else:
            files = make_plots(df, input_dir)
            for f in files:
                print(f"[Plots] zapisano: {f}")

    if args.html:
        html = make_html_report(df, sums, input_dir)
        print(f"[HTML] zapisano: {html}")

if __name__ == "__main__":
    main()
