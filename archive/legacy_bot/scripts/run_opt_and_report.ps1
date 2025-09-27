# scripts\run_opt_and_report.ps1
param(
  [string]$Symbols = "ETH/USDT,BNB/USDT",
  [string]$Timeframe = "15m",
  [int]$MaxBars = 12000
)

$proj = Split-Path -Parent $MyInvocation.MyCommand.Path | Split-Path -Parent
$venvPy = Join-Path $proj ".venv\Scripts\python.exe"

# 1) OPTIMIZE (tu ustaw swoje siatki)
& $venvPy -m backtest.optimize `
  --symbols $Symbols `
  --timeframe $Timeframe --max_bars $MaxBars --min_trades 5 `
  --ema_slow 150 `
  --grid_min_atr_pct "0.12,0.15,0.20" `
  --grid_k_sl "1.2,1.5,1.8" `
  --grid_k_tp1 "1.0,1.2" `
  --grid_k_tp2 "2.0,2.5" `
  --grid_k_tp3 "3.0,3.5,4.0" `
  --grid_p1 "30,40" --grid_p2 "30,40" --grid_p3 "20,30" `
  --grid_trail_act_pct "0.8,1.0" `
  --grid_trail_dist_pct "0.3,0.4"

# 2) RUNNER – (opcjonalnie) tu możesz dodać wczytanie best_preset.json
# Na razie statyczne parametry:
& $venvPy -m backtest.runner `
  --symbols $Symbols `
  --timeframe $Timeframe --max_bars $MaxBars `
  --ema_slow 150 --min_atr_pct 0.15 `
  --capital 10000 --risk_pct 0.5 `
  --k_sl 1.5 --k_tp1 1.0 --k_tp2 2.0 --k_tp3 3.0 `
  --p1 40 --p2 30 --p3 30 `
  --trail_act_pct 0.8 --trail_dist_pct 0.3

# 3) RAPORT (z najnowszego folderu)
& $venvPy -m backtest.report --last --plots --html
