param(
  [Parameter(Mandatory=$true)][string]$Symbols,
  [Parameter(Mandatory=$true)][string]$Timeframe,
  [int]$MaxBars = 12000,
  [int]$EmaSlow = 150,
  [int]$MinTrades = 3,
  [double]$Capital = 10000,
  [double]$RiskPct = 0.5,

  # nadpisania siatki (opcjonalnie)
  [string]$GridMinAtrPct = "0.10,0.12,0.15",
  [string]$GridKSl = "1.0,1.2,1.5",
  [string]$GridKTp1 = "0.8,1.0,1.2",
  [string]$GridKTp2 = "1.5,2.0,2.5",
  [string]$GridKTp3 = "2.5,3.0,3.5",
  [string]$GridP1 = "40,50",
  [string]$GridP2 = "30,40",
  [string]$GridP3 = "20,30",
  [string]$GridTrailActPct = "0.5,0.8,1.0",
  [string]$GridTrailDistPct = "0.2,0.3,0.4"
)

$cmd = @(
  "python","-m","backtest.autotune",
  "--symbols",$Symbols,
  "--timeframe",$Timeframe,
  "--max_bars",$MaxBars,
  "--ema_slow",$EmaSlow,
  "--min_trades",$MinTrades,
  "--capital",$Capital,
  "--risk_pct",$RiskPct,
  "--grid_min_atr_pct",$GridMinAtrPct,
  "--grid_k_sl",$GridKSl,
  "--grid_k_tp1",$GridKTp1,
  "--grid_k_tp2",$GridKTp2,
  "--grid_k_tp3",$GridKTp3,
  "--grid_p1",$GridP1,
  "--grid_p2",$GridP2,
  "--grid_p3",$GridP3,
  "--grid_trail_act_pct",$GridTrailActPct,
  "--grid_trail_dist_pct",$GridTrailDistPct,
  "--run_after","--report"
)

Write-Host "RUN:" ($cmd -join " ")
& $cmd
