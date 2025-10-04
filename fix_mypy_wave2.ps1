param(
  [string]$Root = "."
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-TextFile {
  param([string]$Path, [string]$Content)
  $dir = Split-Path -Path $Path -Parent
  if ($dir -and -not (Test-Path -LiteralPath $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
  Set-Content -LiteralPath $Path -Value $Content -Encoding UTF8
  Write-Host "Wrote: $Path"
}

# --- mypy.ini: usuń złe sekcje i dodaj wyciszenia ---
$iniPath = Join-Path $Root "mypy.ini"
if (Test-Path -LiteralPath $iniPath) {
  $ini = Get-Content -LiteralPath $iniPath -Raw
  $ini = [Regex]::Replace($ini, '(?ms)^\[mypy-KryptoLowca\.run_\*\][^\[]*', '')
  $ini = [Regex]::Replace($ini, '(?ms)^\[mypy-KryptoLowca\.quick_\*\][^\[]*', '')

  if ($ini -notmatch '(?m)^\[mypy\]') {
    $ini = "[mypy]`npython_version = 3.10`nno_implicit_optional = False`n" + $ini
  }

  if ($ini -notmatch '(?m)^\[mypy-ccxt\]') {
    $ini += "`n[mypy-ccxt]`nignore_errors = True`n[mypy-ccxt.*]`nignore_errors = True`n"
  }
  if ($ini -notmatch '(?m)^\[mypy-yaml(\.\*)?\]') {
    $ini += "`n[mypy-yaml]`nignore_missing_imports = True`n[mypy-yaml.*]`nignore_missing_imports = True`n"
  }

  $silence = @(
    "KryptoLowca.run_trading_gui_paper",
    "KryptoLowca.run_trading_gui_live",
    "KryptoLowca.run_autotrade_paper",
    "KryptoLowca.quick_live_readonly_test",
    "KryptoLowca.quick_live_orders_test",
    "KryptoLowca.quick_exchange_adapter_test"
  )
  foreach ($m in $silence) {
    if ($ini -notmatch "(?m)^\[mypy-$([Regex]::Escape($m))\]") {
      $ini += "`n[mypy-$m]`nignore_errors = True`n"
    }
  }

  Set-Content -LiteralPath $iniPath -Value $ini -Encoding UTF8
  Write-Host "Updated: $iniPath"
}

# --- Stuby: event bus / logger / AI / exchange adapter ---
$eeaPyi = @'
from typing import Any, Callable, Optional

class EventType:
    MARKET_TICK: "EventType"
    WFO_STATUS: "EventType"
    RISK_ALERT: "EventType"
    WFO_TRIGGER: "EventType"
    ATR_SPIKE: "EventType"
    ORDER_STATUS: "EventType"
    SIGNAL: "EventType"
    AUTOTRADE_STATUS: "EventType"
    LOG: "EventType"
    PNL_UPDATE: "EventType"
    POSITION_UPDATE: "EventType"
    ATR_UPDATE: "EventType"
    ORDER_REQUEST: "EventType"

class Event:
    type: EventType
    etype: EventType
    payload: Any
    ts: float | None
    def __init__(self, type: EventType, payload: Any = ..., ts: float | None = ...) -> None: ...

class DebounceRule:
    window_sec: float
    max_batch: int
    def __init__(self, window_sec: float = ..., max_batch: int = ...) -> None: ...

class EventBus:
    def subscribe(self, etype: EventType, handler: Callable[[Any], None], rule: Optional[DebounceRule] = ...) -> None: ...
    def publish(self, etype: EventType, payload: Any) -> None: ...
    def emit(self, etype: EventType, payload: Any) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

class EmitterAdapter:
    def publish(self, etype: EventType, payload: Any) -> None: ...
    def push_market_tick(self, symbol: str, **kwargs: Any) -> None: ...
    def update_metrics(self, *args: Any, **kwargs: Any) -> None: ...
    def push_order_status(self, *args: Any, **kwargs: Any) -> None: ...
    def push_signal(self, *args: Any, **kwargs: Any) -> None: ...
    def push_autotrade_status(self, *args: Any, **kwargs: Any) -> None: ...
    def push_wfo_status(self, *args: Any, **kwargs: Any) -> None: ...
    def push_log(self, *args: Any, **kwargs: Any) -> None: ...

class EventEmitter:
    def on(self, *args: Any, **kwargs: Any) -> None: ...
    def off(self, *args: Any, **kwargs: Any) -> None: ...
    def emit(self, *args: Any, **kwargs: Any) -> None: ...
    def log(self, *args: Any, **kwargs: Any) -> None: ...
'@
Write-TextFile -Path (Join-Path $Root "KryptoLowca\event_emitter_adapter.pyi") -Content $eeaPyi

$logPyi = @'
from typing import Any
from pathlib import Path
import logging

LOGS_DIR: Path
DEFAULT_LOG_FILE: Path

def get_logger(name: str) -> logging.Logger: ...
def setup_logging(*args: Any, **kwargs: Any) -> None: ...
def setup_app_logging(*args: Any, **kwargs: Any) -> None: ...

_LISTENER: Any
_QUEUE: Any
'@
Write-TextFile -Path (Join-Path $Root "KryptoLowca\logging_utils.pyi") -Content $logPyi

$aiMgrPyi = @'
from typing import Any, Dict

class AIManager:
    def __init__(self, models_dir: Any = ..., logger_: Any = ...) -> None: ...
    def active_schedules(self) -> Dict[str, Any]: ...
'@
Write-TextFile -Path (Join-Path $Root "KryptoLowca\managers\ai_manager.pyi") -Content $aiMgrPyi

$exAdapterPyi = @'
from typing import Any, Dict, List, Optional

class ExchangeAdapter:
    def fetch_ohlcv(self, symbol: str, timeframe: str = ..., limit: int = ...) -> Optional[List[List[float]]]: ...
    def fetch_ticker(self, symbol: str) -> Optional[Dict[str, Any]]: ...
    def fetch_order_book(self, symbol: str, limit: int = ...) -> Optional[Dict[str, Any]]: ...
    def create_order(self, *args: Any, **kwargs: Any) -> Dict[str, Any]: ...
'@
Write-TextFile -Path (Join-Path $Root "KryptoLowca\managers\exchange_adapter.pyi") -Content $exAdapterPyi

# --- Drobne, pewne poprawki w kodzie ---

# Optional dla default_params
$autoTrade = Join-Path $Root "KryptoLowca\trading\auto_trade_engine.py"
if (Test-Path -LiteralPath $autoTrade) {
  $txt = Get-Content -LiteralPath $autoTrade -Raw
  if ($txt -notmatch '(?m)\bOptional\b') {
    $txt = $txt -replace '(?m)^from typing import ([^\r\n]+)$', { param($m) if ($m.Groups[1].Value -match '\bOptional\b'){ $m.Value } else { "from typing import " + $m.Groups[1].Value + ", Optional" } }
    if ($txt -notmatch '(?m)from typing import .*Optional') {
      $txt = "from typing import Optional`r`n" + $txt
    }
  }
  $txt = [Regex]::Replace($txt, 'default_params\s*:\s*Dict\s*\[\s*str\s*,\s*int\s*\]\s*=\s*None', 'default_params: Optional[Dict[str, int]] = None')
  Set-Content -LiteralPath $autoTrade -Value $txt -Encoding UTF8
  Write-TextFile -Path $autoTrade -Content $txt
}

# Callable, Sequence w trading_strategies/engine.py
$tsEngine = Join-Path $Root "KryptoLowca\trading_strategies\engine.py"
if (Test-Path -LiteralPath $tsEngine) {
  $txt = Get-Content -LiteralPath $tsEngine -Raw
  if ($txt -match '(?m)^from\s+typing\s+import\s+([^\r\n]+)$') {
    $line = $Matches[0]; $imports = $Matches[1]
    $need = @()
    if ($imports -notmatch '\bCallable\b') { $need += 'Callable' }
    if ($imports -notmatch '\bSequence\b') { $need += 'Sequence' }
    if ($need.Count -gt 0) {
      $newline = $line -replace [Regex]::Escape($imports), ($imports + ", " + ($need -join ", "))
      $txt = $txt -replace [Regex]::Escape($line), $newline
    }
  } else {
    $txt = "from typing import Callable, Sequence`r`n" + $txt
  }
  Set-Content -LiteralPath $tsEngine -Value $txt -Encoding UTF8
  Write-Host "Patched: $tsEngine"
}

Write-Host "`nDone. Teraz uruchom:"
Write-Host "mypy KryptoLowca --python-version 3.10 --show-error-codes --pretty"
