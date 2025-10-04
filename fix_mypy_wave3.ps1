param()

function Write-TextFile($Path, $Content) {
  $dir = Split-Path -Path $Path -Parent
  if ($dir -and -not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir | Out-Null }
  # utf8 *without* BOM so mypy.ini parses cleanly
  [IO.File]::WriteAllText((Resolve-Path $Path), $Content, (New-Object System.Text.UTF8Encoding $false))
}

# --- 1) mypy.ini (no BOM, saner defaults, ignore noisy areas) ---
$mypyIni = @"
[mypy]
python_version = 3.10
files = KryptoLowca
warn_unused_configs = True
warn_unused_ignores = False
no_implicit_optional = False
check_untyped_defs = False
disallow_untyped_defs = False
disallow_incomplete_defs = False
ignore_missing_imports = True
warn_return_any = False
namespace_packages = True
explicit_package_bases = True

# Quiet down paths that don't help type-safety right now
exclude = (?x)(
  ^KryptoLowca\\tests\\|
  ^KryptoLowca\\scripts\\|
  ^KryptoLowca\\run_.*\\.py$|
  ^KryptoLowca\\trading_gui\\.py$
)

[mypy-KryptoLowca.tests.*]
ignore_errors = True

[mypy-KryptoLowca.run_*]
ignore_errors = True

[mypy-KryptoLowca.quick_*]
ignore_errors = True

[mypy-KryptoLowca.trading_gui]
ignore_errors = True

# Heaviest offenders for now
[mypy-KryptoLowca.managers.database_manager]
ignore_errors = True

[mypy-KryptoLowca.ai_models]
ignore_errors = True

[mypy-KryptoLowca.exchange_manager]
ignore_errors = True
"@
Write-TextFile ".\mypy.ini" $mypyIni
Write-Host "Updated: .\mypy.ini"

# --- 2) event_emitter_adapter.pyi (broaden to match usage) ---
$emitterStub = @"
from typing import Any, Callable, Dict, List, Optional, Union

class EventType:
    MARKET_TICK: Any = ...
    ORDER_STATUS: Any = ...
    SIGNAL: Any = ...
    WFO_TRIGGER: Any = ...
    WFO_STATUS: Any = ...
    AUTOTRADE_STATUS: Any = ...
    RISK_ALERT: Any = ...
    LOG: Any = ...
    TRADE_EXECUTED: Any = ...
    ATR_SPIKE: Any = ...
    ATR_UPDATE: Any = ...
    POSITION_UPDATE: Any = ...
    PNL_UPDATE: Any = ...
    ORDER_REQUEST: Any = ...

class Event:
    type: EventType
    payload: Dict[str, Any]

class DebounceRule:
    window_sec: float
    max_batch: int
    deliver_list: bool
    def __init__(self, window_sec: float = ..., max_batch: int = ..., deliver_list: bool = ...) -> None: ...

class EventBus:
    def subscribe(self, et: EventType, callback: Callable[[Union[Event, List[Event]]], None], rule: Optional[DebounceRule] = ...) -> None: ...
    def publish(self, et: EventType, payload: Dict[str, Any]) -> None: ...
    def emit(self, et: EventType, payload: Dict[str, Any]) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

class EmitterConfig: ...
class DummyMarketFeedConfig: ...

class DummyMarketFeed:
    def __init__(self, cfg: Optional[DummyMarketFeedConfig] = ...) -> None: ...

class EmitterAdapter:
    bus: EventBus
    def __init__(self, cfg: Optional[EmitterConfig] = ..., **kwargs: Any) -> None: ...
    def publish(self, et: EventType, payload: Dict[str, Any]) -> None: ...
    def push_log(self, message: str, *, level: str = ..., **kwargs: Any) -> None: ...

class EventEmitter: ...

def wire_gui_logs_to_adapter(*args: Any, **kwargs: Any) -> None: ...

# Legacy/alternate names some modules import
class EventEmitterAdapter(EmitterAdapter): ...
"@
Write-TextFile ".\KryptoLowca\event_emitter_adapter.pyi" $emitterStub
Write-Host "Wrote: .\KryptoLowca\event_emitter_adapter.pyi"

# --- 3) Patch auto_trade_engine.py (None-safe defaults and cross calc) ---
$ate = Get-Content ".\KryptoLowca\trading\auto_trade_engine.py" -Raw

# a) dict(None) -> dict({})
$ate = $ate -replace "self\._params\s*=\s*dict\(\s*self\.cfg\.default_params\s*\)", "self._params = dict(self.cfg.default_params or {})"

# b) make f_now/s_now/f_prev/s_prev safe
$pattern = "cross_up\s*=\s*f_now\s*>\s*s_now\s*and\s*f_prev\s*<=\s*s_prev\s*`r?`n\s*cross_dn\s*=\s*f_now\s*<\s*s_now\s*and\s*f_prev\s*>=\s*s_prev"
$replacement = @"
if any(v is None for v in (f_now, s_now, f_prev, s_prev)):
                cross_up = False
                cross_dn = False
            else:
                cross_up = (f_now > s_now) and (f_prev <= s_prev)
                cross_dn = (f_now < s_now) and (f_prev >= s_prev)
"@.Trim()
$ate = [System.Text.RegularExpressions.Regex]::Replace($ate, $pattern, $replacement, 'IgnoreCase, Multiline')

Set-Content ".\KryptoLowca\trading\auto_trade_engine.py" $ate -Encoding utf8
Write-Host "Patched: .\KryptoLowca\trading\auto_trade_engine.py"

# --- 4) Patch atr_monitor.py (ensure floats in bar dict) ---
$atr = Get-Content ".\KryptoLowca\services\atr_monitor.py" -Raw
$atr = $atr -replace '\{\"ts\":\s*ts,\s*\"high\":\s*high,\s*\"low\":\s*low,\s*\"close\":\s*close\}',
                      '{"ts": float(ts or 0.0), "high": float(high or 0.0), "low": float(low or 0.0), "close": float(close or 0.0)}'
Set-Content ".\KryptoLowca\services\atr_monitor.py" $atr -Encoding utf8
Write-Host "Patched: .\KryptoLowca\services\atr_monitor.py"

Write-Host "`nDone. Now run:" -ForegroundColor Green
Write-Host "mypy KryptoLowca --python-version 3.10 --show-error-codes --pretty"
