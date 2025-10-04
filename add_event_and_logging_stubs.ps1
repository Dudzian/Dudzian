Param(
    [string]$Root = (Get-Location).Path
)

function Write-Text {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$Content
    )
    $dir = Split-Path -Parent $Path
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    # Write UTF-8 without BOM
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $utf8NoBom)
}

$pkg = Join-Path $Root 'KryptoLowca'

# --- logging_utils.pyi (expanded) ---
$loggingStub = @"
from __future__ import annotations
import logging
from typing import Any, Optional

LOGS_DIR: str
DEFAULT_LOG_FILE: str

_LISTENER: Any
_QUEUE: Any

def setup_logging(log_level: int | None = ...) -> None: ...
def setup_app_logging(log_file: str | None = ..., log_level: int | None = ...) -> None: ...
def get_queue_handler() -> Any: ...
def get_level(name: str | int | None = ...) -> int: ...
def get_logger(name: str = ...) -> logging.Logger: ...
"@

Write-Text -Path (Join-Path $pkg 'logging_utils.pyi') -Content $loggingStub

# --- event_emitter_adapter.pyi ---
$emitterStub = @"
from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Iterable, Coroutine, Sequence
from enum import Enum
from dataclasses import dataclass

class EventType(Enum):
    TRADE_EXECUTED = 'TRADE_EXECUTED'
    ORDER_STATUS = 'ORDER_STATUS'
    SIGNAL = 'SIGNAL'
    AUTOTRADE_STATUS = 'AUTOTRADE_STATUS'
    WFO_STATUS = 'WFO_STATUS'
    METRICS = 'METRICS'
    TICK = 'TICK'

@dataclass
class Event:
    type: EventType
    payload: Dict[str, Any]

class EventBus:
    def publish(self, event_type: EventType, payload: Dict[str, Any]) -> None: ...
    def subscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None: ...
    def unsubscribe(self, event_type: EventType, handler: Callable[[Event], None]) -> None: ...

class DebounceRule:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class EmitterConfig:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class EmitterAdapter:
    bus: EventBus
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def publish(self, event_type: EventType, payload: Dict[str, Any]) -> None: ...
    def push_market_tick(self, symbol: str, *, price: float, high: float, low: float, close: float) -> None: ...
    def update_metrics(self, symbol: str, *, pf: float | None = ..., expectancy: float | None = ..., trades: int | None = ...) -> None: ...
    def push_order_status(self, *, oid: str, status: str, symbol: str, filled_qty: float | None = ..., price: float | None = ...) -> None: ...
    def push_signal(self, symbol: str, *, side: str, strength: float) -> None: ...
    def push_autotrade_status(self, *args: Any, **kwargs: Any) -> None: ...
    def push_wfo_status(self, *args: Any, **kwargs: Any) -> None: ...

# Names referenced elsewhere
class EventEmitter:  # alias-like placeholder
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class DummyMarketFeedConfig:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...

class DummyMarketFeed:
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...

def wire_gui_logs_to_adapter(*args: Any, **kwargs: Any) -> None: ...
"@

Write-Text -Path (Join-Path $pkg 'event_emitter_adapter.pyi') -Content $emitterStub

# --- mypy.ini (gentle defaults to quell Optional/Any noise) ---
$mypyIniPath = Join-Path $Root 'mypy.ini'
if (-not (Test-Path $mypyIniPath)) {
    $mypyIni = @"
[mypy]
python_version = 3.10
ignore_missing_imports = True
no_implicit_optional = False
warn_return_any = False
# Keep checks on, but feel free to tighten later module-by-module.

# Silence tests and GUI runners for now (optional, comment out if you want them typed)
[mypy-KryptoLowca.tests.*]
ignore_errors = True
[mypy-KryptoLowca.run_*]
ignore_errors = True
[mypy-KryptoLowca.quick_*]
ignore_errors = True
"@
    Write-Text -Path $mypyIniPath -Content $mypyIni
}

"Stubs updated.`n  - $(Join-Path $pkg 'logging_utils.pyi')`n  - $(Join-Path $pkg 'event_emitter_adapter.pyi')`nConfig: $(Test-Path $mypyIniPath) -> $mypyIniPath"
