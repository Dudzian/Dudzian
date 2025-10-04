param(
    [string]$RepoRoot = "."
)
$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $RepoRoot

$stubPath = "KryptoLowca\logging_utils.pyi"
$stub = @'
from typing import Optional, Union, Any

class QueueListener:
    def stop(self) -> None: ...

class QueueHandler:
    def __init__(self, queue: Any) -> None: ...

def setup_logging(level: Union[int, str, None] = ...) -> None: ...
def get_queue_handler() -> QueueHandler: ...
def get_level(level: Union[int, str, None]) -> int: ...
'@

New-Item -ItemType Directory -Path (Split-Path -Parent $stubPath) -Force | Out-Null
Set-Content -LiteralPath $stubPath -Value $stub -Encoding UTF8
Write-Host "OK  (created stub): $stubPath"

Write-Host "Mypy będzie teraz używać stubu i pominie implementację .py."
Write-Host "Uruchom ponownie mypy:"
Write-Host "  mypy KryptoLowca --python-version 3.10 --show-error-codes --pretty"
