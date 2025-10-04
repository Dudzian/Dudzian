param(
    [string]$RepoRoot = "."
)
$ErrorActionPreference = "Stop"

function Replace-InFile {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$Pattern,
        [Parameter(Mandatory=$true)][string]$Replacement
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        Write-Host "SKIP (not found): $Path"
        return
    }
    $text = Get-Content -Raw -LiteralPath $Path -Encoding UTF8
    $new = [System.Text.RegularExpressions.Regex]::Replace($text, $Pattern, $Replacement, 'Singleline')
    if ($new -ne $text) {
        Copy-Item -LiteralPath $Path -Destination ($Path + ".bak") -Force
        Set-Content -LiteralPath $Path -Value $new -Encoding UTF8
        Write-Host "OK  (patched): $Path"
    } else {
        Write-Host "SKIP (no match): $Path"
    }
}

# Ensure we're in repo root
Set-Location -LiteralPath $RepoRoot
if (-not (Test-Path -LiteralPath "KryptoLowca")) {
    throw "Nie znaleziono folderu 'KryptoLowca' w $((Resolve-Path .).Path). Uruchom skrypt w katalogu repo lub podaj -RepoRoot."
}

# 1) backtest/autotune.py -> cast dla json.load
Replace-InFile -Path "KryptoLowca\backtest\autotune.py" `
    -Pattern "(?m)^(\s*)return\s+json\.load\(f\)\s*$" `
    -Replacement @'
$1from typing import Dict, Any, cast
$1return cast(Dict[str, Any], json.load(f))
'@

# 2) security/secret_store.py -> cast dla json.loads + Optional dla get("value")
Replace-InFile -Path "KryptoLowca\security\secret_store.py" `
    -Pattern "(?m)^(\s*)return\s+json\.loads\(self\.file_path\.read_text\(\)\)\s*$" `
    -Replacement @'
$1from typing import Dict, Any, cast
$1return cast(Dict[str, str], json.loads(self.file_path.read_text()))
'@

Replace-InFile -Path "KryptoLowca\security\secret_store.py" `
    -Pattern "(?m)^(\s*)return\s+json\.loads\(self\._metadata_path\.read_text\(\)\)\s*$" `
    -Replacement @'
$1from typing import Dict, Any, cast
$1return cast(Dict[str, str], json.loads(self._metadata_path.read_text()))
'@

Replace-InFile -Path "KryptoLowca\security\secret_store.py" `
    -Pattern "(?m)^(\s*)return\s+data\.get\(""value""\)\s*$" `
    -Replacement @'
$1from typing import Optional
$1v = data.get("value")  # type: ignore[assignment]
$1return v
'@

# 3) backtest/preset_store.py -> dict_values -> list(...)
Replace-InFile -Path "KryptoLowca\backtest\preset_store.py" `
    -Pattern "(?m)^(\s*)items\s*=\s*\[p\s+for\s+p\s+in\s+items\s+if\s+metric\s+in\s+p\.metrics\]\s*$" `
    -Replacement '$1items = [p for p in list(items) if metric in p.metrics]'

# 4) backtest/runner_preset.py -> poprawny typ zwrotny
Replace-InFile -Path "KryptoLowca\backtest\runner_preset.py" `
    -Pattern "\)\s*->\s*\(Dict\[str,\s*Any\],\s*str\):" `
    -Replacement ') -> "tuple[dict[str, Any], str]":'

# 5) trading_strategies/engine.py -> Callable zamiast builtins.callable (string annotation, bez importu)
Replace-InFile -Path "KryptoLowca\trading_strategies\engine.py" `
    -Pattern "(?m)^(\s*)def\s+rolling_apply_numba\(\s*series:\s*pd\.Series,\s*window:\s*int,\s*func:\s*callable\)\s*->\s*pd\.Series:" `
    -Replacement '$1def rolling_apply_numba(series: pd.Series, window: int, func: "Callable[[Sequence[float]], float]") -> pd.Series:'

# 6) logging_utils.py -> Optional przy domyślnej wartości + poprawka dla atexit.register
Replace-InFile -Path "KryptoLowca\logging_utils.py" `
    -Pattern "(?m)^(\s*)level:\s*int\s*\|\s*str\s*=\s*None," `
    -Replacement '$1level: int | str | None = None,'

Replace-InFile -Path "KryptoLowca\logging_utils.py" `
    -Pattern "atexit\.register\(\s*lambda:\s*_LISTENER\s+and\s+_LISTENER\.stop\(\)\s*\)" `
    -Replacement 'atexit.register(lambda: (_LISTENER is not None) -and (_LISTENER.stop() -eq $null))'

# 7) services/alerting.py -> ignorujemy brak stubów requests (jeśli nie zainstalujesz types-requests)
Replace-InFile -Path "KryptoLowca\services\alerting.py" `
    -Pattern "(?m)^(\s*)import\s+requests\s*$" `
    -Replacement '$1import requests  # type: ignore[import-untyped]'

# 8) STUB dla EmitterAdapter (żeby mypy znał metody używane w innych plikach)
$stubPath = "KryptoLowca\event_emitter_adapter.pyi"
$stub = @'
from typing import Any, Optional, Dict, Callable

_Callback = Callable[[Any], None]

class EmitterAdapter:
    def push_autotrade_status(self, kind: str, *, detail: Optional[Dict[str, Any]] = ..., level: Optional[str] = ...) -> None: ...
    def push_wfo_status(self, kind: str, *, detail: Optional[Dict[str, Any]] = ...) -> None: ...
    def push_log(self, msg: str, *, level: str = ...) -> None: ...
'@
Set-Content -LiteralPath $stubPath -Value $stub -Encoding UTF8
Write-Host "OK  (created): $stubPath"

Write-Host ""
Write-Host "Patch Set #1 zastosowany."
Write-Host "Teraz (w aktywnym venv):"
Write-Host "  1) python -m pip install -U pip"
Write-Host "  2) pip install -U types-requests"
Write-Host "  3) mypy KryptoLowca --python-version 3.10 --show-error-codes --pretty"
