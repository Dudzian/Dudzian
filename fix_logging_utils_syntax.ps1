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
        throw "Nie znaleziono pliku: $Path"
    }
    $text = Get-Content -Raw -LiteralPath $Path -Encoding UTF8
    $new = [System.Text.RegularExpressions.Regex]::Replace($text, $Pattern, $Replacement, 'Singleline')
    if ($new -ne $text) {
        Copy-Item -LiteralPath $Path -Destination ($Path + ".bak2") -Force
        Set-Content -LiteralPath $Path -Value $new -Encoding UTF8
        Write-Host "OK  (patched): $Path"
    } else {
        Write-Host "SKIP (no match): $Path"
    }
}

Set-Location -LiteralPath $RepoRoot

$path = "KryptoLowca\logging_utils.py"
# Fix PowerShell operators accidentally inserted into Python
Replace-InFile -Path $path `
  -Pattern "atexit\.register\(\s*lambda:\s*\(_LISTENER\s+is\s+not\s+None\)\s*-and\s*\(_LISTENER\.stop\(\)\s*-eq\s*\$null\)\s*\)" `
  -Replacement 'atexit.register(lambda: (_LISTENER is not None) and (_LISTENER.stop() is None))'

Write-Host "Hotfix applied. Run mypy again:"
Write-Host "  mypy KryptoLowca --python-version 3.10 --show-error-codes --pretty"
