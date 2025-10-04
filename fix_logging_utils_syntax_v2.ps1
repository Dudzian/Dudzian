param(
    [string]$RepoRoot = "."
)
$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $RepoRoot

$path = "KryptoLowca\logging_utils.py"
if (-not (Test-Path -LiteralPath $path)) {
    throw "Nie znaleziono pliku: $path"
}

$orig = Get-Content -Raw -LiteralPath $path -Encoding UTF8

# Find the broken PowerShell-style line inside Python code and replace whole line
$pattern = '(?m)^\s*atexit\.register\(\s*lambda:.*-and.*-eq\s*\$null\)\s*\)\s*$'
$replacement = 'atexit.register(lambda: (_LISTENER is not None) and (_LISTENER.stop() is None))'

if ($orig -match $pattern) {
    Copy-Item -LiteralPath $path -Destination ($path + ".bak2") -Force
    $fixed = [regex]::Replace($orig, $pattern, $replacement)
    Set-Content -LiteralPath $path -Value $fixed -Encoding UTF8
    Write-Host "OK  (patched): $path"
} else {
    Write-Host "SKIP (nie znaleziono wzorca w): $path"
}

Write-Host "Hotfix applied. Run mypy again:"
Write-Host "  mypy KryptoLowca --python-version 3.10 --show-error-codes --pretty"
