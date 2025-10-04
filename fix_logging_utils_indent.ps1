param(
    [string]$RepoRoot = "."
)
$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $RepoRoot

$path = "KryptoLowca\logging_utils.py"
if (-not (Test-Path -LiteralPath $path)) {
    throw "Nie znaleziono pliku: $path"
}

# Wczytaj linie
$lines = Get-Content -LiteralPath $path -Encoding UTF8

# Znajdź linię z atexit.register oraz z assert _QUEUE
$idxAtexit = ($lines | Select-String -Pattern '^\s*atexit\.register\(' -SimpleMatch:$false).LineNumber
$idxAssert = ($lines | Select-String -Pattern '^\s*assert\s+_QUEUE\s+is\s+not\s+None' -SimpleMatch:$false).LineNumber

if (-not $idxAtexit) {
    Write-Host "SKIP: Nie znaleziono linii atexit.register – brak zmian."
    exit 0
}
if (-not $idxAssert) {
    Write-Host "SKIP: Nie znaleziono linii 'assert _QUEUE is not None' – brak zmian."
    exit 0
}

# Weź wcięcie z linii atexit.register
$atexitLine = $lines[$idxAtexit - 1]
$indent = ($atexitLine -match '^\s*') | Out-Null
$indentStr = $matches[0]

# Ustaw to samo wcięcie dla linii assert
$assertLine = $lines[$idxAssert - 1]
$assertText = $assertLine.TrimStart()

$lines[$idxAssert - 1] = "$indentStr$assertText"

# Zapisz kopię i plik
Copy-Item -LiteralPath $path -Destination ($path + ".bakIndent") -Force
Set-Content -LiteralPath $path -Value $lines -Encoding UTF8

Write-Host "OK  (indentation fixed): $path"
Write-Host "Uruchom ponownie mypy:"
Write-Host "  mypy KryptoLowca --python-version 3.10 --show-error-codes --pretty"
