$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_stage10_soak_identity.ps1")

# Tylko parsowanie: nie wykonuj live, rebootu, uslug ani proby 24h.
$parseTokens = $null
$parseErrors = $null
$null = [System.Management.Automation.Language.Parser]::ParseFile(
  (Join-Path $PSScriptRoot "windows_stage10_live.ps1"),
  [ref]$parseTokens, [ref]$parseErrors)
if ($parseErrors.Count -ne 0) { throw "blad skladni skryptu odmowy Stage 10" }

$expected = @{
  pid = 42
  started = "2026-10-29T00:00:00.0000000Z"
  backend_path = 'C:\Program Files\CryptoHunter\CryptoHunterBackend.exe'
  backend_sha256 = 'a' * 64
}
Assert-Stage10SoakIdentity -Expected $expected -Observed $expected.Clone()
$changed = @(
  @{ pid = 43 },
  @{ started = "2026-10-29T00:00:01.0000000Z" },
  @{ backend_path = 'C:\Other\CryptoHunterBackend.exe' },
  @{ backend_sha256 = 'b' * 64 }
)
$invalid = @(
  @{ pid = 0 },
  @{ pid = $true },
  @{ pid = 42.0 },
  @{ started = " " },
  @{ started = "2026-10-29T00:00:00.0000000+00:00" },
  @{ started = "2026-02-30T00:00:00.0000000Z" },
  @{ started = "nie jest czasem" },
  @{ backend_path = 'CryptoHunterBackend.exe' },
  @{ backend_path = 'C:\Other\..\CryptoHunterBackend.exe' },
  @{ backend_sha256 = 'A' * 64 },
  @{ backend_sha256 = ('a' * 64) + "`n" },
  @{ backend_path = $expected.backend_path + "`n" },
  @{ backend_sha256 = "" }
)
function Assert-Refused([hashtable]$Baseline, [hashtable]$Sample) {
  $refused = $false
  try { Assert-Stage10SoakIdentity -Expected $Baseline -Observed $Sample }
  catch { $refused = $true }
  if (-not $refused) { throw "zaakceptowano zmieniona lub niepelna tozsamosc" }
}
foreach ($change in @($changed + $invalid)) {
  $observed = $expected.Clone()
  foreach ($key in $change.Keys) { $observed[$key] = $change[$key] }
  Assert-Refused -Baseline $expected -Sample $observed
}
foreach ($change in $invalid) {
  $observed = $expected.Clone()
  foreach ($key in $change.Keys) { $observed[$key] = $change[$key] }
  # Nieprawidlowy baseline nie moze zaakceptowac identycznie nieprawidlowej probki.
  Assert-Refused -Baseline $observed -Sample $observed
}
foreach ($key in @('pid', 'started', 'backend_path', 'backend_sha256')) {
  $observed = $expected.Clone()
  $observed.Remove($key)
  Assert-Refused -Baseline $expected -Sample $observed
  Assert-Refused -Baseline $observed -Sample $observed
}
Assert-Refused -Baseline @{} -Sample @{}
Write-Output "Syntetyczna regresja punktowej tozsamosci Stage 10: PASS; brak kwalifikacji fizycznej"
