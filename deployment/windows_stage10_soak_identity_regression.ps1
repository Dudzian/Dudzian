$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "windows_stage10_soak_identity.ps1")
$expected = @{ pid = 42; started = "2026-10-29T00:00:00.0000000Z" }
Assert-Stage10SoakIdentity -Expected $expected -Observed @{
  pid = 42; started = "2026-10-29T00:00:00.0000000Z"
}
$rejected = @(
  @{ pid = 43; started = $expected.started },
  @{ pid = 42; started = "2026-10-29T00:00:01.0000000Z" },
  @{ pid = 0; started = $expected.started },
  @{ pid = $true; started = $expected.started },
  @{ started = $expected.started },
  @{ pid = 42 },
  @{ pid = 42; started = " " }
)
foreach ($observed in $rejected) {
  $refused = $false
  try { Assert-Stage10SoakIdentity -Expected $expected -Observed $observed }
  catch { $refused = $true }
  if (-not $refused) { throw "changed or incomplete soak identity was accepted" }
}
# An invalid baseline cannot make an equally invalid final snapshot acceptable.
$refused = $false
try { Assert-Stage10SoakIdentity -Expected @{} -Observed @{} }
catch { $refused = $true }
if (-not $refused) { throw "missing baseline and final identities were accepted" }
Write-Output "Stage-10 synthetic soak identity regression passed (1 valid, 8 refusals)"
