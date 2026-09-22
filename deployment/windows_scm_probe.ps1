$ErrorActionPreference = "Stop"
$service = "CryptoHunterBackend"
$identity = "NT SERVICE\CryptoHunterBackend"
$harness = Join-Path $PSScriptRoot "windows_test_service.py"
$testRoot = Join-Path $env:ProgramData "CryptoHunter"
$marker = Join-Path $testRoot "scm-health.txt"

function Wait-State([string]$Expected) {
  for ($i = 0; $i -lt 60; $i++) {
    $state = (Get-Service -Name $service -ErrorAction Stop).Status.ToString()
    if ($state -eq $Expected) { return }
    Start-Sleep -Milliseconds 500
  }
  throw "service did not reach $Expected"
}

function Wait-Marker {
  for ($i = 0; $i -lt 30; $i++) {
    if (Test-Path $marker) { return }
    Start-Sleep -Milliseconds 500
  }
  throw "health marker missing"
}

try {
  New-Item -ItemType Directory -Path $testRoot -Force | Out-Null
  & icacls.exe $testRoot /grant "${identity}:(OI)(CI)M" | Out-Null
  if ($LASTEXITCODE -ne 0) { throw "test health directory ACL provisioning failed" }
  $pythonExecutable = (Get-Command python -ErrorAction Stop).Source
  & icacls.exe $pythonExecutable /grant "${identity}:RX" | Out-Null
  & icacls.exe $harness /grant "${identity}:RX" | Out-Null
  python $harness --startup manual --username $identity install
  $configured = (Get-CimInstance Win32_Service -Filter "Name='$service'")
  if ($null -eq $configured -or $configured.Name -ne $service) { throw "SCM name mismatch" }
  if ($configured.StartName -ne $identity) { throw "service identity mismatch: $($configured.StartName)" }
  $serviceHost = [regex]::Match($configured.PathName, '^"([^"]+)"').Groups[1].Value
  if (-not $serviceHost) { throw "cannot resolve pywin32 service host" }
  & icacls.exe $serviceHost /grant "${identity}:RX" | Out-Null
  Start-Service $service
  Wait-State "Running"
  Wait-Marker
  Stop-Service $service
  Wait-State "Stopped"
  if (Test-Path $marker) { throw "health marker survived graceful stop" }
  Start-Service $service
  Wait-State "Running"
  Wait-Marker
  Write-Output "SCM install/query/start/health/graceful-stop/restart verified"
} finally {
  Stop-Service $service -ErrorAction SilentlyContinue
  python $harness remove 2>$null
  Remove-Item $marker -Force -ErrorAction SilentlyContinue
  if ($harness) { & icacls.exe $harness /remove:g $identity 2>$null | Out-Null }
  if ($pythonExecutable) { & icacls.exe $pythonExecutable /remove:g $identity 2>$null | Out-Null }
  if ($serviceHost) { & icacls.exe $serviceHost /remove:g $identity 2>$null | Out-Null }
}
