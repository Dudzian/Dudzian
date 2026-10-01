param(
  [ValidateSet("prepare-reboot", "qualify")][string]$Mode,
  [Parameter(Mandatory)][string]$State,
  [string]$QualifiedManifest,
  [string]$CleanInstallReceipt,
  [int]$SoakHours = 24,
  [string]$Output
)
$ErrorActionPreference = "Stop"
$serviceName = "CryptoHunterBackend"
$contractPath = Join-Path $PSScriptRoot "windows_stage10_lifecycle_contract.json"
$contract = Get-Content -Raw -LiteralPath $contractPath | ConvertFrom-Json
function Backend-Path {
  $svc = Get-CimInstance Win32_Service -Filter "Name='$serviceName'"
  if ($null -eq $svc -or $svc.PathName -notmatch '^"([^\"]+CryptoHunterBackend\.exe)"$') {
    throw "SCM backend path is missing or is not the canonical argument-free executable"
  }
  return $Matches[1]
}
function Installed-Product-Version {
  $roots = @('HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*',
             'HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*')
  $products = @(Get-ItemProperty $roots -ErrorAction SilentlyContinue | Where-Object DisplayName -CEQ 'CryptoHunter')
  if ($products.Count -ne 1 -or [string]::IsNullOrWhiteSpace($products[0].DisplayVersion)) {
    throw "exactly one installed CryptoHunter product version is required"
  }
  return $products[0].DisplayVersion
}
function Service-Snapshot([string]$ExpectedBackendSha256, [string]$ExpectedProductVersion) {
  $svc = Get-CimInstance Win32_Service -Filter "Name='$serviceName'"
  if ($null -eq $svc -or $svc.State -cne "Running" -or $svc.StartMode -cne "Auto" -or $svc.ProcessId -le 0) {
    throw "CryptoHunterBackend is not a running automatic native service"
  }
  $process = Get-Process -Id $svc.ProcessId -ErrorAction Stop
  $backendPath = Backend-Path
  $backendSha256 = (Get-FileHash -Algorithm SHA256 -LiteralPath $backendPath).Hash.ToLowerInvariant()
  if ($backendSha256 -cne $ExpectedBackendSha256) {
    throw "installed backend executable changed or differs from qualified artifact"
  }
  if ((Installed-Product-Version) -cne $ExpectedProductVersion) {
    throw "installed product version changed or differs from qualified artifact"
  }
  $readinessPath = Join-Path $env:ProgramData "CryptoHunter\Runtime\backend-readiness.json"
  $readiness = Get-Content -Raw -LiteralPath $readinessPath | ConvertFrom-Json
  if ($readiness.pid -ne $svc.ProcessId -or $readiness.db_role -cne "freshness_runtime" -or
      $readiness.corehost_lock -ne $true -or $readiness.cross_key_denied -ne $true -or
      $readiness.cross_roles_denied -ne $true) {
    throw "backend readiness contract differs from the live SCM process"
  }
  return @{ pid = [int]$svc.ProcessId; started = $process.StartTime.ToUniversalTime().ToString("o");
            backend_path = $backendPath; backend_sha256 = $backendSha256 }
}
function Machine-Guid { (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Cryptography').MachineGuid }
function Boot-Time { (Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToUniversalTime().ToString("o") }
if ($env:GITHUB_SERVER_URL -cne "https://github.com" -or $env:RUNNER_OS -cne "Windows" -or
    [string]::IsNullOrWhiteSpace($env:GITHUB_SHA) -or [string]::IsNullOrWhiteSpace($env:GITHUB_RUN_ID)) {
  throw "qualification requires revision/run-bound GitHub Windows execution"
}
if ($Mode -ceq "prepare-reboot") {
  if ([string]::IsNullOrWhiteSpace($QualifiedManifest) -or
      [string]::IsNullOrWhiteSpace($CleanInstallReceipt)) {
    throw "BLOCKED_BY_CURRENT_RUN_RUNTIME_PROVENANCE"
  }
  $provenancePath = Join-Path $env:RUNNER_TEMP 'stage10-runtime-provenance.json'
  python -m deployment.windows_stage10_runtime_provenance `
    --manifest $QualifiedManifest --clean-install-receipt $CleanInstallReceipt `
    --installed-backend (Backend-Path) --installed-product-version (Installed-Product-Version) `
    --output $provenancePath
  if ($LASTEXITCODE -ne 0) { throw "BLOCKED_BY_CURRENT_RUN_RUNTIME_PROVENANCE" }
  $provenance = Get-Content -Raw -LiteralPath $provenancePath | ConvertFrom-Json
  $before = Service-Snapshot $provenance.installed_backend_sha256 $provenance.product_version
  @{ schema_version = 1; source_revision = $env:GITHUB_SHA; ci_run_id = $env:GITHUB_RUN_ID
     runner_name = $env:RUNNER_NAME; runner_arch = $env:RUNNER_ARCH; machine_guid = Machine-Guid
     boot_before = Boot-Time; service_before = $before; runtime_provenance = $provenance } |
    ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $State -Encoding utf8NoBOM
  exit 0
}
if ($SoakHours -lt 24) { throw "the canonical bounded lifecycle is at least 24 hours" }
if ([string]::IsNullOrWhiteSpace($Output)) { throw "qualify requires Output" }
$prior = Get-Content -Raw -LiteralPath $State | ConvertFrom-Json
if ($prior.source_revision -cne $env:GITHUB_SHA -or $prior.ci_run_id -cne $env:GITHUB_RUN_ID -or
    $prior.runner_name -cne $env:RUNNER_NAME -or $prior.runner_arch -cne $env:RUNNER_ARCH -or
    $prior.machine_guid -cne (Machine-Guid) -or [datetime]$prior.boot_before -ge [datetime](Boot-Time)) {
  throw "stale continuity state or host did not reboot"
}
$runtimeProvenance = $prior.runtime_provenance
if ($null -eq $runtimeProvenance -or
    $runtimeProvenance.source_revision -cne $env:GITHUB_SHA -or
    $runtimeProvenance.ci_run_id -cne $env:GITHUB_RUN_ID) {
  throw "missing or stale runtime provenance"
}
$afterReboot = Service-Snapshot $runtimeProvenance.installed_backend_sha256 $runtimeProvenance.product_version
$soakIdentity = $afterReboot
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
$soakStartedUtc = [datetime]::UtcNow.ToString("o")
do {
  Start-Sleep -Seconds 60
  $sample = Service-Snapshot $runtimeProvenance.installed_backend_sha256 $runtimeProvenance.product_version
  if ($sample.pid -ne $soakIdentity.pid -or $sample.started -cne $soakIdentity.started) {
    throw "service instance changed during the 24-hour lifecycle qualification"
  }
} while ($stopwatch.Elapsed.TotalHours -lt $SoakHours)
$stopwatch.Stop()
$final = Service-Snapshot $runtimeProvenance.installed_backend_sha256 $runtimeProvenance.product_version
$proofValues = @{}
foreach ($property in $contract.results.PSObject.Properties) {
  $proofs = @{}
  foreach ($proof in $property.Value) { $proofs[$proof] = 'PASS' }
  $proofValues[$property.Name] = @{ status='PASS'; proofs=$proofs; details='' }
}
$proofValues.WINDOWS_REBOOT_RECOVERY.details = "boot changed on machine $($prior.machine_guid); readiness matched SCM PID $($afterReboot.pid)"
$proofValues.WINDOWS_LONG_RUNNING_LIFECYCLE.details = "stable SCM/readiness PID $($soakIdentity.pid); monotonic elapsed $($stopwatch.Elapsed); UTC start $soakStartedUtc"
@{ schema_version=1; source_revision=$env:GITHUB_SHA; ci_provider=$env:GITHUB_SERVER_URL;
   ci_run_id=$env:GITHUB_RUN_ID; runner_os='Windows'; runner_arch=$env:RUNNER_ARCH;
   runner_name=$env:RUNNER_NAME; machine_guid=$prior.machine_guid;
   probe_id=$contract.probe_id; runtime_provenance=$runtimeProvenance; results=$proofValues } |
  ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Output -Encoding utf8NoBOM
