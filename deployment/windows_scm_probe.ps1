param(
  [Parameter(Mandatory = $true)]
  [string]$PythonExecutable,
  [Parameter(Mandatory = $true)]
  [string]$WindowsAcceptanceRunToken,
  [switch]$CleanupOnly
)
$ErrorActionPreference = "Stop"
$service = "CryptoHunterBackend"
$identity = "NT SERVICE\CryptoHunterBackend"
$harness = Join-Path $PSScriptRoot "windows_test_service.py"
$testRoot = Join-Path $env:ProgramData "CryptoHunter"
$marker = Join-Path $testRoot "scm-health.txt"
$ownershipPath = Join-Path $testRoot "windows-acceptance-ownership.json"
$ownershipTemp = "$ownershipPath.$WindowsAcceptanceRunToken.tmp"
$ownershipProofTemp = "$ownershipPath.$WindowsAcceptanceRunToken.prove.tmp"
$serviceHost = $null
$serviceSid = $null
$serviceInstalledByProbe = $false
$intentOwned = $false
$serviceOwnershipProven = $false
$ownership = $null
$stage = "INSTALL"
$primaryFailure = $null
$cleanupFailures = [System.Collections.Generic.List[string]]::new()
$result = [ordered]@{
  WINDOWS_NATIVE_PATH_INTEGRATION = "FAIL"
  WINDOWS_PROTECTED_CONFIGURATION = "FAIL"
  WINDOWS_PROTECTED_STATE_PATHS = "FAIL"
  WINDOWS_ACL_QUALIFICATION = "FAIL"
  WINDOWS_SERVICE_INSTALLATION = "FAIL"
  WINDOWS_AUTOSTART = "FAIL"
  WINDOWS_SERVICE_START = "FAIL"
  WINDOWS_GRACEFUL_STOP = "FAIL"
  WINDOWS_MANUAL_RESTART = "FAIL"
  WINDOWS_AUTOMATIC_CRASH_RESTART = "FAIL"
  cleanup = "FAIL"
  details = ""
}

. (Join-Path $PSScriptRoot "windows_path_qualification.ps1")
. (Join-Path $PSScriptRoot "windows_native_acl.ps1")

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

function Read-MarkerPid {
  if (-not (Test-Path -LiteralPath $marker -PathType Leaf)) { return 0 }
  $value = [string](Get-Content -LiteralPath $marker -Raw)
  if ($value -notmatch '^\d+$') { throw "health marker PID is invalid" }
  return [int]$value
}

function Get-ServiceObservation {
  return Get-CimInstance Win32_Service -Filter "Name='$service'" -ErrorAction Stop
}

function Assert-RunningPidMatch {
  $observed = Get-ServiceObservation
  $markerPid = Read-MarkerPid
  if ($observed.State -cne "Running" -or [int]$observed.ProcessId -le 0) {
    throw "SCM service is not running with a positive PID"
  }
  if ($markerPid -ne [int]$observed.ProcessId) { throw "health marker PID differs from SCM PID" }
  return [int]$observed.ProcessId
}

function Stop-OwnedServiceForCleanup {
  $deadline = [DateTime]::UtcNow.AddSeconds(20)
  while ([DateTime]::UtcNow -lt $deadline) {
    $current = Get-Service $service -ErrorAction SilentlyContinue
    if ($null -eq $current) { return }
    if ($current.Status.ToString() -cne "Stopped") {
      Stop-Service $service -ErrorAction SilentlyContinue
      Start-Sleep -Milliseconds 250
      continue
    }
    Start-Sleep -Milliseconds 2000
    $stable = Get-Service $service -ErrorAction SilentlyContinue
    if ($null -eq $stable -or $stable.Status.ToString() -ceq "Stopped") { return }
  }
  throw "owned service did not remain stopped during cleanup"
}

function Test-ServiceGrant([string]$Target) {
  return Test-NativePrincipalGrant $Target $identity $script:serviceSid
}

function Remove-ServiceGrant([string]$Target) {
  if (-not (Test-ServiceGrant $Target)) { return }
  & icacls.exe $Target /remove:g $identity 2>$null | Out-Null
  if ($LASTEXITCODE -ne 0 -or (Test-ServiceGrant $Target)) {
    throw "ACL removal was not confirmed for $Target"
  }
}

function Save-OwnershipRecord {
  $json = $script:ownership | ConvertTo-Json -Compress
  [System.IO.File]::WriteAllText($script:ownershipTemp, $json, [System.Text.UTF8Encoding]::new($false))
  Move-Item -LiteralPath $script:ownershipTemp -Destination $script:ownershipPath -Force
}

function Test-BaseOwnershipRecord($Record) {
  return (
    $null -ne $Record -and
    $Record.run_token -ceq $WindowsAcceptanceRunToken -and
    $Record.service_name -ceq $service -and
    $Record.service_identity -ceq $identity -and
    $Record.python_executable -ceq $PythonExecutable -and
    $Record.harness -ceq $harness -and
    $Record.ownership_phase -in @("INTENT_CREATED", "SERVICE_PROVEN")
  )
}

function Add-PlannedGrant([string]$Target) {
  if (Test-ServiceGrant $Target) {
    throw "pre-existing service ACL prevents ownership-safe test grant: $Target"
  }
  $script:ownership.grants = @($script:ownership.grants) + @($Target)
  Save-OwnershipRecord
}

try {
  $qualifiedPaths = @($PythonExecutable, $harness, $testRoot)
  if ($qualifiedPaths | Where-Object { -not (Test-FullyQualifiedWindowsPath $_) }) {
    throw "native Windows path qualification failed"
  }
  if ($WindowsAcceptanceRunToken -cnotmatch '^[a-f0-9]{64}$') {
    throw "invalid Windows acceptance run token"
  }
  if (-not (Test-Path -LiteralPath $PythonExecutable -PathType Leaf)) {
    throw "canonical Python executable is not an existing file"
  }
  if (-not (Test-Path -LiteralPath $harness -PathType Leaf)) {
    throw "SCM harness is not an existing file"
  }
  New-Item -ItemType Directory -Path $testRoot -Force | Out-Null

  if ($CleanupOnly) {
    $configured = Get-CimInstance Win32_Service -Filter "Name='$service'" -ErrorAction SilentlyContinue
    if (Test-Path -LiteralPath $ownershipPath -PathType Leaf) {
      try { $ownership = Get-Content -LiteralPath $ownershipPath -Raw | ConvertFrom-Json } catch {
        $cleanupFailures.Add("ownership record is unreadable")
      }
    }
    if (-not (Test-BaseOwnershipRecord $ownership)) {
      $cleanupFailures.Add("current-run service ownership is not proven")
    } elseif ($configured) {
      $expectedPath = [string]$ownership.service_path_name
      $expectedHost = [string]$ownership.service_host
      $expectedSid = [string]$ownership.service_sid
      $actualHost = [regex]::Match($configured.PathName, '^"([^"]+)"').Groups[1].Value
      $actualSid = ([System.Security.Principal.NTAccount]::new($identity)).Translate(
        [System.Security.Principal.SecurityIdentifier]
      ).Value
      if (
        $ownership.ownership_phase -cne "SERVICE_PROVEN" -or
        $ownership.strict_create_result -cne "CREATED" -or
        $configured.Name -cne $service -or
        $configured.StartName -cne $identity -or
        -not $expectedPath -or $configured.PathName -cne $expectedPath -or
        -not $expectedHost -or $actualHost -cne $expectedHost -or
        -not $expectedSid -or $actualSid -cne $expectedSid
      ) {
        $cleanupFailures.Add("current-run service configuration ownership is not proven")
      } else {
        $intentOwned = $true
        $serviceOwnershipProven = $true
        $serviceInstalledByProbe = $true
        $serviceHost = $actualHost
        $serviceSid = $actualSid
      }
    } else {
      $intentOwned = $true
      $serviceHost = [string]$ownership.service_host
      $serviceSid = [string]$ownership.service_sid
    }
    $primaryFailure = "[CLEANUP] timeout recovery cleanup requested"
  } else {
    # Ownership is established only after the clean-host absence qualification.
    if (Get-Service $service -ErrorAction SilentlyContinue) {
      throw "acceptance requires the test service to be absent before installation"
    }
    if (Test-Path -LiteralPath $marker) {
      throw "pre-existing health marker prevents ownership-safe acceptance"
    }
    if (Test-Path -LiteralPath $ownershipPath) {
      throw "pre-existing ownership record prevents acceptance"
    }
    $ownership = [ordered]@{
      run_token = $WindowsAcceptanceRunToken
      service_name = $service
      service_identity = $identity
      python_executable = $PythonExecutable
      harness = $harness
      ownership_phase = "INTENT_CREATED"
      strict_create_result = $null
      service_path_name = $null
      service_host = $null
      service_sid = $null
      grants = @()
    }
    $ownershipJson = $ownership | ConvertTo-Json -Compress
    $ownershipStream = [System.IO.File]::Open(
      $ownershipPath, [System.IO.FileMode]::CreateNew,
      [System.IO.FileAccess]::Write, [System.IO.FileShare]::None
    )
    try {
      $ownershipBytes = [System.Text.UTF8Encoding]::new($false).GetBytes($ownershipJson)
      $ownershipStream.Write($ownershipBytes, 0, $ownershipBytes.Length)
    } finally { $ownershipStream.Dispose() }
    $intentOwned = $true

    $strictCreateOutput = @(
      & $PythonExecutable $harness acceptance-install-strict --username $identity 2>&1
    )
    $strictCreateExitCode = $LASTEXITCODE
    if ($strictCreateExitCode -ne 0) {
      throw "strict create-only service installation failed: $($strictCreateOutput -join '; ')"
    }
    try {
      $strictCreatePayload = $strictCreateOutput[-1] | ConvertFrom-Json
    } catch { throw "strict create-only installer returned invalid result" }
    if ($strictCreatePayload.strict_create_result -cne "CREATED") {
      throw "strict create-only installer did not return CREATED"
    }
    $strictCreateResult = [string]$strictCreatePayload.strict_create_result
    $configured = Get-CimInstance Win32_Service -Filter "Name='$service'"
    if ($null -eq $configured -or $configured.Name -cne $service) { throw "SCM query/name mismatch" }
    if ($configured.StartName -cne $identity) { throw "service identity mismatch: $($configured.StartName)" }
    $serviceHost = [regex]::Match($configured.PathName, '^"([^"]+)"').Groups[1].Value
    if (-not $serviceHost) { throw "cannot resolve pywin32 service host" }
    if (-not (Test-Path -LiteralPath $serviceHost -PathType Leaf)) {
      throw "pywin32 service host is not an existing file"
    }
    $serviceSid = ([System.Security.Principal.NTAccount]::new($identity)).Translate(
      [System.Security.Principal.SecurityIdentifier]
    ).Value
    & $PythonExecutable -m deployment.windows_service_ownership `
      --record $ownershipPath `
      --run-token $WindowsAcceptanceRunToken `
      --python-executable $PythonExecutable `
      --harness $harness `
      --strict-create-result $strictCreateResult `
      --observed-name $configured.Name `
      --observed-start-name $configured.StartName `
      --observed-path-name $configured.PathName `
      --observed-service-host $serviceHost `
      --observed-service-sid $serviceSid
    if ($LASTEXITCODE -ne 0) { throw "service ownership qualification failed" }
    $ownership = Get-Content -LiteralPath $ownershipPath -Raw | ConvertFrom-Json
    if ($ownership.ownership_phase -cne "SERVICE_PROVEN") {
      throw "service ownership did not transition to SERVICE_PROVEN"
    }
    $serviceOwnershipProven = $true
    $serviceInstalledByProbe = $true

    $stage = "NATIVE_PATHS_DACL"
    $stage4Output = @(
      & $PythonExecutable -m deployment.windows_dacl_provision provision `
        --record $ownershipPath --run-token $WindowsAcceptanceRunToken 2>&1
    )
    if ($LASTEXITCODE -ne 0) {
      throw "Stage-4 native path/DACL provisioning failed: $($stage4Output -join '; ')"
    }
    try { $stage4ProvisionPayload = $stage4Output[-1] | ConvertFrom-Json } catch {
      throw "Stage-4 provision helper returned invalid result"
    }
    if (
      $stage4ProvisionPayload.CONFIG_PROVISIONED -cne "PASS" -or
      $stage4ProvisionPayload.STATE_PROVISIONED -cne "PASS" -or
      $stage4ProvisionPayload.RUNTIME_PROVISIONED -cne "PASS"
    ) { throw "Stage-4 provisioning did not pass" }

    $stage = "READ_ONLY_DACL_QUALIFICATION"
    $stage4QualificationOutput = @(
      & $PythonExecutable -m deployment.windows_dacl_qualification `
        --record $ownershipPath --run-token $WindowsAcceptanceRunToken 2>&1
    )
    if ($LASTEXITCODE -ne 0) {
      throw "Stage-4 read-only qualification failed: $($stage4QualificationOutput -join '; ')"
    }
    try { $stage4QualificationPayload = $stage4QualificationOutput[-1] | ConvertFrom-Json } catch {
      throw "Stage-4 read-only qualifier returned invalid result"
    }
    if (
      $stage4QualificationPayload.NATIVE_PATHS -cne "PASS" -or
      $stage4QualificationPayload.CONFIG_DACL -cne "PASS" -or
      $stage4QualificationPayload.STATE_DACL -cne "PASS" -or
      $stage4QualificationPayload.RUNTIME_DACL -cne "PASS" -or
      $stage4QualificationPayload.READ_ONLY_QUALIFIER -cne "PASS"
    ) { throw "Stage-4 read-only DACL qualification did not pass" }
    $ownership = Get-Content -LiteralPath $ownershipPath -Raw | ConvertFrom-Json
    $result.WINDOWS_NATIVE_PATH_INTEGRATION = "PASS"
    $result.WINDOWS_PROTECTED_CONFIGURATION = "PASS"
    $result.WINDOWS_PROTECTED_STATE_PATHS = "PASS"
    $result.WINDOWS_ACL_QUALIFICATION = "PASS"

    $stage = "AUTOSTART_QUERY"
    $recoveryOutput = @(
      & $PythonExecutable -m deployment.windows_service_recovery `
        --record $ownershipPath --run-token $WindowsAcceptanceRunToken 2>&1
    )
    if ($LASTEXITCODE -ne 0) {
      throw "native SCM autostart/recovery qualification failed: $($recoveryOutput -join '; ')"
    }
    try { $recoveryPayload = $recoveryOutput[-1] | ConvertFrom-Json } catch {
      throw "native SCM recovery helper returned invalid result"
    }
    if ([int]$recoveryPayload.start_type -ne 2) { throw "SCM start type is not AUTO_START" }
    $result.WINDOWS_AUTOSTART = "PASS"

    Add-PlannedGrant $testRoot
    & icacls.exe $testRoot /grant "${identity}:(OI)(CI)M" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "test health directory ACL provisioning failed" }
    Add-PlannedGrant $PythonExecutable
    & icacls.exe $PythonExecutable /grant "${identity}:RX" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Python ACL provisioning failed" }
    Add-PlannedGrant $harness
    & icacls.exe $harness /grant "${identity}:RX" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "harness ACL provisioning failed" }
    Add-PlannedGrant $serviceHost
    & icacls.exe $serviceHost /grant "${identity}:RX" | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "service host ACL provisioning failed" }
    $result.WINDOWS_SERVICE_INSTALLATION = "PASS"

    $stage = "START"
    Start-Service $service
    Wait-State "Running"
    Wait-Marker
    $result.WINDOWS_SERVICE_START = "PASS"

    $stage = "STOP"
    Stop-Service $service
    Wait-State "Stopped"
    if (Test-Path $marker) { throw "health marker survived graceful stop" }
    $result.WINDOWS_GRACEFUL_STOP = "PASS"

    $stage = "RESTART"
    Start-Service $service
    Wait-State "Running"
    Wait-Marker
    [void](Assert-RunningPidMatch)
    $result.WINDOWS_MANUAL_RESTART = "PASS"

    $stage = "CRASH_RESTART"
    $originalPid = Assert-RunningPidMatch
    Stop-Process -Id $originalPid -Force -ErrorAction Stop
    $deadline = [DateTime]::UtcNow.AddSeconds(45)
    $crashProven = $false
    $lastState = "Unknown"
    $currentPid = 0
    $markerPid = 0
    while ([DateTime]::UtcNow -lt $deadline) {
      Start-Sleep -Milliseconds 250
      $observed = Get-ServiceObservation
      $lastState = [string]$observed.State
      $currentPid = [int]$observed.ProcessId
      $markerPid = Read-MarkerPid
      $oldGone = $null -eq (Get-Process -Id $originalPid -ErrorAction SilentlyContinue)
      if ($oldGone -and $lastState -ceq "Running" -and $currentPid -gt 0 -and
          $currentPid -ne $originalPid -and $markerPid -eq $currentPid) {
        $crashProven = $true
        break
      }
    }
    if (-not $crashProven) {
      throw "automatic recovery deadline expired; state=$lastState original_pid=$originalPid current_pid=$currentPid marker_pid=$markerPid"
    }
    $verifyRecovery = @(
      & $PythonExecutable -m deployment.windows_service_recovery --verify-only `
        --record $ownershipPath --run-token $WindowsAcceptanceRunToken 2>&1
    )
    if ($LASTEXITCODE -ne 0) {
      throw "post-crash SCM configuration verification failed: $($verifyRecovery -join '; ')"
    }
    $result.WINDOWS_AUTOMATIC_CRASH_RESTART = "PASS"

    $stage = "POST_RECOVERY_GRACEFUL_STOP"
    Stop-Service $service
    Wait-State "Stopped"
    if (Test-Path $marker) { throw "health marker survived post-recovery graceful stop" }
    Start-Sleep -Milliseconds 2000
    if ((Get-Service -Name $service).Status.ToString() -cne "Stopped" -or (Test-Path $marker)) {
      throw "graceful stop incorrectly triggered SCM recovery"
    }
    $result.details = "SCM install/autostart/recovery/start/health/graceful-stop/manual-restart/crash-restart verified"
  }
} catch {
  if (-not $primaryFailure) { $primaryFailure = "[$stage] $($_.Exception.Message)" }
} finally {
  if ($intentOwned) {
    if (Test-Path -LiteralPath $ownershipPath -PathType Leaf) {
      try { $ownership = Get-Content -LiteralPath $ownershipPath -Raw | ConvertFrom-Json } catch {
        $cleanupFailures.Add("ownership record cannot be reloaded for cleanup")
      }
    }
    if ($null -ne $ownership -and $null -ne $ownership.path_security_plan) {
      try {
        & $PythonExecutable -m deployment.windows_dacl_provision cleanup `
          --record $ownershipPath --run-token $WindowsAcceptanceRunToken | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "Stage-4 helper rejected cleanup" }
        $ownership = Get-Content -LiteralPath $ownershipPath -Raw | ConvertFrom-Json
      } catch { $cleanupFailures.Add("Stage-4: $($_.Exception.Message)") }
    }
    if (-not $serviceOwnershipProven -and (Get-Service $service -ErrorAction SilentlyContinue)) {
      $cleanupFailures.Add("unproven service exists after install attempt; left untouched")
    }
    if (-not $serviceOwnershipProven -and (Test-Path -LiteralPath $marker)) {
      $cleanupFailures.Add("unowned health marker exists after install attempt; left untouched")
    }
    if ($serviceOwnershipProven -and $serviceInstalledByProbe) {
      try {
        Stop-Service $service -ErrorAction SilentlyContinue
        Stop-OwnedServiceForCleanup
      } catch {
        $cleanupFailures.Add("stop: $($_.Exception.Message)")
      }
    }
    if ($serviceOwnershipProven) {
      try { Remove-Item $marker -Force -ErrorAction Stop } catch {
        if (Test-Path $marker) { $cleanupFailures.Add("marker: $($_.Exception.Message)") }
      }
    }
    foreach ($target in @($ownership.grants)) {
      try { Remove-ServiceGrant ([string]$target) } catch {
        $cleanupFailures.Add("ACL: $($_.Exception.Message)")
      }
    }
    try {
      if ($serviceOwnershipProven -and $serviceInstalledByProbe -and (Get-Service $service -ErrorAction SilentlyContinue)) {
        & $PythonExecutable $harness remove 2>$null
        if ($LASTEXITCODE -ne 0) { throw "service removal failed" }
      }
      if ($serviceOwnershipProven -and $serviceInstalledByProbe -and (Get-Service $service -ErrorAction SilentlyContinue)) {
        throw "service remains installed"
      }
    } catch { $cleanupFailures.Add("service: $($_.Exception.Message)") }

    if ($serviceOwnershipProven -and (Test-Path $marker)) {
      $cleanupFailures.Add("owned health marker remains")
    }
    foreach ($target in @($ownership.grants)) {
      try {
        if (Test-ServiceGrant ([string]$target)) {
          $cleanupFailures.Add("run-owned service ACL remains: $target")
        }
      } catch { $cleanupFailures.Add("ACL verification: $($_.Exception.Message)") }
    }
    try { Remove-Item -LiteralPath $ownershipTemp -Force -ErrorAction SilentlyContinue } catch {
      $cleanupFailures.Add("ownership temporary record removal failed")
    }
    try { Remove-Item -LiteralPath $ownershipProofTemp -Force -ErrorAction SilentlyContinue } catch {
      $cleanupFailures.Add("ownership proof record removal failed")
    }
    try { Remove-Item -LiteralPath $ownershipPath -Force -ErrorAction Stop } catch {
      $cleanupFailures.Add("ownership record removal failed: $($_.Exception.Message)")
    }
    if (Test-Path -LiteralPath $ownershipPath) {
      $cleanupFailures.Add("ownership record remains")
    }
  }
  if ($cleanupFailures.Count -eq 0) { $result.cleanup = "PASS" }
}

$result | ConvertTo-Json -Compress | Write-Output
$reportedFailures = [System.Collections.Generic.List[string]]::new()
if ($primaryFailure) { $reportedFailures.Add($primaryFailure) }
if ($cleanupFailures.Count -gt 0) {
  $reportedFailures.Add("[CLEANUP] $($cleanupFailures -join '; ')")
}
if ($reportedFailures.Count -gt 0) { Write-Error ($reportedFailures -join "; ") }
