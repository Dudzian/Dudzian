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
$runtime = Join-Path $testRoot "Runtime"
$treeMarker = Join-Path $runtime "process-tree.json"
$treeGate = Join-Path $runtime "process-tree.gate"
$logs = Join-Path $testRoot "Logs"
$logFile = Join-Path $logs "backend.log"
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
$startupDiagnosticSinceUtc = $null
$startupServicePid = 0
$cleanupFailures = [System.Collections.Generic.List[string]]::new()
$startupDiagnostics = [System.Collections.Generic.List[string]]::new()
$diagnosticFailures = [System.Collections.Generic.List[string]]::new()
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
  WINDOWS_PROCESS_TREE = "FAIL"
  WINDOWS_NO_ORPHAN_CHILDREN = "FAIL"
  WINDOWS_PERSISTENT_LOGGING = "FAIL"
  cleanup = "FAIL"
  details = ""
}

. (Join-Path $PSScriptRoot "windows_path_qualification.ps1")
. (Join-Path $PSScriptRoot "windows_native_acl.ps1")

function Invoke-CapturedPython {
  param([string[]]$Arguments)

  $savedErrorActionPreference = $ErrorActionPreference
  $captured = @()
  $exitCode = $null
  try {
    $ErrorActionPreference = "Continue"
    $captured = @(
      & $PythonExecutable @Arguments 2>&1 |
        ForEach-Object { $_.ToString() }
    )
    $exitCode = $LASTEXITCODE
  } finally {
    $ErrorActionPreference = $savedErrorActionPreference
  }

  return [pscustomobject]@{
    Output = $captured
    ExitCode = $exitCode
  }
}

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

function Get-Tree([int]$ServicePid) {
  for ($i = 0; $i -lt 40; $i++) {
    if (Test-Path -LiteralPath $treeMarker -PathType Leaf) {
      try {
        $tree = Get-Content -LiteralPath $treeMarker -Raw | ConvertFrom-Json
        if ([int]$tree.service_pid -eq $ServicePid -and [int]$tree.child_pid -gt 0 -and
            [int]$tree.grandchild_pid -gt 0) { return $tree }
      } catch {}
    }
    Start-Sleep -Milliseconds 500
  }
  throw "tree creation deadline expired service_pid=$ServicePid"
}

function Get-ProcessTreeStartupDiagnostic([int]$ServicePid, [DateTime]$SinceUtc) {
  $lines = [System.Collections.Generic.List[string]]::new()
  $observed = Get-CimInstance Win32_Service -Filter "Name='$service'" -ErrorAction Stop
  $lines.Add("PROCESS_TREE_START_DIAGNOSTIC scm_state=$($observed.State) scm_pid=$([int]$observed.ProcessId)")

  $markerExists = Test-Path -LiteralPath $marker
  $markerPid = 0
  if ($markerExists) {
    $markerItem = Get-Item -LiteralPath $marker -Force -ErrorAction Stop
    if (-not $markerItem.PSIsContainer -and
        -not $markerItem.Attributes.HasFlag([IO.FileAttributes]::ReparsePoint)) {
      $markerValue = [string](Get-Content -LiteralPath $marker -Raw -ErrorAction Stop)
      if ($markerValue -match '^\d+$') { $markerPid = [int]$markerValue }
    }
  }
  $treeExists = Test-Path -LiteralPath $treeMarker -PathType Leaf
  $gateExists = Test-Path -LiteralPath $treeGate -PathType Leaf
  $lines.Add("PROCESS_TREE_START_DIAGNOSTIC health_marker=$markerExists health_pid=$markerPid process_tree_json=$treeExists process_tree_gate=$gateExists")

  foreach ($name in @("backend.log", "backend.log.1", "backend.log.2", "backend.log.3", "backend.log.4")) {
    $candidate = Join-Path $logs $name
    if (-not (Test-Path -LiteralPath $candidate)) { continue }
    $item = Get-Item -LiteralPath $candidate -Force -ErrorAction Stop
    if ($item.PSIsContainer -or $item.Attributes.HasFlag([IO.FileAttributes]::ReparsePoint)) {
      $lines.Add("PROCESS_TREE_START_DIAGNOSTIC log=$name skipped=non_regular")
      continue
    }
    foreach ($match in @(Select-String -LiteralPath $candidate -SimpleMatch "PROCESS_TREE_START_FAILURE" -ErrorAction Stop)) {
      $lines.Add("PROCESS_TREE_START_DIAGNOSTIC log=$name event=$($match.Line)")
    }
  }

  try {
    $eventQuery = @{ LogName = "Application"; StartTime = $SinceUtc }
    foreach ($event in @(Get-WinEvent -FilterHashtable $eventQuery -MaxEvents 128 -ErrorAction Stop)) {
      $messageLine = @(
        ([string]$event.Message -split "`r?`n") |
          Where-Object { $_ -match 'PROCESS_TREE_START_FAILURE' } |
          Select-Object -First 1
      )
      if ($messageLine.Count -ne 1) { continue }
      $messageLine = [string]$messageLine[0]
      if ($messageLine -notmatch 'service_name=CryptoHunterBackend(?:\s|$)') { continue }
      if ($ServicePid -gt 0 -and
          $messageLine -notmatch "service_pid=$ServicePid(?:\s|$)") { continue }
      $lines.Add("PROCESS_TREE_START_DIAGNOSTIC event_log=Application event=$messageLine")
    }
  } catch {
    $script:diagnosticFailures.Add(
      "[PROCESS_TREE_START_DIAGNOSTIC_EVENT_LOG] $($_.Exception.Message)"
    )
  }
  return $lines
}

function Assert-Tree($Tree, [string]$Phase) {
  $qualify = Invoke-CapturedPython -Arguments @(
    "-m", "deployment.windows_process_tree_qualification", "--runtime", $runtime,
    "--service-pid", [string]$Tree.service_pid)
  if ($qualify.ExitCode -ne 0) { throw "$Phase tree qualifier failed: $($qualify.Output -join '; ')" }
  $child = Get-CimInstance Win32_Process -Filter "ProcessId=$($Tree.child_pid)"
  $grandchild = Get-CimInstance Win32_Process -Filter "ProcessId=$($Tree.grandchild_pid)"
  if ($null -eq $child -or [int]$child.ParentProcessId -ne [int]$Tree.service_pid -or
      $null -eq $grandchild -or [int]$grandchild.ParentProcessId -ne [int]$Tree.child_pid) {
    throw "$Phase parent relationship mismatch"
  }
}

function Wait-TreeGone($Tree, [string]$Phase) {
  for ($i = 0; $i -lt 60; $i++) {
    $childAlive = $null -ne (Get-Process -Id $Tree.child_pid -ErrorAction SilentlyContinue)
    $grandchildAlive = $null -ne (Get-Process -Id $Tree.grandchild_pid -ErrorAction SilentlyContinue)
    $jobAlive = $true
    $check = Invoke-CapturedPython -Arguments @(
      "-m", "deployment.windows_process_tree_qualification", "--runtime", $runtime,
      "--service-pid", [string]$Tree.service_pid, "--expect-absent")
    if ($check.ExitCode -eq 0) { $jobAlive = $false }
    if (-not $childAlive -and -not $grandchildAlive -and -not $jobAlive) { return }
    Start-Sleep -Milliseconds 250
  }
  throw "$Phase orphan deadline expired service_pid=$($Tree.service_pid) child_pid=$($Tree.child_pid) grandchild_pid=$($Tree.grandchild_pid) job_name=$($Tree.job_name)"
}

function Assert-LogEvent([string]$Event, [int]$ServicePid) {
  $found = $false
  foreach ($name in @("backend.log", "backend.log.1", "backend.log.2", "backend.log.3", "backend.log.4")) {
    $candidate = Join-Path $logs $name
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
      $item = Get-Item -LiteralPath $candidate -Force
      if ($item.Attributes.HasFlag([IO.FileAttributes]::ReparsePoint)) {
        throw "log family contains a reparse point: $name"
      }
      if ((Get-Content -LiteralPath $candidate -Raw) -match "pid=$ServicePid .*$Event") { $found = $true }
    }
  }
  if (-not $found) { throw "log event missing event=$Event pid=$ServicePid" }
}

function Remove-OwnedProcessTreeArtifacts($Record) {
  $runtimePlan = @($Record.path_security_plan.targets | Where-Object { $_.role -ceq "RUNTIME" })
  $sentinelPath = Join-Path $runtime ".stage4-ownership.json"
  if ($runtimePlan.Count -ne 1 -or -not (Test-Path -LiteralPath $sentinelPath -PathType Leaf)) {
    throw "Runtime ownership proof is missing"
  }
  $sentinel = Get-Content -LiteralPath $sentinelPath -Raw | ConvertFrom-Json
  if (
    $sentinel.run_token -cne $WindowsAcceptanceRunToken -or
    $sentinel.role -cne "RUNTIME" -or
    $sentinel.service_sid -cne $Record.service_sid -or
    -not [string]::Equals([string]$sentinel.canonical_path, [string]$runtimePlan[0].path,
      [StringComparison]::OrdinalIgnoreCase)
  ) { throw "Runtime ownership sentinel mismatch" }
  foreach ($name in @("process-tree.json", "process-tree.gate", "process-tree.json.tmp", "process-tree.tmp")) {
    $target = Join-Path $runtime $name
    if (Test-Path -LiteralPath $target) {
      $item = Get-Item -LiteralPath $target -Force
      if ($item.PSIsContainer -or $item.Attributes.HasFlag([IO.FileAttributes]::ReparsePoint)) {
        throw "unsafe process-tree transient: $name"
      }
      Remove-Item -LiteralPath $target -Force -ErrorAction Stop
    }
  }
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
  $json = $script:ownership | ConvertTo-Json -Depth 10 -Compress
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
    $stage4Run = Invoke-CapturedPython -Arguments @(
      "-m",
      "deployment.windows_dacl_provision",
      "provision",
      "--record",
      $ownershipPath,
      "--run-token",
      $WindowsAcceptanceRunToken
    )
    $stage4Output = @($stage4Run.Output)
    if ($stage4Run.ExitCode -ne 0) {
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
    $stage4QualificationRun = Invoke-CapturedPython -Arguments @(
      "-m",
      "deployment.windows_dacl_qualification",
      "--record",
      $ownershipPath,
      "--run-token",
      $WindowsAcceptanceRunToken
    )
    $stage4QualificationOutput = @($stage4QualificationRun.Output)
    if ($stage4QualificationRun.ExitCode -ne 0) {
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

    $stage = "STAGE5_LOGS_PROVISION"
    $loggingRun = Invoke-CapturedPython -Arguments @(
      "-m", "deployment.windows_logging_provision", "provision", "--record", $ownershipPath,
      "--run-token", $WindowsAcceptanceRunToken)
    if ($loggingRun.ExitCode -ne 0) { throw "Stage-5 Logs provisioning failed: $($loggingRun.Output -join '; ')" }
    $stage = "STAGE5_LOGS_QUALIFICATION"
    $loggingQualification = Invoke-CapturedPython -Arguments @(
      "-m", "deployment.windows_logging_qualification", "--record", $ownershipPath,
      "--run-token", $WindowsAcceptanceRunToken)
    if ($loggingQualification.ExitCode -ne 0) { throw "Stage-5 Logs read-only qualification failed" }

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
    $startupDiagnosticSinceUtc = [DateTime]::UtcNow
    Start-Service $service
    Wait-State "Running"
    Wait-Marker
    $pid1 = Assert-RunningPidMatch
    $startupServicePid = $pid1
    $stage = "PROCESS_TREE_START"
    $tree1 = Get-Tree $pid1
    Assert-Tree $tree1 "START"
    $stage = "PERSISTENT_LOGGING_START"
    Assert-LogEvent "SERVICE_START" $pid1
    $result.WINDOWS_SERVICE_START = "PASS"

    $stage = "STOP"
    Stop-Service $service
    Wait-State "Stopped"
    if (Test-Path $marker) { throw "health marker survived graceful stop" }
    $stage = "NO_ORPHAN_GRACEFUL_STOP"
    Wait-TreeGone $tree1 "GRACEFUL_STOP"
    $stage = "PERSISTENT_LOGGING_START"
    Assert-LogEvent "SERVICE_STOP" $pid1
    $result.WINDOWS_GRACEFUL_STOP = "PASS"

    $stage = "RESTART"
    Start-Service $service
    Wait-State "Running"
    Wait-Marker
    $pid2 = Assert-RunningPidMatch
    $stage = "PROCESS_TREE_MANUAL_RESTART"
    $tree2 = Get-Tree $pid2
    Assert-Tree $tree2 "MANUAL_RESTART"
    if ($pid2 -eq $pid1) { throw "manual restart reused service PID" }
    $stage = "PERSISTENT_LOGGING_MANUAL_RESTART"
    Assert-LogEvent "SERVICE_START" $pid1
    Assert-LogEvent "SERVICE_START" $pid2
    $result.WINDOWS_MANUAL_RESTART = "PASS"

    $stage = "CRASH_RESTART"
    $originalPid = Assert-RunningPidMatch
    $crashedTree = $tree2
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
    $stage = "NO_ORPHAN_CRASH_RESTART"
    Wait-TreeGone $crashedTree "CRASH_RESTART"
    $stage = "PROCESS_TREE_CRASH_RESTART"
    $tree3 = Get-Tree $currentPid
    Assert-Tree $tree3 "CRASH_RESTART"
    $stage = "PERSISTENT_LOGGING_CRASH_RESTART"
    Assert-LogEvent "SERVICE_START" $pid1
    Assert-LogEvent "SERVICE_STOP" $pid1
    Assert-LogEvent "SERVICE_START" $pid2
    Assert-LogEvent "SERVICE_START" $currentPid
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
    $stage = "NO_ORPHAN_FINAL_STOP"
    Wait-TreeGone $tree3 "POST_RECOVERY_GRACEFUL_STOP"
    $stage = "PERSISTENT_LOGGING_FINAL_STOP"
    Assert-LogEvent "SERVICE_START" $pid1
    Assert-LogEvent "SERVICE_STOP" $pid1
    Assert-LogEvent "SERVICE_START" $pid2
    Assert-LogEvent "SERVICE_START" $currentPid
    Assert-LogEvent "SERVICE_STOP" $currentPid
    $allowedLogs = @("backend.log", "backend.log.1", "backend.log.2", "backend.log.3", "backend.log.4", ".stage5-logging-ownership.json")
    foreach ($entry in Get-ChildItem -LiteralPath $logs -Force) {
      if ($entry.PSIsContainer -or $entry.Name -cnotin $allowedLogs -or
          $entry.Attributes.HasFlag([IO.FileAttributes]::ReparsePoint)) {
        throw "invalid Logs entry: $($entry.Name)"
      }
    }
    if ((Get-Item -LiteralPath $logFile).Length -le 0) { throw "backend.log is empty" }
    $result.WINDOWS_PROCESS_TREE = "PASS"
    $result.WINDOWS_NO_ORPHAN_CHILDREN = "PASS"
    $result.WINDOWS_PERSISTENT_LOGGING = "PASS"
    $result.details = "SCM install/autostart/recovery/start/health/graceful-stop/manual-restart/crash-restart verified"
  }
} catch {
  if (-not $primaryFailure) { $primaryFailure = "[$stage] $($_.Exception.Message)" }
  if (
    ($stage -ceq "PROCESS_TREE_START" -and
      $_.Exception.Message -like "tree creation deadline expired*") -or
    $stage -ceq "START"
  ) {
    try {
      # Capture and publish while the owned Logs tree still exists; cleanup follows.
      foreach ($line in @(
        Get-ProcessTreeStartupDiagnostic $startupServicePid $startupDiagnosticSinceUtc
      )) {
        $startupDiagnostics.Add([string]$line)
        Write-Output $line
      }
    } catch {
      $diagnosticFailures.Add("[PROCESS_TREE_START_DIAGNOSTIC] $($_.Exception.Message)")
    }
  }
} finally {
  if ($intentOwned) {
    if (Test-Path -LiteralPath $ownershipPath -PathType Leaf) {
      try { $ownership = Get-Content -LiteralPath $ownershipPath -Raw | ConvertFrom-Json } catch {
        $cleanupFailures.Add("ownership record cannot be reloaded for cleanup")
      }
    }
    if ($serviceOwnershipProven -and $serviceInstalledByProbe) {
      try {
        Stop-Service $service -ErrorAction SilentlyContinue
        Stop-OwnedServiceForCleanup
      } catch { $cleanupFailures.Add("stop: $($_.Exception.Message)") }
    }
    if ($serviceOwnershipProven -and $null -ne $ownership.path_security_plan) {
      try { Remove-OwnedProcessTreeArtifacts $ownership } catch {
        $cleanupFailures.Add("Stage-5 process tree: $($_.Exception.Message)")
      }
    }
    if ($null -ne $ownership -and $null -ne $ownership.path_security_plan) {
      if ($null -ne $ownership.logging_security_plan) {
        try {
          $loggingCleanup = Invoke-CapturedPython -Arguments @(
            "-m", "deployment.windows_logging_provision", "cleanup", "--record", $ownershipPath,
            "--run-token", $WindowsAcceptanceRunToken)
          if ($loggingCleanup.ExitCode -ne 0) {
            throw "Stage-5 helper rejected cleanup: $($loggingCleanup.Output -join '; ')"
          }
          $ownership = Get-Content -LiteralPath $ownershipPath -Raw | ConvertFrom-Json
        } catch { $cleanupFailures.Add("Stage-5 Logs: $($_.Exception.Message)") }
      }
      try {
        $stage4CleanupRun = Invoke-CapturedPython -Arguments @(
          "-m",
          "deployment.windows_dacl_provision",
          "cleanup",
          "--record",
          $ownershipPath,
          "--run-token",
          $WindowsAcceptanceRunToken
        )
        if ($stage4CleanupRun.ExitCode -ne 0) {
          throw "Stage-4 helper rejected cleanup: $($stage4CleanupRun.Output -join '; ')"
        }
        $ownership = Get-Content -LiteralPath $ownershipPath -Raw | ConvertFrom-Json
      } catch { $cleanupFailures.Add("Stage-4: $($_.Exception.Message)") }
    }
    if (-not $serviceOwnershipProven -and (Get-Service $service -ErrorAction SilentlyContinue)) {
      $cleanupFailures.Add("unproven service exists after install attempt; left untouched")
    }
    if (-not $serviceOwnershipProven -and (Test-Path -LiteralPath $marker)) {
      $cleanupFailures.Add("unowned health marker exists after install attempt; left untouched")
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
if ($startupDiagnostics.Count -gt 0) {
  $reportedFailures.Add($startupDiagnostics -join "; ")
}
if ($diagnosticFailures.Count -gt 0) {
  $reportedFailures.Add($diagnosticFailures -join "; ")
}
if ($cleanupFailures.Count -gt 0) {
  $reportedFailures.Add("[CLEANUP] $($cleanupFailures -join '; ')")
}
if ($reportedFailures.Count -gt 0) { Write-Error ($reportedFailures -join "; ") }
