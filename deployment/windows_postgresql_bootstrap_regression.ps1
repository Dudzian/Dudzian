[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Root,
    [Parameter(Mandatory = $true)][string]$Report,
    [string]$PythonExecutable = (Get-Command python).Source
)

$ErrorActionPreference = 'Stop'
$taskName = 'CryptoHunter-Isolated-PG-' + [Guid]::NewGuid().ToString('N')
$registered = $false
$exitCode = 1
try {
    $repository = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
    $script = Join-Path $repository 'tests/deployment/windows_postgresql_bootstrap_native.py'
    $rootPath = [IO.Path]::GetFullPath($Root)
    $reportPath = [IO.Path]::GetFullPath($Report)
    $tempRoot = [IO.Path]::GetFullPath($env:RUNNER_TEMP).TrimEnd('\') + '\'
    if (-not $rootPath.StartsWith($tempRoot, [StringComparison]::OrdinalIgnoreCase)) {
        throw 'isolated root must be inside runner temporary storage'
    }
    if ($rootPath -match '["\r\n]' -or $reportPath -match '["\r\n]' -or $script -match '["\r\n]') {
        throw 'invalid task argument'
    }
    $arguments = '"{0}" --root "{1}" --report "{2}" --runner-temp "{3}"' -f $script, $rootPath, $reportPath, $tempRoot.TrimEnd('\')
    $action = New-ScheduledTaskAction -Execute $PythonExecutable -Argument $arguments -WorkingDirectory $repository
    $principal = New-ScheduledTaskPrincipal -UserId 'S-1-5-18' -LogonType ServiceAccount -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 8)
    $task = New-ScheduledTask -Action $action -Principal $principal -Settings $settings
    Register-ScheduledTask -TaskName $taskName -InputObject $task | Out-Null
    $registered = $true
    Start-ScheduledTask -TaskName $taskName
    $deadline = (Get-Date).AddMinutes(8)
    do {
        Start-Sleep -Seconds 2
        $info = Get-ScheduledTaskInfo -TaskName $taskName
        $state = (Get-ScheduledTask -TaskName $taskName).State
        if ((Get-Date) -ge $deadline) { throw 'isolated task timeout' }
    } while ($info.LastRunTime.Year -lt 2000 -or $state -in @('Running', 'Queued'))
    if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) { throw 'isolated report absent' }
    # The Python script emits only fixed diagnostic enums and booleans. Do not
    # display task exception text, command lines, child stderr or temp paths.
    $result = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json
    $categories = @(
        'CONNECTED', 'WINDOWS_ACCESS_DENIED', 'DEPENDENCY_UNAVAILABLE',
        'PROOF_ASSERTION_FAILED', 'SCRAM_PASSWORD_REJECTED', 'LIBPQ_NO_PASSWORD',
        'TLS_CERTIFICATE_REJECTED', 'TLS_HOSTNAME_REJECTED', 'LISTENER_UNAVAILABLE',
        'HBA_REJECTED', 'UNCLASSIFIED_FAILURE', 'SETUP', 'UNEXPECTED_STAGE',
        'CREATE_LOCAL_PKI', 'QUALIFY_LOCAL_PKI', 'POSTGRES_BOOTSTRAP_CREDENTIALS',
        'POSTGRES_INITDB', 'POSTGRES_BOOTSTRAP_START', 'POSTGRES_CREATE_DATABASE',
        'POSTGRES_PROVISION_FRESHNESS_AUTHORITY', 'POSTGRES_QUALIFY_FRESHNESS_AUTHORITY',
        'POSTGRES_FINAL_HBA_WRITE', 'POSTGRES_FINAL_HBA_QUALIFY',
        'POSTGRES_FINAL_ADMIN_REJECTION', 'POSTGRES_BOOTSTRAP_STOP',
        'VALIDATE_PLATFORM', 'IMPORT_WIN32_JOB', 'CREATE_JOB_OBJECT', 'QUERY_JOB_OBJECT',
        'CONFIGURE_JOB_OBJECT', 'ASSIGN_JOB_OBJECT', 'VALIDATE_ROOT', 'IMPORT_POSTGRESQL_DRIVER',
        'IMPORT_BOOTSTRAP', 'VALIDATE_WINDOWS_ARCH', 'REQUIRE_LOCAL_SYSTEM',
        'SELECT_LOOPBACK_PORT', 'PROTECT_TEMPORARY_ROOT', 'PREPARE_TEMPORARY_DIRECTORIES',
        'IMPORT_ERROR', 'ATTRIBUTE_ERROR', 'KEY_ERROR', 'PROOF_FAILURE', 'PYWIN32_ERROR',
        'PROVISION_ERROR', 'OTHER_ERROR'
    )
    foreach ($field in @('stage', 'failure', 'failure_type', 'bootstrap_failure', 'passfile_connection',
        'memory_password_connection', 'missing_password', 'wrong_password', 'wrong_ca', 'tls_disabled')) {
        $value = $result.$field
        if ($value -is [string] -and $value -cin $categories) {
            Write-Output ('NATIVE_POSTGRESQL_{0} = {1}' -f $field.ToUpperInvariant(), $value)
        }
    }
    if ($result.failure_winerror -is [long] -and $result.failure_winerror -in @(5, 6, 87, 1008, 1300, 1314)) {
        Write-Output ('NATIVE_POSTGRESQL_FAILURE_WINERROR = {0}' -f $result.failure_winerror)
    }
    foreach ($field in @('effective_local_system', 'listener_available', 'bootstrap_directory_qualified',
        'final_hba_matches', 'credentials_removed', 'postmaster_stopped', 'temporary_root_removed')) {
        $value = $result.$field
        if ($value -is [bool]) {
            Write-Output ('NATIVE_POSTGRESQL_{0} = {1}' -f $field.ToUpperInvariant(), $value.ToString().ToUpperInvariant())
        }
    }
    if ($result.result -eq 'PASS' -and $info.LastTaskResult -eq 0) {
        Write-Output 'NATIVE_POSTGRESQL_BOOTSTRAP = PASS'
        $exitCode = 0
    } else {
        Write-Output 'NATIVE_POSTGRESQL_BOOTSTRAP = FAIL'
    }
} catch {
    Write-Output 'NATIVE_POSTGRESQL_BOOTSTRAP_HARNESS = FAIL'
} finally {
    if ($registered) {
        Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction SilentlyContinue
    }
}
exit $exitCode
