param(
  [string]$QtQpaPlatform = "offscreen",
  [string]$QtOpenGL = "software",
  [string]$QtQuickBackend = "software",
  [string]$CrashDumpDir = "var/crashdumps",
  [string]$ResultsDir = "test-results/qml"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ciIni = Join-Path $PSScriptRoot "ci/pytest-ci.ini"

$env:QT_QPA_PLATFORM = $QtQpaPlatform
$env:QT_OPENGL = $QtOpenGL
$env:QT_QUICK_BACKEND = $QtQuickBackend
if ($QtQpaPlatform -eq "offscreen") {
  $env:DUDZIAN_DISABLE_QTCHARTS = "1"
} else {
  Remove-Item Env:DUDZIAN_DISABLE_QTCHARTS -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force $ResultsDir | Out-Null

$runSuffix = ""
if ($env:GITHUB_RUN_ID) {
  $runSuffix = $env:GITHUB_RUN_ID
  if ($env:GITHUB_RUN_ATTEMPT) {
    $runSuffix = "$runSuffix-$env:GITHUB_RUN_ATTEMPT"
  }
}
$runTag = ""
if ($runSuffix) {
  $runTag = "-$runSuffix"
}
$consoleLog = Join-Path $ResultsDir ("pytest-console{0}.log" -f $runTag)
$logFilePath = Join-Path $ResultsDir ("pytest{0}.log" -f $runTag)
$junitPath = Join-Path $ResultsDir ("pytest{0}.xml" -f $runTag)
if ($env:CI) {
  $junitPath = Join-Path "var/ci" ("junit-qml{0}.xml" -f $runTag)
}
$formatArgForLog = {
  param([string]$value)
  $escaped = $value -replace "`", "``"
  $escaped = $escaped -replace '"', '`"'
  if ($escaped -match '[\s"`]') {
    return '"' + $escaped + '"'
  }
  return $escaped
}
$maxfail = 1
$maxfailInputValid = $true
if ($env:QML_TEST_MAXFAIL) {
  $parsedMaxfail = 0
  if (-not [int]::TryParse($env:QML_TEST_MAXFAIL, [ref]$parsedMaxfail) -or $parsedMaxfail -lt 1) {
    Write-Warning "Invalid QML_TEST_MAXFAIL '$($env:QML_TEST_MAXFAIL)'; using default 1."
    $maxfailInputValid = $false
  } else {
    $maxfail = $parsedMaxfail
  }
}
$pytestArgs = @(
  "-c", $ciIni,
  "-m", "qml",
  "--maxfail=$maxfail",
  "--disable-warnings",
  "--junitxml", $junitPath,
  "--durations=15",
  "--log-file", $logFilePath,
  "--log-file-level=INFO"
)
if ($env:CI) {
  $pytestArgs += @("-rA", "-vv")
}
if ($env:QML_TEST_K) {
  $pytestArgs += @("-k", $env:QML_TEST_K)
}

$prevPluginAutoload = $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD
$exitCode = 0
try {
  Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD -ErrorAction SilentlyContinue
  New-Item -ItemType Directory -Force $CrashDumpDir | Out-Null
  New-Item -ItemType Directory -Force (Split-Path -Parent $consoleLog) | Out-Null
  New-Item -ItemType Directory -Force (Split-Path -Parent $junitPath) | Out-Null
  New-Item -ItemType Directory -Force (Split-Path -Parent $logFilePath) | Out-Null
  if ($env:CI) {
    $env:QT_MESSAGE_PATTERN = "%{time hh:mm:ss.zzz} %{type} %{category} %{function}:%{line} - %{message}"
    $env:QT_LOGGING_RULES = "qt.qpa.*=true;qt.qml.*=true;qt.quick.*=true"
    $env:PYTHONFAULTHANDLER = "1"
  }
  Write-Host "Qt env:"
  Write-Host "  QT_QPA_PLATFORM=$env:QT_QPA_PLATFORM"
  Write-Host "  QT_OPENGL=$env:QT_OPENGL"
  Write-Host "  QT_QUICK_BACKEND=$env:QT_QUICK_BACKEND"
  Write-Host "  QT_LOGGING_RULES=$env:QT_LOGGING_RULES"
  Write-Host "  QT_MESSAGE_PATTERN=$env:QT_MESSAGE_PATTERN"
  Write-Host "Python env:"
  Write-Host "  PYTHONFAULTHANDLER=$env:PYTHONFAULTHANDLER"
  & python -c "import sys; print('python:', sys.executable)"
  & python -m pip --version
  Write-Host "PYTEST_DISABLE_PLUGIN_AUTOLOAD exists in PS? " (Test-Path Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD)
  & python -c "import os; print('PYTEST_DISABLE_PLUGIN_AUTOLOAD in child:', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD' in os.environ, os.environ.get('PYTEST_DISABLE_PLUGIN_AUTOLOAD'))"
  & python -m pip show pytest-xdist
  $ver = & python -m pytest --version
  if ($ver -notmatch "(?i)\\bxdist\\b") {
    Write-Warning "Preflight: xdist nie pojawia się w 'pytest --version'."
  }
  $getPytestHelp = {
    param(
      [string]$ciIni,
      [string[]]$extraArgs = @(),
      [switch]$fullySilent
    )
    $outputText = ""
    if ($fullySilent) {
      $tempPath = [System.IO.Path]::GetTempFileName()
      try {
        & python -m pytest -c $ciIni @extraArgs -h 1>$tempPath 2>$null
        $outputText = Get-Content -Path $tempPath -Raw -ErrorAction SilentlyContinue
      } finally {
        Remove-Item -Path $tempPath -ErrorAction SilentlyContinue
      }
    } else {
      $output = & python -m pytest -c $ciIni @extraArgs -h 2>$null
      $outputText = ($output -join "`n")
    }
    return ($outputText -replace "`r`n", "`n" -replace "`r", "`n")
  }
  $helpBase = & $getPytestHelp -ciIni $ciIni
  $helpXdist = ""
  $helpForked = ""
  $probeErrorPreference = $ErrorActionPreference
  try {
    $ErrorActionPreference = "Continue"
    try {
      $helpXdist = & $getPytestHelp -ciIni $ciIni -extraArgs @("-p", "xdist.plugin") -fullySilent
    } catch {
      $helpXdist = ""
    }
    try {
      $helpForked = & $getPytestHelp -ciIni $ciIni -extraArgs @("-p", "pytest_forked") -fullySilent
    } catch {
      $helpForked = ""
    }
  } finally {
    $ErrorActionPreference = $probeErrorPreference
  }
  $boxedBase = $helpBase | Select-String -Quiet -Pattern "\-\-boxed"
  $boxedXdist = $helpXdist | Select-String -Quiet -Pattern "\-\-boxed"
  $forkedForked = $helpForked | Select-String -Quiet -Pattern "\-\-forked"
  $autoloadAllowed = -not (Test-Path Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD)
  $xdistImportOk = $false
  if ($boxedXdist -or $boxedBase) {
    & python -c "import xdist.plugin" 2>$null
    if ($LASTEXITCODE -eq 0) {
      $xdistImportOk = $true
    } else {
      Write-Warning "Preflight: nie mogę zaimportować xdist.plugin."
    }
  }
  $isolationAdded = $false
  $isolationSelected = "none"
  if ($boxedXdist -and $xdistImportOk) {
    $pytestArgs += @("-p", "xdist.plugin", "--boxed")
    $isolationAdded = $true
    $isolationSelected = "xdist(-p)"
  } elseif ($boxedBase) {
    if ($autoloadAllowed -and $xdistImportOk) {
      $pytestArgs += @("--boxed")
      $isolationAdded = $true
      $isolationSelected = "xdist(autoload)"
    } elseif ($xdistImportOk) {
      $pytestArgs += @("-p", "xdist.plugin", "--boxed")
      $isolationAdded = $true
      $isolationSelected = "xdist(-p)"
    }
  }
  if (-not $isolationAdded) {
    if ($forkedForked) {
      $pytestArgs += @("-p", "pytest_forked", "--forked")
      $isolationAdded = $true
      $isolationSelected = "forked"
    } else {
      Write-Warning "pytest isolation unavailable: --boxed and --forked not detected; running without isolation."
    }
  }
  if ($env:CI) {
    $pytestArgs += @("--cache-clear")
    $hasTimeout = $helpBase | Select-String -Quiet -Pattern "\-\-timeout="
    if ($hasTimeout) {
      $pytestArgs += @("--timeout=300", "--timeout-method=thread")
    } else {
      Write-Warning "pytest-timeout not available; skipping global timeout."
    }
  }

  Write-Host "pytest config override: -c $ciIni"
  Write-Host ("pytest isolation selected: {0}" -f $isolationSelected)
  if ($env:QML_TEST_MAXFAIL) {
    if ($maxfailInputValid) {
      Write-Host "pytest effective: --maxfail=$maxfail (input=$($env:QML_TEST_MAXFAIL) valid)"
    } else {
      Write-Host "pytest effective: --maxfail=$maxfail (input=$($env:QML_TEST_MAXFAIL) invalid)"
    }
  } else {
    Write-Host "pytest effective: --maxfail=$maxfail"
  }
  if ($env:QML_TEST_K) {
    Write-Host ("pytest effective: -k={0}" -f (& $formatArgForLog $env:QML_TEST_K))
  } else {
    Write-Host "pytest effective: -k=<none>"
  }
  $displayArgs = foreach ($arg in $pytestArgs) {
    & $formatArgForLog $arg
  }
  Write-Host ("pytest args: {0}" -f ($displayArgs -join " "))
  $procdump = Get-Command procdump.exe -ErrorAction SilentlyContinue
  if ($procdump) {
    Write-Host "Running QML tests with ProcDump ($($procdump.Source))."
    $procdumpVersion = (Get-Item $procdump.Source).VersionInfo.FileVersion
    if ($procdumpVersion) {
      Write-Host "ProcDump version: $procdumpVersion"
    }
    $procdumpExceptionArgs = @("-e")
    $fc = ""
    if ($env:PROC_DUMP_FIRST_CHANCE) {
      $fc = $env:PROC_DUMP_FIRST_CHANCE.Trim().ToLowerInvariant()
    }
    if ($fc -and $fc -ne "0" -and $fc -ne "false" -and $fc -ne "no" -and $fc -ne "off") {
      $procdumpExceptionArgs = @("-e", "1")
    }
    $procdumpArgs = @(
      "-accepteula",
      "-ma"
    ) + $procdumpExceptionArgs + @(
      "-n", "1",
      "-x", $CrashDumpDir,
      "--",
      "python",
      "-m",
      "pytest"
    ) + $pytestArgs
    & $procdump.Source @procdumpArgs 2>&1 | Tee-Object -FilePath $consoleLog
    $exitCode = $LASTEXITCODE
  } else {
    Write-Warning "ProcDump not found in PATH. Install from https://learn.microsoft.com/sysinternals/downloads/procdump"
    Write-Host "Running QML tests without crash dump capture."
    & python -m pytest @pytestArgs 2>&1 | Tee-Object -FilePath $consoleLog
    $exitCode = $LASTEXITCODE
  }
  Write-Host "Crashdump dir content:"
  Get-ChildItem -Path $CrashDumpDir -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object Name, Length, LastWriteTime |
    Format-Table -AutoSize
  $logPatterns = @(
    "Fatal Python error",
    "access violation",
    "0xC0000005",
    "EXCEPTION_RECORD",
    "STACK_TEXT",
    "Qt FATAL",
    "QML Import",
    "QSG",
    "D3D",
    "OpenGL"
  )
  $logFiles = @($consoleLog, $logFilePath) | Where-Object { $_ -and (Test-Path $_) } | Select-Object -Unique
  foreach ($logFile in $logFiles) {
    Write-Host "Crash signature scan for $logFile"
    $matches = Select-String -Path $logFile -Pattern $logPatterns -SimpleMatch -CaseSensitive:$false -ErrorAction SilentlyContinue
    if ($matches) {
      $matches | Select-Object -Last 20 | ForEach-Object {
        Write-Host ("[{0}:{1}] ({2}) {3}" -f (Split-Path $logFile -Leaf), $_.LineNumber, $_.Pattern, $_.Line)
      }
    } else {
      Write-Host "No crash signature matches found."
    }
    Write-Host "Tail 20 lines from $logFile"
    Get-Content -Path $logFile -Tail 20 -ErrorAction SilentlyContinue
  }
  Write-Host "pytest/procdump exitCode=$exitCode"
} finally {
  if ($null -ne $prevPluginAutoload -and $prevPluginAutoload -ne "") {
    $env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = $prevPluginAutoload
  } else {
    Remove-Item Env:PYTEST_DISABLE_PLUGIN_AUTOLOAD -ErrorAction SilentlyContinue
  }
}
exit $exitCode
